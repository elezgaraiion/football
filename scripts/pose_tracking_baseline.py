"""
Baseline: pose detection + persistent tracking + team classification on a match clip.

Offside-detection pipeline, Month 1:
  - Detect every player in a video and estimate their body keypoints.
  - Assign each player a persistent track ID that survives across frames,
    even through occlusion or players crossing paths.
  - Classify every player's team from the shirt colour, and repair ID swaps between
    rival players in duels (see team_classification.py).
  - Repair IDs that the online tracker split, with offline tracklet stitching that never
    joins players of different teams (see track_stitching.py).
  - Export (a) an annotated video with final IDs and teams for a visual sanity check,
    (b) a structured JSON with per-frame keypoints, team and track ID, which the
    homography and offside stages will consume, and
    (c) a MOTChallenge-format text file, so tracking can be scored against
    SoccerNet ground truth (HOTA / IDF1 / ID switches) with TrackEval.

team_classification.py and track_stitching.py must sit next to this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from team_classification import (TeamConfig, TeamModel, grass_hue, repair_team_swaps, torso_lab, track_purity,
                                 vote_teams)
from track_stitching import StitchConfig, stitch_tracks

# COCO-17 skeleton (0-based keypoint indices), the same one Ultralytics draws
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
            (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

# Drawing colours (BGR) and labels per team; marker colours, not the kit colours
TEAM_STYLE = {
    "A": ((255, 255, 0), "A"),         # cyan
    "B": ((255, 0, 255), "B"),         # magenta
    "other": ((0, 255, 255), "otro"),  # yellow: referees, goalkeepers, staff
    None: ((200, 200, 200), "?"),      # grey: not enough colour evidence
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose + tracking + team baseline for match clips")
    parser.add_argument("--source", required=True, help="Path to the input video (.mp4/.mkv)")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="Ultralytics pose checkpoint")
    parser.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="bytetrack.yaml, botsort.yaml, or the path to your own tuned tracker YAML. "
        "BoT-SORT compensates for camera motion, which matters on panning broadcast footage.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Detection confidence threshold. Keep it at or below the tracker's track_low_thresh "
        "(0.1 by default) so its second, low-score association stage receives detections. "
        "To limit new IDs, raise new_track_thresh in the tracker YAML instead.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution in pixels")
    parser.add_argument(
        "--kpt-thresh",
        type=float,
        default=0.5,
        help="Keypoints with a visibility score below this are exported as null (their score is kept).",
    )
    parser.add_argument(
        "--stitch-max-gap",
        type=float,
        default=3.0,
        help="Join a track that ends with one that starts up to this many seconds later at a "
        "compatible position (offline tracklet stitching). 0 disables it.",
    )
    parser.add_argument("--no-teams", action="store_true", help="Disable team classification and swap repair")
    parser.add_argument(
        "--team-min-run",
        type=int,
        default=5,
        help="Labelled frames in a row needed for a team change along a track to count as an ID swap.",
    )
    parser.add_argument("--device", default=None, help="Inference device, e.g. 0 or cpu (default: auto)")
    parser.add_argument("--half", action="store_true", help="FP16 inference (roughly 2x faster on CUDA GPUs)")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output files")
    return parser.parse_args()


def keypoints_for(result, n: int, thresh: float) -> tuple[list, list]:
    """Per-detection keypoints as (xy, conf), with xy = None for keypoints scored below `thresh`.

    Pose models still output a position for occluded or out-of-frame keypoints, just with a low
    visibility score (older Ultralytics versions set those to (0, 0) instead). Without the score,
    later stages can't tell a real ankle from a guessed one, which matters a lot for offside calls.
    """
    kp = result.keypoints
    if kp is None:
        return [None] * n, [None] * n

    xy = kp.xy.cpu().tolist()
    conf = kp.conf.cpu().tolist() if kp.conf is not None else None

    out_xy, out_conf = [], []
    for i, person in enumerate(xy):
        person_conf = conf[i] if conf is not None else [None] * len(person)
        out_xy.append([
            None if (c is not None and c < thresh) else [round(x, 1), round(y, 1)]
            for (x, y), c in zip(person, person_conf)
        ])
        out_conf.append(None if conf is None else [round(c, 3) for c in person_conf])
    return out_xy, out_conf


def draw_legend(img, team_model: TeamModel | None) -> None:
    """Top-left legend: marker colour, learnt kit colour (teams only) and label."""
    rows = [("A", "Equipo A"), ("B", "Equipo B"), ("other", "No encaja"), (None, "Sin clasificar")]
    font = cv2.FONT_HERSHEY_SIMPLEX
    x2, y2 = 230, 16 + 26 * len(rows)
    img[8:y2, 8:x2] = (img[8:y2, 8:x2] * 0.45 + 30 * 0.55).astype(img.dtype)   # translucent background
    for i, (team, text) in enumerate(rows):
        y = 16 + 26 * i
        cv2.rectangle(img, (16, y), (34, y + 18), TEAM_STYLE[team][0], -1)
        if team_model is not None and team in ("A", "B"):
            cv2.rectangle(img, (40, y), (58, y + 18), team_model.bgr(team), -1)
            cv2.rectangle(img, (40, y), (58, y + 18), (255, 255, 255), 1)
        cv2.putText(img, text, (66, y + 14), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def render_video(source_path: Path, frames: list[dict], out_path: Path, fps: float, size: tuple[int, int],
                 team_model: TeamModel | None, show_teams: bool) -> None:
    """Draw boxes, skeletons, final IDs and teams on top of the source video."""
    cap = cv2.VideoCapture(str(source_path))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(f"OpenCV could not read {source_path} or create {out_path}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        for fr in tqdm(frames, desc="Dibujando vídeo", unit="frame"):
            ok, img = cap.read()
            if not ok:
                break
            for p in fr["players"]:
                color, team_text = TEAM_STYLE[p.get("team")]
                x1, y1, x2, y2 = (int(round(v)) for v in p["bbox_xyxy"])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                kpts = p.get("keypoints_xy") or []
                for i, j in SKELETON:
                    if i < len(kpts) and j < len(kpts) and kpts[i] and kpts[j]:
                        pt_i = (int(kpts[i][0]), int(kpts[i][1]))
                        pt_j = (int(kpts[j][0]), int(kpts[j][1]))
                        cv2.line(img, pt_i, pt_j, color, 2, cv2.LINE_AA)
                for pt in kpts:
                    if pt:
                        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color, -1, cv2.LINE_AA)

                label = f"ID {p['track_id']} {team_text}" if show_teams else f"ID {p['track_id']}"
                (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
                top = max(y1 - th - 8, 0)
                cv2.rectangle(img, (x1, top), (x1 + tw + 6, top + th + 8), color, -1)
                cv2.putText(img, label, (x1 + 3, top + th + 3), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            if show_teams:
                draw_legend(img, team_model)
            writer.write(img)
    finally:
        cap.release()
        writer.release()


def to_h264(src: Path, dst: Path) -> None:
    """OpenCV's mp4v output doesn't play in browsers or Colab; re-encode to H.264 when ffmpeg exists."""
    if shutil.which("ffmpeg") is None:
        src.replace(dst)
        print("ffmpeg not found: annotated video left as mp4v (may not play in a browser).")
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(dst)],
        check=True,
    )
    src.unlink()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_video_path = output_dir / f"{source_path.stem}_tracked.mp4"
    raw_video_path = output_dir / f"{source_path.stem}_tracked.mp4v.mp4"
    keypoints_json_path = output_dir / f"{source_path.stem}_keypoints.json"
    mot_path = output_dir / f"{source_path.stem}_mot.txt"

    probe = cv2.VideoCapture(str(source_path))
    if not probe.isOpened():
        raise FileNotFoundError(f"OpenCV could not open video: {source_path}")

    fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()

    use_teams = not args.no_teams
    team_cfg = TeamConfig(min_run=args.team_min_run)
    model = YOLO(args.model)
    precision = {"quantize": 16} if args.half else {}   # FP16; replaces the deprecated `half` argument

    # --- 1. Detection + online tracking (+ shirt colour of every detection) ---
    frames_out: list[dict] = []
    for frame_idx, result in enumerate(tqdm(
        model.track(
            source=str(source_path),
            conf=args.conf,
            imgsz=args.imgsz,
            tracker=args.tracker,
            device=args.device,
            persist=True,
            stream=True,
            verbose=False,
            **precision,
        ),
        total=total_frames or None,
        desc="Procesando vídeo",
        unit="frame",
    )):
        frame_record: dict = {"frame": frame_idx, "players": []}
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            boxes_xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            kpts_xy, kpts_conf = keypoints_for(result, len(track_ids), args.kpt_thresh)
            grass_h = grass_hue(result.orig_img) if use_teams else None

            for i, (track_id, box, conf, xy, kc) in enumerate(zip(track_ids, boxes_xyxy, confs, kpts_xy, kpts_conf)):
                others = boxes_xyxy[:i] + boxes_xyxy[i + 1:]
                frame_record["players"].append({
                    "track_id": track_id,
                    "raw_track_id": track_id,
                    "conf": round(conf, 3),
                    "bbox_xyxy": [round(v, 1) for v in box],
                    "keypoints_xy": xy,
                    "keypoints_conf": kc,
                    "torso_lab": torso_lab(result.orig_img, box, xy, others, grass_h, team_cfg) if use_teams else None,
                })

        frames_out.append(frame_record)
    n_frames = len(frames_out)
    raw_ids = {p["track_id"] for fr in frames_out for p in fr["players"]}

    # --- 2. Teams: learn the two kits, label every detection, repair swaps between rivals ---
    team_model, swaps, splits, team_skip_reason = None, [], [], None
    if use_teams:
        team_model = TeamModel.fit([p["torso_lab"] for fr in frames_out for p in fr["players"] if p["torso_lab"]],
                                   team_cfg)
        if team_model is None:
            team_skip_reason = "not enough measurable shirts to learn the two kits"
        elif not team_model.separated:
            kits = team_model.to_dict()
            team_skip_reason = (f"the two shirt colours learnt are not clearly separated "
                                f"(A = Lab{tuple(kits['A']['lab'])}, B = Lab{tuple(kits['B']['lab'])}, "
                                f"valley ratio {kits['valley_ratio']:.2f} >= {team_cfg.max_valley_ratio}), "
                                f"so team labels would be unreliable")
            team_model = None
    purity = None
    for fr in frames_out:
        for p in fr["players"]:
            p["team_frame"] = team_model.classify(p["torso_lab"]) if team_model else None
    if team_model:
        purity = track_purity(frames_out, team_cfg.min_votes)
        if purity < team_cfg.min_track_purity:
            team_skip_reason = (f"the colour groups don't follow the players (only {purity:.0%} of each track's "
                                f"labels agree with its own team, below {team_cfg.min_track_purity:.0%}), "
                                f"so they are probably lighting changes rather than kits")
            team_model = None
            for fr in frames_out:
                for p in fr["players"]:
                    p["team_frame"] = None
    if team_model:
        frames_out, swaps, splits = repair_team_swaps(frames_out, fps, team_cfg)
    for fr in frames_out:
        for p in fr["players"]:
            p["segment_id"] = p["track_id"]
    segment_ids = {p["track_id"] for fr in frames_out for p in fr["players"]}
    segment_teams = vote_teams(frames_out, team_cfg.min_votes) if team_model else None

    # --- 3. Offline tracklet stitching (never across teams) ---
    stitch_cfg = StitchConfig(max_gap_s=args.stitch_max_gap)
    links = []
    if args.stitch_max_gap > 0:
        frames_out, links = stitch_tracks(frames_out, fps, stitch_cfg, segment_teams)
    final_teams = vote_teams(frames_out, team_cfg.min_votes) if team_model else {}
    for fr in frames_out:
        for p in fr["players"]:
            p["team"] = final_teams.get(p["track_id"])

    # --- 4. Outputs ---
    render_video(source_path, frames_out, raw_video_path, fps, (width, height), team_model, team_model is not None)
    to_h264(raw_video_path, annotated_video_path)

    with open(keypoints_json_path, "w") as f:
        json.dump({
            "source": str(source_path),
            "fps": fps,
            "source_width": width,
            "source_height": height,
            "model": args.model,
            "tracker": args.tracker,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "keypoint_visibility_threshold": args.kpt_thresh,
            "teams": {
                "enabled": team_model is not None,
                "skipped_reason": team_skip_reason,
                "track_purity": None if purity is None else round(purity, 3),
                "config": asdict(team_cfg),
                "model": team_model.to_dict() if team_model else None,
                "swaps": swaps,
                "splits": splits,
                "track_teams": {str(k): v for k, v in sorted(final_teams.items())},
            },
            "stitching": {
                "enabled": args.stitch_max_gap > 0,
                "config": asdict(stitch_cfg),
                "links": [asdict(link) for link in links],
            },
            "frames": frames_out,
        }, f)

    # MOTChallenge: frame (1-based), id, left, top, width, height, conf, -1, -1, -1
    mot_lines = [
        f"{fr['frame'] + 1},{p['track_id']},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{p['conf']:.3f},-1,-1,-1"
        for fr in frames_out
        for p in fr["players"]
        for x1, y1, x2, y2 in [p["bbox_xyxy"]]
    ]
    mot_path.write_text("\n".join(mot_lines) + ("\n" if mot_lines else ""))

    # --- Summary ---
    track_lengths = Counter(p["track_id"] for fr in frames_out for p in fr["players"])
    frames_with_detections = sum(1 for fr in frames_out if fr["players"])
    detection_rate = frames_with_detections / n_frames if n_frames else 0.0
    avg_players = sum(len(fr["players"]) for fr in frames_out) / n_frames if n_frames else 0.0
    short_tracks = sum(1 for n in track_lengths.values() if n < fps)
    segment_start = {}
    for fr in frames_out:
        for p in fr["players"]:
            segment_start.setdefault(p["segment_id"], fr["frame"])

    print(f"\nProcessed {n_frames} frames ({width}x{height} @ {fps:.1f}fps, imgsz={args.imgsz}).")
    print(f"Frames with >=1 player detected: {frames_with_detections}/{n_frames} ({detection_rate:.1%})")
    print(f"Tracked players per frame (mean): {avg_players:.1f}")

    if team_model:
        labels = Counter(p["team_frame"] for fr in frames_out for p in fr["players"])
        total = sum(labels.values()) or 1
        kits = team_model.to_dict()
        print(f"\nTeams learnt from the clip: A = Lab{tuple(kits['A']['lab'])}, B = Lab{tuple(kits['B']['lab'])} "
              f"(valley ratio {kits['valley_ratio']:.2f}, lower = better separated; "
              f"track purity {purity:.0%}, higher = labels follow the players)")
        print("Detections per frame label: " + ", ".join(
            f"{name} {labels.get(key, 0) / total:.0%}"
            for key, name in [("A", "A"), ("B", "B"), ("other", "other"), (None, "unclassified")]))
        if min(team_model.sizes) < 0.15 * sum(team_model.sizes):
            print("WARNING: one team colour has very few detections; check the legend colours in the video.")
        print(f"Team swaps repaired: {len(swaps)}")
        for s in swaps:
            print(f"  t={s['frame'] / fps:5.1f}s  IDs {s['track_ids'][0]} <-> {s['track_ids'][1]}  "
                  f"({' / '.join(s['teams'])}, {s['distance']:.2f} body heights apart)")
        print(f"Team changes without a partner (track split): {len(splits)}")
        for s in splits:
            print(f"  t={s['frame'] / fps:5.1f}s  ID {s['track_id']} split -> new ID {s['new_track_id']} ({s['teams']})")
        team_counts = Counter(final_teams.values())
        print(f"Final tracks per team: A {team_counts.get('A', 0)}, B {team_counts.get('B', 0)}, "
              f"other {team_counts.get('other', 0)}, unclassified {team_counts.get(None, 0)}")
    elif use_teams:
        print(f"\nTeams: {team_skip_reason}; team steps skipped.")

    if args.stitch_max_gap > 0:
        print(f"\nStitching links: {len(links)}")
        for link in sorted(links, key=lambda l: segment_start.get(l.to_id, 0)):
            final = next(p["track_id"] for fr in frames_out for p in fr["players"] if p["segment_id"] == link.to_id)
            print(f"  t={segment_start.get(link.to_id, 0) / fps:5.1f}s  segment {link.to_id} -> ID {final}  "
                  f"(gap {link.gap_frames} frames, {link.distance:.2f} body heights)")

    print(f"\nUnique track IDs: {len(raw_ids)} from the tracker -> {len(segment_ids)} after team repair "
          f"-> {len(track_lengths)} final")
    if track_lengths:
        print(
            f"Median track length: {statistics.median(track_lengths.values()):.0f} frames; "
            f"tracks shorter than 1 s: {short_tracks} (many short tracks = fragmentation / ID switches)"
        )
    print(f"Annotated video -> {annotated_video_path}")
    print(f"Keypoints JSON  -> {keypoints_json_path}")
    print(f"MOT tracks      -> {mot_path}")


if __name__ == "__main__":
    main()
