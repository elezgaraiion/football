"""
Baseline: pose detection + persistent identities + team classification on a match clip.

Offside-detection pipeline, Month 1:
  - Detect every player in a video and estimate their body keypoints.
  - Classify every player's team from kit colours (shirt + shorts), with appearance groups joined
    into teams through trajectories, so a big number in the other team's colour doesn't fool it
    (team_classification.py).
  - Give each player one ID from the moment they appear, kept through duels, crossings and
    short absences by position continuity and team (identity_tracking.py). The tracker's IDs
    are followed where they are reliable and overridden where they are not.
  - Export (a) an annotated video with final IDs and teams for a visual sanity check,
    (b) a structured JSON with per-frame keypoints, team and ID, which the homography and
    offside stages will consume, and
    (c) a MOTChallenge-format text file, so tracking can be scored against
    SoccerNet ground truth (HOTA / IDF1 / ID switches) with TrackEval.

team_classification.py and identity_tracking.py must sit next to this script.
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

from identity_tracking import IdentityConfig, assign_identities
from team_classification import (TeamConfig, TeamModel, grass_hsv, label_teams, lab_to_bgr, shorts_color,
                                 torso_color, vote_teams)

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
MAX_LISTED_EVENTS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose + identities + teams baseline for match clips")
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
        "--max-gap",
        type=float,
        default=3.0,
        help="A player who disappears keeps their ID for up to this many seconds and gets it back if they "
        "reappear near where they were expected.",
    )
    parser.add_argument("--no-teams", action="store_true", help="Don't classify teams (identities use position only)")
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


def draw_legend(img, team_colors: dict) -> None:
    """Top-left legend: marker colour, learnt kit colour (teams only) and label."""
    rows = [("A", "Equipo A"), ("B", "Equipo B"), ("other", "No encaja"), (None, "Sin clasificar")]
    font = cv2.FONT_HERSHEY_SIMPLEX
    x2, y2 = 230, 16 + 26 * len(rows)
    img[8:y2, 8:x2] = (img[8:y2, 8:x2] * 0.45 + 30 * 0.55).astype(img.dtype)   # translucent background
    for i, (team, text) in enumerate(rows):
        y = 16 + 26 * i
        cv2.rectangle(img, (16, y), (34, y + 18), TEAM_STYLE[team][0], -1)
        if team in team_colors:
            cv2.rectangle(img, (40, y), (58, y + 18), lab_to_bgr(team_colors[team]), -1)
            cv2.rectangle(img, (40, y), (58, y + 18), (255, 255, 255), 1)
        cv2.putText(img, text, (66, y + 14), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def render_video(source_path: Path, frames: list[dict], out_path: Path, fps: float, size: tuple[int, int],
                 team_colors: dict | None) -> None:
    """Draw boxes, skeletons, final IDs and teams on top of the source video."""
    cap = cv2.VideoCapture(str(source_path))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(f"OpenCV could not read {source_path} or create {out_path}")

    show_teams = team_colors is not None
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
                draw_legend(img, team_colors)
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


def print_events(title: str, events: list[dict], fps: float, fmt) -> None:
    print(f"{title}: {len(events)}")
    for e in sorted(events, key=lambda e: e["frame"])[:MAX_LISTED_EVENTS]:
        print(f"  t={e['frame'] / fps:5.1f}s  {fmt(e)}")
    if len(events) > MAX_LISTED_EVENTS:
        print(f"  ... {len(events) - MAX_LISTED_EVENTS} more in the JSON")


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
    team_cfg = TeamConfig()
    id_cfg = IdentityConfig(max_gap_s=args.max_gap)
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
            grass = grass_hsv(result.orig_img) if use_teams else None

            for i, (track_id, box, conf, xy, kc) in enumerate(zip(track_ids, boxes_xyxy, confs, kpts_xy, kpts_conf)):
                others = boxes_xyxy[:i] + boxes_xyxy[i + 1:]
                frame_record["players"].append({
                    "track_id": track_id,
                    "raw_track_id": track_id,
                    "conf": round(conf, 3),
                    "bbox_xyxy": [round(v, 1) for v in box],
                    "keypoints_xy": xy,
                    "keypoints_conf": kc,
                    "torso_color": torso_color(result.orig_img, box, xy, others, grass, team_cfg) if use_teams else None,
                    "shorts_color": shorts_color(result.orig_img, box, xy, others, grass, team_cfg) if use_teams else None,
                })

        frames_out.append(frame_record)
    n_frames = len(frames_out)
    raw_ids = {p["raw_track_id"] for fr in frames_out for p in fr["players"]}

    # --- 2. Teams: learn the two kits from the clip, then label every detection using appearance
    #        groups joined into teams by trajectories (robust to numbers in the other team's colour) ---
    team_model, team_skip_reason = None, None
    if use_teams:
        team_model = TeamModel.fit([p["torso_color"] for fr in frames_out for p in fr["players"] if p["torso_color"]],
                                   team_cfg)
        if team_model is None:
            team_skip_reason = "not enough measurable shirts to learn the two kits"
        elif not team_model.separated:
            kits = team_model.to_dict()
            team_skip_reason = (f"the two shirt colours learnt are not clearly separated "
                                f"(A = Lab{tuple(kits['A']['lab'])}, B = Lab{tuple(kits['B']['lab'])}, "
                                f"valley ratio {kits['valley_ratio']:.2f} >= {team_cfg.max_valley_ratio})")
            team_model = None
    for fr in frames_out:
        for p in fr["players"]:
            p["team_kit"] = team_model.classify(p["torso_color"]) if team_model else None   # nearest kit only
    team_labels = None
    if team_model:
        team_labels = label_teams(frames_out, team_model, team_cfg)   # team_frame + team_weight per detection
        if not team_labels.supported:
            team_skip_reason = team_labels.reason
            team_model = None
    if team_model is None:
        for fr in frames_out:
            for p in fr["players"]:
                p["appearance_group"], p["team_frame"], p["team_weight"] = None, None, 0.0
    groups = team_labels.groups if team_model else []
    team_colors = team_labels.colors if team_model else {}

    # --- 3. One identity per player (tracker continuity + position + team) ---
    frames_out, id_events = assign_identities(frames_out, fps, id_cfg, label_key="team_frame",
                                              weight_key="team_weight")
    final_teams = vote_teams(frames_out, team_cfg.min_votes, src="team_frame") if team_model else {}
    for fr in frames_out:
        for p in fr["players"]:
            p["team"] = final_teams.get(p["track_id"])

    # --- 4. Outputs ---
    render_video(source_path, frames_out, raw_video_path, fps, (width, height),
                 team_colors if team_model else None)
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
                "trajectory_split_cut": None if team_labels is None or team_labels.ncut is None
                else round(team_labels.ncut, 3),
                "config": asdict(team_cfg),
                "model": team_model.to_dict() if team_model else None,
                "appearance_groups": [g.to_dict() for g in groups],
                "track_teams": {str(k): v for k, v in sorted(final_teams.items())},
            },
            "identities": {
                "config": asdict(id_cfg),
                **id_events,
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

    print(f"\nProcessed {n_frames} frames ({width}x{height} @ {fps:.1f}fps, imgsz={args.imgsz}).")
    print(f"Frames with >=1 player detected: {frames_with_detections}/{n_frames} ({detection_rate:.1%})")
    print(f"Tracked players per frame (mean): {avg_players:.1f}")

    if team_model:
        kits = team_model.to_dict()
        total = sum(len(fr["players"]) for fr in frames_out) or 1
        share = lambda key: ", ".join(
            f"{name} {sum(p[key] == k for fr in frames_out for p in fr['players']) / total:.0%}"
            for k, name in [("A", "A"), ("B", "B"), ("other", "other"), (None, "unclassified")])
        fmt_lab = lambda lab: f"Lab({lab[0]:.0f},{lab[1]:.0f},{lab[2]:.0f})" if lab else "-"
        print(f"\nTeams learnt from the clip: A = {fmt_lab(team_colors.get('A'))}, B = {fmt_lab(team_colors.get('B'))} "
              f"(main shirt colour of each team; kit colours valley ratio {kits['valley_ratio']:.2f}, "
              f"lower = better separated)")
        print(f"Team split respected by isolated players: cut {team_labels.ncut:.2f} "
              f"(0 = no isolated player ever crosses it; must be <= {team_cfg.max_ncut})")
        print("Appearance groups (shirt / shorts) joined into teams through isolated trajectories:")
        for g in sorted(groups, key=lambda g: (str(g.team), -g.size)):
            shorts = "-" if g.shorts is None else f"Lab({g.shorts[0]:.0f},{g.shorts[1]:.0f},{g.shorts[2]:.0f})"
            print(f"  group {g.id}: team {g.team:<5} {g.size:5d} detections  shirt Lab({g.shirt[0]:.0f},{g.shirt[1]:.0f},"
                  f"{g.shirt[2]:.0f}) light {g.shirt[3]:.0f}% dark {g.shirt[4]:.0f}%  shorts {shorts}  "
                  f"confidence {g.confidence:.2f}")
        print(f"Labels by nearest kit only:      {share('team_kit')}")
        print(f"Labels used (groups + segments): {share('team_frame')}")
        if min(team_model.sizes) < 0.15 * sum(team_model.sizes):
            print("WARNING: one team colour has very few detections; check the legend colours in the video.")
        team_counts = Counter(final_teams.values())
        print(f"Final IDs per team: A {team_counts.get('A', 0)}, B {team_counts.get('B', 0)}, "
              f"other {team_counts.get('other', 0)}, unclassified {team_counts.get(None, 0)}")
    elif use_teams:
        print(f"\nTeams: {team_skip_reason}; identities use position only.")

    print()
    print_events("Tracker ID changes overruled (the player kept his ID)", id_events["tracker_switches"], fps,
                 lambda e: f"ID {e['id']}: tracker ID {e['tracker_ids'][0]} -> {e['tracker_ids'][1]}")
    print_events("Players recovered under a new tracker ID", id_events["reidentifications"], fps,
                 lambda e: f"ID {e['id']} (tracker ID {e['tracker_ids'][0]} -> {e['tracker_ids'][1]}) back after "
                           f"{e['gap_frames']} frames, {e['distance']:.2f} body heights from where expected")

    print(f"\nUnique IDs: {len(raw_ids)} from the tracker -> {len(track_lengths)} final")
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
