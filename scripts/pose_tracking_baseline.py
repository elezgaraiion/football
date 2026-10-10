"""
Baseline: pose detection + persistent tracking on a match clip.

Week 1-2 goal (Month 1 of the offside-detection pipeline):
  - Detect every player in a video and estimate their body keypoints.
  - Assign each player a persistent track ID that survives across frames,
    even through occlusion or players crossing paths.
  - Repair IDs that the online tracker split, with offline tracklet stitching
    (see track_stitching.py, which must sit next to this script).
  - Export (a) an annotated video with the final IDs for a visual sanity check,
    (b) a structured JSON with per-frame keypoints per track ID, which
    Week 3 (team clustering) and Week 4 (homography) will consume, and
    (c) a MOTChallenge-format text file, so tracking can be scored against
    SoccerNet ground truth (HOTA / IDF1 / ID switches) with TrackEval.
"""

from __future__ import annotations

import argparse
import colorsys
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

from track_stitching import StitchConfig, stitch_tracks

# COCO-17 skeleton (0-based keypoint indices), the same one Ultralytics draws
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
            (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose + tracking baseline for match clips")
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


def id_color(track_id: int) -> tuple[int, int, int]:
    """A distinct, stable BGR colour per ID (golden-ratio hue spacing)."""
    r, g, b = colorsys.hsv_to_rgb((track_id * 0.618033988749895) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def render_video(source_path: Path, frames: list[dict], out_path: Path, fps: float, size: tuple[int, int]) -> None:
    """Draw boxes, skeletons and the final (stitched) IDs on top of the source video."""
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
                color = id_color(p["track_id"])
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

                label = f"ID {p['track_id']}"
                (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
                top = max(y1 - th - 8, 0)
                cv2.rectangle(img, (x1, top), (x1 + tw + 6, top + th + 8), color, -1)
                cv2.putText(img, label, (x1 + 3, top + th + 3), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
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

    model = YOLO(args.model)
    precision = {"quantize": 16} if args.half else {}   # FP16; replaces the deprecated `half` argument

    # --- 1. Detection + online tracking ---
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

            for track_id, box, conf, xy, kc in zip(track_ids, boxes_xyxy, confs, kpts_xy, kpts_conf):
                frame_record["players"].append({
                    "track_id": track_id,
                    "conf": round(conf, 3),
                    "bbox_xyxy": [round(v, 1) for v in box],
                    "keypoints_xy": xy,
                    "keypoints_conf": kc,
                })

        frames_out.append(frame_record)
    n_frames = len(frames_out)

    # --- 2. Offline tracklet stitching ---
    raw_ids = {p["track_id"] for fr in frames_out for p in fr["players"]}
    stitch_cfg = StitchConfig(max_gap_s=args.stitch_max_gap)
    if args.stitch_max_gap > 0:
        frames_out, links = stitch_tracks(frames_out, fps, stitch_cfg)
    else:
        links = []
        for fr in frames_out:
            for p in fr["players"]:
                p["raw_track_id"] = p["track_id"]

    # --- 3. Outputs ---
    render_video(source_path, frames_out, raw_video_path, fps, (width, height))
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

    print(f"\nProcessed {n_frames} frames ({width}x{height} @ {fps:.1f}fps, imgsz={args.imgsz}).")
    print(f"Frames with >=1 player detected: {frames_with_detections}/{n_frames} ({detection_rate:.1%})")
    print(f"Tracked players per frame (mean): {avg_players:.1f}")
    if args.stitch_max_gap > 0:
        print(f"Unique track IDs: {len(raw_ids)} from the tracker -> {len(track_lengths)} after stitching "
              f"({len(links)} links, max gap {args.stitch_max_gap:g} s)")
    else:
        print(f"Unique track IDs: {len(track_lengths)} (stitching disabled)")
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
