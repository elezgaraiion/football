"""
Baseline: pose detection + persistent tracking on a match clip.

Week 1-2 goal (Month 1 of the offside-detection pipeline):
  - Detect every player in a video and estimate their body keypoints.
  - Assign each player a persistent track ID that survives across frames,
    even through occlusion or players crossing paths.
  - Export (a) an annotated video for a visual sanity check,
    (b) a structured JSON with per-frame keypoints per track ID, which
    Week 3 (team clustering) and Week 4 (homography) will consume, and
    (c) a MOTChallenge-format text file, so tracking can be scored against
    SoccerNet ground truth (HOTA / IDF1 / ID switches) with TrackEval.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


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
        default=0.3,
        help="Detection confidence threshold. Keep it at or below the tracker's track_low_thresh "
        "(0.1 by default) so its second, low-score association stage receives detections.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution in pixels")
    parser.add_argument(
        "--kpt-thresh",
        type=float,
        default=0.5,
        help="Keypoints with a visibility score below this are exported as null (their score is kept).",
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

    writer = cv2.VideoWriter(str(raw_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create the output video: {raw_video_path}")
    model = YOLO(args.model)

    frames_out: list[dict] = []
    mot_lines: list[str] = []
    frame_idx = 0

    try:
        for result in tqdm(
            model.track(
                source=str(source_path),
                conf=args.conf,
                imgsz=args.imgsz,
                tracker=args.tracker,
                device=args.device,
                half=args.half,
                persist=True,
                stream=True,
                verbose=False,
            ),
            total=total_frames or None,
            desc="Procesando vídeo",
            unit="frame",
        ):
            writer.write(result.plot())
            frame_record: dict = {"frame": frame_idx, "players": []}
            boxes = result.boxes

            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                boxes_xyxy = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                kpts_xy, kpts_conf = keypoints_for(result, len(track_ids), args.kpt_thresh)

                for track_id, box, conf, xy, kc in zip(track_ids, boxes_xyxy, confs, kpts_xy, kpts_conf):
                    x1, y1, x2, y2 = box
                    frame_record["players"].append({
                        "track_id": track_id,
                        "conf": round(conf, 3),
                        "bbox_xyxy": [round(v, 1) for v in box],
                        "keypoints_xy": xy,
                        "keypoints_conf": kc,
                    })
                    # MOTChallenge: frame (1-based), id, left, top, width, height, conf, -1, -1, -1
                    mot_lines.append(
                        f"{frame_idx + 1},{track_id},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{conf:.3f},-1,-1,-1"
                    )

            frames_out.append(frame_record)
            frame_idx += 1
    finally:
        writer.release()

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
            "frames": frames_out,
        }, f)

    mot_path.write_text("\n".join(mot_lines) + ("\n" if mot_lines else ""))

    # --- Summary ---
    track_lengths = Counter(p["track_id"] for fr in frames_out for p in fr["players"])
    frames_with_detections = sum(1 for fr in frames_out if fr["players"])
    detection_rate = frames_with_detections / frame_idx if frame_idx else 0.0
    avg_players = sum(len(fr["players"]) for fr in frames_out) / frame_idx if frame_idx else 0.0
    short_tracks = sum(1 for n in track_lengths.values() if n < fps)

    print(f"\nProcessed {frame_idx} frames ({width}x{height} @ {fps:.1f}fps, imgsz={args.imgsz}).")
    print(f"Frames with >=1 player detected: {frames_with_detections}/{frame_idx} ({detection_rate:.1%})")
    print(f"Tracked players per frame (mean): {avg_players:.1f}")
    print(f"Unique track IDs seen: {len(track_lengths)}")
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
