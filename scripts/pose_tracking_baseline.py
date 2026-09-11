```python
"""
Baseline: pose detection + persistent tracking on a match clip.

Week 1-2 goal (Month 1 of the offside-detection pipeline):
  - Detect every player in a video and estimate their body keypoints.
  - Assign each player a persistent track ID that survives across frames,
    even through occlusion or players crossing paths.
  - Export (a) an annotated video for a visual sanity check, and
    (b) a structured JSON with per-frame keypoints per track ID, which
    Week 3 (team clustering) and Week 4 (homography) will consume.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose + tracking baseline for match clips")
    parser.add_argument("--source", required=True, help="Path to the input video (.mp4/.mkv)")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="Ultralytics pose checkpoint")
    parser.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution in pixels")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output files")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_video_path = output_dir / f"{source_path.stem}_tracked.mp4"
    keypoints_json_path = output_dir / f"{source_path.stem}_keypoints.json"

    probe = cv2.VideoCapture(str(source_path))
    if not probe.isOpened():
        raise FileNotFoundError(f"OpenCV could not open video: {source_path}")
    
    fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()

    writer = cv2.VideoWriter(str(annotated_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model = YOLO(args.model)

    frames_out: list[dict] = []
    frame_idx = 0

    for result in tqdm(
        model.track(
            source=str(source_path),
            conf=args.conf,
            imgsz=args.imgsz,
            tracker=args.tracker,
            persist=True,
            stream=True,
            verbose=False,
        ),
        total=total_frames,
        desc="Procesando vídeo",
        unit="frame"
    ):
        writer.write(result.plot())
        frame_record: dict = {"frame": frame_idx, "players": []}
        has_tracks = result.boxes is not None and result.boxes.id is not None
        
        if has_tracks:
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes_xyxy = result.boxes.xyxy.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            keypoints_xy = result.keypoints.xy.cpu().tolist() if result.keypoints is not None else [None] * len(track_ids)

            for track_id, box, kpts, conf in zip(track_ids, boxes_xyxy, keypoints_xy, confs):
                frame_record["players"].append({
                    "track_id": track_id, "conf": round(conf, 3), "bbox_xyxy": box, "keypoints_xy": kpts
                })

        frames_out.append(frame_record)
        frame_idx += 1

    writer.release()

    with open(keypoints_json_path, "w") as f:
        json.dump({
            "source": str(source_path),
            "fps": fps,
            "source_width": width,
            "source_height": height,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "frames": frames_out,
        }, f)

    unique_ids = {player["track_id"] for fr in frames_out for player in fr["players"]}
    frames_with_detections = sum(1 for fr in frames_out if fr["players"])
    detection_rate = frames_with_detections / frame_idx if frame_idx else 0.0

    print(f"\nProcessed {frame_idx} frames ({width}x{height} @ {fps:.1f}fps, imgsz={args.imgsz}).")
    print(f"Frames with >=1 player detected: {frames_with_detections}/{frame_idx} ({detection_rate:.1%})")
    print(f"Unique track IDs seen: {len(unique_ids)}")
    print(f"Annotated video -> {annotated_video_path}")
    print(f"Keypoints JSON  -> {keypoints_json_path}")

if __name__ == "__main__":
    main()