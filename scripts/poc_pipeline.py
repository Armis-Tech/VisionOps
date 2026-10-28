"""Vision AI POC: offline vehicle detection, tracking, counting, and SQLite persistence.

This deliberately does not claim helmet, triple-riding, or ANPR detection.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

VEHICLE_CLASSES = {1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            detected_at TEXT NOT NULL,
            source_video TEXT NOT NULL,
            frame_number INTEGER NOT NULL,
            vehicle_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            UNIQUE(run_id, track_id)
        )
    """)
    connection.commit()
    return connection


def analyze_video(args: argparse.Namespace) -> None:
    source = args.video.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is missing. Install with: python -m pip install ultralytics"
        ) from exc

    output = args.output.resolve()
    if source == output:
        raise ValueError("Output path must differ from input video path.")
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)  # A named Ultralytics model may download on first use.
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {source}")

    writer = None
    connection = None
    run_id = uuid.uuid4().hex
    frames_seen = defaultdict(int)
    class_votes = defaultdict(Counter)
    confidence_totals = defaultdict(float)
    saved_tracks = set()
    processed = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 <= fps <= 240 else 25.0
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            raise RuntimeError("Video opened but contains no readable frames.")
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {output}")
        connection = create_database(args.db.resolve())

        frame = first_frame
        while frame is not None:
            processed += 1
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(VEHICLE_CLASSES),
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )[0]
            annotated = result.plot()
            boxes = result.boxes
            if boxes is not None and len(boxes) and boxes.id is not None:
                classes = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                for class_id, track_id, confidence in zip(classes, track_ids, confidences):
                    if class_id not in VEHICLE_CLASSES:
                        continue
                    frames_seen[track_id] += 1
                    class_votes[track_id][class_id] += 1
                    confidence_totals[track_id] += float(confidence)
                    if frames_seen[track_id] < args.min_frames or track_id in saved_tracks:
                        continue
                    voted_class = class_votes[track_id].most_common(1)[0][0]
                    mean_confidence = confidence_totals[track_id] / frames_seen[track_id]
                    connection.execute(
                        """INSERT INTO detections
                        (run_id, track_id, detected_at, source_video,
                         frame_number, vehicle_type, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (run_id, track_id, utc_now(), str(source), processed,
                         VEHICLE_CLASSES[voted_class], mean_confidence),
                    )
                    connection.commit()
                    saved_tracks.add(track_id)

            cv2.putText(
                annotated, f"Frame: {processed} | Counted: {len(saved_tracks)}",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            writer.write(annotated)
            if args.show:
                cv2.imshow("Vehicle POC (press q to stop)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            ok, next_frame = cap.read()
            frame = next_frame if ok else None

        if processed == 0:
            raise RuntimeError("No frames were processed.")
        totals = dict(connection.execute(
            "SELECT vehicle_type, COUNT(*) FROM detections "
            "WHERE run_id=? GROUP BY vehicle_type", (run_id,)
        ).fetchall())
        print(f"Run ID: {run_id}")
        print(f"Frames processed: {processed}")
        print(f"Unique tracked vehicles (approximate): {len(saved_tracks)}")
        print(f"Vehicle counts: {totals}")
        print(f"Database: {args.db.resolve()}")
        print(f"Processed video: {output}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if connection is not None:
            connection.close()
        if args.show:
            cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline traffic vehicle-counting POC")
    parser.add_argument("--video", type=Path, required=True, help="Input MP4 video")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics YOLO weights")
    parser.add_argument("--output", type=Path, default=Path("processed_video.mp4"))
    parser.add_argument("--db", type=Path, default=Path("poc_detections.db"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-frames", type=int, default=6)
    parser.add_argument("--show", action="store_true", help="Show preview window")
    args = parser.parse_args()
    if not 0 < args.conf <= 1 or args.imgsz < 32 or args.min_frames < 1:
        parser.error("Require 0 < conf <= 1, imgsz >= 32 and min-frames >= 1")
    try:
        analyze_video(args)
        return 0
    except (OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
        print(f"POC failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
