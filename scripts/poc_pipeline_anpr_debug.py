"""Offline vehicle tracking + experimental license plate recognition (ANPR).

Debug version: draws ALL plate detector boxes and optionally saves OCR crops.
This changes visual diagnostics only; it does not guarantee higher OCR accuracy.

Requires a separate trained license-plate YOLO checkpoint (--plate-model).
ANPR outputs are candidates that require human verification, not enforcement evidence.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

# Reuse the tested OCR normalization/preprocessing helpers from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poc_support import choose_plate, normalize_plate, ocr_plate_variants

VEHICLES = {1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, track_id INTEGER NOT NULL,
        detected_at TEXT NOT NULL, source_video TEXT NOT NULL,
        frame_number INTEGER NOT NULL, vehicle_type TEXT NOT NULL,
        confidence REAL NOT NULL, UNIQUE(run_id, track_id))""")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(detections)")}
    if "plate_text" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN plate_text TEXT DEFAULT ''")
    if "plate_confidence" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN plate_confidence REAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS plate_reads (
        id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, track_id INTEGER NOT NULL,
        frame_number INTEGER NOT NULL, plate_text TEXT NOT NULL,
        ocr_confidence REAL NOT NULL, plate_detector_confidence REAL NOT NULL)""")
    conn.commit()
    return conn


def ocr_plate(reader, crop) -> tuple[str, float]:
    """Recognize a plate from original, contrast, sharp, and threshold crops."""
    return ocr_plate_variants(reader, crop)


def matched_track(plate_box, vehicles: dict[int, tuple[int, int, int, int]]) -> int | None:
    """Associate plate with the smallest enclosing tracked vehicle box."""
    x1, y1, x2, y2 = plate_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    candidates = [(max(1, bx2 - bx1) * max(1, by2 - by1), tid)
                  for tid, (bx1, by1, bx2, by2) in vehicles.items()
                  if bx1 <= cx <= bx2 and by1 <= cy <= by2]
    return min(candidates)[1] if candidates else None


def accepted_plate(votes: Counter, best_scores: dict[str, float], min_votes: int):
    return choose_plate(votes, best_scores, min_votes)


def analyze(args: argparse.Namespace) -> None:
    if not args.video.is_file():
        raise FileNotFoundError(f"Input video not found: {args.video}")
    if not args.plate_model.is_file():
        raise FileNotFoundError(
            f"License-plate weights missing: {args.plate_model}. "
            "Download trained plate-detector weights first; yolo11n.pt cannot detect plates."
        )
    if args.output.resolve() == args.video.resolve():
        raise ValueError("Input and output videos must be different files")
    try:
        from ultralytics import YOLO
        import easyocr
    except ImportError as exc:
        raise RuntimeError("Install with: python -m pip install ultralytics easyocr opencv-python") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vehicle_model = YOLO(args.model)
    plate_model = YOLO(str(args.plate_model))
    reader = easyocr.Reader(["en"], gpu=False)
    print(f"Plate model classes: {plate_model.names}", flush=True)
    print("OCR results are unverified plate candidates; no enforcement use.", flush=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {args.video}")
    writer = None
    conn = None
    run_id = uuid.uuid4().hex
    frames_seen = defaultdict(int)
    vehicle_votes = defaultdict(Counter)
    total_conf = defaultdict(float)
    counted = set()
    plate_votes = defaultdict(Counter)
    best_plate_scores = defaultdict(dict)
    frames_processed = 0
    plate_boxes_found = 0
    plate_boxes_filtered = 0
    associated_boxes = 0
    ocr_reads = 0
    debug_saved = 0
    if args.debug_crops:
        args.debug_crops.mkdir(parents=True, exist_ok=True)

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not 1 <= fps <= 240:
            fps = 25.0
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Video opens but has no readable frames")
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {args.output}")
        conn = create_database(args.db)

        while frame is not None:
            frames_processed += 1
            result = vehicle_model.track(frame, persist=True, tracker="bytetrack.yaml",
                                         classes=list(VEHICLES), conf=args.conf,
                                         imgsz=args.imgsz, verbose=False)[0]
            annotated = result.plot()
            boxes = result.boxes
            current_vehicles: dict[int, tuple[int, int, int, int]] = {}
            if boxes is not None and len(boxes) and boxes.id is not None:
                cls_ids = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                coordinates = boxes.xyxy.cpu().tolist()
                for class_id, tid, confidence, xyxy in zip(cls_ids, track_ids, confidences, coordinates):
                    if class_id not in VEHICLES:
                        continue
                    current_vehicles[tid] = tuple(map(int, xyxy))
                    frames_seen[tid] += 1
                    vehicle_votes[tid][class_id] += 1
                    total_conf[tid] += float(confidence)
                    if frames_seen[tid] >= args.min_frames and tid not in counted:
                        best_class = vehicle_votes[tid].most_common(1)[0][0]
                        conn.execute("""INSERT INTO detections
                            (run_id,track_id,detected_at,source_video,frame_number,vehicle_type,confidence)
                            VALUES(?,?,?,?,?,?,?)""",
                            (run_id,tid,utc_now(),str(args.video),frames_processed,
                             VEHICLES[best_class],total_conf[tid]/frames_seen[tid]))
                        conn.commit()
                        counted.add(tid)

            # Plate inference only on sampled frames to keep CPU testing manageable.
            if frames_processed % args.ocr_every == 0:
                presult = plate_model.predict(frame, conf=args.plate_conf,
                                              imgsz=args.plate_imgsz, verbose=False)[0]
                plate_boxes = presult.boxes
                if plate_boxes is not None and len(plate_boxes):
                    for xyxy, det_conf in zip(plate_boxes.xyxy.cpu().tolist(),
                                              plate_boxes.conf.cpu().tolist()):
                        plate_boxes_found += 1
                        x1, y1, x2, y2 = map(int, xyxy)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        # Always draw plate detections, including unmatched and unreadable.
                        if x2 <= x1 or y2 <= y1:
                            continue
                        width, height = x2 - x1, y2 - y1
                        tid = matched_track((x1, y1, x2, y2), current_vehicles)
                        label = f"PLATE {det_conf:.2f} | "
                        if width < args.min_plate_width or height < 8:
                            plate_boxes_filtered += 1
                            label += "too small"
                        elif tid is None:
                            label += "no vehicle match"
                        else:
                            associated_boxes += 1
                            pad_x = max(2, round(width * 0.08))
                            pad_y = max(2, round(height * 0.15))
                            xa, ya = max(0, x1 - pad_x), max(0, y1 - pad_y)
                            xb, yb = min(w, x2 + pad_x), min(h, y2 + pad_y)
                            crop = frame[ya:yb, xa:xb]
                            if args.debug_crops and debug_saved < args.debug_max_crops:
                                crop_path = args.debug_crops / f"frame_{frames_processed:06d}_track_{tid}.jpg"
                                if cv2.imwrite(str(crop_path), crop):
                                    debug_saved += 1
                            plate, score = ocr_plate(reader, crop)
                            if plate and score >= args.ocr_conf:
                                ocr_reads += 1
                                plate_votes[tid][plate] += 1
                                best_plate_scores[tid][plate] = max(
                                    score, best_plate_scores[tid].get(plate, 0.0))
                                conn.execute("""INSERT INTO plate_reads
                                    (run_id,track_id,frame_number,plate_text,
                                     ocr_confidence,plate_detector_confidence)
                                    VALUES(?,?,?,?,?,?)""",
                                    (run_id,tid,frames_processed,plate,score,float(det_conf)))
                                winner = accepted_plate(plate_votes[tid],
                                                        best_plate_scores[tid],args.min_plate_votes)
                                if winner and tid in counted:
                                    conn.execute("""UPDATE detections SET plate_text=?, plate_confidence=?
                                        WHERE run_id=? AND track_id=?""",
                                        (winner[0],winner[1],run_id,tid))
                                conn.commit()
                                label += f"track {tid}: {plate} ({score:.2f})"
                            else:
                                label += f"track {tid}: OCR unreadable"
                        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,255), 2)
                        cv2.putText(annotated, label, (max(0, x1), max(18,y1-9)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0,255,255), 2)

            for tid, (x1,y1,_,_) in current_vehicles.items():
                winner = accepted_plate(plate_votes[tid], best_plate_scores[tid],
                                        args.min_plate_votes)
                if winner:
                    cv2.putText(annotated, f"Track {tid} | plate candidate {winner[0]}",
                                (max(0,x1),max(20,y1-12)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,(0,255,0),2)
            cv2.putText(annotated,
                        f"Frame {frames_processed} | Vehicles {len(counted)} | OCR reads {ocr_reads}",
                        (12,32),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            writer.write(annotated)
            if args.show:
                cv2.imshow("Vehicle + ANPR POC (q to stop)",annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            ok, next_frame = cap.read()
            frame = next_frame if ok else None

        # Plates sometimes become readable before a track qualifies for counting.
        for tid in counted:
            winner = accepted_plate(plate_votes[tid], best_plate_scores[tid],
                                    args.min_plate_votes)
            if winner:
                conn.execute("""UPDATE detections SET plate_text=?,plate_confidence=?
                    WHERE run_id=? AND track_id=?""",(winner[0],winner[1],run_id,tid))
        conn.commit()
        print(f"Run ID: {run_id}")
        print(f"Processed frames: {frames_processed}")
        print(f"Counted vehicle tracks (approximate): {len(counted)}")
        print(f"Plate boxes detected: {plate_boxes_found}")
        print(f"Plate boxes filtered as too small: {plate_boxes_filtered}")
        print(f"Plate boxes linked to vehicles: {associated_boxes}")
        print(f"Debug plate crops saved: {debug_saved}")
        print(f"OCR candidate reads: {ocr_reads}")
        print("Saved tracks with plate candidates:")
        rows = conn.execute("""SELECT track_id,vehicle_type,plate_text,plate_confidence
            FROM detections WHERE run_id=? AND plate_text IS NOT NULL AND plate_text!=''
            ORDER BY track_id""",(run_id,)).fetchall()
        for tid, vehicle, plate, score in rows:
            print(f"  Track {tid}: {vehicle}, {plate} (OCR score {score:.2f})")
        if not rows:
            print("  None confirmed by repeat reads. Try clearer footage or inspect plate detector output.")
        print(f"Output: {args.output.resolve()}")
        print(f"Database: {args.db.resolve()}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if conn is not None:
            conn.close()
        if args.show:
            cv2.destroyAllWindows()


def main() -> int:
    p = argparse.ArgumentParser(description="Offline vehicle counting + experimental ANPR")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--plate-model", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/anpr_processed.mp4"))
    p.add_argument("--db", type=Path, default=Path("outputs/anpr_test.db"))
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--plate-conf", type=float, default=0.25)
    p.add_argument("--plate-imgsz", type=int, default=1280)
    p.add_argument("--ocr-conf", type=float, default=0.25)
    p.add_argument("--ocr-every", type=int, default=5)
    p.add_argument("--min-frames", type=int, default=6)
    p.add_argument("--min-plate-width", type=int, default=35)
    p.add_argument("--min-plate-votes", type=int, default=2)
    p.add_argument("--debug-crops", type=Path, default=None,
                   help="Save up to --debug-max-crops matched plate crops for inspection")
    p.add_argument("--debug-max-crops", type=int, default=80)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    if not (0 < args.conf <= 1 and 0 < args.plate_conf <= 1 and
            0 <= args.ocr_conf <= 1 and args.imgsz >= 32 and
            args.plate_imgsz >= 32 and args.ocr_every >= 1 and
            args.min_frames >= 1 and args.min_plate_votes >= 1 and
            args.min_plate_width >= 1 and args.debug_max_crops >= 0):
        p.error("Invalid thresholds or sampling settings")
    try:
        analyze(args)
        return 0
    except (OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
        print(f"ANPR POC failed: {exc}",file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
