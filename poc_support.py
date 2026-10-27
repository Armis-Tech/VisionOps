"""Small, testable helpers shared by the VisionOps desktop and CLI POCs.

No Qt, Ultralytics, or model imports here. Importing this module never opens
cameras, loads model weights, or modifies a database.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from typing import Iterable

# Demo only: standard state/RTO Indian plate pattern. Some valid plates use
# other formats (BH, diplomatic, temporary, older). Those remain unverified.
INDIAN_STANDARD_PLATE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")


def normalize_plate(text: str) -> str:
    """Normalize OCR text, without guessing or silently changing characters."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def plausible_indian_plate(text: str) -> bool:
    return bool(INDIAN_STANDARD_PLATE.fullmatch(normalize_plate(text)))


def should_count_vehicle(
    age: int,
    history: Iterable[tuple[float, float]],
    previous_y: float | None,
    current_y: float,
    count_line_y: float | None,
    min_movement_px: float = 25,
    min_age: int = 6,
) -> bool:
    """Simple POC gate to exclude stationary tracks and one-frame detections.

    If a counting line is configured, an observed crossing is required.
    Otherwise, the tracked object's image center must move at least the
    configured pixel distance. Movement is NOT compensated for camera motion.
    """
    points = list(history)
    if age < (3 if count_line_y is not None else min_age) or len(points) < 2:
        return False
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    displacement_sq = dx * dx + dy * dy
    if displacement_sq < min_movement_px * min_movement_px:
        return False
    if count_line_y is None:
        return True
    return (previous_y is not None and
            (previous_y < count_line_y <= current_y or
             previous_y > count_line_y >= current_y))


def choose_plate(votes: Counter[str], scores: dict[str, float], minimum_votes: int = 2):
    """Return strongest repeated *plausible* OCR text, else None.

    Confidence is OCR confidence, not an accuracy guarantee. We never
    hallucinate or auto-correct digits and letters.
    """
    if minimum_votes < 1:
        raise ValueError("minimum_votes must be positive")
    choices = [(count, scores.get(text, 0.0), text)
               for text, count in votes.items()
               if count >= minimum_votes and plausible_indian_plate(text)]
    if not choices:
        return None
    count, confidence, text = max(choices)
    return text, confidence


def ensure_column(connection: sqlite3.Connection, table: str, name: str, definition: str) -> None:
    """Idempotent column migration for a known, hardcoded table/field."""
    if table not in {"detections", "violations"} or name not in {
        "run_id", "track_id", "frame_number", "video_path"
    }:
        raise ValueError("Unsafe migration target")
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if name not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def prepare_plate_images(crop):
    """Return several OCR inputs with different contrast characteristics.

    The original crop is included because sharpening/thresholding can erase
    thin lettering in an otherwise clear image.
    """
    import cv2
    if crop is None or crop.size == 0:
        return []
    enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY) if len(enlarged.shape) == 3 else enlarged
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    thresholded = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 9,
    )
    return [enlarged, enhanced, sharp, thresholded]


def ocr_plate_variants(reader, crop) -> tuple[str, float]:
    """Read each preprocessed crop and select best plausible plate candidate.

    Requires an EasyOCR-compatible reader. Returns ('', 0) for uncertain reads.
    A read can be wrong even when it matches the format, so caller must apply
    multi-frame voting and human verification before enforcement.
    """
    best: tuple[str, float] = ("", 0.0)
    for variant in prepare_plate_images(crop):
        parts = [item for item in reader.readtext(
            variant, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ) if len(item) >= 3]
        if not parts:
            continue
        # EasyOCR may split "TS10 UA 1675" into 2-3 separate boxes.
        # This is a single detected plate crop, not the full video frame.
        parts.sort(key=lambda item: (round(min(p[1] for p in item[0]) / 35),
                                     min(p[0] for p in item[0])))
        candidates = [(normalize_plate(item[1]), float(item[2])) for item in parts]
        if len(parts) > 1:
            joined = normalize_plate("".join(str(item[1]) for item in parts))
            joined_conf = min(float(item[2]) for item in parts)
            candidates.append((joined, joined_conf))
        for candidate, confidence in candidates:
            if plausible_indian_plate(candidate) and confidence > best[1]:
                best = candidate, confidence
    return best
