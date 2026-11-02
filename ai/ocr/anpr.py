"""Reusable ANPR pipeline: plate detection, OCR, and multi-frame voting.

The standalone POC scripts in ``scripts/`` keep working because the per-crop
OCR helpers still live in :mod:`poc_support`. This module adds structured
results, plate-to-vehicle association, and per-track voting so that the Qt
dashboard and the CLI pipelines share one implementation.

ANPR outputs are *candidates*. Operators must verify them before enforcement.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

# These helpers are already covered by tests under tests/test_poc_support.py.
from poc_support import (  # noqa: E402  (keep import at top for readability)
    choose_plate as _choose_plate,
    normalize_plate,
    ocr_plate_variants,
    plausible_indian_plate,
)

BBox = Tuple[int, int, int, int]


@dataclass
class AnprResult:
    """One ANPR observation for a tracked vehicle at a specific frame."""

    track_id: int
    plate_text: str
    ocr_confidence: float
    plate_detection_confidence: float
    frame_no: int
    vote_count: int
    is_stable_candidate: bool


@dataclass
class AnprDebugMetrics:
    """Counters reported for offline runs. Not an accuracy metric."""

    plate_boxes_detected: int = 0
    plate_boxes_filtered: int = 0
    plate_boxes_linked: int = 0
    ocr_attempts: int = 0
    ocr_non_empty: int = 0
    stable_candidates: int = 0
    unique_tracks_with_plate: int = 0

    def as_lines(self) -> list[str]:
        return [
            f"Plate boxes detected: {self.plate_boxes_detected}",
            f"Plate boxes filtered (too small / invalid): {self.plate_boxes_filtered}",
            f"Plate boxes linked to tracked vehicles: {self.plate_boxes_linked}",
            f"OCR attempts: {self.ocr_attempts}",
            f"OCR non-empty candidates: {self.ocr_non_empty}",
            f"Stable plate candidates (unique tracks promoted): {self.stable_candidates}",
            f"Unique tracks with stable candidate: {self.unique_tracks_with_plate}",
        ]


def associate_plate(plate_box: BBox, vehicles: dict[int, BBox]) -> Optional[int]:
    """Return the ``track_id`` of the smallest vehicle box containing the plate center.

    Only one track is returned per call. Callers that process multiple plates in
    one frame should prevent repeat associations (see :meth:`AnprPipeline.process_frame`).
    """
    x1, y1, x2, y2 = plate_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    candidates = [
        (max(1, bx2 - bx1) * max(1, by2 - by1), tid)
        for tid, (bx1, by1, bx2, by2) in vehicles.items()
        if bx1 <= cx <= bx2 and by1 <= cy <= by2
    ]
    return min(candidates)[1] if candidates else None


class PlateVoteTracker:
    """Per-track OCR observation history and stable candidate selector.

    Candidates must appear ``min_votes`` times (across different frames or
    preprocessing variants) before :meth:`stable_candidate` returns a value.
    Nothing is auto-corrected: ``TS10UA1675`` and ``TS1OUA167S`` are treated
    as distinct observations.
    """

    def __init__(self, min_votes: int = 2):
        if int(min_votes) < 1:
            raise ValueError("min_votes must be >= 1")
        self.min_votes = int(min_votes)
        self.votes: dict[int, Counter[str]] = defaultdict(Counter)
        self.best_scores: dict[int, dict[str, float]] = defaultdict(dict)
        self.history: dict[int, list[tuple[int, str, float, float]]] = defaultdict(list)
        self.first_frame: dict[int, int] = {}
        self.last_frame: dict[int, int] = {}

    def record(
        self,
        track_id: int,
        frame_no: int,
        plate_text: str,
        ocr_confidence: float,
        detection_confidence: float,
    ) -> str:
        """Append an observation and return the normalized plate text."""
        text = normalize_plate(plate_text)
        if not text:
            return ""
        self.votes[track_id][text] += 1
        prior = self.best_scores[track_id].get(text, 0.0)
        self.best_scores[track_id][text] = max(prior, float(ocr_confidence))
        self.history[track_id].append(
            (int(frame_no), text, float(ocr_confidence), float(detection_confidence))
        )
        self.first_frame.setdefault(track_id, int(frame_no))
        self.last_frame[track_id] = int(frame_no)
        return text

    def stable_candidate(self, track_id: int) -> Optional[tuple[str, float, int]]:
        """Return ``(plate_text, best_confidence, vote_count)`` once stable."""
        choice = _choose_plate(self.votes[track_id], self.best_scores[track_id], self.min_votes)
        if choice is None:
            return None
        plate_text, confidence = choice
        return plate_text, confidence, self.votes[track_id][plate_text]

    def vote_count(self, track_id: int, plate_text: str) -> int:
        return self.votes[track_id][normalize_plate(plate_text)]

    def tracks_with_stable_candidate(self) -> list[int]:
        return [tid for tid in self.votes if self.stable_candidate(tid) is not None]


class AnprPipeline:
    """License-plate detection + OCR + per-track voting.

    Model loading is deferred until the first frame that would run inference,
    so dashboards start quickly and tests can inject fakes. Any model-loading
    or inference failure disables ANPR for the rest of the run but never
    raises back into the host worker.
    """

    def __init__(
        self,
        plate_model_path: str | Path,
        *,
        ocr_every: int = 5,
        plate_conf: float = 0.25,
        plate_imgsz: int = 1280,
        min_plate_width: int = 35,
        min_plate_height: int = 8,
        min_votes: int = 2,
        ocr_min_conf: float = 0.25,
        pad_x_pct: float = 0.08,
        pad_y_pct: float = 0.15,
        plate_model: Any = None,
        ocr_reader: Any = None,
        yolo_factory: Any = None,
    ) -> None:
        self.plate_model_path = Path(plate_model_path)
        self.ocr_every = max(1, int(ocr_every))
        self.plate_conf = float(plate_conf)
        self.plate_imgsz = int(plate_imgsz)
        self.min_plate_width = int(min_plate_width)
        self.min_plate_height = int(min_plate_height)
        self.ocr_min_conf = float(ocr_min_conf)
        self.pad_x_pct = float(pad_x_pct)
        self.pad_y_pct = float(pad_y_pct)
        self.tracker = PlateVoteTracker(min_votes)
        self.metrics = AnprDebugMetrics()
        self._plate_model = plate_model
        self._reader = ocr_reader
        self._yolo_factory = yolo_factory
        self.enabled = plate_model is not None and ocr_reader is not None
        self.load_error: Optional[str] = None

    # ------------------------------------------------------------------ loading
    def _ensure_loaded(self) -> bool:
        if self.enabled:
            return True
        if self.load_error is not None:
            return False
        try:
            if self._plate_model is None:
                if not self.plate_model_path.is_file():
                    raise FileNotFoundError(
                        f"License-plate model not found: {self.plate_model_path}"
                    )
                factory = self._yolo_factory
                if factory is None:
                    from ultralytics import YOLO  # type: ignore[import-not-found]

                    factory = YOLO
                self._plate_model = factory(str(self.plate_model_path))
            if self._reader is None:
                import easyocr  # type: ignore[import-not-found]

                self._reader = easyocr.Reader(["en"], gpu=False)
            self.enabled = True
            return True
        except Exception as exc:  # noqa: BLE001 - we deliberately degrade gracefully
            self.enabled = False
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    def should_run(self, frame_no: int) -> bool:
        if not self._ensure_loaded():
            return False
        return int(frame_no) % self.ocr_every == 0

    # ----------------------------------------------------------------- per-frame
    def process_frame(
        self,
        frame,
        frame_no: int,
        current_vehicles: dict[int, BBox],
    ) -> list[AnprResult]:
        """Detect plates in ``frame``, OCR each, and update per-track votes."""
        if not current_vehicles or not self.should_run(frame_no):
            return []

        height, width = frame.shape[:2]
        try:
            prediction = self._plate_model.predict(
                frame, conf=self.plate_conf, imgsz=self.plate_imgsz, verbose=False
            )[0]
        except Exception as exc:  # noqa: BLE001 - keep host worker alive
            self.enabled = False
            self.load_error = f"plate predict failed: {exc}"
            return []

        boxes = getattr(prediction, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy_list = boxes.xyxy.cpu().tolist()
        conf_list = boxes.conf.cpu().tolist()

        results: list[AnprResult] = []
        used_tracks: set[int] = set()
        for xyxy, det_conf in zip(xyxy_list, conf_list):
            self.metrics.plate_boxes_detected += 1
            x1, y1, x2, y2 = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            if x2 <= x1 or y2 <= y1:
                self.metrics.plate_boxes_filtered += 1
                continue
            plate_w, plate_h = x2 - x1, y2 - y1
            if plate_w < self.min_plate_width or plate_h < self.min_plate_height:
                self.metrics.plate_boxes_filtered += 1
                continue
            track_id = associate_plate((x1, y1, x2, y2), current_vehicles)
            if track_id is None or track_id in used_tracks:
                # Avoid double-linking one plate to multiple vehicles in a frame.
                continue
            used_tracks.add(track_id)
            self.metrics.plate_boxes_linked += 1

            pad_x = max(2, round(plate_w * self.pad_x_pct))
            pad_y = max(2, round(plate_h * self.pad_y_pct))
            xa, ya = max(0, x1 - pad_x), max(0, y1 - pad_y)
            xb, yb = min(width, x2 + pad_x), min(height, y2 + pad_y)
            crop = frame[ya:yb, xa:xb]

            self.metrics.ocr_attempts += 1
            try:
                plate_text, ocr_conf = ocr_plate_variants(self._reader, crop)
            except Exception as exc:  # noqa: BLE001
                self.load_error = f"ocr failed: {exc}"
                continue
            if not plate_text or ocr_conf < self.ocr_min_conf:
                continue
            self.metrics.ocr_non_empty += 1

            had_stable = self.tracker.stable_candidate(track_id) is not None
            self.tracker.record(track_id, frame_no, plate_text, ocr_conf, float(det_conf))
            stable = self.tracker.stable_candidate(track_id)
            is_stable = stable is not None
            if is_stable and not had_stable:
                self.metrics.stable_candidates += 1
            vote_count = self.tracker.vote_count(track_id, plate_text)
            results.append(
                AnprResult(
                    track_id=int(track_id),
                    plate_text=normalize_plate(plate_text),
                    ocr_confidence=float(ocr_conf),
                    plate_detection_confidence=float(det_conf),
                    frame_no=int(frame_no),
                    vote_count=int(vote_count),
                    is_stable_candidate=bool(is_stable),
                )
            )
        return results

    # --------------------------------------------------------------- introspection
    def stable_candidate(self, track_id: int) -> Optional[tuple[str, float, int]]:
        return self.tracker.stable_candidate(track_id)

    def finalize_metrics(self) -> AnprDebugMetrics:
        self.metrics.unique_tracks_with_plate = len(
            self.tracker.tracks_with_stable_candidate()
        )
        return self.metrics
