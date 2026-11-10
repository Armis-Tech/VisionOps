"""Reusable helmet-violation candidate logic.

This module handles per-track helmet/no-helmet observation, spatial rider
association, and temporal voting. It is intentionally decoupled from Qt
and from Ultralytics so it can be tested without loading model weights.

All output is a *candidate*. Operators must verify before any enforcement.

Typical usage::

    observer = HelmetObserver(min_no_helmet_votes=3, min_conf=0.30)

    # each frame:
    observations = observer.process_frame(
        frame_no=frame_no,
        motorcycles={track_id: (x1, y1, x2, y2), ...},
        persons=[(x1, y1, x2, y2), ...],         # from primary COCO model
        no_helmet_detections=[(conf, bbox), ...], # from violation model
    )
    for obs in observations:
        if obs.is_stable_candidate:
            ...  # save violation candidate

    metrics = observer.finalize_metrics()
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

BBox = tuple[int, int, int, int]   # x1, y1, x2, y2

HELMET = "helmet"
NO_HELMET = "no_helmet"
UNKNOWN = "unknown"

# NAME_HINTS reproduced here so describe_violation_model() is self-contained
# and callable without importing main.py (which pulls Qt + Ultralytics).
_NAME_HINTS: dict[str, tuple[str, ...]] = {
    "helmet": ("no_helmet", "nohelmet", "without_helmet", "no_hardhat"),
    "belt": ("no_seatbelt", "no_seat_belt", "noseatbelt", "without_seatbelt", "without_seat_belt"),
    "lights": ("police_light", "beacon", "siren"),
    "rooftop": ("person_on_top", "person_on_roof", "rooftop", "on_roof"),
    "triple": ("triple_riding", "triple"),
}
_KNOWN_VIOLATIONS = {"helmet", "belt", "wrongway", "lights", "rooftop", "triple"}


# --------------------------------------------------------------------------- data types

@dataclass
class HelmetObservation:
    """One helmet status observation for a tracked motorcycle."""
    track_id: int
    helmet_status: str          # "helmet" | "no_helmet" | "unknown"
    confidence: float
    frame_no: int
    is_stable_candidate: bool   # True once min_no_helmet_votes reached for this track
    reason: str                 # human-readable explanation for debugging


@dataclass
class HelmetDebugMetrics:
    """Counters for offline runs. Not an accuracy metric."""
    motorcycles_observed: int = 0
    rider_associations_attempted: int = 0
    helmet_observations: int = 0
    no_helmet_observations: int = 0
    unknown_observations: int = 0
    stable_no_helmet_candidates: int = 0
    candidate_violations_queued: int = 0  # unique tracks added to the pending save list
    candidate_violations_saved: int = 0   # violations successfully committed to SQLite

    def as_lines(self) -> list[str]:
        return [
            f"Motorcycles observed (total across frames): {self.motorcycles_observed}",
            f"Rider association attempts: {self.rider_associations_attempted}",
            f"Helmet observations: {self.helmet_observations}",
            f"No-helmet observations: {self.no_helmet_observations}",
            f"Unknown/inconclusive observations: {self.unknown_observations}",
            f"Stable no-helmet candidates (unique tracks promoted): {self.stable_no_helmet_candidates}",
            f"Candidate violations queued: {self.candidate_violations_queued}",
            f"Candidate violations saved to DB: {self.candidate_violations_saved}",
        ]


@dataclass
class ViolationModelInfo:
    """Diagnostic snapshot of a violation model's capabilities."""
    model_path: str
    exists: bool
    load_error: Optional[str]
    class_names: dict[int, str]              # index -> raw class name from model
    violation_map: dict[int, Optional[str]]  # index -> violation key (or None)
    helmet_class_indices: list[int]          # indices whose violation key is "helmet"
    has_helmet_capability: bool

    def as_lines(self) -> list[str]:
        lines = [f"Violation model: {self.model_path}"]
        if not self.exists:
            lines.append("  File status: MISSING")
            lines.append("  Helmet detection: UNAVAILABLE (place violations.pt at the above path)")
            return lines
        if self.load_error:
            lines.append(f"  File status: load error — {self.load_error}")
            lines.append("  Helmet detection: UNAVAILABLE")
            return lines
        lines.append(f"  File status: loaded  ({len(self.class_names)} classes)")
        for idx, name in sorted(self.class_names.items()):
            vkey = self.violation_map.get(idx)
            tag = f"  → violation:{vkey}" if vkey else ""
            lines.append(f"    [{idx:2d}] {name}{tag}")
        if self.has_helmet_capability:
            indices_str = ", ".join(str(i) for i in self.helmet_class_indices)
            lines.append(f"  Helmet detection: AVAILABLE (class indices: {indices_str})")
        else:
            lines.append(
                "  Helmet detection: UNAVAILABLE — no class maps to 'helmet' violation.\n"
                "  Expected a class name containing one of: "
                + str(_NAME_HINTS["helmet"])
                + "\n  Retrain violations.pt to add a 'no_helmet' class to enable this feature."
            )
        return lines


# --------------------------------------------------------------------------- spatial helpers

def associate_rider(
    moto_box: BBox,
    persons: list[BBox],
    expand_pct: float = 0.15,
) -> list[BBox]:
    """Return person boxes whose centre falls in the motorcycle's rider region.

    The search region extends *upward* by one motorcycle height (the rider
    sits above the vehicle) and expands horizontally by ``expand_pct`` on
    each side. Persons far away from the motorcycle are excluded.

    This replicates the spatial filter used by the triple-riding rule but
    returns the actual person boxes instead of a count.
    """
    x1, y1, x2, y2 = moto_box
    bw = x2 - x1
    bh = y2 - y1
    ex = expand_pct * bw
    # Search box: same horizontal width (+ margin), extended upward by one bike height.
    sx1 = x1 - ex
    sy1 = y1 - bh      # rider sits above the motorcycle
    sx2 = x2 + ex
    sy2 = y2            # not below the motorcycle axle

    riders = []
    for p in persons:
        cx = (p[0] + p[2]) / 2
        cy = (p[1] + p[3]) / 2
        if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
            riders.append(p)
    return riders


def no_helmet_on_rider(
    no_helmet_box: BBox,
    rider_box: BBox,
) -> bool:
    """Return True if the no_helmet detection centre falls within the rider's bbox.

    A purpose-built no_helmet model detects the bare head/neck region.
    If the centre of that detection is within the rider's full bounding box,
    the detection belongs to this rider.
    """
    cx = (no_helmet_box[0] + no_helmet_box[2]) / 2
    cy = (no_helmet_box[1] + no_helmet_box[3]) / 2
    rx1, ry1, rx2, ry2 = rider_box
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def head_region(person_box: BBox, head_pct: float = 0.35) -> BBox:
    """Return bbox of the upper *head_pct* of a person detection.

    Exported for callers that want to perform tighter association than the
    default centre-in-person check used by :class:`HelmetObserver`.
    """
    x1, y1, x2, y2 = person_box
    head_h = max(1, int((y2 - y1) * head_pct))
    return (x1, y1, x2, y1 + head_h)


# --------------------------------------------------------------------------- observer

class HelmetObserver:
    """Accumulates per-motorcycle helmet/no-helmet observations with spatial filtering.

    Spatial association:
      1. For each motorcycle track, find persons whose centre overlaps the
         expanded motorcycle rider region (``associate_rider``).
      2. For each associated rider, check whether any no_helmet detection
         has its centre inside that rider's bounding box.

    Temporal stability:
      A track must accumulate ``min_no_helmet_votes`` no_helmet observations
      *within the most recent ``window_frames`` frames* before
      :meth:`process_frame` returns an observation with
      ``is_stable_candidate=True``. Observations older than ``window_frames``
      are discarded, so a handful of scattered detections spread across a long
      video do not trigger a false positive.

    Duplicate prevention:
      Once a track has been promoted (violation recorded), its ``track_id`` is
      stored in :attr:`promoted_tracks`. The caller is still responsible for
      preventing a second ``save_violation`` call (see the ``flagged`` set in
      ``FrameAnalyzer``). ``promoted_tracks`` is used here solely for the
      ``stable_no_helmet_candidates`` counter.
    """

    def __init__(
        self,
        *,
        min_no_helmet_votes: int = 3,
        min_conf: float = 0.30,
        moto_expand_pct: float = 0.15,
        window_frames: int = 30,
    ) -> None:
        if int(min_no_helmet_votes) < 1:
            raise ValueError("min_no_helmet_votes must be >= 1")
        if int(window_frames) < 1:
            raise ValueError("window_frames must be >= 1")
        self.min_no_helmet_votes = int(min_no_helmet_votes)
        self.min_conf = float(min_conf)
        self.moto_expand_pct = float(moto_expand_pct)
        self.window_frames = int(window_frames)

        # Sliding-window state: parallel deques of (frame_no, confidence) per track.
        self._hit_frames: dict[int, deque] = defaultdict(deque)
        self._hit_confs: dict[int, deque] = defaultdict(deque)
        self.promoted_tracks: set[int] = set()
        self.metrics = HelmetDebugMetrics()

    def process_frame(
        self,
        frame_no: int,
        motorcycles: dict[int, BBox],
        persons: list[BBox],
        no_helmet_detections: list[tuple[float, BBox]],
    ) -> list[HelmetObservation]:
        """Process one frame; return one observation per motorcycle with evidence.

        Parameters
        ----------
        frame_no:
            Current frame index (used for logging/evidence only).
        motorcycles:
            Mapping from vehicle *track_id* (primary COCO tracker) to bbox.
        persons:
            All person bounding boxes from the primary COCO model in this frame.
        no_helmet_detections:
            ``(confidence, bbox)`` pairs from the violation model for every
            detection whose class maps to the "helmet" violation key.
        """
        results: list[HelmetObservation] = []
        self.metrics.motorcycles_observed += len(motorcycles)

        # Filter by confidence once so inner loop is clean.
        valid_dets = [(cf, bbox) for cf, bbox in no_helmet_detections if cf >= self.min_conf]

        for track_id, moto_box in motorcycles.items():
            # Discard observations that have left the sliding window.
            while (self._hit_frames[track_id]
                   and frame_no - self._hit_frames[track_id][0] > self.window_frames):
                self._hit_frames[track_id].popleft()
                self._hit_confs[track_id].popleft()

            self.metrics.rider_associations_attempted += 1
            riders = associate_rider(moto_box, persons, self.moto_expand_pct)

            if not riders:
                self.metrics.unknown_observations += 1
                results.append(HelmetObservation(
                    track_id=track_id,
                    helmet_status=UNKNOWN,
                    confidence=0.0,
                    frame_no=frame_no,
                    is_stable_candidate=False,
                    reason="no rider visible near motorcycle",
                ))
                continue

            best_conf = 0.0
            matched_rider = None
            for rider in riders:
                for det_conf, det_box in valid_dets:
                    if no_helmet_on_rider(det_box, rider):
                        if det_conf > best_conf:
                            best_conf = det_conf
                            matched_rider = rider

            if matched_rider is not None:
                self._hit_frames[track_id].append(frame_no)
                self._hit_confs[track_id].append(best_conf)
                self.metrics.no_helmet_observations += 1

                recent_count = len(self._hit_frames[track_id])
                best_recent_conf = max(self._hit_confs[track_id])

                already_promoted = track_id in self.promoted_tracks
                now_stable = recent_count >= self.min_no_helmet_votes
                if now_stable and not already_promoted:
                    self.promoted_tracks.add(track_id)
                    self.metrics.stable_no_helmet_candidates += 1

                results.append(HelmetObservation(
                    track_id=track_id,
                    helmet_status=NO_HELMET,
                    confidence=best_recent_conf,
                    frame_no=frame_no,
                    is_stable_candidate=now_stable,
                    reason=(
                        f"no_helmet on associated rider "
                        f"({recent_count}/{self.min_no_helmet_votes} "
                        f"in last {self.window_frames} frames)"
                    ),
                ))
            else:
                # Rider found but no no_helmet detection overlaps it.
                self.metrics.unknown_observations += 1
                results.append(HelmetObservation(
                    track_id=track_id,
                    helmet_status=UNKNOWN,
                    confidence=0.0,
                    frame_no=frame_no,
                    is_stable_candidate=False,
                    reason="rider present but no overlapping no_helmet detection",
                ))

        return results

    def finalize_metrics(self) -> HelmetDebugMetrics:
        return self.metrics


# --------------------------------------------------------------------------- diagnostic

def _norm_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


def _build_vmap(
    names: dict[int, str],
    custom: dict[str, str],
) -> dict[int, Optional[str]]:
    out: dict[int, Optional[str]] = {}
    for idx, name in names.items():
        n = _norm_name(name)
        key: Optional[str] = custom.get(n)
        if key is None:
            key = next(
                (k for k, hints in _NAME_HINTS.items() if any(h in n for h in hints)),
                None,
            )
        out[idx] = key if key in _KNOWN_VIOLATIONS else None
    return out


def describe_violation_model(
    model_path: str | Path,
    map_path: str | Path | None = None,
    *,
    _model: Any = None,  # injectable fake for unit tests
) -> ViolationModelInfo:
    """Return a :class:`ViolationModelInfo` describing the model's violation classes.

    This is a diagnostic helper — it is safe to call without Qt or a GPU.
    When ``violations.pt`` is missing or cannot load, it still returns a
    populated object (with ``exists=False`` or ``load_error`` set) rather
    than raising.

    Parameters
    ----------
    model_path:
        Path to ``violations.pt`` or any compatible YOLO weights file.
    map_path:
        Optional path to ``violation_classes.json`` for custom name overrides.
    _model:
        Injectable fake model for testing. If provided, ``model_path`` is used
        only as a label; the fake's ``.names`` dict is used instead.
    """
    import json

    model_path = Path(model_path)

    # Load custom name overrides if provided.
    custom: dict[str, str] = {}
    if map_path is not None and Path(map_path).exists():
        try:
            raw = json.loads(Path(map_path).read_text(encoding="utf-8"))
            custom = {_norm_name(k): str(v) for k, v in raw.items()}
        except (OSError, ValueError):
            pass

    # Injected fake bypasses all file/loader logic.
    if _model is not None:
        raw_names: dict[int, str] = {int(k): str(v) for k, v in _model.names.items()}
        vmap = _build_vmap(raw_names, custom)
        helmet_idx = [i for i, k in vmap.items() if k == "helmet"]
        return ViolationModelInfo(
            model_path=str(model_path),
            exists=True,
            load_error=None,
            class_names=raw_names,
            violation_map=vmap,
            helmet_class_indices=helmet_idx,
            has_helmet_capability=bool(helmet_idx),
        )

    if not model_path.exists():
        return ViolationModelInfo(
            model_path=str(model_path),
            exists=False,
            load_error=None,
            class_names={},
            violation_map={},
            helmet_class_indices=[],
            has_helmet_capability=False,
        )

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
        model = YOLO(str(model_path))
        raw_names = {int(k): str(v) for k, v in model.names.items()}
    except Exception as exc:  # noqa: BLE001
        return ViolationModelInfo(
            model_path=str(model_path),
            exists=True,
            load_error=f"{type(exc).__name__}: {exc}",
            class_names={},
            violation_map={},
            helmet_class_indices=[],
            has_helmet_capability=False,
        )

    vmap = _build_vmap(raw_names, custom)
    helmet_idx = [i for i, k in vmap.items() if k == "helmet"]
    return ViolationModelInfo(
        model_path=str(model_path),
        exists=True,
        load_error=None,
        class_names=raw_names,
        violation_map=vmap,
        helmet_class_indices=helmet_idx,
        has_helmet_capability=bool(helmet_idx),
    )
