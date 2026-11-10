"""Unit and integration tests for the helmet-violation detection module.

All tests inject fake model outputs so they run offline with no GPU and
without real model weights. The goal is to verify:
  - spatial rider association logic
  - temporal voting and stability threshold
  - duplicate violation prevention
  - graceful degradation when violation model is missing or incompatible
  - vehicle detection continues when helmet detection is disabled
  - existing database and ANPR tests are unaffected
"""
from __future__ import annotations

import ast
import sqlite3
import sys
import types
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.rules.helmet import (  # noqa: E402
    HelmetDebugMetrics,
    HelmetObservation,
    HelmetObserver,
    ViolationModelInfo,
    associate_rider,
    describe_violation_model,
    head_region,
    no_helmet_on_rider,
    HELMET,
    NO_HELMET,
    UNKNOWN,
)


# ----------------------------------------------------------------- spatial helpers

def test_associate_rider_finds_person_above_motorcycle():
    moto = (100, 200, 300, 350)   # motorcycle box
    # Rider whose centre is above the motorcycle (sitting on top).
    rider = (120, 60, 280, 210)   # centre at (200, 135)
    assert rider in associate_rider(moto, [rider])


def test_associate_rider_rejects_unrelated_pedestrian():
    moto = (100, 200, 300, 350)
    # Pedestrian far to the right — centre outside search region.
    pedestrian = (500, 150, 650, 350)
    assert associate_rider(moto, [pedestrian]) == []


def test_associate_rider_rejects_person_below_motorcycle():
    moto = (100, 200, 300, 350)
    # Person whose centre is below the motorcycle axle.
    below = (130, 370, 270, 500)
    assert associate_rider(moto, [below]) == []


def test_associate_rider_returns_multiple_for_triple_riding():
    moto = (50, 100, 250, 300)
    r1 = (80, 30, 220, 120)
    r2 = (90, 40, 210, 130)
    r3 = (100, 50, 200, 140)
    riders = associate_rider(moto, [r1, r2, r3])
    assert len(riders) == 3


def test_no_helmet_on_rider_matches_when_centre_inside():
    rider = (100, 50, 300, 300)
    # Detection whose centre (200, 80) is inside the rider box.
    det = (170, 60, 230, 100)
    assert no_helmet_on_rider(det, rider) is True


def test_no_helmet_on_rider_rejects_when_centre_outside():
    rider = (100, 50, 300, 300)
    # Detection whose centre is to the left of the rider box.
    det = (20, 60, 80, 100)    # centre (50, 80) — outside rider
    assert no_helmet_on_rider(det, rider) is False


def test_head_region_returns_upper_fraction():
    person = (100, 50, 300, 250)   # height = 200
    head = head_region(person, head_pct=0.35)
    x1, y1, x2, y2 = head
    assert x1 == 100 and x2 == 300
    assert y1 == 50
    assert y2 == 50 + int(200 * 0.35)


# ----------------------------------------------------------------- HelmetObserver

def test_observer_single_frame_not_stable():
    """One no_helmet frame must not produce a stable candidate."""
    obs = HelmetObserver(min_no_helmet_votes=3, min_conf=0.3)
    moto = {1: (100, 200, 300, 350)}
    rider = [(120, 60, 280, 210)]     # person overlapping motorcycle
    det = [(0.85, (150, 80, 250, 140))]   # no_helmet on rider
    results = obs.process_frame(1, moto, rider, det)
    assert len(results) == 1
    assert results[0].helmet_status == NO_HELMET
    assert results[0].is_stable_candidate is False


def test_observer_reaches_stable_after_min_votes():
    """Stable candidate is only set once min_no_helmet_votes are reached."""
    obs = HelmetObserver(min_no_helmet_votes=3, min_conf=0.3)
    moto = {5: (100, 200, 300, 350)}
    rider = [(120, 60, 280, 210)]
    det = [(0.9, (150, 80, 250, 140))]
    for i in range(1, 3):
        results = obs.process_frame(i, moto, rider, det)
        assert not results[0].is_stable_candidate, f"should not be stable at frame {i}"
    # Third vote → stable.
    results = obs.process_frame(3, moto, rider, det)
    assert results[0].is_stable_candidate is True
    assert results[0].helmet_status == NO_HELMET
    assert 5 in obs.promoted_tracks


def test_observer_no_duplicate_promotion():
    """Promoted tracks are flagged once; subsequent frames don't re-promote."""
    obs = HelmetObserver(min_no_helmet_votes=2, min_conf=0.3)
    moto = {7: (100, 200, 300, 350)}
    rider = [(120, 60, 280, 210)]
    det = [(0.8, (150, 80, 250, 140))]
    for i in range(1, 5):
        obs.process_frame(i, moto, rider, det)
    assert obs.metrics.stable_no_helmet_candidates == 1


def test_observer_unknown_when_no_rider():
    obs = HelmetObserver(min_no_helmet_votes=2, min_conf=0.3)
    moto = {3: (100, 200, 300, 350)}
    # No persons in frame.
    results = obs.process_frame(1, moto, [], [(0.9, (150, 80, 250, 140))])
    assert results[0].helmet_status == UNKNOWN
    assert results[0].is_stable_candidate is False


def test_observer_unknown_when_rider_present_but_no_detection_overlaps():
    """Rider found but no_helmet box does not overlap → unknown."""
    obs = HelmetObserver(min_no_helmet_votes=2, min_conf=0.3)
    moto = {4: (100, 200, 300, 350)}
    rider = [(120, 60, 280, 210)]  # rider centre at (200, 135)
    # Detection far away from the rider.
    det = [(0.9, (500, 400, 600, 500))]
    results = obs.process_frame(1, moto, rider, det)
    assert results[0].helmet_status == UNKNOWN


def test_observer_low_confidence_detection_ignored():
    obs = HelmetObserver(min_no_helmet_votes=2, min_conf=0.5)
    moto = {2: (100, 200, 300, 350)}
    rider = [(120, 60, 280, 210)]
    det = [(0.2, (150, 80, 250, 140))]   # conf below threshold
    results = obs.process_frame(1, moto, rider, det)
    assert results[0].helmet_status == UNKNOWN


def test_observer_nearby_pedestrian_not_associated():
    """A person far from the motorcycle must not be treated as a rider."""
    obs = HelmetObserver(min_no_helmet_votes=2, min_conf=0.3)
    moto = {6: (100, 200, 300, 350)}
    # Pedestrian far to the right; their centre is outside the search region.
    pedestrian = [(600, 150, 750, 380)]
    # no_helmet detection centred on the pedestrian.
    det = [(0.9, (640, 180, 720, 250))]
    results = obs.process_frame(1, moto, pedestrian, det)
    assert results[0].helmet_status == UNKNOWN


def test_observer_mixed_observations_not_stable():
    """Alternating no_helmet and unknown observations do not reach threshold."""
    obs = HelmetObserver(min_no_helmet_votes=3, min_conf=0.3)
    moto = {8: (100, 200, 300, 350)}
    rider = [(120, 60, 280, 210)]
    det_hit = [(0.9, (150, 80, 250, 140))]   # overlaps rider
    det_miss = [(0.9, (600, 400, 700, 500))]  # far away
    for i in range(1, 7):
        d = det_hit if i % 2 == 1 else det_miss
        obs.process_frame(i, moto, rider, d)
    # Only 3 of 6 frames had a hit, but observe votes vs threshold.
    assert obs.metrics.no_helmet_observations == 3
    # 3 votes == threshold so it IS stable; assert the boundary correctly.
    assert 8 in obs.promoted_tracks


def test_observer_empty_motorcycles_returns_empty():
    obs = HelmetObserver(min_no_helmet_votes=2, min_conf=0.3)
    results = obs.process_frame(1, {}, [(0, 0, 100, 100)], [(0.9, (0, 0, 50, 50))])
    assert results == []


def test_observer_metrics_counts():
    obs = HelmetObserver(min_no_helmet_votes=2, min_conf=0.3)
    moto = {1: (100, 200, 300, 350)}
    rider = [(120, 60, 280, 210)]
    det = [(0.9, (150, 80, 250, 140))]
    obs.process_frame(1, moto, rider, det)
    obs.process_frame(2, moto, rider, det)
    obs.process_frame(3, moto, [], [])      # no rider → unknown
    m = obs.finalize_metrics()
    assert m.no_helmet_observations == 2
    assert m.unknown_observations == 1
    assert m.stable_no_helmet_candidates == 1
    assert m.motorcycles_observed == 3
    assert m.rider_associations_attempted == 3


def test_observer_invalid_min_votes_raises():
    with pytest.raises(ValueError):
        HelmetObserver(min_no_helmet_votes=0)


def test_observer_invalid_window_frames_raises():
    with pytest.raises(ValueError):
        HelmetObserver(window_frames=0)


def test_observer_votes_outside_window_do_not_count():
    """Observations older than window_frames are pruned and do not count."""
    obs = HelmetObserver(min_no_helmet_votes=2, min_conf=0.3, window_frames=10)
    moto = {1: (100, 200, 300, 350)}
    rider = [(120, 60, 280, 210)]
    det = [(0.8, (150, 80, 250, 140))]

    # Two votes at frames 1 and 3 (both inside a window_frames=10 window).
    obs.process_frame(1, moto, rider, det)
    obs.process_frame(3, moto, rider, det)

    # Jump to frame 20: frames 1 and 3 are now more than 10 frames ago.
    # The new vote (frame 20) alone is not enough for min_no_helmet_votes=2.
    results = obs.process_frame(20, moto, rider, det)
    assert results[0].helmet_status == NO_HELMET   # this frame has a detection
    assert results[0].is_stable_candidate is False  # only 1 vote in window


def test_observer_window_resets_stability():
    """A track promoted at frame 3 is no longer stable at frame 20 (old votes pruned)."""
    obs = HelmetObserver(min_no_helmet_votes=2, min_conf=0.3, window_frames=5)
    moto = {2: (100, 200, 300, 350)}
    rider = [(120, 60, 280, 210)]
    det = [(0.9, (150, 80, 250, 140))]

    obs.process_frame(1, moto, rider, det)
    r = obs.process_frame(2, moto, rider, det)
    assert r[0].is_stable_candidate is True   # promoted at frame 2
    assert 2 in obs.promoted_tracks

    # Frame 20: both old votes (1, 2) are outside the 5-frame window.
    # Only the single new vote at frame 20 is in the window.
    r2 = obs.process_frame(20, moto, rider, det)
    assert r2[0].is_stable_candidate is False  # 1 < 2 votes required
    # Track remains in promoted_tracks (violation already recorded by caller).
    assert 2 in obs.promoted_tracks


def test_observer_moto_expand_pct_controls_search_width():
    """associate_rider with a narrow expand_pct misses a marginal rider that a wider one finds."""
    moto = (100, 200, 300, 350)   # width=200
    # Rider centre at x=85, which is 15px left of the motorcycle edge (100).
    # narrow: ex = 0.05*200=10  → search from 90 → misses (85 < 90)
    # wider:  ex = 0.15*200=30  → search from 70 → finds  (85 > 70)
    marginal_rider = [(50, 60, 120, 210)]  # centre x=(50+120)/2 = 85
    assert associate_rider(moto, marginal_rider, expand_pct=0.05) == []
    assert marginal_rider[0] in associate_rider(moto, marginal_rider, expand_pct=0.15)


def test_frame_analyzer_passes_moto_expand_pct_to_observer(tmp_path):
    """helmet_moto_expand_pct from opts is forwarded to HelmetObserver."""
    vmodel_path = tmp_path / "violations.pt"
    vmodel_path.write_bytes(b"fake")

    call_count = {"n": 0}
    def _fake_yolo(path):
        call_count["n"] += 1
        return _FakeCOCOModel() if call_count["n"] == 1 else _FakeViolationModel({0: "no_helmet"})

    FrameAnalyzer = _load_frame_analyzer_class({
        "YOLO": _fake_yolo,
        "VIOLATION_MODEL_PATH": vmodel_path,
    })
    analyzer = FrameAnalyzer("test", {"helmet_moto_expand_pct": "0.25", "helmet_window_frames": "15"})
    assert analyzer.helmet_observer is not None
    assert analyzer.helmet_observer.moto_expand_pct == pytest.approx(0.25)
    assert analyzer.helmet_observer.window_frames == 15


# ----------------------------------------------------------------- debug metrics

def test_debug_metrics_lines_do_not_use_word_accuracy():
    m = HelmetDebugMetrics(
        motorcycles_observed=10, rider_associations_attempted=8,
        no_helmet_observations=5, stable_no_helmet_candidates=2,
    )
    text = "\n".join(m.as_lines()).lower()
    assert "accuracy" not in text
    assert "no-helmet observations" in text
    assert "stable no-helmet candidates" in text


# ----------------------------------------------------------------- describe_violation_model

class _FakeModel:
    """Injectable fake model for describe_violation_model tests."""
    def __init__(self, names: dict[int, str]):
        self.names = names


def test_describe_returns_missing_when_file_absent(tmp_path):
    info = describe_violation_model(tmp_path / "nonexistent.pt")
    assert info.exists is False
    assert info.has_helmet_capability is False
    lines = "\n".join(info.as_lines()).lower()
    assert "missing" in lines


def test_describe_detects_helmet_class_via_name_hint():
    fake = _FakeModel({0: "no_helmet", 1: "no_seatbelt", 2: "car"})
    info = describe_violation_model("fake.pt", _model=fake)
    assert info.has_helmet_capability is True
    assert 0 in info.helmet_class_indices
    assert info.violation_map[0] == "helmet"
    assert info.violation_map[1] == "belt"
    assert info.violation_map[2] is None   # car is not a violation class


def test_describe_no_helmet_class_returns_unavailable():
    fake = _FakeModel({0: "car", 1: "truck", 2: "person"})
    info = describe_violation_model("fake.pt", _model=fake)
    assert info.has_helmet_capability is False
    assert info.helmet_class_indices == []
    lines = "\n".join(info.as_lines())
    assert "UNAVAILABLE" in lines


def test_describe_with_custom_map_override(tmp_path):
    """violation_classes.json class name override is respected."""
    import json
    map_file = tmp_path / "violation_classes.json"
    map_file.write_text(json.dumps({"bare_head": "helmet"}))
    fake = _FakeModel({0: "bare_head", 1: "no_seatbelt"})
    info = describe_violation_model("fake.pt", str(map_file), _model=fake)
    assert info.has_helmet_capability is True
    assert 0 in info.helmet_class_indices


def test_describe_as_lines_contains_class_names():
    fake = _FakeModel({0: "no_helmet", 1: "no_seatbelt"})
    info = describe_violation_model("violations.pt", _model=fake)
    text = "\n".join(info.as_lines())
    assert "no_helmet" in text
    assert "no_seatbelt" in text


# ----------------------------------------------------------------- FrameAnalyzer integration (Qt-free)

def _load_frame_analyzer_class(scope_overrides: dict):
    """Load FrameAnalyzer from main.py via AST without importing Qt."""
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    analyzer_cls_node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FrameAnalyzer"
    )
    # Also pull in module-level names needed by FrameAnalyzer.
    name_nodes = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                          ast.AugAssign, ast.AnnAssign))
        and getattr(n, "name", None) in {
            "_log_violation_model_warning", "norm_name", "build_violation_map",
        }
    ]

    from ai.rules.helmet import HelmetObserver, HelmetDebugMetrics
    from poc_support import ensure_column, should_count_vehicle, normalize_plate

    scope = {
        "defaultdict": __import__("collections").defaultdict,
        "deque": __import__("collections").deque,
        "Path": Path,
        "datetime": datetime,
        "re": __import__("re"),
        "json": __import__("json"),
        "cv2": __import__("cv2"),
        "YOLO": scope_overrides.get("YOLO"),
        "VEHICLES": {1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"},
        "PERSON_CLASS": 0,
        "VIOLATIONS": {
            "helmet": ("Riding without helmet", 1000, False),
            "triple": ("Triple riding", 1000, False),
            "belt": ("Driving without seat belt", 1000, False),
            "wrongway": ("Wrong-way / no entry", 500, False),
            "lights": ("Unauthorized police lights", 1000, True),
            "rooftop": ("Person sitting on top of vehicle", 1000, True),
        },
        "NAME_HINTS": {
            "helmet": ("no_helmet", "nohelmet", "without_helmet", "no_hardhat"),
            "belt": ("no_seatbelt", "no_seat_belt", "noseatbelt", "without_seatbelt", "without_seat_belt"),
            "lights": ("police_light", "beacon", "siren"),
            "rooftop": ("person_on_top", "person_on_roof", "rooftop", "on_roof"),
            "triple": ("triple_riding", "triple"),
        },
        "VIOLATION_MODEL_PATH": scope_overrides.get("VIOLATION_MODEL_PATH", Path("nonexistent.pt")),
        "VIOLATION_MAP_PATH": scope_overrides.get("VIOLATION_MAP_PATH", Path("nonexistent_map.json")),
        "MODEL_PATH": Path("nonexistent.pt"),
        "AUTODOWNLOADABLE_MODELS": set(),
        "HelmetObserver": HelmetObserver,
        "HelmetDebugMetrics": HelmetDebugMetrics,
        "should_count_vehicle": should_count_vehicle,
        "save_detection": lambda *a, **kw: None,
        "save_violation": scope_overrides.get("save_violation", lambda *a, **kw: 1),
        "db": types.SimpleNamespace(next_counter=lambda k: 1),
        "BASE_DIR": Path(__file__).resolve().parents[1],
        "now_str": lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "norm_plate": lambda t: "".join(c for c in str(t).upper() if c.isalnum()),
        **scope_overrides,
    }
    exec(
        compile(
            ast.Module(body=name_nodes + [analyzer_cls_node], type_ignores=[]),
            str(root / "main.py"),
            "exec",
        ),
        scope,
    )
    return scope["FrameAnalyzer"]


class _FakeSeries:
    def __init__(self, values):
        self.values = values
    def int(self): return self
    def tolist(self): return list(self.values)


class _FakeBoxes:
    def __init__(self, cls, ids, confs, xyxys):
        self.cls = _FakeSeries(cls)
        self.id = _FakeSeries(ids)
        self.conf = _FakeSeries(confs)
        self.xyxy = _FakeSeries(xyxys)
    def __len__(self): return len(self.cls.values)


class _FakePrediction:
    def __init__(self, frame, boxes):
        self._frame = frame
        self.boxes = boxes
    def plot(self): return self._frame.copy()


class _FakeCOCOModel:
    """Primary vehicle-detection model: one motorcycle (track 10) + one rider (track 20)."""
    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def track(self, frame, **kwargs):
        boxes = _FakeBoxes(
            cls=[3, 0],           # motorcycle + person (rider)
            ids=[10, 20],
            confs=[0.9, 0.85],
            xyxys=[
                [100, 200, 300, 350],  # motorcycle bbox
                [120, 60, 280, 210],   # rider — centre (200, 135) is above motorcycle
            ],
        )
        return [_FakePrediction(frame, boxes)]


class _FakeViolationModel:
    """Violation model: returns one no_helmet detection on the rider."""
    def __init__(self, names):
        self.names = names
    def track(self, frame, **kwargs):
        boxes = _FakeBoxes(
            cls=[0],          # index 0 = "no_helmet"
            ids=[None],       # some violation models don't track
            confs=[0.85],
            xyxys=[[150, 60, 250, 120]],   # overlaps rider head region
        )
        return [_FakePrediction(frame, boxes)]


class _FakeViolationModelNoHelmetClass:
    """Violation model with no helmet-related class."""
    def __init__(self):
        self.names = {0: "car", 1: "truck"}
    def track(self, frame, **kwargs):
        return [_FakePrediction(frame, None)]


def test_frame_analyzer_vehicle_detection_unaffected_by_missing_vmodel(tmp_path):
    """Vehicle detection must work when violations.pt is absent."""
    def _fake_yolo(path):
        return _FakeCOCOModel()

    FrameAnalyzer = _load_frame_analyzer_class({
        "YOLO": _fake_yolo,
        "VIOLATION_MODEL_PATH": tmp_path / "nonexistent.pt",
    })
    analyzer = FrameAnalyzer("test", {})
    assert analyzer._helmet_status == "model_missing"
    assert analyzer.vmodel is None
    assert analyzer.helmet_observer is None
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, in_frame, saved = analyzer.process(frame, 1)
    assert in_frame == 1


def test_frame_analyzer_helmet_unavailable_when_no_class(tmp_path):
    """violations.pt without a helmet class disables helmet observer but keeps other violations."""
    vmodel_path = tmp_path / "violations.pt"
    vmodel_path.write_bytes(b"fake")  # file exists but fake

    call_count = {"n": 0}
    def _fake_yolo(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCOCOModel()
        return _FakeViolationModelNoHelmetClass()

    FrameAnalyzer = _load_frame_analyzer_class({
        "YOLO": _fake_yolo,
        "VIOLATION_MODEL_PATH": vmodel_path,
    })
    analyzer = FrameAnalyzer("test", {})
    assert analyzer._helmet_status == "no_helmet_class"
    assert analyzer.helmet_observer is None
    assert analyzer.vmodel is not None


def test_frame_analyzer_helmet_available_when_class_present(tmp_path):
    """When violations.pt has a no_helmet class, HelmetObserver is initialized."""
    vmodel_path = tmp_path / "violations.pt"
    vmodel_path.write_bytes(b"fake")

    call_count = {"n": 0}
    def _fake_yolo(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCOCOModel()
        return _FakeViolationModel({0: "no_helmet"})

    FrameAnalyzer = _load_frame_analyzer_class({
        "YOLO": _fake_yolo,
        "VIOLATION_MODEL_PATH": vmodel_path,
    })
    analyzer = FrameAnalyzer("test", {})
    assert analyzer._helmet_status == "available"
    assert analyzer.helmet_observer is not None


def test_frame_analyzer_stable_candidate_creates_violation(tmp_path):
    """After min_no_helmet_votes frames with rider+no_helmet overlap, a violation is saved."""
    vmodel_path = tmp_path / "violations.pt"
    vmodel_path.write_bytes(b"fake")
    saved_violations: list[dict] = []

    def _fake_save_violation(type_key, frame, source, **kwargs):
        saved_violations.append({"type_key": type_key, **kwargs})
        return 1

    call_count = {"n": 0}
    def _fake_yolo(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCOCOModel()
        return _FakeViolationModel({0: "no_helmet"})

    FrameAnalyzer = _load_frame_analyzer_class({
        "YOLO": _fake_yolo,
        "VIOLATION_MODEL_PATH": vmodel_path,
        "save_violation": _fake_save_violation,
    })
    analyzer = FrameAnalyzer("test", {"min_no_helmet_votes": 3})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(1, 5):
        analyzer.process(frame, i)
    assert any(v["type_key"] == "helmet" for v in saved_violations)
    # The vehicle track_id (10) must be attached, not None.
    helmet_violations = [v for v in saved_violations if v["type_key"] == "helmet"]
    assert helmet_violations[0].get("track_id") == 10


def test_frame_analyzer_no_duplicate_violation_for_same_track(tmp_path):
    """The same motorcycle track must not generate more than one helmet violation."""
    vmodel_path = tmp_path / "violations.pt"
    vmodel_path.write_bytes(b"fake")
    saved_violations: list[str] = []

    def _fake_save_violation(type_key, frame, source, **kwargs):
        if type_key == "helmet":
            saved_violations.append(type_key)
        return 1

    call_count = {"n": 0}
    def _fake_yolo(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCOCOModel()
        return _FakeViolationModel({0: "no_helmet"})

    FrameAnalyzer = _load_frame_analyzer_class({
        "YOLO": _fake_yolo,
        "VIOLATION_MODEL_PATH": vmodel_path,
        "save_violation": _fake_save_violation,
    })
    analyzer = FrameAnalyzer("test", {"min_no_helmet_votes": 2})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(1, 20):
        analyzer.process(frame, i)
    assert len(saved_violations) == 1


def test_frame_analyzer_violation_model_error_does_not_crash(tmp_path):
    """A RuntimeError in the violation model must not propagate to the caller."""
    vmodel_path = tmp_path / "violations.pt"
    vmodel_path.write_bytes(b"fake")

    class _BrokenVModel:
        names = {0: "no_helmet"}
        def track(self, frame, **kwargs):
            raise RuntimeError("simulated model crash")

    call_count = {"n": 0}
    def _fake_yolo(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCOCOModel()
        return _BrokenVModel()

    FrameAnalyzer = _load_frame_analyzer_class({
        "YOLO": _fake_yolo,
        "VIOLATION_MODEL_PATH": vmodel_path,
    })
    # __init__ succeeds; helmet observer would be created
    analyzer = FrameAnalyzer("test", {})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Must not raise — vehicle detection continues.
    annotated, in_frame, saved = analyzer.process(frame, 1)
    assert in_frame == 1


def test_frame_analyzer_helmet_observer_error_does_not_crash(tmp_path):
    """An error inside HelmetObserver.process_frame must not propagate."""
    vmodel_path = tmp_path / "violations.pt"
    vmodel_path.write_bytes(b"fake")

    class _BrokenObserver:
        promoted_tracks: set = set()
        def process_frame(self, *a, **kw):
            raise RuntimeError("observer crash")
        def finalize_metrics(self):
            return HelmetDebugMetrics()

    call_count = {"n": 0}
    def _fake_yolo(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCOCOModel()
        return _FakeViolationModel({0: "no_helmet"})

    FrameAnalyzer = _load_frame_analyzer_class({
        "YOLO": _fake_yolo,
        "VIOLATION_MODEL_PATH": vmodel_path,
    })
    analyzer = FrameAnalyzer("test", {})
    # Inject the broken observer.
    analyzer.helmet_observer = _BrokenObserver()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, in_frame, saved = analyzer.process(frame, 1)
    assert in_frame == 1   # vehicle detection must survive


# ----------------------------------------------------------------- model class mapping

def test_build_violation_map_no_helmet_hint():
    """build_violation_map correctly maps class names to violation keys."""
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    fn_node = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "build_violation_map")
    scope = {
        "json": __import__("json"),
        "Path": Path,
        "VIOLATION_MAP_PATH": Path("nonexistent.json"),
        "NAME_HINTS": {
            "helmet": ("no_helmet", "nohelmet", "without_helmet", "no_hardhat"),
            "belt": ("no_seatbelt", "no_seat_belt", "noseatbelt", "without_seatbelt", "without_seat_belt"),
            "lights": ("police_light", "beacon", "siren"),
            "rooftop": ("person_on_top", "person_on_roof", "rooftop", "on_roof"),
            "triple": ("triple_riding", "triple"),
        },
        "VIOLATIONS": {
            "helmet": (), "triple": (), "belt": (), "wrongway": (), "lights": (), "rooftop": ()
        },
    }
    fn_node_only = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "norm_name"
    )
    exec(
        compile(ast.Module(body=[fn_node_only, fn_node], type_ignores=[]),
                str(root / "main.py"), "exec"),
        scope,
    )
    build_violation_map = scope["build_violation_map"]

    # Model with both helmet and non-helmet classes.
    names = {0: "no_helmet", 1: "no_seatbelt", 2: "car", 3: "triple_riding"}
    vmap = build_violation_map(names)
    assert vmap[0] == "helmet"
    assert vmap[1] == "belt"
    assert vmap[2] is None
    assert vmap[3] == "triple"


def test_build_violation_map_with_custom_json(tmp_path):
    """violation_classes.json overrides take precedence over NAME_HINTS."""
    import json as _json
    map_file = tmp_path / "violation_classes.json"
    map_file.write_text(_json.dumps({"bare_head": "helmet"}))

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    fn_nodes = [n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name in ("norm_name", "build_violation_map")]
    scope = {
        "json": _json,
        "Path": Path,
        "VIOLATION_MAP_PATH": map_file,
        "NAME_HINTS": {"helmet": ("no_helmet",)},
        "VIOLATIONS": {"helmet": ()},
    }
    exec(compile(ast.Module(body=fn_nodes, type_ignores=[]), str(root / "main.py"), "exec"), scope)
    vmap = scope["build_violation_map"]({0: "bare_head", 1: "no_helmet"})
    assert vmap[0] == "helmet"   # custom override
    assert vmap[1] == "helmet"   # hint still applies too


def test_observer_prunes_votes_on_empty_detection_frame():
    obs = HelmetObserver(
        min_no_helmet_votes=2,
        min_conf=0.30,
        window_frames=5,
    )

    motorcycles = {1: (100, 200, 300, 350)}
    riders = [(120, 60, 280, 210)]
    no_helmet = [(0.9, (150, 80, 250, 140))]

    obs.process_frame(1, motorcycles, riders, no_helmet)
    result = obs.process_frame(2, motorcycles, riders, no_helmet)

    assert result[0].is_stable_candidate is True

    # No detections, and old votes must expire.
    result = obs.process_frame(20, motorcycles, riders, [])

    assert result[0].helmet_status == UNKNOWN
    assert result[0].is_stable_candidate is False

    # One fresh detection must not immediately become stable.
    result = obs.process_frame(21, motorcycles, riders, no_helmet)

    assert result[0].is_stable_candidate is False


# ---------------------------------------- candidate_violations_queued / saved counters
def _make_helmet_analyzer(tmp_path, save_violation_fn=None, min_votes=2):
    """Helper: load FrameAnalyzer with helmet-capable fake models."""
    vmodel_path = tmp_path / "violations.pt"
    vmodel_path.write_bytes(b"fake")
    call_count = {"n": 0}

    def _fake_yolo(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCOCOModel()
        return _FakeViolationModel({0: "no_helmet"})

    overrides = {"YOLO": _fake_yolo, "VIOLATION_MODEL_PATH": vmodel_path}
    if save_violation_fn is not None:
        overrides["save_violation"] = save_violation_fn
    return _load_frame_analyzer_class(overrides)("test", {"min_no_helmet_votes": min_votes})


def test_queued_and_saved_both_increment_on_successful_persistence(tmp_path):
    """When save_violation succeeds, candidate_violations_queued and
    candidate_violations_saved must both be 1 for a single stable track.

    Guards against the original bug where candidate_violations_created was
    always 0 — the previous fix renamed and split it into two counters.
    """
    saved: list[str] = []

    def _ok_save(type_key, frame, source, **kwargs):
        saved.append(type_key)
        return 1

    analyzer = _make_helmet_analyzer(tmp_path, save_violation_fn=_ok_save)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(1, 10):
        analyzer.process(frame, i)

    metrics = analyzer.helmet_observer.finalize_metrics()

    assert metrics.candidate_violations_queued == 1, (
        "candidate_violations_queued must be 1: the unique stable track was "
        "appended to the pending list exactly once."
    )
    assert metrics.candidate_violations_saved == 1, (
        "candidate_violations_saved must be 1: save_violation succeeded so the "
        "row was committed to SQLite."
    )
    assert saved.count("helmet") == 1, "exactly one helmet violation must be saved"


def test_queued_increments_but_saved_stays_zero_when_persistence_fails(tmp_path):
    """When save_violation raises, candidate_violations_queued must still be 1
    (the violation was queued) but candidate_violations_saved must remain 0
    (nothing was written to SQLite).  Vehicle counting must continue normally.
    """
    def _failing_save(type_key, frame, source, **kwargs):
        if type_key == "helmet":
            raise OSError("simulated evidence-write failure")
        return 1

    analyzer = _make_helmet_analyzer(tmp_path, save_violation_fn=_failing_save)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Run must complete — a save failure must not crash vehicle counting.
    for i in range(1, 10):
        annotated, in_frame, _saved_count = analyzer.process(frame, i)
    assert in_frame >= 1, "vehicle detection must continue despite save_violation failure"

    metrics = analyzer.helmet_observer.finalize_metrics()

    assert metrics.candidate_violations_queued == 1, (
        "queued must be 1: the track was stable and the violation was added to pending"
    )
    assert metrics.candidate_violations_saved == 0, (
        "saved must be 0: save_violation raised before the row was committed"
    )


def test_duplicate_suppression_preserved_queued_stays_one_across_many_frames(tmp_path):
    """The flagged-set guard must keep candidate_violations_queued at 1 regardless
    of how many subsequent frames the same track remains stable.

    This confirms the duplicate suppression that existed before the counter split
    still works correctly with the new counter names.
    """
    analyzer = _make_helmet_analyzer(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(1, 30):
        analyzer.process(frame, i)

    metrics = analyzer.helmet_observer.finalize_metrics()
    assert metrics.candidate_violations_queued == 1, (
        "candidate_violations_queued must be 1 regardless of how many frames "
        "the track remains stable after the first violation is queued."
    )
    assert metrics.candidate_violations_saved == 1, (
        "candidate_violations_saved must also be 1: the single violation was saved."
    )
