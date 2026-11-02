"""Unit coverage for the reusable ANPR module and its dashboard integration points.

The tests avoid loading real YOLO weights or EasyOCR. They inject fakes to
exercise voting, association, debug metrics and graceful degradation on
failure, so they run offline on any machine.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.ocr.anpr import (  # noqa: E402
    AnprDebugMetrics,
    AnprPipeline,
    AnprResult,
    PlateVoteTracker,
    associate_plate,
)
from poc_support import ensure_column, normalize_plate  # noqa: E402


# --------------------------------------------------------------------- helpers
class _FakeSeries:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _FakeSeries(xyxy)
        self.conf = _FakeSeries(conf)

    def __len__(self):
        return len(self.xyxy.values)


class _FakePrediction:
    def __init__(self, xyxy, conf):
        self.boxes = _FakeBoxes(xyxy, conf)


class _FakePlateModel:
    """Yields the same plate box every frame."""

    def __init__(self, xyxy, conf):
        self._xyxy = xyxy
        self._conf = conf
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        return [_FakePrediction(self._xyxy, self._conf)]


class _FailingPlateModel:
    def predict(self, frame, **kwargs):
        raise RuntimeError("boom")


class _FakeReader:
    """Simulates EasyOCR by returning a configured plate text each call."""

    def __init__(self, text="TS10UA1675", conf=0.9):
        self.text = text
        self.conf = conf
        self.calls = 0

    def readtext(self, image, detail=1, allowlist=None):
        self.calls += 1
        return [([[0, 0], [10, 0], [10, 10], [0, 10]], self.text, self.conf)]


# ----------------------------------------------------- normalization + voting
def test_normalize_does_not_rewrite_ambiguous_chars():
    assert normalize_plate("ts10 ua 1675") == "TS10UA1675"
    # 0/O and 1/I distinctions remain the caller's responsibility.
    assert normalize_plate("TS1OUA167S") == "TS1OUA167S"


def test_vote_tracker_requires_min_repeats_for_stability():
    tracker = PlateVoteTracker(min_votes=2)
    tracker.record(27, 100, "TS10UA1675", 0.8, 0.9)
    assert tracker.stable_candidate(27) is None
    tracker.record(27, 104, "TS1OUA167S", 0.99, 0.9)
    # The impostor plate has only one vote and is not Indian-standard;
    # the real one has one vote. No stable candidate yet.
    assert tracker.stable_candidate(27) is None
    tracker.record(27, 108, "TS10UA1675", 0.75, 0.9)
    stable = tracker.stable_candidate(27)
    assert stable is not None
    plate, confidence, vote_count = stable
    assert plate == "TS10UA1675"
    assert vote_count == 2
    assert confidence == pytest.approx(0.8)


def test_vote_tracker_preserves_history_for_debug():
    tracker = PlateVoteTracker(min_votes=2)
    tracker.record(5, 10, "TS10UA1675", 0.8, 0.9)
    tracker.record(5, 15, "TS10UA1675", 0.7, 0.9)
    assert tracker.first_frame[5] == 10
    assert tracker.last_frame[5] == 15
    assert len(tracker.history[5]) == 2


# ----------------------------------------------------- plate↔vehicle association
def test_associate_prefers_smallest_containing_box():
    big = (0, 0, 400, 400)
    small = (50, 50, 150, 150)
    # Plate center at (100, 100) sits inside both; prefer the smaller box.
    assert associate_plate((80, 80, 120, 120), {1: big, 2: small}) == 2


def test_associate_returns_none_when_no_box_contains_plate():
    assert associate_plate((500, 500, 520, 520), {1: (0, 0, 100, 100)}) is None


# ------------------------------------------------------- AnprPipeline behaviour
def _make_frame():
    return np.full((400, 400, 3), 180, dtype=np.uint8)


def test_pipeline_builds_stable_candidate_without_duplicating_tracks():
    reader = _FakeReader()
    plate_model = _FakePlateModel(xyxy=[[100, 150, 300, 220]], conf=[0.9])
    pipeline = AnprPipeline(
        "ignored.pt",
        ocr_every=1,
        min_votes=2,
        min_plate_width=50,
        min_plate_height=5,
        ocr_min_conf=0.1,
        plate_model=plate_model,
        ocr_reader=reader,
    )
    vehicles = {7: (50, 50, 350, 350)}
    frame = _make_frame()
    first = pipeline.process_frame(frame, 1, vehicles)
    second = pipeline.process_frame(frame, 2, vehicles)
    assert len(first) == 1 and not first[0].is_stable_candidate
    assert len(second) == 1 and second[0].is_stable_candidate
    assert first[0].track_id == 7 and second[0].track_id == 7
    # Only one track got promoted even after many observations.
    pipeline.process_frame(frame, 3, vehicles)
    pipeline.process_frame(frame, 4, vehicles)
    metrics = pipeline.finalize_metrics()
    assert metrics.unique_tracks_with_plate == 1
    assert metrics.stable_candidates == 1  # promotion counted once per track


def test_pipeline_does_not_double_link_a_plate_to_multiple_tracks():
    reader = _FakeReader()
    plate_model = _FakePlateModel(
        xyxy=[[100, 150, 300, 220], [120, 160, 290, 215]],  # two overlapping boxes
        conf=[0.9, 0.8],
    )
    pipeline = AnprPipeline(
        "ignored.pt",
        ocr_every=1,
        min_votes=1,
        min_plate_width=50,
        min_plate_height=5,
        ocr_min_conf=0.1,
        plate_model=plate_model,
        ocr_reader=reader,
    )
    vehicles = {1: (40, 40, 360, 360), 2: (10, 10, 390, 390)}
    results = pipeline.process_frame(_make_frame(), 1, vehicles)
    # Two boxes detected but both resolve to the smallest enclosing track.
    assert pipeline.metrics.plate_boxes_detected == 2
    assert pipeline.metrics.plate_boxes_linked == 1
    assert len({r.track_id for r in results}) == 1


def test_pipeline_filters_tiny_plates_without_crashing():
    reader = _FakeReader()
    plate_model = _FakePlateModel(xyxy=[[100, 150, 110, 160]], conf=[0.9])
    pipeline = AnprPipeline(
        "ignored.pt",
        ocr_every=1,
        min_votes=1,
        min_plate_width=50,
        plate_model=plate_model,
        ocr_reader=reader,
    )
    vehicles = {1: (0, 0, 400, 400)}
    assert pipeline.process_frame(_make_frame(), 1, vehicles) == []
    assert pipeline.metrics.plate_boxes_filtered == 1


def test_pipeline_degrades_gracefully_when_model_raises():
    pipeline = AnprPipeline(
        "ignored.pt",
        ocr_every=1,
        min_votes=1,
        plate_model=_FailingPlateModel(),
        ocr_reader=_FakeReader(),
    )
    # Model loading reported "enabled" because we injected fakes, but the
    # predict call blows up. The pipeline must swallow it and disable itself.
    out = pipeline.process_frame(_make_frame(), 1, {1: (0, 0, 400, 400)})
    assert out == []
    assert pipeline.enabled is False
    assert pipeline.load_error is not None


def test_pipeline_respects_ocr_every_interval():
    reader = _FakeReader()
    plate_model = _FakePlateModel(xyxy=[[100, 150, 300, 220]], conf=[0.9])
    pipeline = AnprPipeline(
        "ignored.pt",
        ocr_every=5,
        min_votes=1,
        min_plate_width=50,
        min_plate_height=5,
        ocr_min_conf=0.1,
        plate_model=plate_model,
        ocr_reader=reader,
    )
    vehicles = {1: (0, 0, 400, 400)}
    assert pipeline.process_frame(_make_frame(), 1, vehicles) == []
    assert pipeline.process_frame(_make_frame(), 4, vehicles) == []
    assert pipeline.process_frame(_make_frame(), 5, vehicles) != []
    assert plate_model.calls == 1  # only frame 5 ran inference


def test_pipeline_skips_when_no_vehicles():
    reader = _FakeReader()
    plate_model = _FakePlateModel(xyxy=[[100, 150, 300, 220]], conf=[0.9])
    pipeline = AnprPipeline(
        "ignored.pt",
        ocr_every=1,
        min_votes=1,
        plate_model=plate_model,
        ocr_reader=reader,
    )
    assert pipeline.process_frame(_make_frame(), 1, {}) == []
    assert plate_model.calls == 0


# ------------------------------------------------- schema migration integration
def test_ensure_column_accepts_anpr_fields(tmp_path):
    con = sqlite3.connect(tmp_path / "legacy.db")
    con.execute("CREATE TABLE detections (id INTEGER PRIMARY KEY)")
    for name, definition in (
        ("plate_text", "TEXT DEFAULT ''"),
        ("plate_confidence", "REAL"),
        ("plate_vote_count", "INTEGER DEFAULT 0"),
        ("plate_first_frame", "INTEGER"),
        ("plate_last_frame", "INTEGER"),
        ("plate_status", "TEXT DEFAULT 'candidate'"),
    ):
        ensure_column(con, "detections", name, definition)
        ensure_column(con, "detections", name, definition)  # idempotent
    columns = {row[1] for row in con.execute("PRAGMA table_info(detections)")}
    assert {
        "plate_text", "plate_confidence", "plate_vote_count",
        "plate_first_frame", "plate_last_frame", "plate_status",
    } <= columns
    con.close()


def test_ensure_column_still_rejects_unknown_fields(tmp_path):
    con = sqlite3.connect(tmp_path / "db.sqlite")
    con.execute("CREATE TABLE detections (id INTEGER PRIMARY KEY)")
    with pytest.raises(ValueError):
        ensure_column(con, "detections", "arbitrary_column", "TEXT")
    con.close()


# -------------------------------------- debug metrics summary (no accuracy claim)
def test_debug_metrics_render_lines_without_claiming_accuracy():
    metrics = AnprDebugMetrics(
        plate_boxes_detected=10, plate_boxes_filtered=2, plate_boxes_linked=5,
        ocr_attempts=5, ocr_non_empty=3, stable_candidates=1,
        unique_tracks_with_plate=1,
    )
    lines = metrics.as_lines()
    joined = "\n".join(lines).lower()
    assert "plate boxes detected" in joined
    assert "ocr attempts" in joined
    # The debug lines must not use the word "accuracy".
    assert "accuracy" not in joined
