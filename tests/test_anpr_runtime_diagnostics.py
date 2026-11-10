"""Regression tests for ANPR runtime diagnostics and silent-failure fixes.

These tests cover the behavior Sumit flagged on Windows:

* a wrong plate-model path must not silently disable ANPR,
* OCR/model load problems must surface in logs,
* the per-run log must be written even when ANPR never processes a frame,
* plate-to-vehicle association must stay conservative,
* DB updates with zero affected rows must be counted,
* operator-reviewed plates must not be overwritten,
* vehicle detection must continue when the ANPR pipeline fails.

Qt and Ultralytics are never imported. The desktop ``FrameAnalyzer`` and
``VideoWorker`` are loaded by AST and executed in a scope with injected
fakes, which keeps the tests deterministic and CPU-only.
"""
from __future__ import annotations

import ast
import sqlite3
import sys
import threading
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.ocr.anpr import (  # noqa: E402
    AnprDebugMetrics,
    AnprPipeline,
    associate_plate,
)
from poc_support import ensure_column  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


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
    """Yields the same plate box every frame. ``names`` is exposed for diagnostics."""

    def __init__(self, xyxy, conf, names=None):
        self._xyxy = xyxy
        self._conf = conf
        self.names = names if names is not None else {0: "License_Plate"}
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        return [_FakePrediction(self._xyxy, self._conf)]


class _OcrFailingReader:
    def readtext(self, *args, **kwargs):
        raise RuntimeError("ocr-blew-up")


class _EmptyReader:
    def readtext(self, image, detail=1, allowlist=None):
        return []


class _IndianPlateReader:
    def __init__(self, text="TS10UA1675", conf=0.9):
        self.text = text
        self.conf = conf
        self.calls = 0

    def readtext(self, image, detail=1, allowlist=None):
        self.calls += 1
        return [([[0, 0], [10, 0], [10, 10], [0, 10]], self.text, self.conf)]


def _make_frame():
    return np.full((400, 400, 3), 180, dtype=np.uint8)


# --------------------------------------------------- association edge cases
def test_associate_plate_strict_by_default_rejects_slight_overshoot():
    """A plate center 2 pixels outside a motorcycle box must NOT associate by default."""
    vehicle = {7: (100, 100, 180, 200)}  # small moto
    plate_outside = (182, 150, 220, 170)  # center at (201, 160)
    assert associate_plate(plate_outside, vehicle) is None


def test_associate_plate_expand_pct_recovers_boundary_detection():
    """With a small expansion, the same boundary plate associates correctly."""
    vehicle = {7: (100, 100, 180, 200)}
    plate_outside = (182, 150, 220, 170)  # center (201, 160); vehicle expanded by 0.08
    # Width 80 * 0.08 = 6.4, so new x2 ~= 186.4 — still outside.
    assert associate_plate(plate_outside, vehicle, vehicle_expand_pct=0.08) is None
    # With a wider tolerance the center falls inside.
    assert associate_plate(plate_outside, vehicle, vehicle_expand_pct=0.30) == 7


def test_associate_plate_does_not_link_unrelated_vehicle():
    """Expansion must not pull a plate onto a vehicle on the other side of the frame."""
    vehicles = {1: (0, 0, 100, 100), 2: (300, 300, 400, 400)}
    plate = (190, 190, 210, 210)  # center at frame middle, nobody owns it
    assert associate_plate(plate, vehicles, vehicle_expand_pct=0.10) is None


def test_associate_plate_still_picks_smallest_enclosing():
    """Ambiguous overlap: with expansion the smallest enclosing box still wins."""
    vehicles = {1: (0, 0, 400, 400), 2: (50, 50, 150, 150)}
    assert associate_plate((80, 80, 120, 120), vehicles, vehicle_expand_pct=0.05) == 2


# --------------------------------------------------- AnprPipeline diagnostics
def test_pipeline_describes_status_before_any_frame_runs():
    pipeline = AnprPipeline(
        "/does/not/exist/plate.pt",
        plate_model=_FakePlateModel([[1, 1, 2, 2]], [0.9]),
        ocr_reader=_IndianPlateReader(),
    )
    status = pipeline.describe_status()
    # Injected fakes keep the pipeline "enabled" and expose the class names.
    assert status["enabled"] is True
    assert status["model_loaded"] is True
    assert status["ocr_reader_loaded"] is True
    assert status["class_names"] == {0: "License_Plate"}
    assert status["model_file_exists"] is False  # path is bogus
    assert status["load_error"] is None


def test_pipeline_counts_unassociated_plates():
    """Plates in no vehicle must be counted so under-association is visible."""
    plate_model = _FakePlateModel(xyxy=[[300, 300, 360, 320]], conf=[0.9])
    pipeline = AnprPipeline(
        "ignored.pt",
        ocr_every=1,
        min_votes=1,
        min_plate_width=20,
        min_plate_height=5,
        ocr_min_conf=0.1,
        plate_model=plate_model,
        ocr_reader=_IndianPlateReader(),
    )
    vehicles = {1: (0, 0, 100, 100)}  # vehicle nowhere near the plate
    assert pipeline.process_frame(_make_frame(), 1, vehicles) == []
    assert pipeline.metrics.plate_boxes_detected == 1
    assert pipeline.metrics.plate_boxes_unassociated == 1
    assert pipeline.metrics.plate_boxes_linked == 0


def test_pipeline_counts_ocr_failures_without_crashing():
    plate_model = _FakePlateModel(xyxy=[[100, 100, 300, 200]], conf=[0.9])
    pipeline = AnprPipeline(
        "ignored.pt",
        ocr_every=1,
        min_votes=1,
        min_plate_width=20,
        min_plate_height=5,
        ocr_min_conf=0.1,
        plate_model=plate_model,
        ocr_reader=_OcrFailingReader(),
    )
    vehicles = {1: (50, 50, 350, 350)}
    assert pipeline.process_frame(_make_frame(), 1, vehicles) == []
    assert pipeline.metrics.ocr_failures == 1
    assert pipeline.metrics.ocr_attempts == 1
    assert pipeline.load_error is not None


def test_pipeline_counts_empty_ocr_results():
    plate_model = _FakePlateModel(xyxy=[[100, 100, 300, 200]], conf=[0.9])
    pipeline = AnprPipeline(
        "ignored.pt",
        ocr_every=1,
        min_votes=1,
        min_plate_width=20,
        min_plate_height=5,
        ocr_min_conf=0.1,
        plate_model=plate_model,
        ocr_reader=_EmptyReader(),
    )
    vehicles = {1: (50, 50, 350, 350)}
    assert pipeline.process_frame(_make_frame(), 1, vehicles) == []
    assert pipeline.metrics.ocr_attempts == 1
    assert pipeline.metrics.ocr_non_empty == 0


def test_pipeline_pass_through_vehicle_expand_pct():
    """The pipeline forwards vehicle_expand_pct to associate_plate."""
    plate_model = _FakePlateModel(xyxy=[[182, 110, 220, 150]], conf=[0.9])
    pipeline = AnprPipeline(
        "ignored.pt",
        ocr_every=1,
        min_votes=1,
        min_plate_width=20,
        min_plate_height=5,
        ocr_min_conf=0.1,
        vehicle_expand_pct=0.30,
        plate_model=plate_model,
        ocr_reader=_IndianPlateReader(),
    )
    vehicles = {5: (100, 100, 180, 200)}  # plate center ~(201,130) is just outside
    results = pipeline.process_frame(_make_frame(), 1, vehicles)
    assert len(results) == 1
    assert results[0].track_id == 5


def test_pipeline_tracks_frames_submitted_counter():
    plate_model = _FakePlateModel(xyxy=[], conf=[])
    pipeline = AnprPipeline(
        "ignored.pt",
        ocr_every=2,
        min_votes=1,
        plate_model=plate_model,
        ocr_reader=_IndianPlateReader(),
    )
    pipeline.process_frame(_make_frame(), 1, {})  # no vehicles → not submitted
    pipeline.process_frame(_make_frame(), 2, {1: (0, 0, 400, 400)})  # frame 2, ocr_every=2
    pipeline.process_frame(_make_frame(), 3, {1: (0, 0, 400, 400)})  # frame 3 — gated out
    assert pipeline.metrics.frames_submitted_to_anpr == 1


# ----------------------------------------------- FrameAnalyzer integration
def _load_frame_analyzer_class(**scope_overrides):
    """Rebuild FrameAnalyzer in a Qt-free scope with injectable deps."""
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))

    # Pull in helpers FrameAnalyzer needs plus _log_anpr_warning (module-level).
    helper_names = {
        "save_detection", "save_violation", "update_detection_plate",
        "_log_anpr_warning", "build_violation_map", "norm_plate", "now_str",
    }
    helpers = [n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name in helper_names]
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "FrameAnalyzer")

    scope: dict = {
        "defaultdict": __import__("collections").defaultdict,
        "deque": __import__("collections").deque,
        "Path": Path,
        "BASE_DIR": scope_overrides.get("BASE_DIR", ROOT),
        "VIOLATION_MODEL_PATH": scope_overrides.get("VIOLATION_MODEL_PATH", Path("nonexistent_vmodel.pt")),
        "VIOLATION_MAP_PATH": Path("nonexistent_map.json"),
        "MODEL_PATH": Path("primary.pt"),
        "AUTODOWNLOADABLE_MODELS": set(),
        "AnprPipeline": AnprPipeline,
        "VEHICLES": {2: "Car", 3: "Motorcycle"},
        "PERSON_CLASS": 0,
        "VIOLATIONS": {},
        "cv2": __import__("cv2"),
        "np": __import__("numpy"),
        "datetime": datetime,
        "should_count_vehicle": lambda *a, **k: True,
        "db": _FakeDB(),
        "YOLO": scope_overrides.get("YOLO", _FakePrimaryYolo),
        "NAME_HINTS": {},
    }
    scope.update(scope_overrides.pop("extra", {}))
    exec(compile(ast.Module(body=helpers + [cls], type_ignores=[]),
                 "main.py", "exec"), scope)
    return scope["FrameAnalyzer"], scope


class _FakeDB:
    """Minimal in-memory stand-in for the desktop Database used in tests."""

    def __init__(self):
        self._lock = threading.RLock()
        self._con = sqlite3.connect(":memory:")
        self._con.execute(
            "CREATE TABLE detections (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "event_time TEXT, source TEXT, vehicle_type TEXT, confidence REAL, "
            "frame_number INTEGER, video_path TEXT, run_id INTEGER, track_id INTEGER, "
            "plate_text TEXT DEFAULT '', plate_confidence REAL, "
            "plate_vote_count INTEGER DEFAULT 0, plate_first_frame INTEGER, "
            "plate_last_frame INTEGER, plate_status TEXT DEFAULT 'candidate')"
        )
        self._run_seq = 0

    def execute(self, sql, params=()):
        with self._lock:
            cur = self._con.execute(sql, params)
            self._con.commit()
            return cur.lastrowid

    def query(self, sql, params=()):
        with self._lock:
            return self._con.execute(sql, params).fetchall()

    def scalar(self, sql, params=(), default=0):
        rows = self.query(sql, params)
        return rows[0][0] if rows and rows[0][0] is not None else default

    def next_counter(self, key):
        self._run_seq += 1
        return self._run_seq


class _FakePrimaryYolo:
    """YOLO stand-in with COCO-matching names dict; never called by these tests."""

    def __init__(self, *args, **kwargs):
        self.names = {0: "person", 1: "bicycle", 2: "car",
                      3: "motorcycle", 5: "bus", 7: "truck"}

    def track(self, *args, **kwargs):
        class _Res:
            boxes = None

            def plot(self):
                return np.zeros((100, 100, 3), dtype=np.uint8)

        return [_Res()]


def test_analyzer_logs_warning_when_plate_model_missing(tmp_path):
    FrameAnalyzer, _ = _load_frame_analyzer_class(BASE_DIR=tmp_path)
    missing = tmp_path / "no_such_plate.pt"
    analyzer = FrameAnalyzer(
        "Offline: fake.mp4",
        {"anpr_enabled": True, "plate_model_path": missing, "rules": False},
    )
    assert analyzer.anpr is None
    assert analyzer._anpr_status == "model_missing"
    log_text = (tmp_path / "app_errors.log").read_text(encoding="utf-8")
    assert "Plate model file not found" in log_text
    assert str(missing) in log_text


def test_analyzer_disabled_status_when_anpr_not_enabled(tmp_path):
    FrameAnalyzer, _ = _load_frame_analyzer_class(BASE_DIR=tmp_path)
    analyzer = FrameAnalyzer(
        "Offline: fake.mp4",
        {"anpr_enabled": False, "rules": False},
    )
    assert analyzer.anpr is None
    assert analyzer._anpr_status == "disabled"
    # No warning file should be created for an intentional disable.
    assert not (tmp_path / "app_errors.log").exists()


def test_analyzer_init_error_logged(tmp_path):
    FrameAnalyzer, _ = _load_frame_analyzer_class(BASE_DIR=tmp_path)
    plate_path = tmp_path / "fake_plate.pt"
    plate_path.write_bytes(b"")  # file exists so we pass the missing-path branch
    # Pass an invalid min_plate_votes to force AnprPipeline init to raise.
    analyzer = FrameAnalyzer(
        "Offline: fake.mp4",
        {
            "anpr_enabled": True,
            "plate_model_path": plate_path,
            "min_plate_votes": 0,  # < 1 → ValueError in PlateVoteTracker
            "rules": False,
        },
    )
    assert analyzer.anpr is None
    assert analyzer._anpr_status == "init_error"
    assert analyzer._anpr_init_error is not None
    log_text = (tmp_path / "app_errors.log").read_text(encoding="utf-8")
    assert "AnprPipeline construction failed" in log_text


def test_analyzer_configured_status_when_everything_valid(tmp_path):
    FrameAnalyzer, _ = _load_frame_analyzer_class(BASE_DIR=tmp_path)
    plate_path = tmp_path / "plate.pt"
    plate_path.write_bytes(b"stub")
    # The AnprPipeline constructor doesn't load weights eagerly, so it works
    # even with a stub file — load_error is only populated on first inference.
    analyzer = FrameAnalyzer(
        "Offline: fake.mp4",
        {
            "anpr_enabled": True,
            "plate_model_path": plate_path,
            "rules": False,
        },
    )
    assert analyzer.anpr is not None
    assert analyzer._anpr_status == "configured"
    # Model-not-loaded is fine; the file exists and lazy loading will decide.
    assert analyzer._anpr_plate_model_path == plate_path


# ------------------------------------- update_detection_plate rowcount tests
def _load_update_fn():
    """Isolate update_detection_plate with a shared _FakeDB."""
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    func = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "update_detection_plate")
    db = _FakeDB()
    scope = {"db": db}
    exec(compile(ast.Module(body=[func], type_ignores=[]), "main.py", "exec"), scope)
    return scope["update_detection_plate"], db


def test_update_detection_plate_reports_rows_updated():
    update, db = _load_update_fn()
    db._con.execute(
        "INSERT INTO detections(event_time, run_id, track_id, plate_status) "
        "VALUES (?,?,?,?)", ("2026-01-01", 1, 10, "candidate"),
    )
    db._con.commit()
    rows_updated, rows_skipped = update(1, 10, "TS10UA1675", 0.8, 2, 100, 150)
    assert rows_updated == 1
    assert rows_skipped == 0


def test_update_detection_plate_reports_zero_zero_when_row_missing():
    """(0, 0) means no detection row at all — vehicle not yet saved."""
    update, _ = _load_update_fn()
    rows_updated, rows_skipped = update(99, 99, "AP09ZZ9999", 0.5, 1, 1, 2)
    assert rows_updated == 0
    assert rows_skipped == 0  # row doesn't exist, not a protected row


def test_update_detection_plate_distinguishes_reviewed_from_missing():
    """(0, 1) means a row exists but the operator reviewed it — must not be overwritten."""
    update, db = _load_update_fn()
    db._con.execute(
        "INSERT INTO detections(event_time, run_id, track_id, plate_text, plate_status) "
        "VALUES (?,?,?,?,?)", ("2026-01-01", 1, 11, "MH12AB0001", "reviewed"),
    )
    db._con.commit()
    rows_updated, rows_skipped = update(1, 11, "SHOULD_NOT_APPLY", 0.99, 10, 1, 2)
    assert rows_updated == 0
    assert rows_skipped == 1  # the reviewed row is counted, not overwritten
    plate = db._con.execute(
        "SELECT plate_text FROM detections WHERE run_id=1 AND track_id=11"
    ).fetchone()[0]
    assert plate == "MH12AB0001"


# -------------------------------------------- anpr_run.log always produced
def _load_video_worker(tmp_path, analyzer_cls):
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    worker_class = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VideoWorker"
    )

    class _Signal:
        def __init__(self, *a, **kw):
            self.messages = []

        def emit(self, *args):
            self.messages.append(args)

    import cv2
    scope = {
        "QThread": object,
        "Signal": _Signal,
        "cv2": cv2,
        "Path": Path,
        "OUTPUT_DIR": tmp_path,
        "BASE_DIR": tmp_path,
        "datetime": datetime,
        "time": __import__("time"),
        "FrameAnalyzer": analyzer_cls,
    }
    exec(
        compile(ast.Module(body=[worker_class], type_ignores=[]), "main.py", "exec"),
        scope,
    )
    worker_cls = scope["VideoWorker"]
    for name in ("frame_ready", "progress", "data_changed", "error", "finished_info"):
        setattr(worker_cls, name, _Signal())
    return worker_cls


def _write_tiny_mp4_or_skip(path):
    import cv2
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 96),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder not available in test environment")
    for _ in range(3):
        writer.write(np.ones((96, 160, 3), dtype=np.uint8) * 120)
    writer.release()


class _AnalyzerAnprMissing:
    """Simulates a run where ANPR was requested but the model file was missing."""

    unique = 0
    run_id = 42
    anpr = None
    _anpr_status = "model_missing"
    _anpr_plate_model_path = Path("/does/not/exist/plate.pt")
    _anpr_init_error = None
    _anpr_db_updates_attempted = 0
    _anpr_db_rows_updated = 0
    _anpr_db_rows_missing = 0
    _anpr_db_rows_skipped_reviewed = 0

    def __init__(self, *args, **kwargs):
        pass

    def process(self, frame, frame_no):
        import cv2
        return frame.copy(), 0, 0

    def flush_plate_candidates(self):
        return None


class _AnalyzerAnprCrashesEveryFrame:
    """ANPR is loaded but process_frame raises; vehicle counting must continue."""

    unique = 1
    run_id = 43
    _anpr_status = "configured"
    _anpr_plate_model_path = Path("plate.pt")
    _anpr_init_error = None
    _anpr_db_updates_attempted = 0
    _anpr_db_rows_updated = 0
    _anpr_db_rows_not_found = 0

    class _Anpr:
        load_error = "crashed-once"

        def __init__(self):
            self.metrics = AnprDebugMetrics(frames_processed=0)
            from poc_support import normalize_plate  # noqa: F401
            self.plate_model_class_names = {0: "License_Plate"}
            self.plate_model_loaded = True
            self.ocr_reader_loaded = True
            self.enabled = False
            self.ocr_every = 1
            self.plate_conf = 0.25
            self.min_plate_width = 35
            self.min_plate_height = 8
            self.ocr_min_conf = 0.25
            self.vehicle_expand_pct = 0.0

            class _Tracker:
                min_votes = 2

            self.tracker = _Tracker()
            self.plate_model_path = Path("plate.pt")

        def describe_status(self):
            return {
                "model_path": str(self.plate_model_path),
                "model_file_exists": False,
                "model_loaded": True,
                "ocr_reader_loaded": True,
                "enabled": False,
                "load_error": self.load_error,
                "class_names": dict(self.plate_model_class_names),
                "ocr_every": self.ocr_every,
                "plate_conf": self.plate_conf,
                "min_plate_width": self.min_plate_width,
                "min_plate_height": self.min_plate_height,
                "min_votes": self.tracker.min_votes,
                "ocr_min_conf": self.ocr_min_conf,
                "vehicle_expand_pct": self.vehicle_expand_pct,
            }

        def finalize_metrics(self):
            self.metrics.unique_tracks_with_plate = 0
            return self.metrics

    def __init__(self, *args, **kwargs):
        self.anpr = self._Anpr()

    def process(self, frame, frame_no):
        return frame.copy(), 1, 1 if frame_no == 1 else 0

    def flush_plate_candidates(self):
        return None


def test_anpr_run_log_written_even_when_model_missing(tmp_path):
    video = tmp_path / "input.mp4"
    _write_tiny_mp4_or_skip(video)
    worker_cls = _load_video_worker(tmp_path, _AnalyzerAnprMissing)
    worker = worker_cls(str(video), {})
    worker.run()
    log = tmp_path / "anpr_run.log"
    assert log.exists(), "anpr_run.log must be written even when ANPR couldn't start"
    content = log.read_text(encoding="utf-8")
    assert "run_id=42" in content
    assert "ANPR status: model_missing" in content
    assert "/does/not/exist/plate.pt" in content
    assert "Pipeline not constructed" in content


def test_anpr_run_log_includes_pipeline_diagnostics(tmp_path):
    video = tmp_path / "input.mp4"
    _write_tiny_mp4_or_skip(video)
    worker_cls = _load_video_worker(tmp_path, _AnalyzerAnprCrashesEveryFrame)
    worker = worker_cls(str(video), {})
    worker.run()
    content = (tmp_path / "anpr_run.log").read_text(encoding="utf-8")
    assert "ANPR status: configured" in content
    assert "License_Plate" in content
    assert "Plate model loaded: True" in content
    assert "OCR reader loaded: True" in content
    assert "Runtime notice: crashed-once" in content
    # Vehicle detection continued even though ANPR was unhealthy.
    output, unique, stopped, success = worker.finished_info.messages[-1]
    assert success is True
    assert unique == 1


class _AnalyzerAlwaysCrash:
    """Mimics a model that crashes every frame; vehicle counting must survive."""

    unique = 2
    run_id = 44
    _anpr_status = "configured"
    _anpr_plate_model_path = Path("plate.pt")
    _anpr_init_error = None
    _anpr_db_updates_attempted = 0
    _anpr_db_rows_updated = 0
    _anpr_db_rows_missing = 0
    _anpr_db_rows_skipped_reviewed = 0
    anpr = None  # no ANPR state once "load" fails

    def __init__(self, *args, **kwargs):
        pass

    def process(self, frame, frame_no):
        # Vehicle detection succeeds normally regardless of ANPR state.
        return frame.copy(), 1, 1 if frame_no == 1 else 0

    def flush_plate_candidates(self):
        raise RuntimeError("simulated flush crash")


def test_vehicle_detection_continues_when_anpr_flush_raises(tmp_path):
    video = tmp_path / "input.mp4"
    _write_tiny_mp4_or_skip(video)
    worker_cls = _load_video_worker(tmp_path, _AnalyzerAlwaysCrash)
    worker = worker_cls(str(video), {})
    worker.run()
    # The run must still complete successfully; flush errors are swallowed.
    assert not worker.error.messages
    output, unique, stopped, success = worker.finished_info.messages[-1]
    assert success is True
    assert unique == 2


# ------------------------------------------ early-failure ANPR logging
class _AnalyzerThatRaisesOnProcess:
    """FrameAnalyzer builds successfully but raises on the first frame."""

    unique = 0
    run_id = 55
    _anpr_status = "configured"
    _anpr_plate_model_path = Path("plate.pt")
    _anpr_init_error = None
    _anpr_db_updates_attempted = 0
    _anpr_db_rows_updated = 0
    _anpr_db_rows_missing = 0
    _anpr_db_rows_skipped_reviewed = 0
    anpr = None

    def __init__(self, *args, **kwargs):
        pass

    def process(self, frame, frame_no):
        raise RuntimeError("simulated frame-processing crash")

    def flush_plate_candidates(self):
        pass


class _AnalyzerInitRaises:
    """FrameAnalyzer.__init__ raises — simulates YOLO model loading failure."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("simulated FrameAnalyzer init crash")


def test_anpr_log_written_when_analyzer_init_fails(tmp_path):
    """When FrameAnalyzer raises, the early-failure path must log an ANPR entry."""
    video = tmp_path / "input.mp4"
    _write_tiny_mp4_or_skip(video)
    worker_cls = _load_video_worker(tmp_path, _AnalyzerInitRaises)
    worker = worker_cls(str(video), {"anpr_enabled": True,
                                     "plate_model_path": tmp_path / "plate.pt"})
    worker.run()
    # The error signal must fire.
    assert worker.error.messages
    log = tmp_path / "anpr_run.log"
    assert log.exists(), "anpr_run.log must be written even when FrameAnalyzer raises"
    content = log.read_text(encoding="utf-8")
    assert "RUN ABORTED BEFORE ANALYSIS" in content
    assert "plate.pt" in content


def test_anpr_log_written_when_video_fails_to_open(tmp_path):
    """When the video can't be opened (missing file), an early-failure entry is logged."""
    worker_cls = _load_video_worker(tmp_path, _AnalyzerInitRaises)
    worker = worker_cls(str(tmp_path / "nonexistent.mp4"),
                        {"anpr_enabled": True,
                         "plate_model_path": tmp_path / "plate.pt"})
    worker.run()
    assert worker.error.messages
    log = tmp_path / "anpr_run.log"
    assert log.exists(), "anpr_run.log must be written even when the video file is missing"
    content = log.read_text(encoding="utf-8")
    assert "RUN ABORTED BEFORE ANALYSIS" in content


def test_anpr_no_early_log_when_anpr_disabled(tmp_path):
    """When ANPR is disabled in opts, no early-failure log entry should be written."""
    worker_cls = _load_video_worker(tmp_path, _AnalyzerInitRaises)
    worker = worker_cls(str(tmp_path / "nonexistent.mp4"),
                        {"anpr_enabled": False})
    worker.run()
    # The error signal fires (missing video), but no ANPR log is written.
    assert worker.error.messages
    assert not (tmp_path / "anpr_run.log").exists()


def test_anpr_log_written_when_frame_processing_raises(tmp_path):
    """When FrameAnalyzer.process raises on the first frame, the log must still appear."""
    video = tmp_path / "input.mp4"
    _write_tiny_mp4_or_skip(video)
    worker_cls = _load_video_worker(tmp_path, _AnalyzerThatRaisesOnProcess)
    worker = worker_cls(str(video), {"anpr_enabled": True,
                                     "plate_model_path": tmp_path / "plate.pt"})
    worker.run()
    assert worker.error.messages
    log = tmp_path / "anpr_run.log"
    assert log.exists(), "anpr_run.log must be written when frame processing crashes"
    content = log.read_text(encoding="utf-8")
    # When analyzer was built, the full _log_anpr_run path runs — not the fallback.
    assert "ANPR status: configured" in content


# ------------------------------------------ dashboard status text checks
def test_anpr_status_text_does_not_imply_loaded():
    """The return string of _anpr_status_text, in the file-found branch,
    must tell the user that the YOLO model and EasyOCR reader have NOT been
    loaded yet — merely that the file exists.

    We test the string literals in the source rather than executing Qt code.
    """
    src = (ROOT / "main.py").read_text(encoding="utf-8")

    # Locate the _anpr_status_text method source region.
    start = src.index("def _anpr_status_text(")
    # Grab up to the next top-level def/class (generous range).
    end = src.index("\n    def ", start + 1)
    method_src = src[start:end]

    # The "file found" return statement should communicate that neither model
    # nor OCR reader is loaded yet.
    assert "not yet loaded" in method_src.lower() or "not loaded" in method_src.lower(), (
        "The 'file found' status text must tell the user the YOLO model and "
        "EasyOCR reader have not yet loaded.  Update _anpr_status_text to include "
        "'not yet loaded' or equivalent."
    )
    # The phrase "(loads on first frame)" is vague and doesn't mention EasyOCR;
    # it must not appear without a clear "not loaded yet" qualifier.
    if "(loads on first frame)" in method_src:
        assert False, (
            "_anpr_status_text still uses the old '(loads on first frame)' phrasing "
            "which implies the model will load, not that it is not yet loaded.  "
            "Replace it with a phrase that explicitly says neither model nor OCR is loaded."
        )


def test_anpr_info_label_does_not_imply_loaded():
    """The _refresh_anpr_info_label method must similarly avoid implying
    that the model or OCR reader is already loaded when the file merely exists.
    """
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    start = src.index("def _refresh_anpr_info_label(")
    end = src.index("\n    def ", start + 1)
    method_src = src[start:end]

    # The file-found branch must say "NOT loaded" or "not yet loaded".
    assert "not loaded" in method_src.lower() or "not yet loaded" in method_src.lower(), (
        "_refresh_anpr_info_label's 'file found' text must clarify that neither "
        "the YOLO model nor EasyOCR reader are loaded yet."
    )


# ---------------------------------------- unique-row vs operations counter
def test_anpr_updated_tracks_deduplicates_repeated_updates(tmp_path):
    """update_detection_plate called twice for the same (run_id, track_id) must
    register as 1 unique row in _anpr_updated_tracks, not 2.

    This guards against a misleading log that says '5 DB plate rows updated'
    when only 1 unique detection row was written to — the same row is
    re-updated each frame the plate is stable.
    """
    FrameAnalyzer, scope = _load_frame_analyzer_class(BASE_DIR=tmp_path)
    update_fn = scope["update_detection_plate"]
    db = scope["db"]

    # Insert the detection row that ANPR will update.
    db._con.execute(
        "INSERT INTO detections(event_time, run_id, track_id, plate_status) "
        "VALUES (?,?,?,?)", ("2026-01-01", 1, 10, "candidate"),
    )
    db._con.commit()

    # Build a minimal FrameAnalyzer (ANPR disabled; we test counters directly).
    analyzer = FrameAnalyzer("Offline: test.mp4", {"anpr_enabled": False, "rules": False})
    analyzer.run_id = 1

    # Simulate the in-frame update path calling update_detection_plate twice
    # for the same track (once per stable frame).
    track_id = 10
    for i in range(2):
        rows, skipped = update_fn(analyzer.run_id, track_id, "TS10UA1675", 0.9, 2 + i, 100, 150)
        if rows:
            analyzer._anpr_db_rows_updated += rows
            analyzer._anpr_updated_tracks.add(track_id)

    assert len(analyzer._anpr_updated_tracks) == 1, (
        "_anpr_updated_tracks must deduplicate: calling update_detection_plate "
        "twice for the same track must count as 1 unique row, not 2."
    )
    assert analyzer._anpr_db_rows_updated == 2, (
        "_anpr_db_rows_updated must count every UPDATE operation, "
        "including repeat writes to the same row."
    )


def test_anpr_log_distinguishes_unique_rows_from_operations():
    """The anpr_run.log format must report unique detection rows separately
    from the total UPDATE operation count so a user reading the log cannot
    mistake 'N operations' for 'N distinct vehicles with plates'.
    """
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    start = src.index("def _log_anpr_run(")
    end = src.index("\nclass ", start)
    method_src = src[start:end]

    # Must mention unique rows or unique detection rows.
    assert "unique" in method_src.lower(), (
        "_log_anpr_run must include the word 'unique' when reporting the row "
        "update count so it is clear the number is de-duplicated."
    )
    # Must also mention operations so the reader understands why the two numbers differ.
    assert "operation" in method_src.lower(), (
        "_log_anpr_run must mention 'operations' alongside the unique-row count "
        "so the reader understands the distinction."
    )
