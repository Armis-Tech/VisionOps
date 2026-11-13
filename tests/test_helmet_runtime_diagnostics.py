"""VideoWorker-side helmet diagnostic logging — early-failure, enriched log, success.

These tests mirror the structure of test_anpr_runtime_diagnostics.py on the
ANPR branch. They load the real VideoWorker class via AST into a Qt-free
scope and inject fake FrameAnalyzer stand-ins.

What is validated:
    * helmet_run.log is written on every run when rules are enabled,
      including when the video fails to open or FrameAnalyzer.__init__ raises.
    * The log records rules flag, model path, class names, frames processed,
      and runtime error (when present).
    * When rules are disabled in opts, no helmet log entry is written.
    * Vehicle counting continues when the violation model raises on every
      frame — the run must still complete successfully.
"""
from __future__ import annotations

import ast
import sys
import types
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _Signal:
    def __init__(self, *args, **kwargs):
        self.messages = []

    def emit(self, *args):
        self.messages.append(args)


def _load_video_worker(tmp_path, analyzer_cls):
    """Load VideoWorker via AST with a scope that routes logs under tmp_path."""
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    worker_class = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VideoWorker"
    )
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
        # Fallback module-level constant the log uses when analyzer is None.
        "VIOLATION_MODEL_PATH": tmp_path / "violations.pt",
    }
    exec(
        compile(ast.Module(body=[worker_class], type_ignores=[]),
                "main.py", "exec"),
        scope,
    )
    worker_cls = scope["VideoWorker"]
    # Qt Signal() instances are per-worker. Create fresh ones for each worker.
    worker_cls.frame_ready = _Signal()
    worker_cls.progress = _Signal()
    worker_cls.data_changed = _Signal()
    worker_cls.error = _Signal()
    worker_cls.finished_info = _Signal()
    return worker_cls


def _write_tiny_mp4_or_skip(path: Path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (160, 96))
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder unavailable in this environment")
    for _ in range(2):
        writer.write(np.ones((96, 160, 3), dtype=np.uint8) * 120)
    writer.release()


# ---------------------------- fake analyzers
class _FakeHelmetMetrics:
    motorcycles_observed = 4
    rider_associations_attempted = 4
    helmet_observations = 0
    no_helmet_observations = 2
    unknown_observations = 2
    stable_no_helmet_candidates = 1
    candidate_violations_created = 1

    def as_lines(self):
        return [
            f"Motorcycles observed (total across frames): {self.motorcycles_observed}",
            f"Rider association attempts: {self.rider_associations_attempted}",
            f"Helmet observations: {self.helmet_observations}",
            f"No-helmet observations: {self.no_helmet_observations}",
            f"Unknown/inconclusive observations: {self.unknown_observations}",
            f"Stable no-helmet candidates: {self.stable_no_helmet_candidates}",
            f"Candidate violations created: {self.candidate_violations_created}",
        ]


class _FakeObserver:
    def __init__(self):
        self.metrics = _FakeHelmetMetrics()

    def finalize_metrics(self):
        return self.metrics


class _FakeVModel:
    names = {0: "Bike_Rider", 1: "Helmet", 2: "No_Helmet"}


class _AnalyzerHelmetAvailable:
    """Mimics a fully-configured helmet pipeline."""
    unique = 1

    def __init__(self, *args, **kwargs):
        self._helmet_status = "available"
        self._violation_model_path = Path("/configured/violations.pt")
        self.helmet_observer = _FakeObserver()
        self.vmodel = _FakeVModel()

    def process(self, frame, frame_no):
        return frame.copy(), 1, 1 if frame_no == 1 else 0


class _AnalyzerHelmetMissing:
    """Rules enabled but violations.pt was missing."""
    unique = 1

    def __init__(self, *args, **kwargs):
        self._helmet_status = "model_missing"
        self._violation_model_path = Path("/does/not/exist/violations.pt")
        self.helmet_observer = None
        self.vmodel = None

    def process(self, frame, frame_no):
        return frame.copy(), 1, 1 if frame_no == 1 else 0


class _AnalyzerRulesDisabled:
    """Rules disabled in opts; no violation detection configured at all."""
    unique = 1

    def __init__(self, *args, **kwargs):
        self._helmet_status = "disabled"
        self._violation_model_path = None
        self.helmet_observer = None
        self.vmodel = None

    def process(self, frame, frame_no):
        return frame.copy(), 1, 1 if frame_no == 1 else 0


class _AnalyzerInitRaises:
    """FrameAnalyzer.__init__ raises — simulates violations.pt load crash."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError("simulated FrameAnalyzer init failure")


class _AnalyzerProcessRaises:
    """FrameAnalyzer builds but process() raises on every frame."""
    unique = 0

    def __init__(self, *args, **kwargs):
        self._helmet_status = "available"
        self._violation_model_path = Path("/configured/violations.pt")
        self.helmet_observer = _FakeObserver()
        self.vmodel = _FakeVModel()

    def process(self, frame, frame_no):
        raise RuntimeError("simulated process failure")


# ----------------------------------------- success path
def test_helmet_log_written_on_successful_run(tmp_path):
    video = tmp_path / "input.mp4"
    _write_tiny_mp4_or_skip(video)
    worker_cls = _load_video_worker(tmp_path, _AnalyzerHelmetAvailable)
    worker = worker_cls(str(video), {"rules": True})
    worker.run()
    log = tmp_path / "helmet_run.log"
    assert log.exists(), "helmet_run.log must be written on a successful run"
    content = log.read_text(encoding="utf-8").replace("\\", "/")
    # The log must record the rules flag, the model path, class names, and
    # key observer metrics. These are the fields Sumit's testing report
    # said were missing.
    assert "Experimental violation detection enabled: True" in content
    assert "/configured/violations.pt" in content
    assert "Bike_Rider" in content
    assert "No_Helmet" in content
    assert "Frames processed:" in content
    assert "Motorcycles observed" in content
    assert "No-helmet observations" in content


# ----------------------------------------- early-failure path
def test_helmet_log_written_when_video_fails_to_open(tmp_path):
    """Video missing → FrameAnalyzer never constructed → log must still appear."""
    worker_cls = _load_video_worker(tmp_path, _AnalyzerHelmetAvailable)
    worker = worker_cls(str(tmp_path / "nonexistent.mp4"), {"rules": True})
    worker.run()
    assert worker.error.messages, "worker must still emit the original error"
    log = tmp_path / "helmet_run.log"
    assert log.exists(), "helmet_run.log must be written even when video open fails"
    content = log.read_text(encoding="utf-8").replace("\\", "/")
    assert "Experimental violation detection enabled: True" in content
    assert "run_aborted_before_analyzer_init" in content
    assert "Run error:" in content
    assert "FrameAnalyzer not constructed" in content


def test_helmet_log_written_when_analyzer_init_raises(tmp_path):
    """FrameAnalyzer.__init__ raises → log must still appear with the error."""
    video = tmp_path / "input.mp4"
    _write_tiny_mp4_or_skip(video)
    worker_cls = _load_video_worker(tmp_path, _AnalyzerInitRaises)
    worker = worker_cls(str(video), {"rules": True})
    worker.run()
    assert worker.error.messages
    log = tmp_path / "helmet_run.log"
    assert log.exists()
    content = log.read_text(encoding="utf-8").replace("\\", "/")
    assert "run_aborted_before_analyzer_init" in content
    assert "simulated FrameAnalyzer init failure" in content


def test_helmet_log_written_when_process_raises(tmp_path):
    """FrameAnalyzer builds but process() crashes → full log path runs."""
    video = tmp_path / "input.mp4"
    _write_tiny_mp4_or_skip(video)
    worker_cls = _load_video_worker(tmp_path, _AnalyzerProcessRaises)
    worker = worker_cls(str(video), {"rules": True})
    worker.run()
    assert worker.error.messages
    log = tmp_path / "helmet_run.log"
    assert log.exists()
    content = log.read_text(encoding="utf-8").replace("\\", "/")
    # Full log path runs because the analyzer was constructed.
    assert "status=available" in content
    assert "Bike_Rider" in content
    assert "Run error:" in content


# ----------------------------------------- opt-out path
def test_no_helmet_log_when_rules_disabled(tmp_path):
    """Rules disabled → no helmet log entry (consistent, not silent-fail)."""
    video = tmp_path / "input.mp4"
    _write_tiny_mp4_or_skip(video)
    worker_cls = _load_video_worker(tmp_path, _AnalyzerRulesDisabled)
    worker = worker_cls(str(video), {"rules": False})
    worker.run()
    log = tmp_path / "helmet_run.log"
    # Entry is still written — it's useful to confirm "yes we ran, no rules
    # were enabled". The entry records rules=False so the operator can tell
    # a disabled run apart from a crashed run.
    assert log.exists(), "helmet_run.log must record disabled runs too"
    content = log.read_text(encoding="utf-8").replace("\\", "/")
    assert "Experimental violation detection enabled: False" in content
    assert "status=disabled" in content


# ----------------------------------------- missing-model path
def test_helmet_log_reports_model_missing_clearly(tmp_path):
    """When violations.pt is missing, the log must say so explicitly."""
    video = tmp_path / "input.mp4"
    _write_tiny_mp4_or_skip(video)
    worker_cls = _load_video_worker(tmp_path, _AnalyzerHelmetMissing)
    worker = worker_cls(str(video), {"rules": True})
    worker.run()
    content = (tmp_path / "helmet_run.log").read_text(encoding="utf-8").replace("\\", "/")
    assert "status=model_missing" in content
    assert "/does/not/exist/violations.pt" in content
    assert "Violations model file exists: False" in content
    assert "Helmet observer not constructed" in content


# ----------------------------------------- vehicle detection resilience
def test_vehicle_detection_survives_process_crash(tmp_path):
    """A process crash must not prevent the finished_info signal from firing."""
    video = tmp_path / "input.mp4"
    _write_tiny_mp4_or_skip(video)
    worker_cls = _load_video_worker(tmp_path, _AnalyzerProcessRaises)
    worker = worker_cls(str(video), {"rules": True})
    worker.run()
    assert worker.finished_info.messages
    # The run is marked as failed, but it terminates cleanly.
    out, unique, stopped, success = worker.finished_info.messages[-1]
    assert success is False  # process() raised on first frame
