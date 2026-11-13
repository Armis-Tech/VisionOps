"""Smoke tests for the dashboard's ANPR integration points.

We never import Qt here. The desktop Database class is loaded by AST so the
tests exercise the real schema-migration logic without pulling PySide6.
"""
from __future__ import annotations

import ast
import sqlite3
import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poc_support import ensure_column  # noqa: E402


def _load_desktop_database():
    """Rebuild the Database class in isolation (no Qt or Ultralytics import)."""
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Database")
    scope = {
        "sqlite3": sqlite3,
        "threading": threading,
        "ensure_column": ensure_column,
        "now_str": lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datetime": datetime,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(root / "main.py"), "exec"), scope)
    return scope["Database"]


def test_schema_migration_adds_anpr_columns_without_touching_rows(tmp_path):
    """A pre-ANPR database keeps its data; new ANPR columns are added."""
    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE detections ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, event_time TEXT NOT NULL, "
        "source TEXT, vehicle_type TEXT, confidence REAL, frame_number INTEGER, video_path TEXT)"
    )
    con.execute("INSERT INTO detections(event_time, vehicle_type) VALUES ('2026-01-01', 'Car')")
    con.commit()
    con.close()

    Database = _load_desktop_database()
    db = Database(path)
    try:
        columns = {row[1] for row in db.query("PRAGMA table_info(detections)")}
        assert {
            "plate_text", "plate_confidence", "plate_vote_count",
            "plate_first_frame", "plate_last_frame", "plate_status",
        } <= columns
        rows = db.query("SELECT COUNT(*) FROM detections")
        assert rows[0][0] == 1  # existing row preserved
        rows = db.query("SELECT plate_text, plate_status FROM detections")
        assert rows[0][0] in ("", None)
        assert rows[0][1] in ("candidate", None)
    finally:
        db._con.close()


def test_update_detection_plate_only_touches_candidate_rows(tmp_path):
    """A row marked 'reviewed' must not be overwritten by the worker."""
    Database = _load_desktop_database()
    db = Database(tmp_path / "app.db")
    try:
        # Insert two rows: one candidate, one already reviewed.
        db.execute(
            "INSERT INTO detections(event_time, vehicle_type, run_id, track_id, "
            "plate_text, plate_status) VALUES (?,?,?,?,?,?)",
            ("2026-01-01", "Car", 1, 10, "", "candidate"),
        )
        db.execute(
            "INSERT INTO detections(event_time, vehicle_type, run_id, track_id, "
            "plate_text, plate_status) VALUES (?,?,?,?,?,?)",
            ("2026-01-01", "Car", 1, 11, "MH12AB0001", "reviewed"),
        )

        # Simulate update_detection_plate for both tracks.
        update_sql = (
            "UPDATE detections SET plate_text=?, plate_confidence=?, plate_vote_count=?, "
            "plate_first_frame=COALESCE(plate_first_frame,?), plate_last_frame=? "
            "WHERE run_id=? AND track_id=? AND COALESCE(plate_status,'candidate')='candidate'"
        )
        db.execute(update_sql, ("TS10UA1675", 0.8, 3, 100, 150, 1, 10))
        db.execute(update_sql, ("TS99XX9999", 0.9, 3, 100, 150, 1, 11))

        rows = {
            r[0]: r for r in db.query(
                "SELECT track_id, plate_text, plate_vote_count, plate_status FROM detections"
            )
        }
        assert rows[10][1] == "TS10UA1675"
        assert rows[10][2] == 3
        assert rows[11][1] == "MH12AB0001"  # unchanged
        assert rows[11][3] == "reviewed"
    finally:
        db._con.close()


# ----------------------------------------------------- VideoWorker safety smoke
class _NoopAnpr:
    def __init__(self):
        self.load_error = None
        self.finalized = False

    def finalize_metrics(self):
        from ai.ocr.anpr import AnprDebugMetrics
        self.finalized = True
        return AnprDebugMetrics()


class _AnalyzerWithFlushError:
    """Simulates an analyzer that raises during flush_plate_candidates()."""

    def __init__(self, *args, **kwargs):
        self.unique = 1
        self.anpr = _NoopAnpr()

    def process(self, frame, frame_no):
        return frame.copy(), 1, 1 if frame_no == 1 else 0

    def flush_plate_candidates(self):
        raise RuntimeError("simulated ANPR flush error")


def _write_tiny_mp4_or_skip(path):
    """VideoWorker always writes mp4v; skip when the local build cannot encode it."""
    import pytest

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (160, 96),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder not available in test environment")
    for _ in range(3):
        writer.write(np.ones((96, 160, 3), dtype=np.uint8) * 120)
    writer.release()


def _make_video_worker(tmp_path, analyzer_cls):
    """Rebuild VideoWorker with injected deps, Qt-free."""
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    worker_class = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VideoWorker"
    )

    class _Signal:
        def __init__(self, *args):
            self.messages = []

        def emit(self, *args):
            self.messages.append(args)

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


def test_video_worker_survives_anpr_flush_error(tmp_path):
    """An ANPR flush exception must NOT turn a completed run into a failure."""
    video_path = tmp_path / "input.mp4"
    _write_tiny_mp4_or_skip(video_path)
    worker_cls = _make_video_worker(tmp_path, _AnalyzerWithFlushError)
    worker = worker_cls(str(video_path), {})
    worker.run()
    # No error signal emitted and finished_info reports success.
    assert not worker.error.messages
    output, unique, stopped, success = worker.finished_info.messages[-1]
    assert success is True
    assert stopped is False
    assert unique == 1
