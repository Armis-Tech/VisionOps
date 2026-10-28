"""Test actual desktop VideoWorker failure/success paths without needing Qt/models."""
from __future__ import annotations

import ast
import types
from pathlib import Path

import cv2
import numpy as np


class MockSignal:
    def __init__(self, *args):
        self.messages = []

    def emit(self, *args):
        self.messages.append(args)


class MockThread:
    pass


def make_video_worker(tmp_path, analyzer_type):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse(
        (root / 'main.py').read_text(encoding="utf-8")
    )
    worker_class = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == 'VideoWorker'
    )
    scope = dict(
        QThread=MockThread,
        Signal=MockSignal,
        cv2=cv2,
        Path=Path,
        OUTPUT_DIR=tmp_path,
        datetime=__import__('datetime').datetime,
        time=__import__('time'),
        FrameAnalyzer=analyzer_type,
    )
    exec(
        compile(
            ast.Module(body=[worker_class], type_ignores=[]),
            'main.py',
            'exec'
        ),
        scope
    )
    worker = scope['VideoWorker']

    # Qt's Signal() creates unique bound signals per instance. Simulate here.
    worker.frame_ready = MockSignal()
    worker.progress = MockSignal()
    worker.data_changed = MockSignal()
    worker.error = MockSignal()
    worker.finished_info = MockSignal()

    return worker


def make_input_video(path):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        10,
        (160, 96)
    )
    assert writer.isOpened()

    for _ in range(3):
        writer.write(
            np.ones((96, 160, 3), dtype=np.uint8) * 120
        )

    writer.release()


class WorkingAnalyzer:
    def __init__(self, *args):
        self.unique = 1

    def process(self, frame, frame_no):
        return frame.copy(), 1, 1 if frame_no == 1 else 0


class FailingAnalyzer:
    def __init__(self, *args):
        self.unique = 0

    def process(self, frame, frame_no):
        raise RuntimeError('simulated model failure')


def test_offline_worker_emits_success_only_for_completed_video(tmp_path):
    path = tmp_path / 'input.mp4'
    make_input_video(path)

    cls = make_video_worker(tmp_path, WorkingAnalyzer)
    worker = cls(str(path), {})
    worker.run()

    assert not worker.error.messages

    output, unique, stopped, success = worker.finished_info.messages[-1]

    assert success is True and stopped is False and unique == 1
    assert Path(output).exists()


def test_offline_worker_does_not_call_failed_run_a_success(tmp_path):
    path = tmp_path / 'input.mp4'
    make_input_video(path)

    cls = make_video_worker(tmp_path, FailingAnalyzer)
    worker = cls(str(path), {})
    worker.run()

    assert worker.error.messages
    assert worker.finished_info.messages[-1][-1] is False


def test_missing_video_reports_error(tmp_path):
    cls = make_video_worker(tmp_path, WorkingAnalyzer)
    worker = cls(str(tmp_path / 'not-found.mp4'), {})
    worker.run()

    assert worker.error.messages
    assert worker.finished_info.messages[-1][-1] is False

