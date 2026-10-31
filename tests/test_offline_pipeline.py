"""Offline pipeline smoke test with simulated YOLO detections, no downloaded weights."""
from __future__ import annotations

import argparse
import sqlite3
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.poc_pipeline import analyze_video


class FakeSeries:
    def __init__(self, values):
        self.values = values
    def int(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.cls = FakeSeries([2])
        self.id = FakeSeries([12])
        self.conf = FakeSeries([0.9])
    def __len__(self):
        return 1


class FakePrediction:
    def __init__(self, frame):
        self.frame = frame
        self.boxes = FakeBoxes()
    def plot(self):
        return self.frame.copy()


class FakeYOLO:
    def __init__(self, weights):
        self.weights = weights
    def track(self, frame, **kwargs):
        return [FakePrediction(frame)]


def test_end_to_end_offline_records_and_video(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'ultralytics', types.SimpleNamespace(YOLO=FakeYOLO))
    source = tmp_path / 'input.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 96))
    if not writer.isOpened():
        pytest.skip('OpenCV mp4v encoder not available in test environment')
    for i in range(9):
        frame = np.full((96, 160, 3), 64 + i, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    output = tmp_path / 'processed.mp4'
    database = tmp_path / 'test.db'
    args = argparse.Namespace(video=source, model='fake.pt', output=output, db=database,
                              conf=0.35, imgsz=640, min_frames=6, show=False)
    analyze_video(args)
    assert output.is_file() and output.stat().st_size > 0
    conn = sqlite3.connect(database)
    rows = conn.execute('SELECT run_id,track_id,vehicle_type,frame_number FROM detections').fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][1:] == (12, 'Car', 6)


def test_missing_video_produces_clear_error(tmp_path):
    args = argparse.Namespace(video=tmp_path / 'missing.mp4')
    with pytest.raises(FileNotFoundError):
        analyze_video(args)
