from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from poc_support import (
    choose_plate, ensure_column, normalize_plate, ocr_plate_variants,
    plausible_indian_plate, prepare_plate_images, should_count_vehicle,
)


def test_normalize_does_not_invent_characters():
    assert normalize_plate('ts10 ua 1675') == 'TS10UA1675'
    assert not plausible_indian_plate('TS1OUA167S')
    assert plausible_indian_plate('TS10UA1675')


def test_vehicle_count_requires_motion_and_age():
    stationary = [(10, 10), (12, 11)]
    moving = [(10, 10), (38, 11)]
    assert not should_count_vehicle(6, stationary, 10, 11, None, 25)
    assert not should_count_vehicle(2, moving, 10, 11, None, 25)
    assert should_count_vehicle(6, moving, 10, 11, None, 25)
    assert not should_count_vehicle(6, moving, 10, 11, 50, 25)
    assert should_count_vehicle(6, [(10, 30), (50, 60)], 30, 60, 50, 25)


def test_plate_vote_rejects_unverified_string():
    assert choose_plate(Counter({'TS10UA1675': 2}), {'TS10UA1675': 0.8}, 2) == ('TS10UA1675', 0.8)
    assert choose_plate(Counter({'TS1OUA167S': 5}), {'TS1OUA167S': 0.99}, 2) is None
    assert choose_plate(Counter({'TS10UA1675': 1}), {'TS10UA1675': 0.9}, 2) is None


def test_migration_is_idempotent(tmp_path: Path):
    con = sqlite3.connect(tmp_path / 'test.db')
    con.execute('CREATE TABLE detections (id INTEGER PRIMARY KEY, run_id INTEGER)')
    ensure_column(con, 'detections', 'track_id', 'INTEGER')
    ensure_column(con, 'detections', 'track_id', 'INTEGER')
    assert [r[1] for r in con.execute('PRAGMA table_info(detections)')].count('track_id') == 1
    with pytest.raises(ValueError):
        ensure_column(con, 'detections; DROP TABLE detections;', 'x', 'TEXT')
    con.close()


class SplitTextOCR:
    """Simulates a real EasyOCR response with segmented plate text."""
    def readtext(self, image, detail=1, allowlist=None):
        return [
            ([[1, 1], [100, 1], [100, 25], [1, 25]], 'TS10', 0.88),
            ([[102, 1], [240, 1], [240, 25], [102, 25]], 'UA1675', 0.84),
        ]


def test_ocr_can_join_split_plate_text():
    # Synthetic crop validates text selection, NOT real-world OCR accuracy.
    crop = np.full((20, 100, 3), 255, dtype=np.uint8)
    assert len(prepare_plate_images(crop)) == 4
    text, confidence = ocr_plate_variants(SplitTextOCR(), crop)
    assert text == 'TS10UA1675'
    assert confidence == pytest.approx(0.84)


def test_blank_ocr_crop_returns_no_result():
    assert ocr_plate_variants(SplitTextOCR(), np.empty((0, 0, 3), dtype=np.uint8)) == ('', 0.0)
