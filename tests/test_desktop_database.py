"""Exercise the *real* desktop Database class without importing heavyweight GUI models."""
from __future__ import annotations

import ast
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import pytest

from poc_support import ensure_column


def load_desktop_database_class():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'main.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Database')
    scope = {
        'sqlite3': sqlite3, 'threading': threading,
        'ensure_column': ensure_column,
        'now_str': lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'datetime': datetime,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(root / 'main.py'), 'exec'), scope)
    return scope['Database']


def test_database_migrates_legacy_rows_without_deleting(tmp_path):
    path = tmp_path / 'legacy.db'
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE detections (id INTEGER PRIMARY KEY AUTOINCREMENT, event_time TEXT NOT NULL, '
                'source TEXT, vehicle_type TEXT, confidence REAL, frame_number INTEGER, video_path TEXT)')
    con.execute("INSERT INTO detections(event_time,vehicle_type) VALUES ('2026-01-01','Car')")
    con.commit()
    con.close()
    db = load_desktop_database_class()(path)
    assert db.scalar('SELECT COUNT(*) FROM detections') == 1
    assert db.scalar('SELECT COUNT(*) FROM detections WHERE run_id IS NULL') == 1
    columns = {r[1] for r in db.query('PRAGMA table_info(violations)')}
    assert {'run_id', 'track_id', 'frame_number', 'video_path'} <= columns
    assert {'track_id'} <= {r[1] for r in db.query('PRAGMA table_info(detections)')}
    db._con.close()


def test_atomic_review_and_audit(tmp_path):
    db = load_desktop_database_class()(tmp_path / 'app.db')
    vid = db.execute(
        "INSERT INTO violations(violation_type,plate,status) VALUES (?,?,?)",
        ('helmet', '', 'pending'),
    )
    db.apply_review_action(
        vid, 'pending', 'accepted', 'operator', 'Accept',
        'TS10UA1675', 'Intersection', 'Inspected evidence',
    )
    assert db.query('SELECT plate,status,location FROM violations WHERE id=?', (vid,))[0] == (
        'TS10UA1675', 'accepted', 'Intersection'
    )
    assert db.scalar('SELECT COUNT(*) FROM audit WHERE violation_id=?', (vid,)) == 2
    with pytest.raises(ValueError):
        db.apply_review_action(
            vid, 'pending', 'issued', 'authority', 'Issue',
            'TS10UA1675', 'Elsewhere', 'Invalid stale transition', challan_no='DEMO123', fine_amount=1000,
        )
    assert db.scalar('SELECT status FROM violations WHERE id=?', (vid,)) == 'accepted'
    assert db.scalar('SELECT COUNT(*) FROM audit WHERE violation_id=?', (vid,)) == 2
    db._con.close()


def test_unique_per_run_vehicle_track(tmp_path):
    db = load_desktop_database_class()(tmp_path / 'app.db')
    q = ('INSERT INTO detections(event_time,vehicle_type,run_id,track_id) '
         'VALUES (?,?,?,?)')
    db.execute(q, ('2026-01-01', 'Car', 1, 14))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(q, ('2026-01-01', 'Car', 1, 14))
    db.execute(q, ('2026-01-01', 'Car', 2, 14))
    assert db.scalar('SELECT COUNT(*) FROM detections') == 2
    db._con.close()
