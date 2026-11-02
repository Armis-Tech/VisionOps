"""Read-only health report for the original dashboard database or POC database."""
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect stored VisionOps counts without PySide6")
    parser.add_argument('--db', type=Path, required=True)
    args = parser.parse_args()
    if not args.db.exists():
        parser.error(f"Database does not exist: {args.db}")
    con = sqlite3.connect(f'file:{args.db.resolve()}?mode=ro', uri=True)
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        print(f'Database: {args.db.resolve()}')
        print(f'Tables: {", ".join(sorted(tables))}')
        if 'detections' in tables:
            columns = {row[1] for row in con.execute('PRAGMA table_info(detections)')}
            print(f'Detection rows: {con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]}')
            if 'run_id' in columns:
                legacy = con.execute('SELECT COUNT(*) FROM detections WHERE run_id IS NULL').fetchone()[0]
                print(f'Legacy per-frame rows (excluded from dashboard): {legacy}')
                latest = con.execute('SELECT MAX(run_id) FROM detections').fetchone()[0]
                print(f'Latest run: {latest}')
                if latest is not None:
                    print('Latest-run vehicle counts:')
                    for vehicle, count in con.execute(
                        'SELECT vehicle_type,COUNT(*) FROM detections WHERE run_id=? '
                        'GROUP BY vehicle_type ORDER BY vehicle_type', (latest,)
                    ):
                        print(f'  {vehicle}: {count}')
            else:
                print('Legacy schema: run_id not present; apply updated desktop main.py first.')
        if 'violations' in tables:
            print('Violations by status:')
            for status, count in con.execute('SELECT status,COUNT(*) FROM violations GROUP BY status'):
                print(f'  {status}: {count}')
            cols = {row[1] for row in con.execute('PRAGMA table_info(violations)')}
            if 'challan_no' in cols:
                issued = con.execute('SELECT COUNT(*) FROM violations WHERE challan_no IS NOT NULL').fetchone()[0]
                print(f'Demo challan numbers assigned: {issued}')
        if 'plate_reads' in tables:
            print(f'OCR candidate reads: {con.execute("SELECT COUNT(*) FROM plate_reads").fetchone()[0]}')
    finally:
        con.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
