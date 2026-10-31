"""Webcam or RTSP one-frame connection probe (no AI required).

For RTSP credentials, supply VISIONOPS_RTSP_URL as an environment variable.
Do not commit credentials or full URLs to Git.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import cv2


def open_source(source):
    if isinstance(source, str) and source.lower().startswith('rtsp://'):
        try:
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
            ])
        except (cv2.error, TypeError):
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


def main() -> int:
    parser = argparse.ArgumentParser(description='Test an actual camera before live YOLO inference')
    parser.add_argument('--source', default='0', help='Webcam index, e.g. 0. Avoid putting passwords here.')
    parser.add_argument('--from-env', action='store_true', help='Read RTSP URL from VISIONOPS_RTSP_URL')
    args = parser.parse_args()
    source = os.environ.get('VISIONOPS_RTSP_URL') if args.from_env else args.source
    if not source:
        parser.error('Set VISIONOPS_RTSP_URL or provide --source 0')
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = open_source(source)
    try:
        if not cap.isOpened():
            raise RuntimeError('VideoCapture could not open camera. Check credentials/device permissions.')
        started = time.monotonic()
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f'PASS: frame received ({frame.shape[1]} x {frame.shape[0]})')
                print(f'Elapsed: {time.monotonic() - started:.2f}s')
                return 0
        raise RuntimeError('Camera opened but no readable frames were received.')
    except (RuntimeError, cv2.error) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1
    finally:
        cap.release()


if __name__ == '__main__':
    raise SystemExit(main())
