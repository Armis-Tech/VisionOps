"""CLI diagnostic: inspect violations.pt class names and mapped violation keys.

Usage (from the VisionOps project root):

    python scripts/check_violation_model.py
    python scripts/check_violation_model.py --model path/to/violations.pt
    python scripts/check_violation_model.py --map violation_classes.json

Prints whether helmet/no-helmet detection is available without running the
full dashboard. Safe to run without a GPU; uses CPU inference for the check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.rules.helmet import describe_violation_model  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = BASE_DIR / "violations.pt"
DEFAULT_MAP = BASE_DIR / "violation_classes.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect violation model classes")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="Path to violations.pt (default: %(default)s)")
    parser.add_argument("--map", default=str(DEFAULT_MAP) if DEFAULT_MAP.exists() else None,
                        dest="map_path",
                        help="Path to violation_classes.json override (optional)")
    args = parser.parse_args(argv)

    info = describe_violation_model(args.model, args.map_path)
    for line in info.as_lines():
        print(line)

    return 0 if info.has_helmet_capability else 1


if __name__ == "__main__":
    sys.exit(main())
