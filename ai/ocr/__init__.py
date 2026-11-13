"""ANPR / OCR components reused by the dashboard and the standalone pipelines."""
from ai.ocr.anpr import (
    AnprDebugMetrics,
    AnprPipeline,
    AnprResult,
    PlateVoteTracker,
    associate_plate,
)

__all__ = [
    "AnprDebugMetrics",
    "AnprPipeline",
    "AnprResult",
    "PlateVoteTracker",
    "associate_plate",
]
