# VisionOps

VisionOps is a proof-of-concept traffic intelligence and surveillance platform for processing recorded and live video, detecting and tracking vehicles, storing detection events, reviewing suspected traffic violations, and supporting an operator-driven e-Challan workflow.

The current focus is on building a reliable end-to-end POC before expanding to production-scale multi-camera deployment.

## Current Status

### Working / validated
- Python and OpenCV environment
- YOLO-based vehicle detection
- ByteTrack vehicle tracking
- Offline MP4/MOV video processing
- Annotated output video generation
- SQLite detection persistence
- Run-level and track-level traceability
- PySide6 desktop dashboard
- Detection History
- Vehicle Analytics
- Dashboard refresh flow
- Basic violation review workflow
- Experimental license plate detection
- Experimental OCR / ANPR candidate extraction
- Regression tests for core POC data flow

### Partially working / under validation
- Vehicle classification for local vehicle types
- Unique vehicle counting in crowded scenes
- License plate recognition accuracy
- Vehicle-to-plate association
- Helmet detection
- Triple-riding detection
- Wrong-direction detection

### Pending
- Real RTSP camera validation
- Multi-camera testing
- Reliable helmet and triple-riding accuracy
- Production-grade ANPR
- Government system integration
- Production e-Challan integration
- Enforcement-grade evidence validation

> ANPR and violation detections are currently POC features and should be treated as review candidates, not final enforcement decisions.

## Project Structure

```text
visionops/
├── ai/
├── backend/
├── frontend/
├── infra/
├── docs/
├── scripts/
│   ├── poc_pipeline.py
│   ├── poc_pipeline_anpr.py
│   ├── poc_pipeline_anpr_debug.py
│   ├── camera_probe.py
│   └── db_health.py
├── tests/
├── data/
├── outputs/
├── models/
├── main.py
├── poc_support.py
├── requirements-poc.txt
├── .gitignore
└── README.md
```

## Technology Stack

- Python 3.12
- OpenCV
- Ultralytics YOLO
- ByteTrack
- PyTorch
- SQLite
- PySide6
- EasyOCR for experimental ANPR
- Pytest

Planned components include FastAPI, PostgreSQL, Redis, React, Docker, Prometheus, and Grafana.

## Local Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd visionops
```

### 2. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-poc.txt
```

The current POC uses NumPy 1.26.4 for compatibility with the installed PyTorch build:

```bash
python -m pip install numpy==1.26.4
```

Verify the environment:

```bash
python - <<'PY'
import numpy
import torch
import cv2
from ultralytics import YOLO

print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("Ultralytics: OK")
PY
```

## Model Weights

Place required model files in the project root or `models/` directory.

Example:

```text
yolo11n.pt
models/license_plate.pt
```

Model weights should not be committed to Git.

The default YOLO model is suitable for general vehicle detection but is not sufficient by itself for helmet, seat-belt, or specialized traffic-violation detection.

## Offline Vehicle Detection POC

Place a local test video under `data/`.

Example:

```text
data/visionops_original_20s_test.mp4
```

Run:

```bash
python scripts/poc_pipeline.py \
  --video data/visionops_original_20s_test.mp4 \
  --model yolo11n.pt \
  --output outputs/poc_processed.mp4 \
  --db outputs/poc_detections.db
```

Pipeline:

```text
Video
  ↓
YOLO vehicle detection
  ↓
ByteTrack tracking
  ↓
Unique-track counting
  ↓
SQLite persistence
  ↓
Annotated output video
```

## Desktop Dashboard

Launch:

```bash
python main.py
```

The current dashboard includes:

- Home Dashboard
- Live Camera
- Camera Management
- Offline Analysis
- Vehicle Analytics
- Detection History
- Violation Review
- Reports
- Settings

Recommended validation flow:

```text
Offline Analysis
  ↓
Detection History
  ↓
Vehicle Analytics
  ↓
Home Dashboard
```

The totals shown across these areas should remain logically consistent for the same processing run.

## Experimental ANPR

The ANPR pipeline is kept separate from the main dashboard while recognition accuracy is still being validated.

Run:

```bash
python scripts/poc_pipeline_anpr_debug.py \
  --video data/visionops_original_20s_test.mp4 \
  --model yolo11n.pt \
  --plate-model models/license_plate.pt \
  --output outputs/anpr_debug.mp4 \
  --db outputs/anpr_debug.db \
  --ocr-every 2 \
  --plate-conf 0.25 \
  --min-plate-width 25 \
  --min-plate-votes 2 \
  --debug-crops outputs/plate_crops
```

Current ANPR flow:

```text
Vehicle detection
  ↓
Vehicle tracking
  ↓
License plate detection
  ↓
Plate crop
  ↓
OCR
  ↓
Candidate normalization
  ↓
Multi-frame validation
  ↓
Vehicle association
```

ANPR is experimental. Plate candidates should be manually reviewed.

## Running Tests

Run:

```bash
python -m pytest tests/ -v
```

The test suite covers:
- Database migration
- Detection persistence
- Duplicate-track protection
- Offline worker failures
- Review-state database updates
- OCR candidate handling
- Video processing flow

Passing automated tests does not validate AI accuracy. Model accuracy must be evaluated separately using labeled footage.

## Test Videos

Use different videos for different validation goals.

### Dashboard regression video
Use the original developer test footage to validate:
- Offline analysis
- Detection History
- Vehicle Analytics
- Dashboard refresh
- Basic violation flow

### ANPR regression video
Use footage containing clear visible number plates to validate:
- Plate localization
- OCR preprocessing
- OCR candidate accuracy
- Vehicle-to-plate association

### Violation regression footage
Use clearly labeled examples for:
- No helmet
- Triple riding
- Wrong direction

Keep local videos under `data/` and out of Git.

## Known Limitations

### Vehicle counting
Tracking IDs can fragment in crowded or moving-camera footage. Parked vehicles can also inflate counts if motion or counting-line rules are not used.

### Vehicle classification
Generic COCO-trained models may misclassify local vehicle types such as auto-rickshaws.

### License plates
A clear full-resolution frame does not guarantee a large enough plate crop for OCR. Accuracy depends on plate size, viewing angle, motion blur, compression, crop quality, contrast, and repeated observations across frames.

### Helmet and triple riding
Helmet and triple-riding detection are not yet considered validated. They require suitable model weights, reliable rider-to-motorcycle association, and labeled evaluation footage.

### e-Challan
The current e-Challan workflow is a POC review flow only. It is not a production enforcement or government-integrated system.

## Development Workflow

Do not work directly on `main`.

Create a feature branch:

```bash
git switch -c feature/<name>
```

Before committing:

```bash
git status
python -m pytest tests/ -v
```

Then:

```bash
git add .
git diff --cached --stat
git commit -m "<type>: <description>"
git push -u origin feature/<name>
```

Open a Pull Request and review before merging.

## Git Ignore Policy

Keep these local:

```text
.venv/
.idea/
.vscode/
data/
outputs/
processed_videos/
evidence/
*.pt
*.onnx
*.db
*.sqlite
*.log
.env
```

Do not commit videos, model weights, local databases, generated evidence, secrets, or RTSP credentials.

## Current Development Priorities

1. Stabilize offline detection and tracking
2. Keep database and dashboard totals consistent
3. Improve ANPR preprocessing and multi-frame OCR voting
4. Validate helmet detection using labeled footage
5. Improve triple-riding rider association
6. Validate webcam and RTSP streams
7. Validate Camera Management
8. Integrate verified ANPR results into Detection History and Violation Review
9. Expand to multi-camera POC testing

## POC Success Criteria

Core flow:

```text
Video input
  ↓
Vehicle detection
  ↓
Tracking
  ↓
Detection event
  ↓
Database record
  ↓
Evidence
  ↓
Dashboard / review
```

ANPR flow:

```text
Vehicle track
  ↓
Plate detection
  ↓
OCR candidate
  ↓
Repeated-frame agreement
  ↓
Database association
  ↓
Operator verification
```

AI accuracy should be reported separately from software functionality.

## Project Status

VisionOps is currently an active POC under development.

The core application provides a working foundation for offline video analysis, vehicle detection and tracking, database persistence, dashboard review, and experimental ANPR. The next phase is focused on improving detection accuracy, validating traffic-violation models, and testing live camera workflows.
