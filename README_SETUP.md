# VisionOps POC Recovery and Validation Pack

This package is built from the developer's uploaded **Nexora Vision AI v3 `main.py`** plus the offline/ANPR scripts previously used in VisionOps. It is designed to preserve existing dashboard work, not discard it.

## What is included

- `main.py`: patched **desktop application** with camera, history, dashboard, review and demo challan workflow.
- `poc_support.py`: testable helpers for movement filtering, OCR preprocessing, plate format voting, and SQLite migration.
- `scripts/poc_pipeline.py`: existing independent offline YOLO vehicle POC.
- `scripts/poc_pipeline_anpr_debug.py`: experimental ANPR debugging with padded crops and multi-pass OCR.
- `scripts/db_health.py`: compare stored DB counts without GUI.
- `scripts/camera_probe.py`: test camera input without loading AI.
- `scripts/download_vehicle_weights.py`: fetch standard COCO vehicle weights.
- `tests/`: smoke tests and database regression tests.
- `docs/VALIDATION_CHECKLIST.md`: manual tests for eight original developer-reported issues.

## Critical setup warning

**Make a new Git feature branch and back up the current `main.py` and `traffic_history.db` before copying these files.** Copy into the repository **root** so `poc_support.py` lives next to `main.py`. This app uses a local SQLite DB file in the project root. The patched app performs additive database schema migration when it starts. It does not delete legacy rows automatically.

Do not overwrite the other developer's branch or merge to `main` without a PR.

## 1. Install and run on macOS / PyCharm

In the existing `visionops` repo root:

```bash
git switch -c feature/desktop-poc-integrity
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-anpr.txt
python -m pip install pytest
python -m pytest -q
```

If `git switch` says the branch already exists, use your existing feature branch. If `.venv` already exists, just activate it instead of recreating it.

Get `yolo11n.pt` into the root: `python scripts/download_vehicle_weights.py` (internet may be required on first run). This does not install the **separate** `violations.pt` or `models/license_plate.pt` checkpoints.

Run desktop app:

```bash
python main.py
```

The desktop app has multiple pages. The current `main.py` is a **local PySide6 demo**, not the FastAPI/React enterprise architecture.

## 2. Independent offline video check

Copy the original clean 20-second MP4 to `data/visionops_original_20s_test.mp4`.

```bash
mkdir -p data outputs models
python scripts/poc_pipeline.py \
  --video data/visionops_original_20s_test.mp4 \
  --model yolo11n.pt \
  --output outputs/poc_processed.mp4 \
  --db outputs/poc_detections.db
python scripts/db_health.py --db outputs/poc_detections.db
```

The standalone POC DB is **separate** from the desktop `traffic_history.db`. Its counters will not update the desktop dashboard until you integrate them. Do not copy/merge SQLite files blindly, as their run ID and schema conventions differ.

For the desktop dashboard's own database: `python scripts/db_health.py --db traffic_history.db` after running video from the desktop **Offline Analysis** page.

## 3. Experimental plate OCR

Requires a separate, licensed plate detector checkpoint placed at `models/license_plate.pt`. The general-purpose YOLO vehicle model cannot localize/read plates by itself.

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

The debug script now tests several OCR crop variants, handles split OCR text boxes, requires a plausible standard Indian plate pattern, and uses repeated votes. It still cannot read physically unreadable plates or correct wrong guesses without labeled reference data. A strict pattern rejects some legitimate non-standard registrations (BH series etc.).

**No automatic ANPR-to-desktop integration is claimed.** The two pipelines remain separate until the data contract and source footage have been validated.

## 4. Test live camera / camera management

```bash
python scripts/camera_probe.py --source 0
```

For RTSP, avoid placing credentials on a command line or in Git:

```bash
export VISIONOPS_RTSP_URL='rtsp://username:password@camera-ip:554/stream'
python scripts/camera_probe.py --from-env
unset VISIONOPS_RTSP_URL
```

Then in the desktop app, use **Camera Management** to save/select/test a camera, or **Live Camera** to analyze it. Real RTSP behavior must be verified with the actual device/network.

## 5. Compare dashboard vs database

Test the following in order:

1. From desktop **Offline Analysis**, run the video to completion.
2. Run `python scripts/db_health.py --db traffic_history.db`.
3. Open **Detection History** and filter to **Latest run**. Compare row count by class.
4. Open **Vehicle Analytics**, refresh, and check the same latest-run count.
5. Open **Home Dashboard** with scope **Latest run**. Refresh and compare again.
6. Add a **manual violation candidate**, select it in **Violation Review**, move it through the demonstration workflow, and check audit/history and PDF values.

If these disagree, collect `app_errors.log`, model version, script console output and `db_health.py` result (redact any plate or RTSP credentials before sharing).

## 6. Original issue coverage

| Original report | Patch coverage | Verification still required |
|---|---|---|
| Dashboard refresh | Legacy rows excluded; refreshed SQL totals; timestamp feedback | Verify app counts with same DB |
| Live camera | RTSP open-timeout handling, one-frame camera probe | Test actual webcam/RTSP device |
| Camera management | Saved camera test button and CLI probe | Verify with real selected camera |
| Offline violation accuracy | Reliable worker success/error; movement-filtered track counting; review-only candidate links | Helmet and triple-riding require real data and validated models |
| Analytics refresh | Exclude old inflated per-frame rows; count per run | Verify model IDs and representative footage |
| Detection History | Add track/run/source fields and uniqueness; additive migrations | Inspect original DB and compare manual totals |
| Violation Review | Atomic status/plate/location/audit updates; refresh retained | GUI click-through testing |
| e-Challan refresh | Atomic workflow update, counter refresh, marked demo-only | Test PDF; no actual government/secure issuance |

## 7. What this does not magically fix

- Missing or untrained `violations.pt` model: helmet/seatbelt detection cannot be accurate without it.
- Triple riding heuristic: overlapping people and motorcycles is not proof of rider count.
- Wrong-direction: per-camera calibration and static view are needed.
- OCR text: preprocessing and voting help diagnose errors, but not guarantee accuracy.
- Camera movement, tracker ID switches, and parked vehicles: object counts are approximate until benchmarked.
- Authentication, government integration, legally valid challans or court-certified evidence are **not part of this POC**.

## 8. Next integration step

Get your other developer's PR, apply this patch on a separate branch, and compare side by side. **First merge corrected desktop DB/refresh behavior and unit tests, then evaluate ANPR and violations in separately reviewed PRs.** Do not combine two developers' DB files or assume the independent CLI POC database automatically appears in the desktop dashboard.
