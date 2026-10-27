# VisionOps — Next Steps & Work Distribution

**Date:** October 27, 2026  
**Status:** POC Stabilisation Phase (6–8 weeks)  
**Current state:** main.py and poc_support.py staged for commit; README updated with comprehensive project documentation.

---

## Current Blockers

1. **Staged files need to be committed** — main.py and poc_support.py were restored from feature branch but not yet committed to main
2. **Work assignments unclear** — Sumit and Waseem need clear parallel tracks to avoid conflicts
3. **ANPR accuracy not yet benchmarked** — no baseline for improvement comparison
4. **Live camera validation pending** — RTSP integration has not been tested on real hardware

---

## Immediate Actions (This Week)

### 1. Commit Staged Files to Main (Waseem — 30 min)
```bash
git add main.py poc_support.py
git commit -m "feat: restore desktop POC app and support libraries to main

This restores the working PySide6 dashboard (main.py) and AI pipeline support
(poc_support.py) from feature/poc-integrity-fixes to main. Includes:
- Complete 9-page PySide6 desktop dashboard
- YOLO v11n vehicle detection + ByteTrack tracking
- EasyOCR ANPR pipeline with multi-frame voting
- SQLite persistence with audit logging
- e-Challan violation review workflow

This is the current POC baseline. Next: improve ANPR accuracy with PaddleOCR.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push origin main
```

### 2. Create Feature Branch for ANPR Improvement (Waseem — 5 min)
```bash
git switch -c feature/anpr-paddleocr
```

---

## 6–8 Week Sprint Plan

### Sprint 1: ANPR Accuracy Improvement (Weeks 1–2)

**Owner: Waseem**  
**Goal:** Validate ANPR baseline with EasyOCR, then switch to PaddleOCR and measure improvement

#### Tasks (Parallel where possible)

1. **Benchmark EasyOCR accuracy on test footage (3–4 hours)**
   - Run poc_pipeline_anpr_debug.py on existing test video or test footage
   - Collect plate candidates from ~50 vehicles
   - Measure accuracy manually (correct vs. incorrect OCR reads)
   - Document false-positive and false-negative rate
   - Save screenshot of output for comparison

2. **Install PaddleOCR and create wrapper (4–6 hours)**
   - `pip install paddlepaddle paddlepaddle-gpu paddleocr`
   - Create `ocr_read_plate_paddle()` function in poc_support.py alongside existing EasyOCR
   - Keep preprocessing pipeline (CLAHE, sharpening, adaptive thresholding) — it's OCR-agnostic
   - Add config flag to switch between EasyOCR and PaddleOCR at runtime

3. **Benchmark PaddleOCR on same footage (3–4 hours)**
   - Run with PaddleOCR flag enabled
   - Measure accuracy on the same 50 vehicles
   - Document comparison (EasyOCR vs. PaddleOCR)
   - Expected improvement: +5–15% accuracy on Indian plates

4. **Integrate PaddleOCR as default (2 hours)**
   - Update poc_pipeline_anpr_debug.py to use PaddleOCR by default
   - Retain EasyOCR path for fallback/comparison
   - Add --ocr-backend flag to switch at runtime

**Deliverable:** PaddleOCR ANPR benchmark report (EasyOCR vs. PaddleOCR accuracy on 50+ vehicles)

---

### Sprint 2: Evidence Integrity & Video Buffering (Weeks 2–4)

**Owner: Sumit** (code implementation) + **Waseem** (integration & testing)

#### Tasks

**Sumit's work (2–3 days):**

1. **Add SHA-256 hash to evidence records (2–3 hours)**
   - Add `evidence_hash TEXT` column to violations table schema (poc_support.py line 153)
   - Modify `save_violation()` function to compute SHA-256 hash of JPEG file:
     ```python
     evidence_hash = hashlib.sha256(open(evidence_path, 'rb').read()).hexdigest()
     ```
   - Store hash in violations INSERT
   - Update database migration code if needed

2. **Implement circular frame buffer for video clips (1–2 days)**
   - In OfflineWorker class (main.py), add deque to store last N frames
   - N = fps × pre_buffer_seconds (default: 30 fps × 5 sec = 150 frames)
   - On violation trigger: extract pre-buffer frames + record next 5 seconds post-buffer
   - Use cv2.VideoWriter to save MP4 clip
   - Store clip path in violations table (`clip_path TEXT` column)
   - Handle edge cases: violation at start of video, buffer overflow

3. **Update database schema migration (1 hour)**
   - Add `evidence_hash TEXT` and `clip_path TEXT` columns to violations table
   - Write migration code in poc_support.py to handle existing databases

**Waseem's work (testing & integration):**

1. **Update Violation Review UI in main.py (3–4 hours)**
   - Display evidence hash in violation detail panel (read-only, for verification)
   - Add video clip player widget if clip_path is not null
   - Show clip duration and file size
   - Test hash display and clip playback on existing violations

2. **Regression test the database changes (2 hours)**
   - Run poc_pipeline.py to generate violations
   - Verify evidence_hash is computed correctly (spot-check hashes)
   - Verify clip_path is stored correctly
   - Verify Violation Review displays both hash and clip

**Deliverable:** Working video clip + SHA-256 hash in Violation Review panel; no database errors

---

### Sprint 3: Live RTSP Camera Validation (Weeks 4–6)

**Owner: Waseem** (primary) + **Sumit** (support)

**Goal:** Validate that main.py can ingest from a real IP camera and detect violations in real time

#### Tasks

1. **Set up test camera (2–3 hours)**
   - Use existing IP camera from MHTS infrastructure OR
   - Use a USB webcam as fallback (if no IP camera available)
   - Confirm RTSP/ONVIF URL works: `ffprobe rtsp://camera-url`
   - Test OpenCV can open it: `cv2.VideoCapture("rtsp://camera-url")`

2. **Validate Live Camera Detection in main.py (2–4 hours)**
   - Run main.py and open the "Live Camera" tab
   - Select camera from dropdown or enter RTSP URL
   - Verify stream displays in real time
   - Trigger test violations (drive motorcycle without helmet, triple ride, etc.)
   - Confirm violations are detected and stored to database
   - Check Violation Review shows live violations

3. **Validate Camera Management workflow (2 hours)**
   - Test "Camera Management" tab: add new camera with RTSP URL
   - Test camera list refresh, delete camera
   - Verify persistence to database across app restarts

4. **Document findings (1 hour)**
   - List any stream timeouts, reconnection logic issues
   - Note any latency or dropped frames
   - Confirm detection runs at ≥25 fps for real-time feel

**Deliverable:** Live camera detection working; documented performance metrics (latency, FPS, false-positives per hour)

---

### Sprint 4: Violation Module Accuracy Validation (Weeks 6–8)

**Owner: Sumit** (primary) + **Waseem** (review)

**Goal:** Validate helmet, triple-riding, and wrong-direction accuracy on labelled test footage

#### Tasks

1. **Prepare test footage (1–2 days)**
   - Collect 5–10 short video clips (30–60 sec each) covering:
     - Clear helmet violations (rider + no helmet)
     - Clear triple-riding violations (3+ persons on motorcycle)
     - Clear wrong-direction violations (vehicle moving against traffic)
     - Non-violations (normal traffic, parked vehicles)
   - Label ground truth by hand (frame number + violation type)

2. **Validate helmet detection (1–2 days)**
   - Run poc_pipeline.py with violations.pt model on helmet footage
   - Count true positives (correctly detected helmets) vs. false positives
   - Count false negatives (missed violations)
   - Calculate precision = TP / (TP + FP), recall = TP / (TP + FN)
   - Document accuracy per lighting condition (day/night/shadows)
   - Note: COCO-trained yolo11n cannot detect helmets; need violations.pt model

3. **Validate triple-riding detection (1 day)**
   - Run on triple-riding test footage
   - Measure accuracy (same precision/recall metrics)
   - Identify failure modes (e.g., riders partially outside frame, overlapping riders)
   - Document minimum frame rate needed for reliable detection

4. **Validate wrong-direction detection (1 day)**
   - Run on wrong-direction test footage
   - Measure accuracy
   - Confirm 20-frame history and 60px displacement threshold are sufficient
   - Document edge cases (e.g., U-turns, vehicles stopping mid-intersection)

5. **Generate Accuracy Performance Report (1–2 hours)**
   - Compile accuracy metrics (precision, recall, F1-score) per module
   - Compare against proposal targets: Helmet ≥90%, Triple Riding ≥88%, Wrong Direction ≥92%
   - Document confidence scores (are low-confidence detections accurate?)
   - Recommend threshold adjustments or model retraining

**Deliverable:** Accuracy Performance Report (per-module precision/recall, comparison to proposal targets)

---

## Work Distribution

### Waseem's Track (You)
**Focus:** ANPR accuracy, live camera validation, integration testing, end-to-end pipeline

| Sprint | Task | Effort | Owner |
|--------|------|--------|-------|
| 1 | Benchmark EasyOCR + switch to PaddleOCR | 1–2 weeks | Waseem |
| 2 | Integrate video clip + hash display in UI | 3–4 hours | Waseem |
| 3 | Live RTSP camera validation + testing | 2–4 hours | Waseem |
| 4 | Review accuracy performance report | — | Waseem |

**Total:** ~2 weeks of concentrated effort (mostly ANPR in Sprint 1)

---

### Sumit's Track
**Focus:** Code implementation, database changes, violation module testing

| Sprint | Task | Effort | Owner |
|--------|------|--------|-------|
| 2 | Add SHA-256 hash + video clip buffering | 2–3 days | Sumit |
| 4 | Validate helmet, triple-riding, wrong-direction accuracy | 2–3 days | Sumit |

**Total:** ~1 week of focused implementation across two sprints (non-consecutive)

---

## How to Avoid Conflicts

1. **Different files**
   - Waseem: poc_support.py (ANPR functions), scripts/poc_pipeline_anpr_debug.py
   - Sumit: main.py (UI + video clip player), poc_support.py (database schema + save_violation)
   - **Conflict risk:** Both touching poc_support.py in Sprint 2 — coordinate timing

2. **Branching strategy**
   - Waseem: `feature/anpr-paddleocr` (Sprint 1)
   - Sumit: `feature/evidence-integrity` (Sprint 2)
   - Merge to main when each sprint is complete and tested

3. **Testing gates**
   - Never merge without running: `pytest tests/ -v`
   - Always test on existing offline POC video before merging
   - Validate database consistency (test_atomic_review_and_audit)

---

## Definition of Done (Per Sprint)

### Sprint 1 (ANPR)
- [ ] EasyOCR baseline accuracy documented
- [ ] PaddleOCR installed and integrated
- [ ] PaddleOCR shows measurable improvement over EasyOCR
- [ ] poc_pipeline_anpr_debug.py runs with both backends
- [ ] Tests pass: `pytest tests/ -v`
- [ ] PR reviewed and merged to main

### Sprint 2 (Evidence Integrity)
- [ ] evidence_hash column added to schema
- [ ] clip_path column added to schema
- [ ] Database migration handles existing data
- [ ] Violation Review UI displays hash and video clip
- [ ] Video clip plays correctly in player widget
- [ ] Tests pass: `pytest tests/ -v`
- [ ] PR reviewed and merged to main

### Sprint 3 (Live Camera)
- [ ] Live Camera tab successfully connects to RTSP
- [ ] Stream displays in real time without timeouts
- [ ] Violations detected and stored to database
- [ ] Camera Management CRUD works
- [ ] Documented latency and FPS metrics
- [ ] No crashes or errors on reconnection

### Sprint 4 (Accuracy Validation)
- [ ] Test footage labelled and stored
- [ ] Per-module accuracy metrics (precision/recall) calculated
- [ ] Comparison to proposal targets documented
- [ ] Accuracy Performance Report generated
- [ ] Recommendations for next phase documented

---

## Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| PaddleOCR installation fails due to CUDA/GPU | Create Docker container with CUDA pre-installed; use CPU mode if GPU unavailable |
| Violation.pt model unavailable for helmet detection | Use COCO-trained yolo11n for now; note that helmet detection requires custom model retraining |
| No real IP camera available for testing | Use USB webcam as fallback; full RTSP validation deferred if hardware unavailable |
| Database migration breaks existing schema | Write backward-compatible migration; test on copy of traffic_history.db first |
| Live camera stream times out frequently | Tune OpenCV timeout parameters; add automatic reconnection with exponential backoff |

---

## Success Metrics (End of Sprint 4)

1. ✓ ANPR accuracy improved by ≥5% (EasyOCR → PaddleOCR)
2. ✓ Evidence records include SHA-256 hash and video clips
3. ✓ Live camera detection validated on real IP camera
4. ✓ Per-module accuracy metrics documented (vs. proposal targets)
5. ✓ No critical bugs or data loss in database
6. ✓ All tests pass
7. ✓ Comprehensive accuracy performance report ready for Phase 2 approval

---

## Next Quarterly Planning

Once this sprint is complete, Sumit and Waseem should evaluate:
- **Is the POC accuracy acceptable to move forward with Phase 2?**
- **Do we have budget and timeline for Phase 2 (24 weeks, ₹28–32 lakh)?**
- **Should we pilot with a real government/enterprise customer on the POC?**
- **What specific improvements are needed before Phase 2 kickoff?**

---

## How to Track Progress

1. Use git branch naming: `feature/anpr-paddleocr`, `feature/evidence-integrity`, etc.
2. Create GitHub issues per task (or Jira if available)
3. Daily standup: 15 min, async Slack update on blockers
4. Weekly review: check against sprint checklist
5. Sprint retrospective: lessons learned, adjust for next sprint
