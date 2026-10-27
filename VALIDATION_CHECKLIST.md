# VisionOps POC acceptance checklist

Use the **same clean original video**, the same model weights, and the same parameters for both developers. The processed video contains drawn boxes and is **not** a clean input.

## Before running

- [ ] Make a Git branch. Copy `main.py` and back up `traffic_history.db` before replacing existing files.
- [ ] Install requirements, confirm `python -m pytest -q` passes.
- [ ] Download/copy `yolo11n.pt` to repo root. The UI does not automatically fetch missing files.
- [ ] Confirm recorded test video exists at `data/visionops_original_20s_test.mp4`.

## 1. Dashboard / analytics

- [ ] Run one video and record run ID and tracked vehicle count.
- [ ] In desktop app choose **All completed runs** or **Latest run**.
- [ ] Compare dashboard and Vehicle Analytics with `python scripts/db_health.py --db traffic_history.db`.
- [ ] Ensure older legacy per-frame rows are excluded from dashboard totals but still visible in History.
- [ ] Refresh buttons visibly update timestamps, and counts change after new records arrive.
- [ ] Check no duplicate `run_id + track_id` records are created.

## 2. Offline processing

- [ ] MP4 decodes and produces a playable output video.
- [ ] Missing file, corrupt input, missing model, and unavailable output directory all generate error feedback.
- [ ] A failed processing run is **not** labeled complete and does not set progress to 100%.
- [ ] Count only stable tracks that move at least the configured pixel threshold; calibrate for footage.
- [ ] For fixed cameras test an optional count line and compare with manual traffic count.
- [ ] Remember moving cameras can falsely produce motion and tracker fragmentation.

## 3. SQLite / violation history

- [ ] Verify `detections` table has `run_id`, `track_id`, `frame_number` and `video_path`.
- [ ] Verify candidate `violations` have run/track/frame/video identifiers where a link exists.
- [ ] Evidence images exist at paths stored in the database.
- [ ] For automatic candidates from a separate violation model, the model's track ID is NOT automatically the primary vehicle's ID.

## 4. Review and demo challan

- [ ] Add a manual candidate, select it, edit plate and location, accept it as operator.
- [ ] Change demo role to authority, approve, generate demo challan, and refresh.
- [ ] Verify status, fine, challan number, and audit trail remain consistent.
- [ ] Verify stale/invalid status updates do not partially update the record.
- [ ] Export demo PDF and verify its contents.
- [ ] Treat UI dropdown roles and challans as DEMO ONLY. This is not secure authorization, certified evidence, or government integration.

## 5. Camera testing

- [ ] Run `python scripts/camera_probe.py --source 0` on a laptop webcam.
- [ ] Enter and save webcam index or camera URL in Camera Management, select Test camera.
- [ ] Test real RTSP only when camera and network are available, via `VISIONOPS_RTSP_URL` environment variable.
- [ ] Record whether opening, first frame, sustained read, and stop all work.

## 6. ANPR experimental path

- [ ] Supply separate `models/license_plate.pt` weights.
- [ ] Run `scripts/poc_pipeline_anpr_debug.py` on the same original video.
- [ ] Save and inspect cropped plate images. Distinguish unreadable crop from incorrect OCR.
- [ ] Compare original vs enlarged/CLAHE/sharpened/thresholded OCR outputs.
- [ ] Only retain repeated candidates matching the POC plate pattern; do not fill missing characters.
- [ ] Manually verify actual plate characters. A pattern-compliant OCR guess is NOT a verified plate.
- [ ] Do not use this POC output for automatic enforcement.

## 7. Violation accuracy (requires extra assets)

- [ ] Inspect `violations.pt` classes. COCO vehicle model does not detect no-helmet.
- [ ] Label footage with helmet and triple-riding ground truth.
- [ ] For each violation, report TP, FP, FN, precision and recall. Measure separately per camera/video.
- [ ] Improve/customize trained model weights and rider-to-motorcycle association.
- [ ] Keep all violation events pending human review.

**Cannot be verified with code alone:** real RTSP streams, actual helmet/triple-riding accuracy, plate OCR performance on real source footage, government registration lookup, legal admissibility, and authenticated challan issuance.
