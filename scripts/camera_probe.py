## Purpose

This PR restores and migrates the existing POC scripts and tests into the new VisionOps repository.

## Changes

* Restored `.github/`, `ai/`, `backend/`, `frontend/`, `infra/`, and `docs/`.
* Moved `RELEASE_NOTES.md` and `VALIDATION_CHECKLIST.md` into `docs/`.
* Retained existing POC scripts and tests.
* Retained `main.py` and `poc_support.py`.

## Validation Results

* [x] Python tests: 14 passed.
* [x] Main application launches with local YOLO weights.
* [x] POC pipeline executed successfully.
* [x] ANPR debug pipeline executed successfully.
* [x] Plate detection tested: 26 boxes detected, 9 linked to vehicles.
* [ ] OCR text recognition requires further improvement.

## Scope

This PR focuses on restoring and migrating the existing project structure, POC scripts, and tests into the new repository.

ANPR dashboard integration and detection accuracy improvements will be handled separately in future feature branches after this PR is merged.
