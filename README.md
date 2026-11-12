# Human Interaction Analyzer

A local application for reviewing human activity and object interactions in video. Next.js provides playback and annotation editing; FastAPI, SQLite, FFmpeg and pretrained vision models provide storage and analysis.

**Upload → Analyze → Review → Save → Export JSON/JSONL**

Automatic annotations are geometry-based review suggestions. They start as `needs_review`, include observable evidence, and have uncalibrated action confidence represented as `null`. No API key, paid inference service, facial recognition, identity matching, emotion or relationship inference is used.

## Contents

- [Setup](#setup)
- [Run locally](#run-locally)
- [Review workflow](#review-workflow)
- [Automatic behavior suggestions](#automatic-behavior-suggestions)
- [Manual-only behaviors](#manual-only-behaviors)
- [Configuration](#configuration)
- [Analysis and tracking](#analysis-and-tracking)
- [Annotations and storage](#annotations-and-storage)
- [Testing and validation](#testing-and-validation)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)

## Setup

Run commands from the project directory. Install FFmpeg and FFprobe on PATH, and use Python and Node.js versions compatible with the existing dependencies.

### Backend and person model

```bash
python3 -m venv .venv  # Skip if the environment already exists.
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
python backend/setup_model.py
```

The person model is [YOLO11 nano pose](https://docs.ultralytics.com/models/yolo11/), cached as `backend/data/models/yolo11n-pose.pt` (approximately 6 MiB).

### Box and package detection

```bash
.venv/bin/python -m pip install -r backend/requirements-boxes.txt
.venv/bin/python backend/setup_box_model.py
```

This prepares [YOLO-World small](https://docs.ultralytics.com/models/yolo-world/) with a fixed portable-object vocabulary: boxes, cartons, packages, envelopes, bags and other supported manipulation classes. The prepared checkpoint, `backend/data/models/yolov8s-world-boxes.pt`, is approximately 25 MiB. A source checkpoint of approximately 25 MiB remains cached for rebuilding.

Preparation uses a one-time CLIP encoder download of approximately 338 MiB, then removes that encoder cache after saving vocabulary embeddings. Video inference uses the saved embeddings with no text encoder or network access. The setup-only CLIP dependency and its text/tokenizer dependencies are separate from the base requirements.

Use `--force` to rebuild the prepared vocabulary, or `--keep-setup-cache` to retain the encoder for vocabulary experiments. No extra model is downloaded during analysis.

For a smaller general COCO detector without box/packet categories, use:

```bash
.venv/bin/python backend/setup_model.py --objects
```

The prepared box-capable model takes priority when present. An explicit `LOCAL_OBJECT_MODEL_PATH` overrides selection. The COCO fallback cannot reliably classify cardboard boxes or packets and displays that limitation. The prepared vocabulary focuses on portable objects; beds and refrigerators are outside it.

### Frontend

```bash
cd frontend
npm install
cd ..
```

## Run locally

Backend, from the project directory:

```bash
.venv/bin/uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Frontend, in a second terminal:

```bash
cd frontend
npm run dev
```

Open http://localhost:3000. Use one backend process. Restart it after Python changes or model setup, and avoid reload mode during inference. Interrupted jobs become failed on restart and can be retried.

For a production frontend check, run `npm run build` inside `frontend`; `npm start` serves that build. Production output uses `.next-build`, separate from the development `.next` cache.

## Review workflow

1. Click **Upload videos** and select one or more MP4, MOV or WebM files. Files upload sequentially with per-file success/error results; a failed or duplicate file does not stop the batch. Defaults are 50 MiB and 60 seconds per file.
2. Select a video. **Analyze** appears in its **Selected video** controls beside **Reanalyze**, **Save** and **Delete**. Completed results are reused; **Reanalyze** explicitly recomputes them.
3. In **Video & editor**, play the video and toggle **Show sampled person/object detections**. Anonymous IDs and classes describe raw model observations at sampled timestamps.
4. Open **Annotations table** for behavior rows and raw detection samples. Search filters both tables; detection samples are paginated in groups of 50. Switching videos resets the tab, filter and page.
5. Review evidence, edit times/labels/participants, correct object IDs/classes, choose a parent activity, or add missing manual annotations. Mark results **Reviewed** or **Rejected**, then **Save**. The table reflects current edits; saving persists them.
6. Export JSON or JSONL from either tab. Saved/manual/rejected annotations remain protected during reanalysis.

Raw detection boxes and behavior annotations are different outputs. Recognizing a box does not establish that someone lifted it. Manual association corrections change annotations, not the underlying detector observations. A completed analysis can have detection samples but no qualifying behavior suggestions.

## Automatic behavior suggestions

Distances are normalized by estimated body height. Required pose joints generally need confidence of at least 0.6; degenerate limbs and nonfinite data are rejected. Human interaction windows default to four seconds, with sustained evidence requirements. Object histories cover up to six seconds and require multiple actual observations.

### People and movement

| Behavior | Observable evidence | Limitation |
|---|---|---|
| Approaching another person | Sustained decrease in relative image distance | Image motion does not establish social intent or physical depth. |
| Moving apart | Sustained increase in relative image distance | Perspective and camera movement affect distance. |
| Walking together | Nearby tracks with similar velocity and stable separation | Co-motion is not proof of gait or shared purpose. |
| Standing close together | Nearby tracks with low image velocity | Proximity does not establish a relationship. |
| Raising a hand | Confident wrist held above its shoulder | Held pose does not prove raising onset. |
| Waving | Raised wrist oscillation with at least two meaningful reversals and 0.15-height lateral excursion | A stationary raised hand and small jitter are excluded; short/fast waves may be missed. |
| Pointing | Stable lateral arm extension, near-straight segments, reach of at least 0.3 heights and another torso along the ray | No finger pose; upward/object-directed pointing is not covered. Participants: pointer, possible target. |
| Following | Delayed agreement with another person's turning trajectory over at least 2.4 seconds, within 0.2 heights | Straight same-direction walking is insufficient. Participants: follower candidate, leader candidate. |
| Handshaking | Previously separated wrists converge between bodies within 0.12 heights, below shoulders along torso axes, with sustained correlated oscillation | High contact, static proximity and brief contact are excluded; occluded/cropped hands can prevent suggestions. |
| Hugging | Close torsos, reciprocal bent-arm reach and sustained wrapping after an observed non-wrapped state | Standing close or unilateral reaching is insufficient; clips starting mid-hug may be missed. |
| Pushing | Bent-to-extended arm near another torso, followed within 0.4–1.2 seconds by increased recipient speed and displacement exceeding 0.25 heights | Does not establish force, cause or aggression. Participants: initiator candidate, recipient. |
| Group activity | At least three nearby tracks moving in aligned directions with stable spacing for at least 1.6 seconds | Coordinated-motion proxy; stationary groups and unrelated co-moving pedestrians remain limitations. |
| Standing | Upright torso, extended legs below hips and low image motion | Requires reliable lower-body pose. |
| Walking | Lateral body movement with alternating knee positions | Primarily a side-view proxy; frontal walking or hidden legs may be missed. |
| Bending | Sustained torso inclination relative to visible hips | Viewpoint and foreshortening affect the geometry. |
| Reaching | Increasing shoulder-to-wrist distance into a nearly straight arm | No object association or contact is implied. |

Waving and handshake oscillation rules require at least 4 FPS. Following compares delays of approximately 0.6, 1.0 and 1.4 seconds through a shared turn. Handshake candidates require at least one wrist-motion reversal and more than 0.05-height excursion. Pair rules require reasonably comparable body scales. Different gestures, pairs and interactions can overlap.

### Objects and activity sequences

| Behavior | Observable evidence | Limitation |
|---|---|---|
| Reaching toward an object | Shrinking wrist/object gap, arm extension and approximately stationary object | Mere body approach is insufficient; contact remains unconfirmed. |
| Picking up / lifting | Stationary visible object gains a wrist association, then rises more than 0.18 heights with that association | Sparse sampling, hidden hands and nonvertical motion can prevent recognition. |
| Holding | Stable geometry after observed acquisition/carrying, or sustained two-arm support around a torso-level object | One nearby wrist is insufficient. Two confident wrists, visible arm support and stable relative position are required for the support-pose alternative; physical support remains unconfirmed. |
| Carrying | Sustained single-person wrist association and person/object co-motion exceeding 0.2 heights | Co-motion is not proof of physical support or ownership. |
| Walking while carrying | Carrying evidence plus sustained lateral body displacement | Gait and frontal/depth motion remain uncertain. |
| Placing / putting down | Hand-associated descent, wrist proximity when the object first settles, then visible separation and stationary object | Support surface is unconfirmed; early separation followed by falling/resting is excluded. |
| Releasing | Sustained wrist association ends while wrist and stationary object remain visible | Missing wrists or disappearing objects never imply release. |
| Handing over / receiving objects | Same continuous object changes donor-only → shared → recipient-only associations, with displacement and visible separation | Gaps, third-person associations or reversed phases invalidate transfer. Participants: donor, recipient. |
| Joint carrying / lifting | Two people maintain wrist proximity while both move with the same object | Shared support is unconfirmed; large/occluded objects and more than two holders are missed. |
| Moving objects between locations | Complete pickup → carrying → placement evidence for one uninterrupted object/person epoch and different source/destination positions | Creates a parent activity; partial/interrupted sequences and durations above 20 seconds are excluded. |

Manipulation rules operate on plausible portable classes such as boxes, packages, bags, bottles, cups, books, phones, laptops, utensils, fruit and balls. Package/envelope/parcel synonyms normalize to `package`; cartons normalize to `cardboard box`. Detection and subtype classification can still be wrong.

## Manual-only behaviors

All labels remain available as free text in the existing editor. Native label suggestions include activities that are not automatically recognized.

| Category | Manual-only labels |
|---|---|
| Object manipulation | Dropping; pushing/pulling objects; dragging/sliding; stacking/unstacking; opening/closing; inserting/removing; throwing/catching |
| Human movement | Running; sitting; standing up; sitting down; crouching; kneeling; turning; jumping; walking backward |
| Collaboration | Airborne passing; helping or assisting with object movement |
| Tasks | Packing/unpacking; sorting/organizing; loading/unloading; repeated manipulation sequences and task outcomes |

Human-to-human pushing is separate from manual-only object pushing/pulling. Generic containers, disappearance, proximity and pose alone do not establish these actions or successful task completion.

## Configuration

Export settings in the backend shell using [.env.example](.env.example) or [sample.env](sample.env) as a reference. Python does not implicitly load the root `.env`. Frontend API configuration uses `frontend/.env.local` or its shell environment.

| Variable | Default | Purpose |
|---|---|---|
| `LOCAL_MODEL_PATH` | `backend/data/models/yolo11n-pose.pt` | Person pose weights; use an absolute override path |
| `LOCAL_SAMPLE_FPS` | `5` | Pose sampling rate, 1–15 FPS |
| `LOCAL_IMAGE_SIZE` | `960` | Pose input; multiple of 32 in 128–1280 |
| `LOCAL_DEVICE` | `auto` | Validated CPU path; explicit device overrides remain supported |
| `LOCAL_DETECTION_CONFIDENCE` | `0.25` | Person detection threshold |
| `OBJECT_DETECTION_ENABLED` | `true` | Set `false` to disable object inference |
| `LOCAL_OBJECT_MODEL_PATH` | Prepared box model when present; otherwise nano COCO | Explicit object-model override |
| `OBJECT_SAMPLE_FPS` | `2`, capped by pose rate when unset | Object sampling, 0.5–min(pose FPS, 5); sparse rates may not support interaction rules |
| `OBJECT_IMAGE_SIZE` | `640` | Object input; multiple of 32 in 128–1280 |
| `OBJECT_DETECTION_CONFIDENCE` | `0.3` | Object detection threshold |
| `BEHAVIOR_MIN_SECONDS` | `1` | Human-rule minimum duration; temporal suggestions require at least 0.6 seconds |
| `BEHAVIOR_WINDOW_SECONDS` | `4` | Human history window, 3–10 seconds; must cover minimum duration |
| `BEHAVIOR_MERGE_GAP_SECONDS` | `0.4` | Human event merge gap, 0–1 second |
| `BEHAVIOR_CLOSE_HEIGHTS` | `0.8` | Pair proximity threshold in body heights |
| `BEHAVIOR_SPEED_HEIGHTS` | `0.15` | Pair motion threshold in heights/second |
| `MAX_VIDEO_SIZE_BYTES` | `52428800` | Per-file upload limit |
| `MAX_VIDEO_DURATION_SECONDS` | `60` | Per-file duration limit |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Frontend API address |
| `DATABASE_PATH`, `UPLOAD_ROOT` | Existing local database/upload paths | Optional overrides, used for test isolation |

The pinned Ultralytics release warns about pose issues on Apple MPS. Auto therefore uses CPU; explicit MPS selection is not validated for these suggestions.

## Analysis and tracking

Each enabled model loads once per backend process. One FFmpeg stream decodes/resizes frames in memory, respecting display orientation and sampling onto a fixed timestamp grid. Pose runs at its configured rate; one selected object detector consumes a subset of those same frames. No extracted frames are permanently stored and no second video decoder runs for objects.

[ByteTrack](https://docs.ultralytics.com/modes/track/) supplies anonymous `person_N` and `object_N` segments. Trackers reset between videos. Missing observations, implausible jumps, abrupt scale/class changes and ambiguous object overlaps end reliable continuity. IDs can fragment or switch during crossings; they are not unique-person/object counts and do not reconnect identity through occlusion.

Background optical flow excludes previous person boxes, checks forward/backward consistency and fits a RANSAC affine transform. Measurements distinguish static background, measured camera motion, unknown background motion and possible scene cuts. Reliable transforms compensate pan, moderate rotation and zoom in the trajectory coordinates; displayed detection boxes retain their original video coordinates. Unknown intervals break trajectory history and rebase the next reliable interval instead of being treated as confirmed camera movement.

Waving, pointing, handshake/hug geometry, reaching, bending and relative two-arm holding remain enabled on raw poses when background motion is unknown. Following, pushing, group motion, walking and object trajectories require reliable coordinates. Scene changes reset both histories and renew track aliases. This is image-based compensation rather than full 3D stabilization; hidden/textureless backgrounds, severe perspective changes and occlusion remain limitations.

The old blanket camera-warning message has been replaced. Compensated camera motion is not itself an analysis failure. Background uncertainty is reported when it affects most samples and no behavior qualifies; pose-relative rules are still evaluated. Console metrics separate unknown, compensated and scene-cut frames. Previously cached warnings/results require Reanalyze.

Repeated evidence merges within continuous histories. Occlusion, ID changes, unreliable trajectory intervals and scene discontinuities prevent cross-gap trajectory merging or activity grouping. Object motion uses a reliable torso anchor when available rather than the detection-box foot point, which can be clipped by the frame. Unrelated bystanders no longer reset an object holder’s history, and an older missing wrist does not invalidate a later continuous valid window. Existing human rules remain active alongside object and movement rules.

## Annotations and storage

SQLite and local video storage remain the persistence layer. No queue service, cloud infrastructure or additional database service is required.

Annotations retain their existing IDs, timestamps, behavior, participants, evidence/description, source, confidence and review status. Object-aware fields are:

```json
{
  "objects": [{"id": "object_1", "class_name": "cardboard box"}],
  "parent_activity_id": null
}
```

Parent activities are ordinary annotation rows referenced by child IDs. Saves preserve stable annotation IDs. The API rejects duplicate/cross-video IDs, dangling or cyclic parents, nonfinite/reversed/negative times and timestamps outside the video. Deleting a parent in the editor clears its children's links. Older save payloads default missing object/parent fields to `[]`/`null`.

Video details also contain timestamped normalized `detections` and `analysis_warnings`. Detection boxes are compact metadata for overlays and tables, not images. JSON/JSONL annotation exports include evidence, object associations and parent links; raw detections remain available through video details.

Startup adds the object/parent/detection/warning columns idempotently to existing tables. Before updating an older schema, it creates `backend/data/human_interaction.pre-object-backup.sqlite3`. Existing videos, annotations and uploads are preserved.

| API behavior | Result |
|---|---|
| `POST /api/videos/{id}/analyze` | Run local inference or reuse completed/saved results |
| `POST /api/videos/{id}/analyze?force=true` | Reanalyze; replace only unreviewed local suggestions |
| Reanalysis with saved/manual/rejected rows | Preserve rows; suppress matching label/person/object/time duplicates and remap preserved parent links |
| Duplicate in-flight analysis | Atomic SQLite claim; return 409 |
| Save/delete during processing | Return 409; editor inputs are disabled while processing |
| Missing/broken object weights | Continue human analysis and manual editing with a visible warning |
| Decode/person-model failure | Visible failed status with retry available; pending replacements roll back |
| Backend restart during analysis | Recover processing jobs as failed for explicit retry |

Weights, model settings, databases/backups, uploads, virtual environments and caches are excluded from Git.

## Testing and validation

### Automated checks

```bash
.venv/bin/python -m pytest backend/tests -q
cd frontend
npm run build
```

Tests use temporary SQLite/upload storage. Real-video cases are opt-in:

```bash
REAL_TEST_VIDEOS="/absolute/clip1.mp4:/absolute/clip2.mp4" \
LOCAL_DEVICE=cpu .venv/bin/python -m pytest backend/tests -q -s
```

`REAL_TEST_VIDEO` accepts a single path. Optional `REAL_HANDSHAKE_VIDEO`, `REAL_BOX_VIDEO`, `REAL_HOLDING_VIDEO` and `REAL_CARRYING_VIDEO` must match a supplied absolute path and require the corresponding handshake, box, holding or carrying result. Otherwise real cases verify execution, timestamps, schema, editing and export without asserting recognition accuracy.

The latest recorded run, **November 11, 2026**, passed **186 tests**, including six real-video API workflows, in 63.84 seconds. Frontend build/type checks passed. An isolated Chrome/Playwright check verified the Selected video Analyze button, both tables, filtering, pagination, corrected live cardboard-box rows, the holding row on `4277372…`, and absence of browser JavaScript errors. The temporary browser-testing tool was not added to application dependencies.

Coverage includes synthetic positive/negative behavior sequences; 0.5×/2× scale and 5/10/15 FPS cases; high-fives, held raised hands, static proximity, accidental contact and parallel walking; missing joints, ID changes and camera/temporal gaps; object pickup/carry/place/transfer/grouping; camera-state separation, pan/zoom compensation, scene cuts, pose-relative rules under unknown motion, stationary-scene pan negatives, and camera-tolerant two-arm holding without invented locomotion; bystander and old-keypoint-gap regressions; model failures; migration; stable IDs; review preservation; parent validation; and JSON/JSONL exports. A shared-stream test verifies one decode, ten pose calls and four object calls at configured 5/2 FPS. Prepared-model inference also passed with socket connections blocked and no text encoder.

Two existing frontend effect-dependency warnings and backend HTTPX/Starlette/asyncio configuration warnings remain. Tests ran in the existing development environment (Python 3.13, Ultralytics 8.3.228, PyTorch 2.15.0); base dependency pins were resolution-checked, not separately benchmarked in a fresh environment.

### Measured results and limits

| Development example | Recorded observation |
|---|---|
| Packing clip `6868104…`, prepared portable-object model | 53 pose frames, 21 object calls, 9.547 seconds CPU, approximately 664 MiB process peak RSS |
| Same clip after live reanalysis | 12 cardboard-box, 3 paper-bag and 25 person observations across samples; no bed/refrigerator observations |
| Packing clip behavior output | No lifting suggestion; cropped pose/fragmented tracks remain limiting. The camera update reduces unavailable trajectory frames from 46/53 to 31/53 and compensates 17 frames |
| Yellow mailer subtype | Sometimes classified as paper bag; package/bag subtype accuracy remains imperfect |
| Box-carrying clip `4277372…` | Holding at 2.6–4.0s and carrying/walking-while-carrying candidates at 3.0–4.0s after camera compensation; unavailable trajectory frames fell from 54/62 to 5/62 |
| Overhead clip `3044843…` | Handshake suggestion at 7.6–9.6 seconds retained through later regressions |
| Meeting clip `5520035…` | Camera compensation reduced unavailable trajectory samples from 61/67 to 10/67; pose-relative reaching remains enabled, while some handshakes are still missed |
| Portrait clip `6170610…` | 13 moving-camera frames compensated and 19/40 frames remain unknown; the former blanket 34/40 warning no longer represents the current analysis |
| Earlier nine-clip nano-object run | Approximately 3–30 seconds CPU; no qualifying manipulation/parent-activity suggestions |

Observation counts include repeated samples and fragmented tracks, not ground-truth object/person totals. Timings include inference/decoding/rules and may include initialization; they exclude upload/browser rendering. RSS is a process high-water mark including models, imports and prior tests, not incremental inference or GPU memory. These runs are not a controlled benchmark or latency guarantee. Earlier small-person testing at 640 missed co-motion that 960 proposed, motivating the current pose resolution.

There is no representative frame-level ground truth, calibrated action confidence, measured precision/recall or identity-switch rate. Holding and carrying now have a real-clip positive example; positive pickup/placement, transfer and activity-sequence cases remain primarily synthetic. Real clips validate execution and integration and demonstrate some object and carrying candidates, not calibrated or comprehensive activity recognition. The packing example fixes category coverage on that clip, not general classification accuracy.

### Manual checks

1. Use a well-lit stationary-camera clip with visible full bodies, wrists and objects. Keep each phase visible for roughly 1–2 seconds.
2. Compare wave versus held hand, handshake versus high-five, hug versus standing close, following through a turn versus straight co-motion, and group movement versus stationary/unrelated people.
3. Show an object still → reach → pickup → carry → place elsewhere → hand withdrawal. Check before/after evidence and parent links. Compare against static proximity, independent motion, hidden hands and disappearing objects.
4. For transfer, show donor-only → shared → recipient-only contact-like geometry. For joint carrying, both people/object must co-move with visible hands. Use existing suitable footage for push review.
5. Inspect playback, raw detections and behavior tables separately. Correct associations/times/labels, mark Reviewed or Rejected, save, export both formats and reanalyze to confirm saved rows remain protected.
6. Record expected events and match suggestions by interval and participants before counting false positives/misses. Add manual annotations when evidence is insufficient; no prediction does not prove no interaction occurred.

## Troubleshooting

| Symptom | Check |
|---|---|
| Model is not configured / old API-key message | Restart the backend with the current local code; no API key is required |
| Boxes/packets appear as unrelated objects | Prepare the box-capable model, restart, remove any unintended explicit model override and Reanalyze cached clips |
| Object visible, no lifting annotation | Check full wrist/body visibility, track continuity and the camera diagnostic. A clip starting with an already-held box does not establish pickup onset; two-arm support can still qualify as holding. |
| Missing/corrupt weights | Run the relevant setup command; remove only the corrupted cached file before downloading it again |
| Empty annotation table | Inspect detection samples separately; add manual behavior labels if no suggestion qualifies |
| Failed upload | Check codec, extension, size/duration and duplicate status; per-file errors do not stop a multi-upload batch |
| Video controls visible but unresponsive | Refresh the page and check that development JavaScript assets load; production builds must use the separate cache |
| Build stalls on cloud-evicted development files | Keep `.next` excluded from TypeScript checks; production route types are generated in `.next-build` |

## Project structure

| Path | Responsibility |
|---|---|
| [backend/app/services/local_analysis.py](backend/app/services/local_analysis.py) | Shared frame stream, cached models, original rules, motion gate and metrics |
| [backend/app/services/camera_motion.py](backend/app/services/camera_motion.py) | Background estimates and compensated trajectory coordinates |
| [backend/app/services/temporal_behaviors.py](backend/app/services/temporal_behaviors.py) | Human gesture/interaction windows and event merging |
| [backend/app/services/object_tracking.py](backend/app/services/object_tracking.py) | Selected object model, vocabulary-aware tracking and anonymous IDs |
| [backend/app/services/object_activities.py](backend/app/services/object_activities.py) | Object association, manipulation/movement rules and parent activities |
| [backend/app/services/video_service.py](backend/app/services/video_service.py) | Upload validation, FFprobe, annotation records and exports |
| [backend/app/api/routes.py](backend/app/api/routes.py) | Existing upload, analysis, save, playback and export APIs |
| [backend/app/database.py](backend/app/database.py), [models.py](backend/app/models.py), [schemas.py](backend/app/schemas.py) | SQLite initialization/migration and data contracts |
| [backend/setup_model.py](backend/setup_model.py), [setup_box_model.py](backend/setup_box_model.py) | Explicit one-time local model preparation |
| [frontend/app/page.tsx](frontend/app/page.tsx) | Multiple uploads, per-video controls, tabs, editor, tables and export |
| [frontend/components/DetectionOverlay.tsx](frontend/components/DetectionOverlay.tsx) | Sampled person/object boxes on playback |
| [backend/tests](backend/tests) | Existing regressions and focused model/rule/API/storage tests |
