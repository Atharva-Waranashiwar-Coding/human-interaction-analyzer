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

## Extension baseline and compatibility

Baseline inspected on November 12, 2026 at commit `532ae8d`, clean `main` tracking `origin/main`. Before implementation, backend verification passed 180 tests with one optional real-video case skipped. Frontend lint and TypeScript passed with two existing React effect dependency warnings.

Architecture: Next.js App Router provides sequential multi-upload, selected-video controls, the existing playback/detection overlay, annotation editor/table, and JSON/JSONL export. FastAPI stores videos and annotations in SQLite through SQLAlchemy, with additive migrations and local FFmpeg/FFprobe upload validation. A serialized local YOLO pose/ByteTrack and sparse object-detector pass consumes one in-memory sampled-frame stream. Camera transforms feed existing temporal person, movement, manipulation, and pickup/carry/place grouping rules. Normalized raw detections are cached on each video; extracted inference frames are not permanently stored. Analysis jobs preserve user-saved annotations and rejected suggestions.

Reusable extension points are the cached detection list, `Annotation` records and stable IDs, the bulk review endpoint, export serializer, background analysis job, and existing player. Compatibility risks include bulk-save replacement of annotations, source changes on review, reanalysis preserving manual records, legacy detection caches lacking camera reliability, and tracking discontinuities. Object pushing/pulling and complex packing/sorting/loading remain manual; direct object handover and a conservative relocation parent already exist. New capabilities must not imply that all listed tasks are automatically recognized or that image overlap proves containment/contact.

## Scene and activity extension

### What was added

The extension reuses the sampled detection cache, existing action annotations, SQLite database, bulk review/save API, export path, and single playback component. It introduces no model, service, dependency, second video pipeline, automatic retraining, or paid API. Original behavior rules remain intact; the detection cache additionally records `camera_state` for each object-sampling timestamp. Existing caches remain usable for static entity relations, while motion and region transitions require explicit reliable static-camera evidence.

New annotation types are `spatial`, `activity`, and `audio`; legacy records default to `behavior`. These types use the same editor, IDs, review statuses, and persistence as existing annotations. Automatic additions always begin as `needs_review`, with `confidence: null` and observable evidence. The Scene and activities tab summarizes the advanced annotations and opens the existing editor for review. The editor defaults to behavior rows, with a type filter for advanced rows. Region outlines appear in the existing detection overlay. Search results seek that same player rather than opening another player.

### Spatial relationships

`spatial.py` generates sustained `near`, `overlapping`, `above`, `below`, `left`, `right`, `moves toward`, and `moves away` suggestions. Distances use person height when available, otherwise object height. It requires at least three observations and 0.8 seconds of sustained evidence. Both directions of entity relations are explicit. Long interruptions, missing entities, scene cuts, and camera-state changes break histories. Regions require static-camera evidence; moving or unknown cameras suppress region analysis and trajectory relations. Static pairwise relations can still be inspected on older caches.

Configure a table, shelf, container, floor, workspace, or other image region with an ID, label, kind, and `[left, top, right, bottom]` box in normalized square letterbox coordinates. For a landscape video with aspect ratio `r`, visible normalized image bounds are `x=0..1`, `y=(1-1/r)/2..(1+1/r)/2`. For portrait video, visible bounds are `x=(1-r)/2..(1+r)/2`, `y=0..1`. Region outlines provide a visual check. IDs must be unique and cannot collide with tracked entity IDs.

`enters region` and `exits region` require stable center membership on both sides, with no intervening missing observation or unreliable camera estimate. Region membership is supporting evidence; it is never described as physical containment. `possible on-surface placement` additionally requires an existing placement action, a uniquely supported table/shelf region, and a stationary object tail. Overlap alone never produces placement. Surface height, depth, support, and contact remain unconfirmed.

### Configurable activity recognition

`sequences.py` defines small `ActivityDefinition` and `Step` records. Each definition specifies ordered labels, optional steps, a total time window, and a maximum inter-step gap. Matching preserves participant IDs and object IDs, references the existing action IDs, and records `complete`, `partial`, or `interrupted` plus missing required steps. Optional steps are never fabricated. Separate identities can produce concurrent activities; tracking-ID changes cannot be silently joined. Scene cuts and missing tracked entities between steps interrupt a sequence.

Default candidates are:

- `object_relocation`: pickup/lift → carry or walking while carrying → place, with optional release, within 20 seconds and a three-second inter-step gap.
- `object_transfer`: references the existing direct handover/receiving action. It does not add a new handover detector.
- `repeated_pickup_placement`: at least two nonoverlapping complete relocation patterns by the same participants within 60 seconds. It does not establish packing, sorting, loading, or unloading.

Activities reference their evidence in `details.action_ids` rather than copying the actions. Source and destination regions are included only when repeated observations uniquely support the same region around the first/last action. Missing or overlapping region evidence leaves these fields absent. A complete sequence means the rule found its required action annotations, not that the task succeeded. The earlier `moving objects between locations` parent annotation and `parent_activity_id` remain unchanged for compatibility.

Packing/unpacking, sorting, loading/unloading, shelf-specific tasks, helping, and object pushing/pulling remain manual unless independently supported annotations are added. The sequence definitions are Python configuration records; this phase does not add a workflow designer or automatically infer task intent.

### Feedback and evaluation

The server captures immutable `original_suggestion`, `rule_version`, per-annotation `review_history`, and optional `reviewer_feedback`. Saving Reviewed is recorded as accepted or corrected according to changes from the original suggestion; Rejected records rejection. Submitted copies of provenance fields are accepted for round trips but cannot overwrite server provenance. A video-level review log preserves rejected removals after their annotation row is deleted. Reanalysis and cached enrichment preserve saved/corrected/rejected rows.

Evaluation requires separate, independent manual annotations with `evaluation_role: "ground_truth"` and `review_status: "reviewed"`. A prediction cannot simply be relabeled as ground truth. The evaluation request explicitly declares covered behaviors, interval, scenario, and whether ground truth is exhaustive for that scope. Without this declaration or usable reviewed data, the API returns a reason and no scores. Declaring exhaustive scope includes responsibility for labeling missed actions and for verifying genuinely empty intervals.

Reviewed original predictions, including rejected suggestions, are matched one-to-one against independent ground truth using exact behavior and entity ID/class sets with temporal intersection-over-union of at least 0.5 by default. Maximum-cardinality matching avoids greedy double-counting. Results include TP, FP, FN, precision, recall, F1, per-behavior metrics, and excluded unreviewed predictions. Undefined ratios are `null`, not invented zeros. Changes in IDs can affect matching and must be corrected consistently. Metrics cover reviewed predictions in the declared scope, not unreviewed predictions or the entire dataset. `/api/evaluation` accepts multiple video scopes and aggregates counts and metrics by scenario; per-video and per-behavior results remain available in its response. No training runs automatically.

### Structured activity search

Search uses the current SQLite annotations and JSON metadata. Filters cover behavior/activity label, annotation type, exact person count, object category, participant ID, object ID, review status, interval overlap, video, and ordered action sequence. Participant/object filters require the association in the same annotation. Sequence filters use referenced action IDs in order and support up to eight labels, separated by `>`. Queries are parameterized and paginated (default 50, maximum 100; UI pages use 25). Results include video, timestamp interval, entities, status, and an editor link. Search includes saved legacy annotations and clearly displays unreviewed results. Unsaved editor changes are excluded until Save. There are no embeddings, natural-language search, or new search services.

### Optional audio

Audio is disabled by default. Only clicking Inspect audio metadata or explicitly calling its endpoint invokes an audio-stream FFprobe probe. Results cache codec, sample rate, channels, and track start/end intervals synchronized to video timestamps. Missing stream duration uses a documented video-duration fallback. No audio is extracted or permanently stored. No speech recognition, speaker identity, emotion, intent, or automatic audio-event model is added. Manual `audio` annotations use the same timestamps, editor, review, search, and export as other annotations. Videos without audio continue normally and report `has_audio: false` only after an explicit probe.

### Additive API and schema changes

Existing endpoints and required fields retain their behavior. New endpoints are:

| Method and path | Purpose |
|---|---|
| `PUT /api/videos/{id}/regions` | Validate/save regions and regenerate cached suggestions |
| `POST /api/videos/{id}/understanding` | Regenerate spatial and activity suggestions without model inference |
| `GET /api/videos/{id}/feedback` | Read persisted feedback history |
| `POST /api/videos/{id}/evaluation` | Evaluate an explicit video scope |
| `POST /api/evaluation` | Evaluate multiple scopes and aggregate by scenario |
| `GET /api/activities/search` | Search annotations and activity sequences |
| `POST /api/videos/{id}/audio` | Explicitly probe/cache audio metadata |

SQLite migrations add `scene_regions`, `audio_metadata`, and `review_log` to videos. Annotation additions are `annotation_kind`, `details`, `original_suggestion`, `review_history`, `reviewer_feedback`, `rule_version`, and `evaluation_role`. Defaults preserve existing records. Migration is idempotent and makes a local backup before altering an existing schema; an existing pre-object backup is preserved and a separate pre-understanding backup is used. No tables are added. JSON exports retain original keys and add annotation metadata plus regions/audio/review log at video level. JSONL remains one annotation per line with additive fields. Linked action IDs must exist in the same saved set and point to behavior annotations; parent activity validation remains in place.

### Files changed

New backend services: `spatial.py`, `sequences.py`, `annotation_data.py`, `understanding.py`, `evaluation.py`, `activity_search.py`, and `audio.py`; new router: `backend/app/api/understanding.py`. Integration changes: `models.py`, `schemas.py`, `database.py`, `main.py`, existing `api/routes.py`, `services/video_service.py`, and the cached camera-state field in `services/local_analysis.py`. Existing detector/tracker, person temporal, manipulation and camera-estimation modules are reused.

New frontend components: `ActivitySearch.tsx`, `UnderstandingPanel.tsx`, shared `lib/types.ts`; existing `app/page.tsx` and `DetectionOverlay.tsx` provide navigation, editor integration, region overlays, and timestamp seeking. Tests are added in `test_spatial.py`, `test_sequences.py`, `test_evaluation.py`, `test_understanding_api.py`, and `test_audio.py`. The existing optional real-video workflow now checks the source appropriate to each annotation kind while retaining the original `local_model` contract for behavior annotations.

### Manual verification steps

1. Start the app using the existing local setup. Restart the backend after this update so additive migrations run. Upload real MP4/MOV/WebM clips, select one, Analyze, and confirm playback, detections, old behavior suggestions, manual editing, Save, and JSON/JSONL export still work. Analyze again should use saved results; Reanalyze should preserve saved/rejected rows.
2. Open Scene and activities. For a static-camera clip, configure a floor/workspace and table/shelf region using the documented coordinates and check the violet outlines in Video & editor. Save regions. Inspect relation evidence, timestamps, and entity IDs; an overlapping box without a placement action must not produce a surface-placement claim. Moving-camera clips must not produce region-entry/exit or directional motion suggestions from unsupported screen motion.
3. Use a clip with pickup, carrying, and placement on the same visible track. Save or correct the underlying action annotations, then Update suggestions from saved results. Confirm grouped action IDs and status. Remove a required action or interrupt a track to inspect a partial/interrupted candidate rather than fabricated completion. Check simultaneous actions by different people remain separate.
4. Open Review in editor on a spatial or activity row. Correct its label/entities/times, add reviewer feedback, mark Reviewed or Rejected, and Save. Reload and check the original suggestion and history count; export retains provenance. Updating cached suggestions must preserve the reviewed row.
5. Add separate manual action annotations from independently reviewed video. Select Independent manual ground truth and Reviewed, then Save. In Evaluation, set covered labels, scope, scenario and exhaustive labeling checkbox. Calculate and inspect TP/FP/FN and defined/undefined metrics. Without complete labeling, confirm the explanatory no-score response. Create a deliberately missing action to verify FN accounting; a rejected false suggestion should contribute FP against complete independent truth.
6. Expand Search activities across videos. Search by label, object class, participant/object association, person count, status, and interval; search `picking up / lifting > carrying > placing / putting down` for sequences. Open a result and confirm the existing player seeks its timestamp and its annotation is editable. Check a nonexistent label returns zero matches.
7. On an audio and a silent clip, explicitly Inspect audio metadata and verify track presence and intervals. Add a manual audio event, edit its times/description, review, Save, search by audio type, and export. Until inspection is requested, no audio processing occurs.

### Performance and remaining limitations

The extension performs only cached-box arithmetic, ordered annotation matching, small SQLite queries, and explicit FFprobe metadata inspection. On the two existing real-video caches (21 and 25 sampled detection frames), a single spatial-plus-sequence pass measured approximately 3.5 ms and 0.3 ms respectively, excluding database I/O and inference. These small samples are not a general performance benchmark. Real video model passes remained the same single pipeline; their measured local analysis times were approximately 8.6 and 8.7 seconds in the first integration run. New processing adds no model memory or frame storage; annotations and review history add SQLite storage proportional to evidence/review volume.

Geometry remains two-dimensional. Regions are image anchored, not world coordinates; moving-camera region mapping and physical containment remain unsupported. Legacy caches lack camera reliability and need Reanalyze for trajectory/region rules. Occlusions and ID changes conservatively fragment workflows. Advanced task intent, comprehensive packing/sorting/loading recognition, automatic audio classification, training, calibrated model accuracy, and a visual workflow designer remain future work. Representative independent ground truth is still needed before drawing recognition-quality conclusions. Real-video checks validate execution and integration, not accuracy of every new relation or workflow.


### Extension validation results

The full extension regression run passed **210 tests including two real-video API workflows** in **23.70 seconds** on November 12, 2026. This covers 208 deterministic/API tests and two explicitly supplied real clips; no real-video case was skipped in that run. The suite includes additive migration and legacy export checks, spatial motion/occlusion/cut negatives, sequence completeness and identity continuity, accept/reject/correct provenance, deleted-review retention, independent-ground-truth matching, scoped/scenario evaluation, structured and ordered-sequence searches, optional audio probing with an actual generated audio track, and silent-video compatibility. The existing Starlette/httpx deprecation warning remains; it does not fail tests.

Frontend lint, TypeScript checking and the production build passed. An isolated Chrome session using a temporary SQLite database and a copied real clip verified advanced activity review, persisted feedback, region configuration, absence of invented evaluation scores without ground truth, audio metadata and manual event saving, structured search, and existing-player seeking to 2 seconds, with **zero browser JavaScript errors**. Scripted manual action rows exercised the UI sequence controls; those rows are not recognition ground truth. User videos/databases were read or copied for testing rather than edited. Test media, temporary databases, browser artifacts, and weights remain outside Git.

The extension was developed and verified on `feature/spatial-activity-understanding`. No deployment or production configuration change was performed. Restart the local backend to activate the additive schema migration and endpoints; the frontend uses the existing development/build commands. No known regression was found in the completed checks. Exhaustive recognition quality across real scenes remains unverified and requires representative independent ground truth.
