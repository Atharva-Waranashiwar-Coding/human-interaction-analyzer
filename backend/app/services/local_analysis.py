"""Local pose inference. No network access occurs during analysis."""
from __future__ import annotations

import itertools
import math
import os
import subprocess
import threading
import time
from pathlib import Path

from .temporal_behaviors import TemporalSuggestions
from .object_activities import ObjectActivities, MovementSuggestions
from .object_tracking import get_object_model, ObjectTracker
from .camera_motion import CameraEstimate,TrajectoryCoordinates,estimate_camera,transform_observations

MODEL_PATH = Path(os.getenv("LOCAL_MODEL_PATH", str(Path(__file__).resolve().parents[2] / "data/models/yolo11n-pose.pt")))
LOCK = threading.Lock()
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise RuntimeError("Local pose weights missing. Run: python backend/setup_model.py")
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        config = MODEL_PATH.parent / "config"
        config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config))
        os.environ.setdefault("MPLCONFIGDIR", str(config / "matplotlib"))
        from ultralytics import YOLO, settings
        settings.update({"sync": False})
        _model = YOLO(str(MODEL_PATH), task="pose")
    return _model


def device():
    # This pinned Ultralytics release warns of pose errors on MPS. Use the
    # validated CPU path in auto mode; explicit device overrides remain possible.
    choice = os.getenv("LOCAL_DEVICE", "auto")
    return "cpu" if choice == "auto" else choice


def frames(path, duration, fps, size):
    """FFmpeg emits resized BGR frames on a fixed timestamp grid, in memory only."""
    import numpy as np
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-an", "-vf",
               f"setpts=PTS-STARTPTS,fps={fps}:start_time=0,scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    count = 0
    try:
        while True:
            raw = bytearray()
            while len(raw) < size * size * 3:
                chunk = process.stdout.read(size * size * 3 - len(raw))
                if not chunk:
                    break
                raw.extend(chunk)
            if not raw:
                break
            if len(raw) != size * size * 3:
                raise RuntimeError("Incomplete decoded frame")
            timestamp = count / fps
            count += 1
            if timestamp < duration:
                yield timestamp, np.frombuffer(raw, dtype=np.uint8).reshape(size, size, 3).copy()
        if process.wait() != 0:
            raise RuntimeError("FFmpeg could not decode the video")
        if not count:
            raise RuntimeError("No decodable video frames")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


class Rules:
    """Conservative screen-space evidence, never semantic relationship inference."""
    def __init__(self, fps, split_relative=False):
        self.fps = fps
        self.temporal = TemporalSuggestions(fps,mode="trajectory" if split_relative else "all")
        self.relative_temporal = TemporalSuggestions(fps,mode="relative") if split_relative else None
        self.previous = {}
        self.active = {}
        self.output = []
        self.hold = float(os.getenv("BEHAVIOR_MIN_SECONDS", "1"))
        self.close = float(os.getenv("BEHAVIOR_CLOSE_HEIGHTS", "0.8"))
        self.speed = float(os.getenv("BEHAVIOR_SPEED_HEIGHTS", "0.15"))

    def step(self, t, people, camera_motion=False,relative_people=None,scene_cut=False):
        if self.relative_temporal is not None:
            self.relative_temporal.step(t,relative_people if relative_people is not None else people,scene_cut)
        self.temporal.step(t, people, camera_motion)
        if scene_cut:
            for key in list(self.active):self.finish(key)
            self.previous={}
        candidates = {}
        for pid, p in (relative_people if relative_people is not None else people).items():
            k = p["pose"]
            for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
                if all(k[i][2] >= .6 for i in (shoulder, elbow, wrist)) and k[wrist][1] < k[shoulder][1] - .15 * p["height"]:
                    candidates[("raising a hand", (pid,))] = "Wrist held above shoulder; raising onset is not established."
        for a, b in itertools.combinations(sorted(people), 2):
            p, q = people[a], people[b]
            scale = (p["height"] + q["height"]) / 2
            if scale <= 0 or max(p["height"], q["height"]) / min(p["height"], q["height"]) > 1.5:
                continue
            distance = math.dist(p["center"], q["center"]) / scale
            oldp, oldq = self.previous.get(a), self.previous.get(b)
            if not oldp or not oldq or camera_motion:
                continue
            dt = t - oldp[0]
            if dt <= 0 or dt > 1.5 / self.fps:
                continue
            vp = [(p["center"][i] - oldp[1]["center"][i]) / scale / dt for i in range(2)]
            vq = [(q["center"][i] - oldq[1]["center"][i]) / scale / dt for i in range(2)]
            sp, sq = math.hypot(*vp), math.hypot(*vq)
            old_distance = math.dist(oldp[1]["center"], oldq[1]["center"]) / scale
            change = (distance - old_distance) / dt
            label = None
            if distance < self.close and max(sp, sq) < self.speed:
                label = "standing close together"
            elif distance < 3 and change < -self.speed:
                label = "approaching another person"
            elif distance < 3 and change > self.speed:
                label = "moving apart"
            elif distance < 1.5 and min(sp, sq) > self.speed and sum(x*y for x, y in zip(vp, vq)) / (sp*sq) > .9 and abs(change) < self.speed:
                label = "walking together"
            if label:
                candidates[(label, (a, b))] = "Sustained relative screen-space motion/proximity; verify perspective and camera motion."
        for key in list(self.active):
            if key not in candidates:
                self.finish(key)
        for key, description in candidates.items():
            if key not in self.active:
                self.active[key] = [t, t, 1, description]
            else:
                self.active[key][1] = t
                self.active[key][2] += 1
        self.previous = {pid: (t, p) for pid, p in people.items()}

    def finish(self, key):
        start, end, count, description = self.active.pop(key)
        if count >= 3 and end - start >= self.hold:
            self.output.append(dict(start_time=round(start, 4), end_time=round(end, 4), behavior=key[0], participants=list(key[1]), description=description, source="local_model", confidence=None, review_status="needs_review"))

    def finish_all(self):
        for key in list(self.active):
            self.finish(key)
        return sorted(self.output + self.temporal.results() + (self.relative_temporal.results() if self.relative_temporal else []), key=lambda e: (e["start_time"], e["behavior"], e["participants"]))


def background_camera_motion(previous,current,boxes,fps):
    return estimate_camera(previous,current,boxes,fps)


class AnalysisResult(list):
    """Keep the existing list interface while carrying review-overlay metadata."""
    def __init__(self, predictions, detections=None, warnings=None, metrics=None):
        super().__init__(predictions)
        self.detections=detections or []
        self.warnings=warnings or []
        self.metrics=metrics or {}


def analyze(path, metadata):
    import cv2
    import numpy as np
    started = time.perf_counter()
    fps = float(os.getenv("LOCAL_SAMPLE_FPS", "5"))
    size = int(os.getenv("LOCAL_IMAGE_SIZE", "960"))
    if not 1 <= fps <= 15 or size < 128 or size > 1280 or size % 32:
        raise ValueError("LOCAL_SAMPLE_FPS must be 1..15; LOCAL_IMAGE_SIZE must be 128..1280 and divisible by 32")
    # The singleton predictor/tracker is mutable: serialize all jobs, reset between videos.
    with LOCK:
        model = get_model()
        if model.predictor is not None and hasattr(model.predictor, "trackers"):
            for tracker in model.predictor.trackers:
                tracker.reset()
        object_fps=float(os.getenv("OBJECT_SAMPLE_FPS",str(min(2,fps))))
        object_size=int(os.getenv("OBJECT_IMAGE_SIZE","640"))
        if object_size<128 or object_size>1280 or object_size%32:raise ValueError("OBJECT_IMAGE_SIZE must be 128..1280 and divisible by 32")
        if not .5<=object_fps<=min(fps,5):raise ValueError("OBJECT_SAMPLE_FPS must be .5..min(LOCAL_SAMPLE_FPS,5)")
        object_model,warning=get_object_model()
        warnings=[warning] if warning else []
        object_tracker=ObjectTracker(object_model,object_fps) if object_model is not None else None
        object_rules=ObjectActivities(object_fps)
        movement=MovementSuggestions(fps,mode="trajectory")
        relative_movement=MovementSuggestions(fps,mode="relative")
        coordinates=TrajectoryCoordinates()
        camera_unknown_frames=0
        camera_compensated_frames=0
        scene_cut_frames=0
        detections=[]
        next_object_time=0.
        object_frames=0
        rules = Rules(fps,split_relative=True)
        previous_gray = None
        previous_boxes = []
        seen = set()
        aliases = {}
        last_seen = {}
        frame_count = 0
        peak_people = 0
        camera_gated_frames = 0
        selected_device = device()
        for t, frame in frames(path, float(metadata["duration"]), fps, size):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            estimate = (background_camera_motion(previous_gray,gray,previous_boxes,fps)
                        if previous_gray is not None else CameraEstimate("static",np.eye(3)))
            if isinstance(estimate,bool):estimate=CameraEstimate("unknown" if estimate else "static",None if estimate else np.eye(3))
            inverse=coordinates.update(estimate)
            limited=inverse is None
            cut=estimate.state=="cut"
            camera_gated_frames+=int(limited)
            camera_unknown_frames+=int(estimate.state=="unknown")
            camera_compensated_frames+=int(estimate.state=="moving" and not limited)
            scene_cut_frames+=int(cut)
            previous_gray = gray
            if cut:
                aliases.clear();last_seen.clear()
                if model.predictor is not None and hasattr(model.predictor,"trackers"):
                    for tracker in model.predictor.trackers:tracker.reset()
                if object_tracker is not None:object_tracker.reset_tracks()
            result = model.track(frame, persist=True, tracker="bytetrack.yaml", device=selected_device, imgsz=size, conf=float(os.getenv("LOCAL_DETECTION_CONFIDENCE", ".25")), verbose=False, save=False)[0]
            previous_boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
            people = {}
            if result.boxes is not None and result.boxes.id is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy()
                poses = result.keypoints.data.cpu().numpy()
                for box, raw_id, pose in zip(boxes, ids, poses):
                    if not np.isfinite(box).all() or not np.isfinite(pose).all():
                        continue
                    raw_id = int(raw_id)
                    # A discontinuity gets a fresh anonymous ID, even if ByteTrack reuses its ID.
                    if raw_id not in aliases or t - last_seen[raw_id] > 1.5 / fps:
                        aliases[raw_id] = f"person_{len(seen)+1}"
                        seen.add(aliases[raw_id])
                    last_seen[raw_id] = t
                    height = float(box[3]-box[1])
                    if height <= 0:
                        continue
                    people[aliases[raw_id]] = dict(center=((float(box[0])+float(box[2]))/2, float(box[3])), height=height, pose=pose, box=box.tolist())
            peak_people = max(peak_people, len(people))
            stable_people,_=transform_observations(people,{},inverse)
            rules.step(t,stable_people,limited,relative_people=people,scene_cut=cut)
            movement.step(t,stable_people,limited)
            relative_movement.step(t,people,cut)
            if limited:object_rules.reset(relative=cut)
            if t+1e-6>=next_object_time:
                next_object_time+=1/object_fps
                objects={}
                if object_tracker is not None:
                    try:
                        objects=object_tracker.step(t,frame,selected_device,object_size)
                        object_frames+=1
                    except Exception as exc:
                        warnings.append(f"Object detection interrupted ({type(exc).__name__}); person analysis continues.")
                        object_tracker=None
                        object_rules.reset()
                if cut:object_rules.reset()
                else:
                    _,stable_objects=transform_observations({},objects,inverse)
                    object_rules.step(t,stable_people,stable_objects,limited,relative_people=people,relative_objects=objects)
                boxes=[dict(id=pid,class_name='person',box=[v/size for v in p['box']]) for pid,p in people.items()]
                boxes += [dict(id=oid,class_name=o['class_name'],box=[v/size for v in o['box']]) for oid,o in objects.items()]
                detections.append(dict(time=round(t,4),items=boxes))
            frame_count += 1
        predictions = sorted(rules.finish_all()+movement.results()+relative_movement.results()+object_rules.results(),key=lambda e:(e["start_time"],e["behavior"]))
        if not predictions and frame_count and camera_unknown_frames/frame_count>.5:
            warnings.append("Background motion could not be estimated reliably for much of this video. Pose-relative gestures and holding were still evaluated; motion-based actions need more reliable background and track evidence.")
        if scene_cut_frames:
            warnings.append(f"Detected {scene_cut_frames} possible scene changes; behavior histories were reset at those boundaries.")
        import resource
        metrics = dict(seconds=round(time.perf_counter()-started, 3), sampled_frames=frame_count, object_inference_frames=object_frames, object_tracks=object_tracker.count if object_tracker else 0, camera_gated_frames=camera_gated_frames, camera_unknown_frames=camera_unknown_frames, camera_compensated_frames=camera_compensated_frames, scene_cut_frames=scene_cut_frames, peak_tracked_people=peak_people, anonymous_tracks=len(seen), annotations=len(predictions), device=selected_device,
                       process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if __import__('sys').platform == 'darwin' else 1024))
        print("Local analysis:", metrics, flush=True)
        return AnalysisResult(predictions,detections,warnings,metrics)
