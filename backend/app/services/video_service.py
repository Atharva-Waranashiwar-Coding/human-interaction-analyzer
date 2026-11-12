from __future__ import annotations

import json
import uuid
import os
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Any, TypedDict

from fastapi import HTTPException, UploadFile

from ..database import SessionLocal
from ..models import Annotation, Video
from .annotation_data import serialize_annotation, snapshot, AUTOMATIC_SOURCES

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}
MAX_VIDEO_SIZE_BYTES = int(os.getenv("MAX_VIDEO_SIZE_BYTES", "52428800"))
MAX_VIDEO_DURATION_SECONDS = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "60"))
DEFAULT_SAMPLE_INTERVAL_SECONDS = float(os.getenv("SAMPLE_INTERVAL_SECONDS", "2.5"))
UPLOAD_ROOT = Path(os.getenv("UPLOAD_ROOT", str(Path(__file__).resolve().parents[1] / "data" / "uploads")))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


class UnsupportedVideoError(ValueError):
    pass


class VideoMetadata(TypedDict):
    duration: float
    width: int
    height: int
    codec: str
    file_size: int


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._") or "video"
    return f"{stem}_{os.urandom(4).hex()}"


def _ffprobe(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,avg_frame_rate",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip() or "ffprobe failed"
        raise UnsupportedVideoError(f"Video metadata could not be read: {error_text}")
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams", [])
    format_info = payload.get("format", {})
    width = int(streams[0].get("width", 0)) if streams else 0
    height = int(streams[0].get("height", 0)) if streams else 0
    duration = float(format_info.get("duration") or 0.0)
    codec = (streams[0].get("codec_name") if streams else "unknown") or "unknown"
    file_size = int(format_info.get("size") or 0)
    return {"duration": duration, "width": width, "height": height, "codec": codec, "file_size": file_size}


def validate_upload(file: UploadFile) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A video file is required.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
    if file.size and file.size > MAX_VIDEO_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"Video exceeds the {MAX_VIDEO_SIZE_BYTES} byte limit.")
    return {"suffix": suffix, "filename": file.filename}


async def save_uploaded_video(file: UploadFile) -> tuple[Path, str, dict[str, Any]]:
    validation = validate_upload(file)
    destination = UPLOAD_ROOT / safe_filename(file.filename)
    try:
        with destination.open("wb") as handle:
            total = 0
            while chunk := await file.read(65536):
                total += len(chunk)
                if total > MAX_VIDEO_SIZE_BYTES:
                    raise HTTPException(status_code=400, detail="Video exceeds configured byte limit.")
                handle.write(chunk)
        if not destination.stat().st_size:
            raise HTTPException(status_code=400, detail="Uploaded video is empty.")
        try:
            metadata = _ffprobe(destination)
        except UnsupportedVideoError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        import math
        if not math.isfinite(metadata["duration"]) or not 0 < metadata["duration"] <= MAX_VIDEO_DURATION_SECONDS:
            raise HTTPException(status_code=400, detail="Video duration is invalid or exceeds configured limit.")
        if metadata["width"] <= 0 or metadata["height"] <= 0:
            raise HTTPException(status_code=400, detail="Video dimensions are invalid.")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    content_hash = sha256_file(destination)
    return destination, content_hash, metadata


def get_video_path_by_id(video_id: str) -> Path:
    file_path = UPLOAD_ROOT / f"{video_id}*"
    matches = sorted(UPLOAD_ROOT.glob(f"{video_id}*"))
    if not matches:
        raise FileNotFoundError(f"Video {video_id} does not exist on disk.")
    return matches[0]


def sample_video_frames(video_path: Path, duration: float) -> list[Path]:
    if duration <= 0:
        return []
    max_frames = max(1, min(12, int(duration / max(DEFAULT_SAMPLE_INTERVAL_SECONDS, 0.1))))
    sample_times = []
    interval = max((duration / max_frames), DEFAULT_SAMPLE_INTERVAL_SECONDS)
    current = 0.0
    while current < duration and len(sample_times) < max_frames:
        sample_times.append(min(current, max(duration - 0.1, 0.0)))
        current += interval
    if not sample_times:
        sample_times = [0.0]
    if sample_times[-1] != max(duration - 0.1, 0.0):
        sample_times.append(max(duration - 0.1, 0.0))
    sample_times = sorted(set(round(t, 3) for t in sample_times))

    frames: list[Path] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="human_interaction_frames_"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    for index, time_value in enumerate(sample_times[:12]):
        output = temp_dir / f"frame_{index}.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            str(time_value),
            "-i",
            str(video_path),
            "-vf",
            "scale=960:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease",
            "-frames:v",
            "1",
            str(output),
        ]
        subprocess.run(command, check=False)
        if output.exists() and output.stat().st_size > 0:
            frames.append(output)
    return frames


def parse_model_response(raw_text: str) -> list[dict[str, Any]]:
    content = raw_text.strip()
    if "```" in content:
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError("Model response was not valid JSON.")
    if isinstance(payload, dict):
        items = payload.get("annotations") or payload.get("predictions") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    if not isinstance(items, list):
        raise ValueError("Model output must be a list or object with an annotations field.")

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "start_time": float(item.get("start_time", 0.0)),
                "end_time": float(item.get("end_time", 0.0)),
                "behavior": str(item.get("behavior", "unknown")).strip() or "unknown",
                "participants": list(item.get("participants") or []),
                "description": str(item.get("description", "")).strip(),
                "confidence": item.get("confidence"),
                "review_status": str(item.get("review_status") or "needs_review"),
                "source": str(item.get("source") or "model"),
            }
        )
    return normalized


def run_model_inference(video_path: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    from .local_analysis import analyze
    return analyze(video_path, metadata)


def create_annotation_records(video_id: str, prediction_list: list[dict[str, Any]]) -> list[Annotation]:
    annotations: list[Annotation] = []
    for item in prediction_list:
        item = dict(item, id=item.get("id") or uuid.uuid4().hex)
        version = item.get("rule_version") or ("local-rules-v1" if item.get("source")=="local_model" else None)
        annotation = Annotation(
            id=item.get("id"),
            annotation_kind=item.get("annotation_kind", "behavior"),
            details=json.dumps(item.get("details") or {}),
            original_suggestion=json.dumps(snapshot(dict(item, rule_version=version))) if item.get('source') in AUTOMATIC_SOURCES else '{}',
            rule_version=version,
            objects=json.dumps(item.get("objects") or []),
            parent_activity_id=item.get("parent_activity_id"),
            video_id=video_id,
            start_time=float(item.get("start_time", 0.0)),
            end_time=float(item.get("end_time", 0.0)),
            behavior=str(item.get("behavior", "unknown")).lower(),
            participants=json.dumps(item.get("participants") or []),
            description=str(item.get("description", "")),
            source=str(item.get("source") or "model"),
            confidence=item.get("confidence"),
            review_status=str(item.get("review_status") or "needs_review"),
        )
        if annotation.end_time < annotation.start_time:
            annotation.end_time = annotation.start_time
        annotations.append(annotation)
    return annotations


def build_export_payload(video: Video, annotations: list[Annotation]) -> dict[str, Any]:
    return {
        "video_id": video.id,
        "duration": float(video.duration),
        "annotations": [serialize_annotation(item) for item in annotations],
        "scene_regions": json.loads(video.scene_regions or '[]'),
        "audio_metadata": json.loads(video.audio_metadata or '{}'),
        "review_log": json.loads(video.review_log or '[]'),
    }


def remove_video_storage(video: Video) -> None:
    storage_path = Path(video.storage_path)
    if storage_path.exists():
        storage_path.unlink(missing_ok=True)


def clear_frames(frame_paths: list[Path]) -> None:
    for path in frame_paths:
        if path.exists():
            path.unlink(missing_ok=True)
