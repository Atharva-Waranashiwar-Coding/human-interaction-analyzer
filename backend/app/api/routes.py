from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..database import SessionLocal, get_db
from ..models import Annotation, Video
from ..schemas import AnnotationBulkUpdate
from ..services.video_service import (
    build_export_payload,
    create_annotation_records,
    get_video_path_by_id,
    remove_video_storage,
    run_model_inference,
    save_uploaded_video,
    sample_video_frames,
    _ffprobe,
)

router = APIRouter(prefix="/api")


def serialize_annotation(annotation: Annotation) -> dict:
    return {
        "id": annotation.id,
        "start_time": float(annotation.start_time),
        "end_time": float(annotation.end_time),
        "behavior": annotation.behavior,
        "participants": json.loads(annotation.participants or "[]"),
        "objects": json.loads(annotation.objects or "[]"),
        "parent_activity_id": annotation.parent_activity_id,
        "description": annotation.description,
        "source": annotation.source,
        "confidence": annotation.confidence,
        "review_status": annotation.review_status,
    }


def process_video_analysis(video_id: str) -> None:
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if video is None:
            return
        target_path = Path(video.storage_path)
        if not target_path.exists():
            video.analysis_status = "failed"
            video.status = "error"
            video.status_message = "Uploaded video file was not found."
            db.commit()
            return
        try:
            metadata = _ffprobe(target_path)
            predictions = run_model_inference(target_path, metadata)
            db.refresh(video)
            video.detections = json.dumps(getattr(predictions,"detections",[]))
            video.analysis_warnings = json.dumps(getattr(predictions,"warnings",[]))
            # Preserve every user-saved annotation, including rejected predictions.
            db.query(Annotation).filter(Annotation.video_id == video.id, Annotation.source == "local_model", Annotation.review_status == "needs_review").delete()
            if predictions:
                protected = db.query(Annotation).filter(Annotation.video_id == video.id).all()
                records = [item for item in create_annotation_records(video.id,predictions)
                           if 0 <= item.start_time <= item.end_time <= video.duration]
                remap = {}
                pending = []
                for item in records:
                    item.id = item.id or uuid.uuid4().hex
                    duplicate = next((existing for existing in protected
                        if existing.behavior == item.behavior and existing.participants == item.participants
                        and existing.objects == item.objects and existing.start_time <= item.end_time
                        and existing.end_time >= item.start_time),None)
                    if duplicate:
                        remap[item.id] = duplicate.id
                    else:
                        pending.append(item)
                available = {item.id for item in protected+pending}
                for item in pending:
                    parent = remap.get(item.parent_activity_id,item.parent_activity_id)
                    item.parent_activity_id = parent if parent in available else None
                    db.add(item)
                video.analysis_status = "completed"
                video.status = "annotated"
                video.status_message = "Behavioral analysis completed."
            else:
                video.analysis_status = "completed"
                video.status = "ready"
                video.status_message = "Local analysis completed with no supported behaviors detected. Manual annotation remains available."
        except Exception as exc:  # pragma: no cover - defensive branch
            db.rollback()
            video = db.get(Video,video_id)
            if video is None:return
            video.analysis_status = "failed"
            video.status = "error"
            video.status_message = f"Analysis failed: {exc}"
        db.commit()
    finally:
        db.close()


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@router.post("/videos")
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    try:
        saved_path, content_hash, metadata = await save_uploaded_video(file)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime only
        raise HTTPException(status_code=500, detail=f"Failed to save video: {exc}") from exc

    existing = db.query(Video).filter(Video.content_hash == content_hash).first()
    if existing is not None:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="Duplicate video already exists in storage.")

    video = Video(
        original_name=file.filename or "untitled",
        stored_name=saved_path.name,
        storage_path=str(saved_path),
        content_hash=content_hash,
        mime_type=file.content_type or "video/mp4",
        size_bytes=saved_path.stat().st_size,
        duration=float(metadata.get("duration") or 0.0),
        width=int(metadata.get("width") or 0),
        height=int(metadata.get("height") or 0),
        status="uploaded",
        analysis_status="pending",
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return {
        "id": video.id,
        "original_name": video.original_name,
        "status": video.status,
        "analysis_status": video.analysis_status,
        "duration": video.duration,
        "width": video.width,
        "height": video.height,
        "file_size_bytes": video.size_bytes,
    }


@router.get("/videos")
def list_videos(db: Session = Depends(get_db)) -> list[dict]:
    videos = db.query(Video).order_by(Video.created_at.desc()).all()
    results = []
    for video in videos:
        results.append(
            {
                "id": video.id,
                "original_name": video.original_name,
                "duration": float(video.duration),
                "width": video.width,
                "height": video.height,
                "status": video.status,
                "analysis_status": video.analysis_status,
                "message": video.status_message,
            }
        )
    return results


@router.get("/videos/{video_id}")
def get_video(video_id: str, db: Session = Depends(get_db)) -> dict:
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")
    annotations = db.query(Annotation).filter(Annotation.video_id == video.id).order_by(Annotation.start_time.asc()).all()
    return {
        "id": video.id,
        "original_name": video.original_name,
        "duration": float(video.duration),
        "width": video.width,
        "height": video.height,
        "status": video.status,
        "analysis_status": video.analysis_status,
        "message": video.status_message,
        "annotations": [serialize_annotation(item) for item in annotations],
        "detections": json.loads(video.detections or "[]"),
        "analysis_warnings": json.loads(video.analysis_warnings or "[]"),
    }


@router.get("/videos/{video_id}/file")
def get_video_file(video_id: str, db: Session = Depends(get_db)) -> FileResponse:
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")
    path = Path(video.storage_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file is missing from storage.")
    return FileResponse(path, media_type=video.mime_type or "video/mp4", filename=video.original_name)


@router.get("/videos/{video_id}/annotations")
def get_annotations(video_id: str, db: Session = Depends(get_db)) -> list[dict]:
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")
    annotations = db.query(Annotation).filter(Annotation.video_id == video_id).order_by(Annotation.start_time.asc()).all()
    return [serialize_annotation(item) for item in annotations]


@router.put("/videos/{video_id}/annotations")
def update_annotations(video_id: str, payload: AnnotationBulkUpdate, db: Session = Depends(get_db)) -> dict:
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")
    if video.analysis_status == "processing":
        raise HTTPException(status_code=409, detail="Wait for analysis to finish before saving annotations.")
    if any(item.end_time > video.duration for item in payload.annotations):
        raise HTTPException(status_code=422, detail="Annotation timestamps exceed video duration.")
    ids = [item.id or uuid.uuid4().hex for item in payload.annotations]
    if len(ids) != len(set(ids)):
        raise HTTPException(status_code=422, detail="Annotation IDs must be unique.")
    if ids and db.query(Annotation).filter(Annotation.id.in_(ids),Annotation.video_id != video_id).first():
        raise HTTPException(status_code=422, detail="Annotation ID belongs to another video.")
    parents = {aid:item.parent_activity_id for aid,item in zip(ids,payload.annotations)}
    for aid,parent in parents.items():
        seen={aid}
        while parent:
            if parent not in parents or parent in seen:
                raise HTTPException(status_code=422, detail="Parent activity must exist in these annotations without cycles.")
            seen.add(parent);parent=parents[parent]
    db.query(Annotation).filter(Annotation.video_id == video_id).delete()
    for index, item in enumerate(payload.annotations):
        annotation = Annotation(
            id=ids[index],
            objects=json.dumps([obj.model_dump() for obj in item.objects]),
            parent_activity_id=item.parent_activity_id,
            video_id=video_id,
            start_time=float(item.start_time),
            end_time=float(item.end_time),
            behavior=str(item.behavior).lower(),
            participants=json.dumps(item.participants or []),
            description=item.description,
            source="manual",
            confidence=item.confidence,
            review_status=item.review_status or "needs_review",
        )
        if annotation.end_time < annotation.start_time:
            annotation.end_time = annotation.start_time
        db.add(annotation)
    video.status = "annotated"
    if video.analysis_status == "completed":
        video.analysis_status = "manual_review"
    video.status_message = "User-edited annotations saved."
    db.commit()
    return {"updated": len(payload.annotations), "status": "saved"}


@router.post("/videos/{video_id}/analyze")
def analyze_video(video_id: str, background_tasks: BackgroundTasks, force: bool = False, db: Session = Depends(get_db)) -> dict:
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")
    if video.analysis_status == "processing":
        raise HTTPException(status_code=409, detail="Video analysis is already in progress.")
    if video.analysis_status in {"completed", "manual_review"} and not force:
        return {"id": video_id, "status": video.analysis_status, "message": "Using saved analysis. Use force=true to reanalyze."}
    # Atomic claim prevents concurrent requests from scheduling the same video.
    claimed = db.query(Video).filter(Video.id == video_id, Video.analysis_status != "processing").update({"analysis_status": "processing", "status": "processing", "status_message": "Running local pose analysis."}, synchronize_session=False)
    db.commit()
    if not claimed:
        raise HTTPException(status_code=409, detail="Video analysis is already in progress.")
    background_tasks.add_task(process_video_analysis, video_id)
    return {"id": video_id, "status": "processing", "analysis_status": "processing"}


@router.get("/videos/{video_id}/export")
def export_video(video_id: str, format: str = Query(default="json", pattern="^(json|jsonl)$"), db: Session = Depends(get_db)) -> Response:
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")
    annotations = db.query(Annotation).filter(Annotation.video_id == video.id).order_by(Annotation.start_time.asc()).all()
    payload = build_export_payload(video, annotations)
    if format == "jsonl":
        lines = [json.dumps(item, sort_keys=True) for item in payload["annotations"]]
        return Response(content="\n".join(lines), media_type="application/jsonl")
    return Response(content=json.dumps(payload, indent=2), media_type="application/json")


@router.delete("/videos/{video_id}")
def delete_video(video_id: str, db: Session = Depends(get_db)) -> dict:
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")
    if video.analysis_status == "processing":
        raise HTTPException(status_code=409, detail="Wait for analysis before deleting this video.")
    db.query(Annotation).filter(Annotation.video_id == video_id).delete()
    try:
        if Path(video.storage_path).exists():
            Path(video.storage_path).unlink(missing_ok=True)
    finally:
        db.delete(video)
        db.commit()
    return {"deleted": True, "id": video_id}
