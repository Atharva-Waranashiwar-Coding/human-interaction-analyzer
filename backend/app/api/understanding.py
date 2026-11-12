"""Additive APIs for cached understanding, evaluation, search and optional audio."""
import json
import subprocess
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import Video, Annotation
from ..schemas import RegionUpdate, EvaluationRequest
from ..services.annotation_data import serialize_annotation
from ..services.understanding import refresh_understanding
from ..services.activity_search import search_annotations
from ..services.evaluation import evaluate
from ..services.audio import probe_audio

router=APIRouter(prefix='/api')


def video_or_404(db,video_id, writable=False):
    video=db.get(Video,video_id)
    if not video:raise HTTPException(404,'Video not found.')
    if writable and video.analysis_status=='processing':raise HTTPException(409,'Wait for analysis to finish.')
    return video


@router.put('/videos/{video_id}/regions')
def set_regions(video_id:str,payload:RegionUpdate,db:Session=Depends(get_db)):
    video=video_or_404(db,video_id,True)
    frames=json.loads(video.detections or '[]')
    track_ids={i['id'] for f in frames for i in f.get('items',[])}
    if any(r.id in track_ids for r in payload.regions):raise HTTPException(422,'Region IDs must not collide with tracked entity IDs.')
    video.scene_regions=json.dumps([r.model_dump() for r in payload.regions])
    count=refresh_understanding(db,video);db.commit()
    return dict(regions=json.loads(video.scene_regions),generated=count)


@router.post('/videos/{video_id}/understanding')
def enrich(video_id:str,db:Session=Depends(get_db)):
    video=video_or_404(db,video_id,True)
    count=refresh_understanding(db,video);db.commit()
    return dict(generated=count,cached_detection_frames=len(json.loads(video.detections or '[]')),model_passes=0)


@router.get('/videos/{video_id}/feedback')
def feedback(video_id:str,db:Session=Depends(get_db)):
    video=video_or_404(db,video_id)
    return dict(review_log=json.loads(video.review_log or '[]'))


@router.post('/videos/{video_id}/evaluation')
def evaluation(video_id:str,payload:EvaluationRequest,db:Session=Depends(get_db)):
    video=video_or_404(db,video_id)
    if payload.end_time>video.duration or payload.end_time<payload.start_time:raise HTTPException(422,'Invalid evaluation interval.')
    rows=[serialize_annotation(a) for a in db.query(Annotation).filter(Annotation.video_id==video_id).all()]
    return evaluate(rows,payload,json.loads(video.review_log or '[]'))


@router.post('/evaluation')
def dataset_evaluation(payload:dict[str,EvaluationRequest],db:Session=Depends(get_db)):
    if len(payload)>100:raise HTTPException(422,'Evaluate at most 100 video scopes at once.')
    videos={vid:evaluation(vid,scope,db) for vid,scope in payload.items()}
    scenarios={}
    for result in videos.values():
        if not result['available']:continue
        target=scenarios.setdefault(result['scenario'],dict(true_positives=0,false_positives=0,false_negatives=0))
        for k in target:target[k]+=result['summary'][k]
    for c in scenarios.values():
        tp,fp,fn=c['true_positives'],c['false_positives'],c['false_negatives']
        c.update(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    return dict(videos=videos,scenarios=scenarios)


@router.get('/activities/search')
def search(behavior:str|None=None,annotation_kind:str|None=None,person_count:int|None=Query(None,ge=0),
           object_category:str|None=None,participant_id:str|None=None,object_id:str|None=None,
           review_status:str|None=None,start_time:float|None=Query(None,ge=0),end_time:float|None=Query(None,ge=0),
           video_id:str|None=None,sequence:str|None=None,limit:int=Query(50,ge=1,le=100),offset:int=Query(0,ge=0),db:Session=Depends(get_db)):
    if start_time is not None and end_time is not None and start_time>end_time:raise HTTPException(422,'Invalid time range.')
    steps=[s.strip() for s in (sequence or '').split('>') if s.strip()]
    if len(steps)>8:raise HTTPException(422,'Search sequences support at most 8 steps.')
    return search_annotations(db,behavior=behavior,annotation_kind=annotation_kind,person_count=person_count,
        object_category=object_category,participant_id=participant_id,object_id=object_id,review_status=review_status,
        start_time=start_time,end_time=end_time,video_id=video_id,sequence=steps,limit=limit,offset=offset)


@router.post('/videos/{video_id}/audio')
def audio_metadata(video_id:str,db:Session=Depends(get_db)):
    video=video_or_404(db,video_id,True)
    cached=json.loads(video.audio_metadata or '{}')
    if cached.get('enabled'):return cached
    try:result=probe_audio(Path(video.storage_path),video.duration)
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(422,'Audio metadata could not be read.') from exc
    video.audio_metadata=json.dumps(result);db.commit();return result
