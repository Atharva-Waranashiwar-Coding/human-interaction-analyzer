"""Enrichment of persisted results; never decodes video or invokes models."""
import json
import uuid
from ..models import Annotation
from .annotation_data import serialize_annotation
from .spatial import generate_spatial, surface_suggestions
from .sequences import recognize_activities
from .complex_activities import complex_candidates
from .video_service import create_annotation_records


def refresh_understanding(db, video):
    db.flush()
    rows=db.query(Annotation).filter(Annotation.video_id==video.id).all()
    protected=[serialize_annotation(a) for a in rows if a.source not in {'local_spatial','local_sequence'} or a.review_status!='needs_review']
    actions=[a for a in protected if a['annotation_kind']=='behavior']
    frames=json.loads(video.detections or '[]');regions=json.loads(video.scene_regions or '[]')
    grouped=recognize_activities(actions,frames,regions)
    generated=(generate_spatial(frames,regions)+surface_suggestions(actions,frames,regions)+
               grouped+complex_candidates(actions,frames,regions,grouped))
    db.query(Annotation).filter(Annotation.video_id==video.id,Annotation.source.in_(['local_spatial','local_sequence']),Annotation.review_status=='needs_review').delete(synchronize_session='fetch')
    pending=[]
    for e in generated:
        if e['annotation_kind']=='spatial':e['id']=uuid.uuid5(uuid.NAMESPACE_URL,video.id+e['id']).hex
        if any(a['id']==e['id'] for a in protected):continue
        if not 0<=e['start_time']<=e['end_time']<=video.duration:continue
        # Saved/corrected/rejected suggestions remain protected across enrichment.
        duplicate=False
        for saved in protected:
            original=saved.get('original_suggestion') or saved
            if (original.get('behavior')==e['behavior'] and original.get('participants')==e['participants'] and
                original.get('objects')==e['objects'] and original.get('annotation_kind','behavior')==e['annotation_kind'] and
                original['start_time']<=e['end_time'] and original['end_time']>=e['start_time']):
                duplicate=True;break
        if not duplicate:pending.append(e)
    db.add_all(create_annotation_records(video.id,pending));db.flush()
    return len(pending)
