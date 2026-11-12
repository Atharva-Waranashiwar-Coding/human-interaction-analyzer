"""SQLite-backed structured search using existing annotation JSON columns."""
from sqlalchemy import func, text
from ..models import Annotation, Video
from .annotation_data import serialize_annotation


def search_annotations(db, *, behavior=None, annotation_kind=None, person_count=None, object_category=None,
                       participant_id=None, object_id=None, review_status=None, start_time=None, end_time=None,
                       video_id=None, sequence=(), limit=50, offset=0):
    query=db.query(Annotation,Video).join(Video,Annotation.video_id==Video.id)
    if behavior:query=query.filter(Annotation.behavior.ilike('%'+behavior.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%',escape='\\'))
    if annotation_kind:query=query.filter(Annotation.annotation_kind==annotation_kind)
    if person_count is not None:query=query.filter(func.json_array_length(Annotation.participants)==person_count)
    if review_status:query=query.filter(Annotation.review_status==review_status)
    if video_id:query=query.filter(Annotation.video_id==video_id)
    if start_time is not None:query=query.filter(Annotation.end_time>=start_time)
    if end_time is not None:query=query.filter(Annotation.start_time<=end_time)
    if participant_id:
        query=query.filter(text('EXISTS (SELECT 1 FROM json_each(annotations.participants) p WHERE p.value=:participant_id)')).params(participant_id=participant_id)
    if object_category or object_id:
        tests=[];params={}
        if object_category:tests.append("json_extract(o.value,'$.class_name')=:object_category");params['object_category']=object_category
        if object_id:tests.append("json_extract(o.value,'$.id')=:object_id");params['object_id']=object_id
        query=query.filter(text('EXISTS (SELECT 1 FROM json_each(annotations.objects) o WHERE '+' AND '.join(tests)+')')).params(**params)
    if sequence:
        # Only fixed SQL identifiers are generated; all submitted labels are bound.
        joins=[];conditions=[];params={}
        for i,label in enumerate(sequence):
            joins.append(f"json_each(annotations.details, '$.action_ids') j{i} JOIN annotations a{i} ON a{i}.id=j{i}.value AND a{i}.video_id=annotations.video_id")
            conditions.append(f'a{i}.behavior=:step{i}');params[f'step{i}']=label
            if i:conditions.append(f'CAST(j{i-1}.key AS INTEGER)<CAST(j{i}.key AS INTEGER)')
        query=query.filter(text('EXISTS (SELECT 1 FROM '+ ' CROSS JOIN '.join(joins)+' WHERE '+' AND '.join(conditions)+')')).params(**params)
    total=query.count()
    rows=query.order_by(Video.created_at.desc(),Annotation.start_time,Annotation.id).offset(offset).limit(limit).all()
    return dict(total=total,limit=limit,offset=offset,results=[dict(video_id=v.id,video_name=v.original_name,
        annotation=serialize_annotation(a),link=f'/?video={v.id}&time={a.start_time}') for a,v in rows])
