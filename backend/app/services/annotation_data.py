"""Shared backwards-compatible annotation serialization and review provenance."""
import json
from datetime import datetime, timezone

VERSION = 'local-rules-v1'
AUTOMATIC_SOURCES = {'model', 'local_model', 'local_spatial', 'local_sequence'}


def serialize_annotation(a):
    return dict(id=a.id,start_time=float(a.start_time),end_time=float(a.end_time),behavior=a.behavior,
        participants=json.loads(a.participants or '[]'),objects=json.loads(a.objects or '[]'),
        parent_activity_id=a.parent_activity_id,description=a.description,source=a.source,confidence=a.confidence,
        review_status=a.review_status,annotation_kind=a.annotation_kind or 'behavior',details=json.loads(a.details or '{}'),
        original_suggestion=json.loads(a.original_suggestion or '{}'),review_history=json.loads(a.review_history or '[]'),
        reviewer_feedback=a.reviewer_feedback or '',rule_version=a.rule_version,evaluation_role=a.evaluation_role or 'unspecified')


def snapshot(data):
    return {k:data.get(k) for k in ('id','start_time','end_time','behavior','participants','objects','annotation_kind','details','rule_version','source')}


def review_entry(original, final, prior=None):
    """No submitted provenance is trusted. Capture actual server-side transitions."""
    relevant=('start_time','end_time','behavior','participants','objects','review_status','reviewer_feedback','evaluation_role','details')
    if prior and all(prior.get(k)==final.get(k) for k in relevant):return None
    if final.get('review_status')=='rejected':outcome='rejected'
    elif final.get('review_status')=='reviewed':
        changed=original and any(original.get(k)!=final.get(k) for k in ('behavior','start_time','end_time','participants','objects'))
        outcome='corrected' if changed else 'accepted'
    else:outcome='pending'
    return dict(timestamp=datetime.now(timezone.utc).isoformat(),outcome=outcome,original=original,
        final=snapshot(final),review_status=final.get('review_status'),feedback=final.get('reviewer_feedback',''))
