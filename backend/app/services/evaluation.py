"""Explicit scoped temporal detection evaluation; never uses unreviewed labels as truth."""
from collections import defaultdict


def temporal_iou(a,b):
    intersection=max(0,min(a['end_time'],b['end_time'])-max(a['start_time'],b['start_time']))
    union=max(a['end_time'],b['end_time'])-min(a['start_time'],b['start_time'])
    return intersection/union if union else float(a['start_time']==b['start_time'])


def same_entities(a,b):
    return (set(a.get('participants',[]))==set(b.get('participants',[])) and
            {(o['id'],o['class_name']) for o in a.get('objects',[])}=={(o['id'],o['class_name']) for o in b.get('objects',[])})


def evaluate(annotations, request, review_log=()):
    if not request.complete_ground_truth:
        return dict(available=False,reason='Declare complete independent ground truth for the selected behaviors and interval before computing metrics.')
    labels=set(request.covered_behaviors)
    in_scope=lambda a:a.get('behavior') in labels and request.start_time<=a['start_time'] and a['end_time']<=request.end_time
    truth=[a for a in annotations if a.get('evaluation_role')=='ground_truth' and not a.get('original_suggestion') and a.get('review_status')=='reviewed' and in_scope(a)]
    predictions=[];seen=set()
    for a in annotations:
        original=a.get('original_suggestion')
        if original and a.get('review_status') in {'reviewed','rejected'} and in_scope(original):
            predictions.append(original);seen.add(a['id'])
    for item in review_log:
        original=item.get('original')
        if original and item['annotation_id'] not in seen and item.get('review_status')=='rejected' and in_scope(original):
            predictions.append(original);seen.add(item['annotation_id'])
    if not predictions and not truth:
        return dict(available=False,reason='No reviewed predictions or independent ground-truth annotations in this scope.')
    # Global maximum-cardinality bipartite matching prevents greedy TP undercounts.
    edges={i:sorted([j for j,g in enumerate(truth) if p['behavior']==g['behavior'] and same_entities(p,g) and temporal_iou(p,g)>=request.temporal_iou],key=lambda j:-temporal_iou(p,truth[j])) for i,p in enumerate(predictions)}
    assigned={}
    def augment(i,visited):
        for j in edges[i]:
            if j in visited:continue
            visited.add(j)
            if j not in assigned or augment(assigned[j],visited):assigned[j]=i;return True
        return False
    for i in edges:augment(i,set())
    counts=defaultdict(lambda:dict(true_positives=0,false_positives=0,false_negatives=0))
    matched=set(assigned.values())
    for i,p in enumerate(predictions):counts[p['behavior']]['true_positives' if i in matched else 'false_positives']+=1
    for j,g in enumerate(truth):
        if j not in assigned:counts[g['behavior']]['false_negatives']+=1
    def metrics(c):
        tp,fp,fn=c['true_positives'],c['false_positives'],c['false_negatives']
        return dict(c,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    total={k:sum(c[k] for c in counts.values()) for k in ('true_positives','false_positives','false_negatives')}
    return dict(available=True,scenario=request.scenario,scope=request.model_dump(),matching='Exact behavior, participant IDs and object ID/class sets, with temporal IoU threshold; one-to-one maximum matching.',
                summary=metrics(total),per_behavior={label:metrics(counts[label]) for label in sorted(labels)},
                false_positive_examples=[dict(id=p.get('id'),behavior=p['behavior'],start_time=p['start_time'],end_time=p['end_time'])
                    for i,p in enumerate(predictions) if i not in matched],
                missed_ground_truth_examples=[dict(id=g.get('id'),behavior=g['behavior'],start_time=g['start_time'],end_time=g['end_time'])
                    for j,g in enumerate(truth) if j not in assigned],
                reviewed_predictions=len(predictions),ground_truth_annotations=len(truth),
                unreviewed_predictions_excluded=sum(bool(a.get('original_suggestion')) and a.get('review_status')=='needs_review' and in_scope(a['original_suggestion']) for a in annotations))
