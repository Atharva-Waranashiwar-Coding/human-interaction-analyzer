"""Temporal image-space relationships from cached boxes, without inference."""
import itertools
import math
import uuid

VERSION = 'spatial-v1'
CAUTION = ' Image-space evidence only; contact, containment and physical depth are unconfirmed.'


def center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4 and
            all(isinstance(x, (int, float)) and math.isfinite(x) for x in box) and
            box[0] < box[2] and box[1] < box[3])


def observations(frame):
    return {i['id']: i for i in frame.get('items', []) if i.get('id') and valid_box(i.get('box'))}


def generate_spatial(frames, regions=(), min_duration=.8, max_gap=.8):
    """Require consecutive evidence; never bridge gaps, cuts or unstable camera intervals.

    Configured regions are fixed image areas. Region motion is used only for an
    explicitly static camera. Relative entity distances require comparable sizes.
    Legacy caches without camera state get static relationships only.
    """
    active = {}; events = []; previous = None; epoch = 0
    def emit(label, subject, target, start, end, description, **details):
        if end-start+1e-6 < min_duration: return
        entities = [x for x in (subject, target) if not x.get('region')]
        pids = [x['id'] for x in entities if x['class_name']=='person']
        objects = [dict(id=x['id'], class_name=x['class_name']) for x in entities if x['class_name']!='person']
        key = f'{VERSION}:{epoch}:{label}:{subject["id"]}:{target["id"]}:{start:.4f}:{end:.4f}'
        events.append(dict(id=uuid.uuid5(uuid.NAMESPACE_URL,key).hex, start_time=start,end_time=end,
            behavior=label, participants=pids, objects=objects, annotation_kind='spatial',
            details=dict(subject_id=subject['id'],target_id=target['id'],
                         region_id=target['id'] if target.get('region') else None, epoch=epoch, **details),
            description=description+CAUTION,source='local_spatial',review_status='needs_review',confidence=None,rule_version=VERSION))
    def finish(key):
        start,end,count,subject,target,desc,details = active.pop(key)
        if count >= 3: emit(key[0],subject,target,start,end,desc,**details)
    ordered = sorted(frames,key=lambda f:f['time'])
    for f in ordered:
        t=f['time']; state=f.get('camera_state','unknown'); entities=observations(f)
        # Region geometry is image-anchored; it cannot survive camera motion.
        region_entities = {r['id']:dict(r,class_name=r.get('kind','region'),region=True) for r in regions}
        if previous and (t-previous['time']>max_gap or state=='cut' or state!=previous.get('camera_state','unknown')):
            for key in list(active):finish(key)
            previous=None;epoch+=1
        candidates={}
        pairs = list(itertools.permutations(sorted(entities),2))
        pairs += [(eid,rid) for eid in sorted(entities) for rid in region_entities]
        all_entities = dict(region_entities,**entities)
        old=observations(previous) if previous else {}
        for aid,bid in pairs:
            a,b=all_entities[aid],all_entities[bid]; ab,bb=a['box'],b['box']
            ca,cb=center(ab),center(bb)
            person_heights=[e['box'][3]-e['box'][1] for e in (a,b) if e['class_name']=='person']
            scale=sum(person_heights)/len(person_heights) if person_heights else max(ab[3]-ab[1],bb[3]-bb[1])
            if scale<=0:continue
            gap=math.hypot(max(ab[0]-bb[2],bb[0]-ab[2],0),max(ab[1]-bb[3],bb[1]-ab[3],0))/scale
            if gap>3:continue
            overlap=min(ab[2],bb[2])>max(ab[0],bb[0]) and min(ab[3],bb[3])>max(ab[1],bb[1])
            labels=[]
            if gap<.25:labels.append(('near','Sustained bounding-box proximity.'))
            if overlap:labels.append(('overlapping','Sustained overlap of image bounding boxes.'))
            if ab[2]<bb[0]:labels.append(('left','Subject bounding box stays left of target.'))
            if ab[0]>bb[2]:labels.append(('right','Subject bounding box stays right of target.'))
            if ab[3]<bb[1]:labels.append(('above','Subject bounding box stays above target.'))
            if ab[1]>bb[3]:labels.append(('below','Subject bounding box stays below target.'))
            if b.get('region') and state!='static':labels=[]
            for label,desc in labels:candidates[(label,aid,bid)]=(a,b,desc,{})
            # Directionality is evaluated both ways for entity pairs.
            if previous and state=='static' and previous.get('camera_state')=='static' and aid in old:
                oldb=old.get(bid,b if b.get('region') else None)
                if oldb:
                    dt=t-previous['time']
                    delta=(math.dist(ca,cb)-math.dist(center(old[aid]['box']),center(oldb['box'])))/scale
                    if 0<dt<=max_gap and abs(delta)/dt>.12:
                        label='moves toward' if delta<0 else 'moves away'
                        candidates[(label,aid,bid)]=(a,b,'Sustained relative distance change with a static-camera estimate.',{})
                    if b.get('region'):
                        inside=lambda c:bb[0]<=c[0]<=bb[2] and bb[1]<=c[1]<=bb[3]
                        was=inside(center(old[aid]['box']));now=inside(ca)
                        if was!=now:
                            # Boundary transition requires at least .8 seconds of stable membership on both sides.
                            candidates[('membership',aid,bid)]=(a,b,'',{'inside':now})
        # Stable region membership is tracked separately to debounce entries/exits.
        for aid,a in entities.items():
            for rid,b in region_entities.items():
                if state!='static':continue
                c=center(a['box']);r=b['box'];inside=r[0]<=c[0]<=r[2] and r[1]<=c[1]<=r[3]
                candidates[('inside region' if inside else 'outside region',aid,rid)]=(a,b,'Center membership in a configured image region.',{'inside':inside})
        for key in list(active):
            if key not in candidates:finish(key)
        for key,(a,b,desc,details) in candidates.items():
            if key[0]=='membership':continue
            if key not in active:active[key]=[t,t,1,a,b,desc,details]
            else:active[key][1]=t;active[key][2]+=1
        previous=f
    for key in list(active):finish(key)
    # Membership is supporting evidence, not a containment assertion.
    memberships=[e for e in events if e['behavior'] in {'inside region','outside region'}]
    for a,b in itertools.pairwise(sorted(memberships,key=lambda x:(x['details']['subject_id'],x['details']['target_id'],x['start_time']))):
        if (a['details']['subject_id'],a['details']['target_id'])!=(b['details']['subject_id'],b['details']['target_id']):continue
        if a['details']['epoch']!=b['details']['epoch'] or a['behavior']==b['behavior'] or b['start_time']-a['end_time']>max_gap:continue
        interval=[f for f in ordered if a['end_time']<=f['time']<=b['start_time']]
        if any(f.get('camera_state')!='static' or a['details']['subject_id'] not in observations(f) for f in interval):continue
        label='enters region' if b['behavior']=='inside region' else 'exits region'
        event=dict(b,id=uuid.uuid5(uuid.NAMESPACE_URL,a['id']+b['id']).hex,behavior=label,start_time=a['end_time'],end_time=b['start_time'],description='Stable center membership changed across a configured image-region boundary.'+CAUTION)
        events.append(event)
    return [e for e in events if e['behavior'] not in {'inside region','outside region'}]


def region_at(frames, regions, object_id, timestamp, tolerance=.6):
    """Return one supported region, rejecting overlaps, missing tracks and moving views."""
    nearby=[f for f in frames if abs(f['time']-timestamp)<=tolerance]
    if len(nearby)<2 or any(f.get('camera_state')!='static' for f in nearby):return None
    points=[]
    for f in nearby:
        entity=observations(f).get(object_id)
        if entity is None:return None
        points.append(center(entity['box']))
    matches=[r['id'] for r in regions if all(r['box'][0]<=x<=r['box'][2] and r['box'][1]<=y<=r['box'][3] for x,y in points)]
    return matches[0] if len(matches)==1 else None


def surface_suggestions(actions, frames, regions):
    rows=[]
    for a in actions:
        if a.get('behavior')!='placing / putting down' or a.get('review_status')=='rejected':continue
        for obj in a.get('objects',[]):
            oid=obj['id'];rid=region_at(frames,regions,oid,a['end_time'])
            region=next((r for r in regions if r['id']==rid and r.get('kind') in {'table','shelf'}),None)
            if region is None:continue
            tail=[f for f in frames if a['end_time']-.8<=f['time']<=a['end_time']]
            boxes=[observations(f).get(oid) for f in tail]
            if len(boxes)<3 or any(b is None for b in boxes) or tail[-1]['time']-tail[0]['time']<.6:continue
            height=max(boxes[0]['box'][3]-boxes[0]['box'][1],.01)
            if max(math.dist(center(b['box']),center(boxes[0]['box'])) for b in boxes)>.08*height:continue
            rows.append(dict(a,id=uuid.uuid5(uuid.NAMESPACE_URL,'surface:'+a['id']+rid).hex,
                behavior='possible on-surface placement',annotation_kind='spatial',source='local_spatial',
                details=dict(subject_id=oid,target_id=rid,region_id=rid,action_ids=[a['id']]),
                description='A placement action coincides with a stable object center in a manually configured surface region.'+CAUTION,
                review_status='needs_review',confidence=None,rule_version=VERSION))
    return rows
