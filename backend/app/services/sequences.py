"""Small configurable ordered sequence matcher over existing annotation IDs."""
from dataclasses import dataclass
import uuid
from .spatial import region_at, observations

VERSION='sequences-v1'


@dataclass(frozen=True)
class Step:
    labels: tuple[str, ...]
    optional: bool = False


@dataclass(frozen=True)
class ActivityDefinition:
    activity_type: str
    steps: tuple[Step, ...]
    window: float = 20
    max_gap: float = 3


DEFAULT_DEFINITIONS=(
    ActivityDefinition('object_relocation',(
        Step(('picking up / lifting',)), Step(('carrying','walking while carrying')),
        Step(('placing / putting down',)), Step(('releasing',),optional=True))),
    ActivityDefinition('object_transfer',(Step(('handing over / receiving objects',)),)),
)


def continuity(a):
    return (tuple(a.get('participants',[])),tuple(sorted(o['id'] for o in a.get('objects',[]))))


def interrupted(frames, a, b):
    ids=set(a.get('participants',[]))|{o['id'] for o in a.get('objects',[])}
    between=[f for f in frames if a['end_time']<=f['time']<=b['start_time']]
    return any(f.get('camera_state')=='cut' or not ids.issubset(observations(f)) for f in between) or any(y['time']-x['time']>.8 for x,y in zip(between,between[1:]))


def recognize_activities(actions, frames=(), regions=(), definitions=DEFAULT_DEFINITIONS):
    usable=sorted([a for a in actions if a.get('annotation_kind','behavior')=='behavior' and a.get('review_status')!='rejected' and a.get('id') and a.get('objects')],key=lambda a:(a['start_time'],a['end_time'],a['id']))
    output=[]
    for definition in definitions:
        for first in usable:
            if first['behavior'] not in definition.steps[0].labels:continue
            matched=[first]; missing=[]; status='complete'; last=first; blocked=False
            for step in definition.steps[1:]:
                candidates=[a for a in usable if a['id'] not in {m['id'] for m in matched} and
                    a['behavior'] in step.labels and continuity(a)==continuity(first) and
                    a['start_time']>=last['start_time'] and a['end_time']>=last['end_time'] and
                    a['end_time']-first['start_time']<=definition.window]
                nxt=next(iter(candidates),None)
                if nxt and (nxt['start_time']-last['end_time']>definition.max_gap or interrupted(frames,last,nxt)):
                    status='interrupted';blocked=True;nxt=None
                if blocked:nxt=None
                if nxt:matched.append(nxt);last=nxt
                elif not step.optional:
                    missing.append(list(step.labels))
                    if status!='interrupted':status='partial'
            # A transfer is already a complete observable sequence within its action.
            key=definition.activity_type+':'+':'.join(m['id'] for m in matched)
            details=dict(activity_type=definition.activity_type,action_ids=[m['id'] for m in matched],
                         sequence_status=status,missing_steps=missing)
            oid=first['objects'][0]['id']
            source=region_at(frames,regions,oid,first['start_time'])
            dest=region_at(frames,regions,oid,last['end_time'])
            if source:details['source_region']=source
            if dest:details['destination_region']=dest
            output.append(dict(id=uuid.uuid5(uuid.NAMESPACE_URL,VERSION+key).hex,
                start_time=first['start_time'],end_time=max(a['end_time'] for a in matched),
                behavior=definition.activity_type,annotation_kind='activity',details=details,
                participants=first['participants'],objects=first['objects'],source='local_sequence',
                description=f'{status.capitalize()} ordered action evidence. Missing actions are not inferred; task purpose and success remain unconfirmed.',
                confidence=None,review_status='needs_review',rule_version=VERSION))
    # Repetition is an observed pattern, without asserting packing, sorting or loading.
    relocations=[e for e in output if e['behavior']=='object_relocation' and e['details']['sequence_status']=='complete']
    for key in {tuple(e['participants']) for e in relocations}:
        batch=[]
        for e in sorted([e for e in relocations if tuple(e['participants'])==key],key=lambda e:e['start_time']):
            if batch and (e['start_time']<batch[-1]['end_time'] or e['end_time']-batch[0]['start_time']>60):batch=[]
            batch.append(e)
            if len(batch)<2:continue
            ids=list(dict.fromkeys(aid for b in batch for aid in b['details']['action_ids']))
            objects={o['id']:o for b in batch for o in b['objects']}
            output.append(dict(e,id=uuid.uuid5(uuid.NAMESPACE_URL,'repeat:'+':'.join(ids)).hex,
                start_time=batch[0]['start_time'],behavior='repeated_pickup_placement',objects=list(objects.values()),
                details=dict(activity_type='repeated_pickup_placement',action_ids=ids,sequence_status='complete',missing_steps=[]),
                description='At least two complete pickup, carry and placement patterns by the same participants; packing, sorting and loading are not established.'))
    return output
