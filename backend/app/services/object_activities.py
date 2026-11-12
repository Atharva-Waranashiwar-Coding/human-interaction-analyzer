"""Conservative person/object evidence on the existing sampled-frame stream."""
from collections import deque
import math
import uuid

from .temporal_behaviors import valid, arm, ARMS, cosine, sub, norm, torso

# General COCO classes are observable; abstract containers/tasks are not.
MANIPULABLE = {'cardboard box','package','plastic package','packet','plastic bag','paper bag','backpack','handbag','suitcase','umbrella','sports ball','frisbee','bottle','cup','bowl',
               'fork','knife','spoon','banana','apple','sandwich','orange','broccoli','carrot','hot dog',
               'pizza','donut','cake','book','cell phone','remote','laptop','scissors','teddy bear','toothbrush','hair drier'}
CAUTION = ' Geometry-based review suggestion; proximity does not establish contact, ownership or task success.'


def distance_to_box(point, box):
    x,y=point[:2]; x1,y1,x2,y2=box
    return math.hypot(max(x1-x,0,x-x2),max(y1-y,0,y-y2))


def center(box):
    return ((box[0]+box[2])/2,(box[1]+box[3])/2)


def hand_gap(p, obj):
    wrists=[p['pose'][i][:2] for i in (9,10) if valid(p,i)]
    return min((distance_to_box(w,obj['box'])/p['height'] for w in wrists),default=math.inf)


def object_arm_reach(p, obj):
    arms=[a for indices in ARMS if (a:=arm(p,indices))]
    nearest=min(arms,key=lambda a:distance_to_box(a[2],obj['box']),default=None)
    return norm(sub(nearest[2],nearest[0]))/p['height'] if nearest else 0


def supported_box_pose(p,obj):
    """Two visible supporting arms around a torso-level object, not one near wrist."""
    if not valid(p,5,6,7,8,9,10,11,12):return False
    box=obj.get('box',())
    if len(box)!=4 or not all(math.isfinite(v) for v in box):return False
    h=p['height'];x1,y1,x2,y2=box;k=p['pose']
    if h<=0 or (x2-x1)/h<.15 or (y2-y1)/h<.12:return False
    if y2>(k[11][1]+k[12][1])/2+.12*h:return False
    arms=[arm(p,ids) for ids in ARMS]
    if not all(arms) or min(a[3] for a in arms)>.85:return False
    if math.dist(k[9][:2],k[10][:2])/h<.08:return False
    return all(distance_to_box(k[i],obj['box'])/h<.04 and
               y1+.45*(y2-y1)<k[i][1]<y2+.04*h for i in (9,10))


class SupportedHolding:
    """Relative pose evidence survives camera translation; never claims locomotion."""
    def __init__(self):
        self.histories={};self.active={};self.events=[]

    def reset(self):
        self.histories.clear();self.active.clear()

    def step(self,t,people,objects):
        seen=set()
        for oid,obj in objects.items():
            if obj.get('class_name') not in MANIPULABLE:continue
            holders=[pid for pid,p in people.items() if supported_box_pose(p,obj)]
            if len(holders)!=1:continue
            pid=holders[0];p=people[pid];key=(oid,pid,obj['class_name']);seen.add(key)
            relative=tuple(v/p['height'] for v in sub(center(obj['box']),torso(p)))
            hist=self.histories.setdefault(key,deque())
            if hist and (t-hist[-1][0]>.8 or norm(sub(relative,hist[-1][1]))>.2 or
                         math.dist(torso(p),hist[-1][2])/p['height']>.65 or
                         max(p['height'],hist[-1][3])/min(p['height'],hist[-1][3])>1.6):
                hist.clear();self.active.pop(key,None)
            hist.append((t,relative,torso(p),p['height']))
            while hist and t-hist[0][0]>1.2+1e-6:hist.popleft()
            if len(hist)<3 or t-hist[0][0]<.8:continue
            if max(math.dist(hist[0][1],x[1]) for x in hist)>.08:continue
            event=self.active.get(key)
            if event and hist[0][0]<=event['end_time']+.1:
                event['end_time']=round(t,4)
            else:
                event=dict(id=uuid.uuid4().hex,start_time=round(hist[0][0],4),end_time=round(t,4),behavior='holding',
                    participants=[pid],objects=[{'id':oid,'class_name':obj['class_name']}],parent_activity_id=None,
                    description='Two confident wrists and supporting arms remained around a torso-level object with stable object-to-torso position. This relative-pose evidence does not establish carrying, walking, pickup or physical support.'+CAUTION,
                    confidence=None,review_status='needs_review',source='local_model')
                self.events.append(event);self.active[key]=event
        for key in list(self.histories):
            if key not in seen:self.histories.pop(key);self.active.pop(key,None)

    def results(self):return [dict(e) for e in self.events]


class ObjectActivities:
    def __init__(self, fps=2):
        self.fps=fps
        self.history={}
        self.events=[]
        self.event_epochs={}
        self.event_positions={}
        self.epochs={}
        self.recent={}
        self.last_t=None
        self.supported_holding=SupportedHolding()

    def reset(self,relative=True):
        if relative:self.supported_holding.reset()
        for oid in list(self.history): self._drop(oid)

    def _drop(self, oid):
        self.history.pop(oid,None)
        self.epochs[oid]=self.epochs.get(oid,0)+1

    def emit(self,label,pids,oid,cls,start,end,evidence):
        if end-start < .35: return
        key=(label,tuple(pids),oid,self.epochs.get(oid,0))
        event=self.recent.get(key)
        history=self.history.get(oid,[])
        origin=min(history,key=lambda r:abs(r[0]-start))[1] if history else None
        current=history[-1][1] if history else None
        if event is not None and start <= event['end_time']+.35:
            event['end_time']=max(event['end_time'],round(end,4))
            if current:
                initial,_,h=self.event_positions[event['id']]
                self.event_positions[event['id']]=(initial,center(current['obj']['box']),h)
            return
        event=dict(id=uuid.uuid4().hex,start_time=round(start,4),end_time=round(end,4),behavior=label,
                   participants=list(pids),objects=[{'id':oid,'class_name':cls}],parent_activity_id=None,
                   description=evidence+CAUTION,confidence=None,review_status='needs_review',source='local_model')
        self.events.append(event);self.recent[key]=event
        self.event_epochs[event['id']]=self.epochs.get(oid,0)
        if origin and current:
            h=max((p['height'] for p in current['people'].values()),default=1)
            self.event_positions[event['id']]=(center(origin['obj']['box']),center(current['obj']['box']),h)

    def step(self,t,people,objects,camera_motion=False,relative_people=None,relative_objects=None):
        if not math.isfinite(t) or t < 0 or (self.last_t is not None and t <= self.last_t):
            self.reset();return
        self.last_t=t
        self.supported_holding.step(t,relative_people if relative_people is not None else people,relative_objects if relative_objects is not None else objects)
        if camera_motion:
            self.reset(relative=False);return
        for oid in list(self.history):
            if oid not in objects: self._drop(oid)
        for oid,obj in objects.items():
            box=obj.get('box',())
            if obj.get('class_name') not in MANIPULABLE or len(box)!=4 or not all(math.isfinite(v) for v in box) or box[2]<=box[0] or box[3]<=box[1]:
                self._drop(oid);continue
            associated={pid:p for pid,p in people.items() if p['height']>0 and hand_gap(p,obj)<.12}
            supported={pid:p for pid,p in associated.items() if supported_box_pose(p,obj)}
            if supported:associated=supported
            # All associations are evidence, not ownership; ambiguous multiple hands stay ambiguous.
            snapshot=dict(obj=dict(box=tuple(box),class_name=obj['class_name']),people={pid:dict(
                center=tuple(torso(p) or p['center']),height=float(p['height']),gap=hand_gap(p,obj),
                reach=object_arm_reach(p,obj),
                wrists=tuple(tuple(p['pose'][i][:2]) for i in (9,10) if valid(p,i)))
                for pid,p in people.items() if p['height']>0},
                near=tuple(sorted(associated)))
            hist=self.history.get(oid)
            if hist:
                pt,previous=hist[-1]
                scale=max((p['height'] for p in people.values()),default=1)
                if t-pt>1.6/self.fps or previous['obj']['class_name']!=obj['class_name'] or math.dist(center(box),center(previous['obj']['box']))/scale>.65:
                    self._drop(oid);hist=None
                elif (any(pid not in snapshot['people'] for pid in previous['near']) or any(
                    math.dist(previous['people'][pid]['center'],snapshot['people'][pid]['center'])/snapshot['people'][pid]['height']>.65 or
                    max(previous['people'][pid]['height'],snapshot['people'][pid]['height'])/min(previous['people'][pid]['height'],snapshot['people'][pid]['height'])>1.5
                    for pid in set(previous['near'])|set(snapshot['near']) if pid in previous['people'] and pid in snapshot['people'])):
                    self._drop(oid);hist=None
            hist=self.history.setdefault(oid,deque());hist.append((t,snapshot))
            while hist and t-hist[0][0]>6: hist.popleft()
            self._evaluate(oid,list(hist))

    def _evaluate(self,oid,history):
        if len(history)<3:return
        t,current=history[-1];cls=current['obj']['class_name']
        recent=[r for r in history if t-r[0]<=1.2+1e-6]
        if len(recent)<3 or recent[-1][0]-recent[0][0]<.8:return
        for pid in current['people']:
            person_history=[]
            for sample in reversed(history):
                person=sample[1]['people'].get(pid)
                if person is None or not math.isfinite(person['gap']):break
                person_history.append(sample)
            person_history.reverse()
            if len(person_history)>=3:self._evaluate_person(oid,pid,person_history)
        self._transfer(oid,history)
        if len(current['near'])==2 and all(r['near']==current['near'] for _,r in recent):
            a,b=current['near'];h=(current['people'][a]['height']+current['people'][b]['height'])/2
            oc=[center(r['obj']['box']) for _,r in recent]
            if math.dist(oc[0],oc[-1])>.2*h and all(
                max(math.dist(sub(center(r['obj']['box']),r['people'][pid]['center']),
                              sub(oc[0],recent[0][1]['people'][pid]['center'])) for _,r in recent)<.1*h for pid in (a,b)):
                self.emit('joint carrying / lifting',[a,b],oid,cls,recent[0][0],t,'Two visible people maintained wrist proximity to the same object while object and both people moved together; shared support is unconfirmed.')

    def _evaluate_person(self,oid,pid,history):
        t,current=history[-1];p=current['people'][pid];cls=current['obj']['class_name']
        recent=[r for r in history if t-r[0]<=1.2+1e-6]
        if len(recent)<3 or recent[-1][0]-recent[0][0]<.8:return
        h=p['height'];now=center(current['obj']['box'])
        if not math.isfinite(p['gap']) or any(not math.isfinite(r['people'][pid]['gap']) for _,r in history):return
        if math.dist(current['obj']['box'][:2],current['obj']['box'][2:])/h>1.2:return
        # Approaching the box with a wrist, while the object stays stationary.
        first=history[0][1]; old=first['people'][pid];before=center(first['obj']['box'])
        if (old['gap']>.25 and p['gap']<.12 and old['gap']-p['gap']>.2 and p['reach']-old['reach']>.1 and
            math.dist(before,now)/h<.08 and len(current['near'])==1):
            self.emit('reaching toward an object',[pid],oid,cls,history[0][0],t,
                      'Visible wrist-to-object distance decreased while the object stayed approximately stationary.')
        near=all(r['near']==(pid,) for _,r in recent)
        if near:
            positions=[center(r['obj']['box']) for _,r in recent]
            relative=[sub(pos,r['people'][pid]['center']) for pos,(_,r) in zip(positions,recent)]
            stable=max(math.dist(relative[0],x) for x in relative)/h<.08
            displacement=math.dist(positions[0],positions[-1])/h
            body_displacement=math.dist(recent[0][1]['people'][pid]['center'],p['center'])/h
            acquired=any(e['behavior'] in {'picking up / lifting','carrying'} and e['participants']==[pid] and e['objects'][0]['id']==oid and self.event_epochs[e['id']]==self.epochs.get(oid,0) for e in self.events)
            if acquired and stable and displacement<.08 and body_displacement<.08:
                self.emit('holding',[pid],oid,cls,recent[0][0],t,'One visible person maintained wrist proximity and stable object/body position; static proximity alone cannot confirm holding.')
            elif stable and displacement>.2 and body_displacement>.2:
                self.emit('carrying',[pid],oid,cls,recent[0][0],t,'A nearby object moved with one person while its position relative to the body stayed stable.')
                delta=sub(p['center'],recent[0][1]['people'][pid]['center'])
                if abs(delta[0])>.18*h and abs(delta[1])<.15*h:
                    self.emit('walking while carrying',[pid],oid,cls,recent[0][0],t,'Sustained lateral person/object co-motion with stable hand association; gait and physical support are unconfirmed.')
        # Conservative planar manipulation: observed wrist and object co-motion
        # with a relatively stationary torso. Does not assert contact or force.
        if near:
            self._planar_manipulation(oid,pid,cls,recent)

        # Before/after transitions require observed states, never missing-object inference.
        for j in range(2,len(history)-2):
            ct,contact=history[j]
            if t-ct>2.5:continue
            pre=history[max(0,j-2):j+1];post=history[j:]
            pre_boxes=[center(r['obj']['box']) for _,r in pre]
            post_boxes=[center(r['obj']['box']) for _,r in post]
            if post[-1][0]-post[0][0]<.8:continue
            if (all(r['people'][pid]['gap']>.2 for _,r in pre[:-1]) and contact['near']==(pid,)
                and all(r['near']==(pid,) for _,r in post) and
                max(math.dist(pre_boxes[0],v) for v in pre_boxes)/h<.08 and
                post_boxes[0][1]-post_boxes[-1][1]>.18*h):
                self.emit('picking up / lifting',[pid],oid,cls,pre[0][0],t,'A previously stationary visible object gained a single wrist association, then moved upward while that association persisted.')
            prior_near=all(r['near']==(pid,) for _,r in pre)
            stationary=max(math.dist(post_boxes[-1],v) for v in post_boxes[-3:])/h<.06
            handled=any(e['behavior'] in {'picking up / lifting','carrying','holding'} and e['participants']==[pid] and e['objects'][0]['id']==oid
                        and e['start_time']<=ct and e['end_time']>=pre[0][0]-.6
                        for e in self.events+self.supported_holding.events)
            if prior_near and handled and current['near']==() and p['gap']>.25 and stationary:
                if pre_boxes[-1][1]-pre_boxes[0][1]>.12*h and post[-3][1]['near']==(pid,):
                    self.emit('placing / putting down',[pid],oid,cls,pre[0][0],t,'A hand-associated object moved downward, then remained visibly stationary as the wrist separated; support surface is unconfirmed.')
                self.emit('releasing',[pid],oid,cls,ct,t,'A previously sustained wrist association ended while the object remained visible and approximately stationary.')

    def _planar_manipulation(self,oid,pid,cls,recent):
        """Suggest planar object motion only with sustained single-hand evidence.

        Restrict to horizontal image motion and a stationary torso to avoid
        mistaking lifting or co-moving carrying for a push/pull. Track epochs,
        camera reliability, and visibility are enforced by the caller.
        """
        if len(recent)<3 or recent[-1][0]-recent[0][0]<.8:
            return
        if any(row['near']!=(pid,) or pid not in row['people'] for _,row in recent):
            return
        first=recent[0][1];last=recent[-1][1]
        h=last['people'][pid]['height']
        if h<=0:
            return
        obj_start=center(first['obj']['box']);obj_end=center(last['obj']['box'])
        dx=(obj_end[0]-obj_start[0])/h
        dy=(obj_end[1]-obj_start[1])/h
        body_start=first['people'][pid]['center'];body_end=last['people'][pid]['center']
        if abs(dx)<.20 or abs(dy)>.10 or math.dist(body_start,body_end)/h>.12:
            return
        # Find one continuously visible wrist following the observed object.
        # Switching between hands can falsely simulate interaction.
        wrist_tracks=[]
        for wrist_idx in (0,1):
            path=[]
            for _,row in recent:
                wrists=row['people'][pid]['wrists']
                if len(wrists)!=2:
                    path=[];break
                wrist=wrists[wrist_idx]
                if distance_to_box(wrist,row['obj']['box'])/row['people'][pid]['height']>=.12:
                    path=[];break
                path.append(wrist)
            if path:
                wrist_tracks.append(path)
        if not wrist_tracks:
            return
        if not any(
            (track[-1][0]-track[0][0])*dx>0 and
            abs((track[-1][0]-track[0][0])/h-dx)<.12 and
            abs((track[-1][1]-track[0][1])/h-dy)<.12
            for track in wrist_tracks
        ):
            return
        # Reject abrupt reversals and intermittent displacement.
        xs=[center(row['obj']['box'])[0] for _,row in recent]
        increments=[b-a for a,b in zip(xs,xs[1:])]
        if sum(step*dx>0 for step in increments)<len(increments)-1:
            return
        # Only a change in body-relative distance distinguishes these labels.
        # Otherwise report the observable sliding motion without intent.
        d0=math.dist(obj_start,body_start)/h
        d1=math.dist(obj_end,body_end)/h
        difference=d1-d0
        if difference>.15:
            label='pushing an object'
            evidence='The visible object and a continuously nearby wrist moved laterally away from a mostly stationary torso. Direction is consistent with pushing but force and contact are unconfirmed.'
        elif difference<-.15:
            label='pulling an object'
            evidence='The visible object and a continuously nearby wrist moved laterally toward a mostly stationary torso. Direction is consistent with pulling but force and contact are unconfirmed.'
        else:
            label='dragging / sliding an object'
            evidence='The visible object and a continuously nearby wrist moved laterally while the torso stayed approximately stationary. Surface contact and sliding are unconfirmed.'
        self.emit(label,[pid],oid,cls,recent[0][0],recent[-1][0],evidence)

    def _transfer(self,oid,hist):
        # Require stable donor-only and recipient-only states, with an observed shared phase.
        if len(hist)<7:return
        t,last=hist[-1];recipient=last['near']
        if len(recipient)!=1:return
        tail=[r for r in hist if t-r[0]<=1.2+1e-6]
        if len(tail)<3 or tail[-1][0]-tail[0][0]<.8 or not all(r['near']==recipient for _,r in tail):return
        for j in range(2,len(hist)-3):
            donor=hist[j][1]['near']
            if len(donor)!=1 or donor==recipient:continue
            if donor[0] not in last['people'] or recipient[0] not in hist[j][1]['people']:continue
            if not math.isfinite(last['people'][donor[0]]['gap']) or last['people'][donor[0]]['gap']<.25:continue
            if not math.isfinite(hist[j][1]['people'][recipient[0]]['gap']) or hist[j][1]['people'][recipient[0]]['gap']<.25:continue
            pre=hist[j-2:j+1]
            if pre[-1][0]-pre[0][0]<.8 or not all(r['near']==donor for _,r in pre):continue
            between=[r for _,r in hist[j+1:] if len(r['near'])==2 and set(r['near'])==set(donor+recipient)]
            if not between:continue
            allowed=(donor,tuple(sorted(donor+recipient)),recipient)
            phases=[allowed.index(r['near']) if r['near'] in allowed else -1 for _,r in hist[j+1:]]
            if -1 in phases or phases!=sorted(phases):continue
            h=last['people'][recipient[0]]['height']
            if math.dist(center(pre[0][1]['obj']['box']),center(last['obj']['box']))/h<.15:continue
            self.emit('handing over / receiving objects',[donor[0],recipient[0]],oid,last['obj']['class_name'],pre[0][0],t,
                      'The same continuously visible object changed from sustained donor-only to shared to recipient-only wrist proximity, with object displacement; transfer/contact is unconfirmed.')
            return

    def results(self):
        events=[dict(e) for e in self.events]
        # Only physically observed pickup -> movement -> placement becomes a parent.
        for oid in {o['id'] for e in events for o in e['objects']}:
            children=[e for e in events if e['objects'][0]['id']==oid]
            for pickup in [e for e in children if e['behavior']=='picking up / lifting']:
                carry=next((e for e in children if e['behavior']=='carrying' and self.event_epochs[e['id']]==self.event_epochs[pickup['id']] and e['participants']==pickup['participants'] and pickup['start_time']<=e['start_time']<=pickup['end_time']+2),None)
                place=next((e for e in children if carry and e['behavior']=='placing / putting down' and self.event_epochs[e['id']]==self.event_epochs[pickup['id']] and e['participants']==pickup['participants'] and carry['start_time']<=e['start_time']<=carry['end_time']+2),None)
                if not carry or not place or any(e.get('parent_activity_id') for e in (pickup,carry,place)):continue
                start_position,_,h=self.event_positions[pickup['id']]
                _,end_position,_=self.event_positions[place['id']]
                if math.dist(start_position,end_position)/h<.25:continue
                if place['end_time']-pickup['start_time']>20:continue
                parent=dict(pickup,id=uuid.uuid5(uuid.NAMESPACE_URL,pickup['id']+carry['id']+place['id']).hex,behavior='moving objects between locations',end_time=place['end_time'],
                            description='Observed pickup, carrying and placement candidates for the same continuous object/person track; task success is unconfirmed.'+CAUTION)
                # Distinct locations must be established; use carrying displacement evidence.
                for e in (pickup,carry,place):e['parent_activity_id']=parent['id']
                events.append(parent)
        for support in self.supported_holding.results():
            duplicate=next((e for e in events if e['behavior']=='holding' and e['participants']==support['participants'] and e['objects']==support['objects'] and e['start_time']<=support['end_time'] and e['end_time']>=support['start_time']),None)
            if duplicate:
                duplicate['start_time']=min(duplicate['start_time'],support['start_time'])
                duplicate['end_time']=max(duplicate['end_time'],support['end_time'])
            else:events.append(support)
        return sorted(events,key=lambda e:(e['start_time'],e['behavior']))


class MovementSuggestions:
    """A small set of supported individual movement/posture candidates."""
    def __init__(self,fps,mode="all"):
        if mode not in {"all","relative","trajectory"}:raise ValueError("Invalid movement mode")
        self.mode=mode
        self.fps=fps;self.history={};self.events=[];self.active={}

    def step(self,t,people,camera_motion=False):
        if camera_motion:self.history.clear();self.active.clear();return
        for pid in list(self.history):
            if pid not in people:self.history.pop(pid);self.active={k:v for k,v in self.active.items() if k[1]!=pid}
        for pid,p in people.items():
            anchor=torso(p)
            if anchor is not None:p=dict(p,center=anchor)
            hist=self.history.setdefault(pid,deque())
            if hist and (t-hist[-1][0]>1.5/self.fps or math.dist(p['center'],hist[-1][1]['center'])/p['height']>.65):
                hist.clear();self.active={k:v for k,v in self.active.items() if k[1]!=pid}
            hist.append((t,p))
            while hist and t-hist[0][0]>2:hist.popleft()
            tail=[r for r in hist if t-r[0]<=1.2+1e-6]
            if len(tail)<3 or t-tail[0][0]<1:continue
            h=p['height'];dx=(p['center'][0]-tail[0][1]['center'][0])/h;dy=(p['center'][1]-tail[0][1]['center'][1])/h
            candidates={}
            if all(valid(q,5,6,11,12,13,14,15,16) for _,q in tail):
                # Standing requires extended legs and a roughly upright torso throughout.
                upright=all(all(arm(q,ids) is not None and arm(q,ids)[3]>.9 for ids in ((11,13,15),(12,14,16))) and
                            abs((q['pose'][5][0]+q['pose'][6][0]-q['pose'][11][0]-q['pose'][12][0])/2)<.15*q['height'] and
                            all(q['pose'][s][1]<q['pose'][hip][1]<q['pose'][knee][1]<q['pose'][ankle][1] for s,hip,knee,ankle in ((5,11,13,15),(6,12,14,16))) for _,q in tail)
                if self.mode!='relative' and upright and math.hypot(dx,dy)<.06 and max(q['height'] for _,q in tail)/min(q['height'] for _,q in tail)<1.12:candidates['standing']='Sustained upright torso, extended legs and little screen-space movement.'
                if self.mode!='relative' and abs(dx)>.25 and abs(dy)<.15:
                    knee_diff=[(q['pose'][13][0]-q['pose'][14][0])/q['height'] for _,q in tail]
                    if min(knee_diff)<-.04 and max(knee_diff)>.04:
                        candidates['walking']='Lateral body displacement with alternating knee positions; direction of attention is unknown.'
                bent=all(abs((q['pose'][5][0]+q['pose'][6][0]-q['pose'][11][0]-q['pose'][12][0])/2)>.2*q['height'] and
                         abs((q['pose'][5][1]+q['pose'][6][1]-q['pose'][11][1]-q['pose'][12][1])/2)<.25*q['height'] for _,q in tail)
                if self.mode!='trajectory' and bent:candidates['bending']='Sustained torso inclination relative to visible hips; task purpose is unknown.'
            for indices in (ARMS if self.mode!='trajectory' else ()):
                if all(arm(q,indices) for _,q in tail):
                    reach=[norm(sub(arm(q,indices)[2],arm(q,indices)[0]))/q['height'] for _,q in tail]
                    if reach[-1]-reach[0]>.15 and arm(p,indices)[3]>.9:candidates['reaching']='Arm extended over time with increasing shoulder-to-wrist distance; no object/contact is implied.'
            for label,evidence in candidates.items():
                key=(label,pid);event=self.active.get(key)
                if event and tail[0][0]<=event['end_time']+.2:event['end_time']=t
                else:
                    event=dict(start_time=tail[0][0],end_time=t,behavior=label,participants=[pid],objects=[],parent_activity_id=None,
                               description=evidence+CAUTION,confidence=None,review_status='needs_review',source='local_model')
                    self.events.append(event);self.active[key]=event

    def results(self):return self.events
