import copy
import math
from types import SimpleNamespace

import pytest

from app.services.object_activities import ObjectActivities,MovementSuggestions
from app.services.object_tracking import ObjectTracker
from test_temporal_behaviors import body


def person(x=0,wrist=(10,60)):
    p=body(x)
    p['pose'][8]=[x+20,50,.95]
    p['pose'][10]=[*wrist,.95]
    p['pose'][9]=[500,500,.95]
    return p


def obj(x=60,y=90,cls='bottle'):
    return {'class_name':cls,'box':[x-5,y-5,x+5,y+5]}


def manipulation(t):
    x=60 if t<=3 else 60+40*min(t-3,2)
    y=90 if t<=1 else 90-50*min(t-1,1)
    if t>5:y=40+50*min(t-5,1)
    px=0 if t<=3 else 40*min(t-3,2)
    wrist=(10,60) if t<1 or t>=6.5 else (x,y)
    return {'p':person(px,wrist)}, {'o':obj(x,y)}


def evaluate(fn,seconds=8,fps=2):
    r=ObjectActivities(fps)
    for i in range(round(seconds*fps)+1):
        p,o=fn(i/fps);r.step(i/fps,p,o)
    return r.results()


def test_pickup_carry_place_release_and_parent():
    events=evaluate(manipulation)
    labels={e['behavior'] for e in events}
    assert {'picking up / lifting','carrying','walking while carrying','placing / putting down','releasing','moving objects between locations'}<=labels
    parent=next(e for e in events if e['behavior']=='moving objects between locations')
    children=[e for e in events if e.get('parent_activity_id')==parent['id']]
    assert len(children)==3
    assert parent['objects']==[{'id':'o','class_name':'bottle'}]
    assert all(e['review_status']=='needs_review' and e['confidence'] is None for e in events)
    assert all('proximity does not establish contact' in e['description'] for e in events)


def test_reaching_toward_object_and_holding_after_acquisition():
    def reach(t):return {'p':person(wrist=(10+min(t,1)*50,90))},{'o':obj()}
    assert any(e['behavior']=='reaching toward an object' for e in evaluate(reach,2))
    def hold(t):
        y=90 if t<=1 else 90-50*min(t-1,1)
        return {'p':person(wrist=(10,60) if t<1 else (60,y))},{'o':obj(60,y)}
    assert any(e['behavior']=='holding' for e in evaluate(hold,5))


def test_static_proximity_is_not_holding_or_pickup():
    events=evaluate(lambda t:({'p':person(wrist=(60,90))},{'o':obj()}),4)
    assert events==[]


def transfer(t):
    x=50 if t<=1 else (75 if t<2 else 100+20*min(t-2,1))
    a=(x,60) if t<2 else (10,60)
    b=(250,60) if t<=1 else (x,60)
    return {'donor':person(0,a),'receiver':person(150,b)},{'o':obj(x,60)}


def test_transfer_requires_before_shared_after_states():
    events=evaluate(transfer,4)
    e=next(e for e in events if e['behavior']=='handing over / receiving objects')
    assert e['participants']==['donor','receiver']
    def no_shared(t):
        p,o=transfer(t)
        if 1<t<2:p['receiver']=person(150,(250,60))
        return p,o
    assert not any(e['behavior']=='handing over / receiving objects' for e in evaluate(no_shared,4))


def test_joint_carrying_and_ambiguous_single_holder():
    def joint(t):
        x=40+30*t
        return {'a':person(30*t,(x,60)),'b':person(80+30*t,(x,60))},{'o':obj(x,60)}
    events=evaluate(joint,3)
    assert any(e['behavior']=='joint carrying / lifting' for e in events)
    assert not any(e['behavior'] in {'carrying','holding'} for e in events)


@pytest.mark.parametrize('case',['no_people','no_objects','unrelated_motion','missing_wrists','camera'])
def test_negative_evidence(case):
    r=ObjectActivities(2)
    for i in range(12):
        t=i/2;p={'p':person()};o={'o':obj(60+30*t,90)}
        if case=='no_people':p={}
        if case=='no_objects':o={}
        if case=='missing_wrists':
            for k in (9,10):p['p']['pose'][k][2]=0
        r.step(t,p,o,camera_motion=case=='camera')
    assert r.results()==[]


@pytest.mark.parametrize('barrier',['missing_object','object_id','person_id','camera','timestamp_gap'])
def test_interrupted_sequences_do_not_group(barrier):
    r=ObjectActivities(2)
    for i in range(17):
        t=i/2;p,o=manipulation(t)
        if barrier=='timestamp_gap' and 3<=t<4:continue
        if barrier=='missing_object' and t==3:o={}
        if barrier=='object_id' and t>=3:o={'new':o.pop('o')}
        if barrier=='person_id' and t>=3:p={'new':p.pop('p')}
        r.step(t,p,o,camera_motion=barrier=='camera' and t==3)
    assert not any(e['behavior']=='moving objects between locations' for e in r.results())


def test_missing_wrist_does_not_fabricate_release():
    def missing(t):
        p,o=manipulation(min(t,5))
        if t>5:
            for k in (9,10):p['p']['pose'][k][2]=0
        return p,o
    assert not any(e['behavior'] in {'releasing','placing / putting down'} for e in evaluate(missing))


def test_unsupported_class_and_disappearance_do_not_imply_actions():
    assert evaluate(lambda t:({'p':person(wrist=(60,90))},{'o':obj(cls='chair')}))==[]
    def disappears(t):
        p,o=manipulation(min(t,5))
        return p,{} if t>5 else o
    assert not any(e['behavior'] in {'dropping','releasing','placing / putting down'} for e in evaluate(disappears))


@pytest.mark.parametrize('scale',[.5,2])
def test_body_size_normalization(scale):
    def scaled(t):
        p,o=manipulation(t)
        for q in p.values():
            q['height']*=scale;q['center']=tuple(x*scale for x in q['center']);q['pose']=[[x*scale,y*scale,c] for x,y,c in q['pose']]
        for q in o.values():q['box']=[x*scale for x in q['box']]
        return p,o
    assert any(e['behavior']=='moving objects between locations' for e in evaluate(scaled))


class Tensor:
    def __init__(self,data):
        import numpy as np
        self.data=np.array(data)
    def cpu(self):return self
    def numpy(self):return self.data


def fake_model(rows):
    calls=[]
    def track(frame,**kwargs):
        calls.append(kwargs)
        row=rows.pop(0)
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor(row[0]),id=Tensor(row[1]),cls=Tensor(row[2])),names={39:'bottle',41:'cup'})]
    return SimpleNamespace(predictor=None,track=track),calls


def test_tracker_class_and_gap_changes_renew_anonymous_ids():
    model,calls=fake_model([([[0,0,10,10]],[1],[39]),([[1,0,11,10]],[1],[39]),([[2,0,12,10]],[1],[41]),([[2,0,12,10]],[1],[41])])
    r=ObjectTracker(model,2)
    first=r.step(0,None,'cpu',640);second=r.step(.5,None,'cpu',640)
    third=r.step(1,None,'cpu',640);fourth=r.step(2,None,'cpu',640)
    assert list(first)==list(second)
    assert len({*first,*third,*fourth})==3
    assert all(0 not in call['classes'] for call in calls)


def test_ambiguous_overlaps_get_no_object_identity():
    model,_=fake_model([([[0,0,10,10],[1,0,11,10]],[1,2],[39,39])])
    assert ObjectTracker(model).step(0,None,'cpu',640)=={}


def test_missing_weights_preserves_manual_workflow(monkeypatch,tmp_path):
    from app.services import object_tracking
    monkeypatch.setattr(object_tracking,'OBJECT_MODEL_PATH',tmp_path/'missing.pt')
    monkeypatch.setenv('OBJECT_DETECTION_ENABLED','true')
    model,warning=object_tracking.get_object_model()
    assert model is None and 'manual' in warning


def full_body(x=0):
    p=person(x)
    for k,dx,y in [(5,-10,20),(6,10,20),(11,-8,60),(12,8,60),(13,-8,80),(14,8,80),(15,-8,100),(16,8,100)]:p['pose'][k]=[x+dx,y,.95]
    return p


@pytest.mark.parametrize('label',['standing','walking','bending','reaching'])
def test_individual_movement_candidates(label):
    r=MovementSuggestions(5)
    for i in range(21):
        t=i/5;p=full_body(40*t if label=='walking' else 0)
        if label=='walking':
            p['pose'][13][0]+=14*math.sin(t*2*math.pi);p['pose'][14][0]-=14*math.sin(t*2*math.pi)
        if label=='bending':
            p['pose'][5]=[30,50,.95];p['pose'][6]=[50,50,.95]
        if label=='reaching':
            p['pose'][8]=[30,36,.95];p['pose'][10]=[25+min(t,1)*35,60,.95]
        r.step(t,{'p':p})
    assert label in {e['behavior'] for e in r.results()}


def test_existing_frame_stream_is_shared_and_object_inference_is_sparse(monkeypatch):
    import numpy as np
    from app.services import local_analysis as local
    calls={'decode':0,'pose':0,'object':0}
    def frames(*args):
        calls['decode']+=1
        for i in range(10):yield i/5,np.zeros((128,128,3),dtype=np.uint8)
    def pose_track(frame,**kwargs):
        calls['pose']+=1
        p=person(40,(60,90))
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([[20,0,80,100]]),id=Tensor([1])),keypoints=SimpleNamespace(data=Tensor([p['pose']])))]
    def object_track(frame,**kwargs):
        calls['object']+=1
        assert kwargs['imgsz']==640
        assert 0 not in kwargs['classes']
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([[55,85,65,95]]),id=Tensor([1]),cls=Tensor([39])),names={39:'bottle'})]
    monkeypatch.setattr(local,'frames',frames)
    monkeypatch.setattr(local,'get_model',lambda:SimpleNamespace(predictor=None,track=pose_track))
    monkeypatch.setattr(local,'get_object_model',lambda:(SimpleNamespace(predictor=None,track=object_track),None))
    monkeypatch.setattr(local,'background_camera_motion',lambda *a:False)
    monkeypatch.setenv('LOCAL_IMAGE_SIZE','128');monkeypatch.setenv('LOCAL_SAMPLE_FPS','5');monkeypatch.setenv('OBJECT_SAMPLE_FPS','2')
    result=local.analyze('unused',{'duration':2})
    assert calls=={'decode':1,'pose':10,'object':4}
    assert len(result.detections)==4
    assert all({d['class_name'] for d in f['items']}=={'person','bottle'} for f in result.detections)
    assert all(0<=v<=1 for f in result.detections for d in f['items'] for v in d['box'])
    assert not result.warnings


def test_object_failure_keeps_person_analysis_and_warning(monkeypatch):
    import numpy as np
    from app.services import local_analysis as local
    def fail(*a,**k):raise RuntimeError('object backend unavailable')
    p=person()
    pose=SimpleNamespace(predictor=None,track=lambda *a,**k:[SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([[0,0,50,100]]),id=Tensor([1])),keypoints=SimpleNamespace(data=Tensor([p['pose']])))])
    monkeypatch.setattr(local,'get_model',lambda:pose)
    monkeypatch.setattr(local,'get_object_model',lambda:(SimpleNamespace(predictor=None,track=fail),None))
    monkeypatch.setattr(local,'frames',lambda *a:((i/5,np.zeros((128,128,3),dtype=np.uint8)) for i in range(10)))
    monkeypatch.setattr(local,'background_camera_motion',lambda *a:False)
    result=local.analyze('unused',{'duration':2})
    assert any('interrupted' in warning for warning in result.warnings)
    assert any(d['class_name']=='person' for frame in result.detections for d in frame['items'])


def test_model_initialization_failure_is_optional(monkeypatch,tmp_path):
    import sys
    from app.services import object_tracking as tracking
    weights=tmp_path/'bad.pt';weights.touch()
    monkeypatch.setattr(tracking,'OBJECT_MODEL_PATH',weights);monkeypatch.setattr(tracking,'_object_model',None)
    monkeypatch.setenv('OBJECT_DETECTION_ENABLED','true')
    def fail(*a,**k):raise RuntimeError('broken weights')
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=fail))
    model,warning=tracking.get_object_model()
    assert model is None and 'initialize' in warning


def test_transfer_is_not_inferred_through_unassociated_gap():
    def interrupted(t):
        p,o=transfer(t)
        if t==2:p['receiver']=person(150,(250,60))
        return p,o
    assert not any(e['behavior']=='handing over / receiving objects' for e in evaluate(interrupted,4))


def test_falling_then_resting_after_early_separation_is_not_placement():
    def drop(t):
        p,o=manipulation(t)
        if t>5.5:p['p']['pose'][10]=[10,60,.95]
        return p,o
    assert not any(e['behavior']=='placing / putting down' for e in evaluate(drop))


def test_custom_vocabulary_keeps_class_zero_and_excludes_person_at_any_index():
    rows=[([[0,0,10,10]],[1],[0])]
    model,_=fake_model(rows)
    model.names={0:'cardboard box',1:'person',2:'packet'}
    def track(*a,**kwargs):
        assert kwargs['classes']==[0,2]
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([[0,0,10,10]]),id=Tensor([1]),cls=Tensor([0])),names=model.names)]
    model.track=track
    assert ObjectTracker(model).step(0,None,'cpu',640)['object_1']['class_name']=='cardboard box'


@pytest.mark.parametrize('cls',['cardboard box','package','packet'])
def test_box_and_packet_classes_use_existing_manipulation_rules(cls):
    def scene(t):
        people,objects=manipulation(t)
        objects['o']['class_name']=cls
        return people,objects
    assert any(e['behavior']=='picking up / lifting' for e in evaluate(scene))


def test_furniture_labels_are_not_rewritten_to_boxes():
    model,_=fake_model([])
    model.names={0:'bed',1:'refrigerator',2:'person'}
    def track(*a,**kwargs):
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([[0,0,10,10]]),id=Tensor([1]),cls=Tensor([0])),names=model.names)]
    model.track=track
    assert ObjectTracker(model).step(0,None,'cpu',640)['object_1']['class_name']=='bed'


def supported_scene(t=0,offset=0):
    p=body(offset)
    for index,x,y in [(7,-35,55),(8,35,55),(9,-15,70),(10,15,70)]:p['pose'][index]=[offset+x,y,.95]
    return {'holder':p},{'box':{'class_name':'cardboard box','box':[offset-25,45,offset+25,75]}}


def test_two_arm_support_can_suggest_holding_without_pickup_onset():
    events=evaluate(supported_scene,3)
    holding=[e for e in events if e['behavior']=='holding']
    assert len(holding)==1
    assert holding[0]['participants']==['holder']
    assert holding[0]['review_status']=='needs_review'
    assert 'Two confident wrists' in holding[0]['description']
    assert not any(e['behavior']=='picking up / lifting' for e in events)


def test_relative_support_survives_camera_translation_without_claiming_carrying():
    rules=ObjectActivities(2)
    for i in range(7):
        t=i/2;people,objects=supported_scene(t,offset=40*t)
        rules.reset(relative=False)  # Matches the per-pose-frame motion gate.
        rules.step(t,people,objects,camera_motion=True)
    assert {e['behavior'] for e in rules.results()}=={'holding'}


@pytest.mark.parametrize('case',['one_wrist','below_hips','ambiguous_holders','missing_arm'])
def test_relative_support_rejects_weak_or_ambiguous_evidence(case):
    def scene(t):
        people,objects=supported_scene(t)
        if case=='one_wrist':people['holder']['pose'][9][2]=.1
        if case=='missing_arm':people['holder']['pose'][7][2]=.1
        if case=='below_hips':objects['box']['box']=[-25,45,25,120]
        if case=='ambiguous_holders':people['other']=copy.deepcopy(people['holder'])
        return people,objects
    assert not any(e['behavior']=='holding' for e in evaluate(scene,3))


def test_unrelated_bystander_does_not_reset_carrier_history():
    def scene(t):
        people,objects=manipulation(t)
        if 3.5<=t<=4.5:people['bystander']=person(500,(500,90))
        return people,objects
    events=evaluate(scene)
    assert any(e['behavior']=='carrying' for e in events)
    assert any(e['behavior']=='moving objects between locations' for e in events)


def test_missing_old_wrist_does_not_poison_later_valid_carrying_window():
    def scene(t):
        people,objects=manipulation(t)
        if t==2.5:
            for index in (9,10):people['p']['pose'][index][2]=0
        return people,objects
    events=evaluate(scene)
    assert any(e['behavior']=='carrying' and e['start_time']>=3 for e in events)


def test_support_history_does_not_bridge_occlusion_or_object_id_change():
    rules=ObjectActivities(2)
    for i in range(8):
        t=i/2;people,objects=supported_scene(t)
        if t==1.5:objects={}
        if t>=2:objects={'new-box':objects.pop('box')}
        rules.step(t,people,objects,camera_motion=True)
    events=[e for e in rules.results() if e['behavior']=='holding']
    assert len(events)==2
    assert events[0]['objects']!=events[1]['objects']


def test_relative_support_does_not_merge_through_implausible_track_jump():
    rules=ObjectActivities(2)
    for i in range(8):
        t=i/2;people,objects=supported_scene(t,offset=1000 if t>=1.5 else 0)
        rules.step(t,people,objects,camera_motion=True)
    events=[e for e in rules.results() if e['behavior']=='holding']
    assert len(events)==2
    assert events[0]['end_time']<events[1]['start_time']


def test_stationary_object_proximity_then_hand_withdrawal_is_not_release():
    def scene(t):
        return {'p':person(wrist=(60,90) if t<2 else (10,60))},{'o':obj()}
    events=evaluate(scene,4)
    assert not any(e['behavior'] in {'releasing','holding','placing / putting down'} for e in events)


def test_planar_pushing_candidate_with_sustained_wrist_object_motion():
    def scene(time):
        x=60+20*time
        p=person(wrist=(x,90))
        # Both wrist tracks are visible, but only one is near the object.
        return {'p':p},{'o':obj(x,90)}
    labels={e['behavior'] for e in evaluate(scene,2)}
    assert 'pushing an object' in labels
    assert 'carrying' not in labels
    assert 'picking up / lifting' not in labels


def test_planar_pulling_candidate_moves_toward_stationary_torso():
    def scene(time):
        x=120-20*time
        return {'p':person(wrist=(x,90))},{'o':obj(x,90)}
    labels={e['behavior'] for e in evaluate(scene,2)}
    assert 'pulling an object' in labels
    assert 'carrying' not in labels


@pytest.mark.parametrize('case',['no_wrist','static','independent','moving_body','moving_camera','track_gap'])
def test_planar_manipulation_rejects_weak_or_interrupted_evidence(case):
    rules=ObjectActivities(2)
    for i in range(5 if case=='track_gap' else 7):
        time=i/2
        x=60+20*time
        p=person(wrist=(x if case!='independent' else 10,90))
        if case=='no_wrist':
            p['pose'][9][2]=0;p['pose'][10][2]=0
        if case=='moving_body':
            p=person(x=20*time,wrist=(x,90))
        objects={'o':obj(60 if case=='static' else x,90)}
        if case=='track_gap' and time==1:
            objects={}
        rules.step(time,{'p':p},objects,camera_motion=case=='moving_camera')
    assert not any(e['behavior'] in {'pushing an object','pulling an object','dragging / sliding an object'}
                   for e in rules.results())
