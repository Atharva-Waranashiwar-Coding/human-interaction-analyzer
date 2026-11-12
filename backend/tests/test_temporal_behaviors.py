"""Synthetic geometry validates rule contracts, not real-world recognition accuracy."""
import copy
import math

import pytest

from app.services.local_analysis import Rules
from app.services.temporal_behaviors import TemporalSuggestions
from app.services.video_service import create_annotation_records, build_export_payload
from app.models import Video
from app.schemas import AnnotationInput


def body(x=0, y=0, scale=1):
    k = [[0., 0., 0.] for _ in range(17)]
    for i, px, py in [(5,-10,40),(6,10,40),(7,-15,65),(8,15,65),
                      (9,-15,85),(10,15,85),(11,-8,80),(12,8,80)]:
        k[i] = [x+px*scale,y+py*scale,.95]
    return dict(center=(x,y+100*scale),height=100*scale,pose=k)


def set_arm(p, side, elbow, wrist):
    e,w = (7,9) if side == 'left' else (8,10)
    p['pose'][e] = [*elbow,.95]
    p['pose'][w] = [*wrist,.95]


def wave(t):
    p = body()
    set_arm(p,'left',(-25,20),(-20+18*math.sin(4*math.pi*t/1.6),0))
    return {'a':p}


def point(t):
    p,q = body(),body(120)
    set_arm(p,'right',(30,40),(50,40))
    return {'a':p,'b':q}


def handshake(t, high=False, pumping=True):
    p,q = body(),body(80)
    separated = t < .4
    y = (10 if high else 70) + (5*math.sin(2*math.pi*t/.8) if pumping else 0)
    set_arm(p,'right',(20,30 if high else 60),(25 if separated else 39,y))
    set_arm(q,'left',(60,30 if high else 60),(55 if separated else 41,y))
    return {'a':p,'b':q}


def hugging(t, wrap=True):
    p,q = body(),body(80 if t < .4 else 35)
    if t >= .4 and wrap:
        set_arm(p,'right',(40,25),(35,55))
        set_arm(q,'left',(-5,25),(0,55))
    return {'a':p,'b':q}


def pushing(t, extension=True, already_moving=False):
    x = 75 + (50*t if already_moving else max(0,t-.6)*50)
    p,q = body(),body(x)
    if t < .6 and extension:
        set_arm(p,'right',(12,65),(30,60))
    else:
        set_arm(p,'right',(40,50),(70,60))
    return {'a':p,'b':q}


def turning_path(t):
    return (60*t,0) if t <= 2 else (120,60*(t-2))


def following(t, turn=True):
    leader = turning_path(t) if turn else (60*t,0)
    follower = turning_path(t-1) if turn else (60*(t-1),0)
    return {'follower':body(*follower),'leader':body(*leader)}


def group(t, coordinated=True):
    return {str(i):body(i*50+30*t, 0 if coordinated else i*30*t) for i in range(3)}


def run(sequence, seconds=3, fps=5, engine=TemporalSuggestions):
    rules = engine(fps)
    for i in range(round(seconds*fps)+1):
        rules.step(i/fps,sequence(i/fps))
    return rules.results() if engine is TemporalSuggestions else rules.finish_all()


CASES = [
    ('waving',wave,3),('pointing',point,3),('handshaking',handshake,3),
    ('hugging',hugging,3),('pushing',pushing,2),('following',following,5),
    ('group activity',group,3),
]


@pytest.mark.parametrize('label,sequence,seconds', CASES)
def test_positive_candidates(label,sequence,seconds):
    events = [e for e in run(sequence,seconds) if e['behavior'] == label]
    assert events, label
    for event in events:
        assert 0 <= event['start_time'] < event['end_time'] <= seconds
        assert event['participants']
        assert event['review_status'] == 'needs_review'
        assert 'Geometry-only' in event['description']
        assert event['confidence'] is None
        assert AnnotationInput(**event).source == 'local_model'


@pytest.mark.parametrize('label,sequence,seconds', [
    ('waving',lambda t:wave(0),3),  # Held raised hand, no oscillation.
    ('pointing',lambda t:{'a':point(t)['a']},3),  # No human target on the ray.
    ('pointing',lambda t:{'a':body(),'b':body(120)},3),  # Bent/down arms.
    ('handshaking',lambda t:handshake(t,high=True),3),  # High-five.
    ('handshaking',lambda t:handshake(t,pumping=False),3),  # Static near wrists.
    ('hugging',lambda t:hugging(t,wrap=False),3),  # Standing close only.
    ('pushing',lambda t:pushing(t,extension=False),2),  # Coincident/accidental contact.
    ('pushing',lambda t:pushing(t,already_moving=True),2),  # Recipient already walking.
    ('following',lambda t:following(t,turn=False),5),  # Same-direction straight walking.
    ('group activity',lambda t:group(0),3),  # Standing near three people.
    ('group activity',lambda t:group(t,coordinated=False),3),
])
def test_misleading_negatives(label,sequence,seconds):
    assert not [e for e in run(sequence,seconds) if e['behavior'] == label]


@pytest.mark.parametrize('label,sequence,seconds', CASES)
@pytest.mark.parametrize('fps',[5,10,15])
@pytest.mark.parametrize('scale',[.5,2])
def test_rules_normalize_body_size_and_time(label,sequence,seconds,fps,scale):
    def scaled(t):
        people = sequence(t)
        for p in people.values():
            p['height'] *= scale
            p['center'] = tuple(x*scale for x in p['center'])
            p['pose'] = [[x*scale,y*scale,c] for x,y,c in p['pose']]
        return people
    assert any(e['behavior'] == label for e in run(scaled,seconds,fps=fps))


def test_jitter_is_not_wave():
    def jitter(t):
        p = wave(0)['a']
        p['pose'][9][0] += math.sin(t*30)*2
        return {'a':p}
    assert not run(jitter)


def test_low_sampling_rate_does_not_claim_hand_oscillations():
    assert not [e for e in run(wave,4,fps=2) if e['behavior']=='waving']
    assert not [e for e in run(handshake,4,fps=2) if e['behavior']=='handshaking']


@pytest.mark.parametrize('label,sequence,seconds', CASES)
def test_camera_motion_suppresses_all_new_candidates(label,sequence,seconds):
    rules = TemporalSuggestions(5)
    for i in range(round(seconds*5)+1):
        rules.step(i/5,sequence(i/5),camera_motion=True)
    assert rules.results() == []


def test_missing_keypoints_suppress_complex_candidates():
    def missing(t):
        people = handshake(t)
        for p in people.values():
            for k in p['pose']: k[2] = .1
        return people
    assert run(missing) == []
    def no_wrist(t):
        people=wave(t)
        people['a']['pose'][9][2] = .1
        return people
    assert run(no_wrist) == []


def test_id_switches_cannot_accumulate_evidence():
    def changed(t):
        return {f'{pid}_{round(t*5)}':p for pid,p in handshake(t).items()}
    assert run(changed) == []


@pytest.mark.parametrize('barrier',['missing','camera','timestamp_gap','jump'])
def test_events_merge_but_never_bridge_safety_barriers(barrier):
    rules = TemporalSuggestions(5)
    for i in range(8): rules.step(i/5,point(i/5))
    assert len([e for e in rules.results() if e['behavior']=='pointing']) == 1
    if barrier == 'missing': rules.step(1.6,{})
    elif barrier == 'camera': rules.step(1.6,point(1.6),camera_motion=True)
    elif barrier == 'jump':
        people=point(1.6)
        for p in people.values(): p['center']=(1000,1000)
        rules.step(1.6,people)
    # No sample at 1.6 for timestamp gap.
    for i in range(9,17): rules.step(i/5,point(i/5))
    events=[e for e in rules.results() if e['behavior']=='pointing']
    assert len(events)==2
    assert events[0]['end_time'] < events[1]['start_time']


def test_long_event_is_merged_and_independent_gestures_overlap():
    def two(t):
        return {**wave(t),'b':{**wave(t)['a'],'pose':copy.deepcopy(wave(t)['a']['pose'])}}
    events=[e for e in run(two,6) if e['behavior']=='waving']
    assert len(events)==2
    assert {tuple(e['participants']) for e in events}=={('a',),('b',)}
    assert all(e['end_time']-e['start_time'] > 4 for e in events)


def test_existing_rules_overlap_new_candidates_and_exports_keep_evidence():
    events=run(wave,3,engine=Rules)
    assert {'waving','raising a hand'} <= {e['behavior'] for e in events}
    records=create_annotation_records('video',events)
    for i,r in enumerate(records): r.id=str(i)
    video=Video(id='video',duration=3)
    exported=build_export_payload(video,records)
    assert any('Geometry-only' in e['description'] for e in exported['annotations'])
    assert all(e['review_status']=='needs_review' for e in exported['annotations'])


def test_non_monotonic_time_resets_without_inventing_event():
    rules=TemporalSuggestions(5)
    for i in range(4): rules.step(i/5,point(i/5))
    rules.step(float('nan'),point(0))
    rules.step(.8,point(.8))
    assert rules.results()==[]


def test_unilateral_reach_is_not_hug():
    def one_arm(t):
        people=hugging(t)
        people['b']=body(80 if t < .4 else 35)
        return people
    assert not [e for e in run(one_arm) if e['behavior']=='hugging']


def test_push_requires_subsequent_recipient_displacement():
    def stationary(t):
        people=pushing(t)
        people['b']=body(75)
        return people
    assert not [e for e in run(stationary,2) if e['behavior']=='pushing']


def test_handshake_requires_visible_convergence():
    assert not [e for e in run(lambda t:handshake(t+.4),3) if e['behavior']=='handshaking']


def test_established_hug_and_handshake_persist_beyond_history_window():
    for label,sequence in [('hugging',hugging),('handshaking',handshake)]:
        events=[e for e in run(sequence,8) if e['behavior']==label]
        assert len(events)==1
        assert events[0]['end_time']==8


def test_config_validation(monkeypatch):
    monkeypatch.setenv('BEHAVIOR_WINDOW_SECONDS','1')
    with pytest.raises(ValueError, match='Temporal window'):
        TemporalSuggestions(5)


def test_hand_contact_outside_shared_body_space_is_not_handshake():
    def outside(t):
        people=handshake(t)
        for p in people.values():
            for wrist in (9,10): p['pose'][wrist][0] -= 100
        return people
    assert not [e for e in run(outside) if e['behavior']=='handshaking']


def test_background_motion_ignores_foreground_person_and_detects_pan_zoom():
    import cv2
    import numpy as np
    from app.services.local_analysis import background_camera_motion
    rng=np.random.default_rng(42)
    background=rng.integers(0,256,(256,256),dtype=np.uint8)
    previous=background.copy()
    previous[40:210,60:210]=40
    current=background.copy()
    current[40:210,70:220]=40
    assert not background_camera_motion(previous,current,[[60,40,210,210]],5)
    pan=cv2.warpAffine(background,np.array([[1.,0.,8.],[0.,1.,0.]]),(256,256))
    assert background_camera_motion(background,pan,[],5)
    zoom=cv2.warpAffine(background,cv2.getRotationMatrix2D((128,128),0,1.04),(256,256))
    assert background_camera_motion(background,zoom,[],5)
    blank=np.zeros((256,256),dtype=np.uint8)
    assert background_camera_motion(blank,blank,[],5)
