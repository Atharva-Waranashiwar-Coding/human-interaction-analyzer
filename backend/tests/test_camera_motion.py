"""Measured motion is compensated; unknown motion only disables trajectory claims."""
import copy
import numpy as np
import pytest

from app.services.camera_motion import CameraEstimate,TrajectoryCoordinates,estimate_camera,transform_observations
from app.services.local_analysis import Rules
from app.services.object_activities import ObjectActivities,MovementSuggestions
from app.services.object_tracking import ObjectTracker
from test_temporal_behaviors import CASES,body
from test_object_activities import supported_scene,fake_model


def translated(people,objects,dx,dy=0):
    people=copy.deepcopy(people);objects=copy.deepcopy(objects)
    for p in people.values():
        p['center']=(p['center'][0]+dx,p['center'][1]+dy)
        p['pose']=[[x+dx,y+dy,c] for x,y,c in p['pose']]
        if 'box' in p:p['box']=[p['box'][0]+dx,p['box'][1]+dy,p['box'][2]+dx,p['box'][3]+dy]
    for o in objects.values():o['box']=[o['box'][0]+dx,o['box'][1]+dy,o['box'][2]+dx,o['box'][3]+dy]
    return people,objects


def test_unknown_background_is_distinct_from_measured_camera_motion():
    blank=np.zeros((256,256),dtype=np.uint8)
    estimate=estimate_camera(blank,blank,[],5)
    assert estimate.state=='unknown'
    assert estimate.transform is None


def test_estimate_recovers_pan_and_compensates_points():
    import cv2
    rng=np.random.default_rng(7)
    previous=rng.integers(0,256,(256,256),dtype=np.uint8)
    pan=np.array([[1.,0.,8.],[0.,1.,3.]])
    current=cv2.warpAffine(previous,pan,(256,256))
    estimate=estimate_camera(previous,current,[],5)
    assert estimate.state=='moving'
    assert np.allclose(estimate.transform[:2,2],[8,3],atol=.6)
    coordinates=TrajectoryCoordinates();coordinates.update(CameraEstimate('static',np.eye(3)))
    inverse=coordinates.update(estimate)
    p,o=supported_scene()
    shifted_p,shifted_o=translated(p,o,8,3)
    stable_p,stable_o=transform_observations(shifted_p,shifted_o,inverse)
    assert np.allclose(stable_p['holder']['center'],p['holder']['center'],atol=.7)
    assert np.allclose(stable_o['box']['box'],o['box']['box'],atol=.7)
    assert shifted_o['box']['box']!=o['box']['box'] # Input/overlay data remains raw.


def test_zoom_compensation_scales_body_height_and_preserves_confidence():
    coordinates=TrajectoryCoordinates();coordinates.update(CameraEstimate('static',np.eye(3)))
    zoom=np.diag([1.1,1.1,1.])
    inverse=coordinates.update(CameraEstimate('moving',zoom))
    p,o=supported_scene();original=copy.deepcopy(p)
    for person in p.values():
        person['height']*=1.1;person['center']=tuple(v*1.1 for v in person['center'])
        person['pose']=[[x*1.1,y*1.1,c] for x,y,c in person['pose']]
    for obj in o.values():obj['box']=[v*1.1 for v in obj['box']]
    stable_p,_=transform_observations(p,o,inverse)
    assert stable_p['holder']['height']==pytest.approx(100)
    assert stable_p['holder']['pose'][9][2]==original['holder']['pose'][9][2]
    assert np.allclose(stable_p['holder']['pose'][9][:2],original['holder']['pose'][9][:2])


def test_reference_rebases_after_unknown_interval():
    coordinates=TrajectoryCoordinates();coordinates.update(CameraEstimate('static',np.eye(3)))
    pan=np.array([[1.,0.,10.],[0.,1.,0.],[0.,0.,1.]])
    assert coordinates.update(CameraEstimate('unknown')) is None
    assert np.allclose(coordinates.update(CameraEstimate('moving',pan)),np.eye(3))


@pytest.mark.parametrize('label,sequence,seconds',[case for case in CASES if case[0] in {'waving','pointing','handshaking','hugging'}])
def test_pose_relative_behaviors_remain_enabled_with_unknown_background(label,sequence,seconds):
    rules=Rules(5,split_relative=True)
    for i in range(round(seconds*5)+1):
        t=i/5;people,_=translated(sequence(t),{},20*t)
        rules.step(t,{},camera_motion=True,relative_people=people)
    assert label in {e['behavior'] for e in rules.finish_all()}


def test_unknown_background_does_not_enable_trajectory_behaviors():
    rules=Rules(5,split_relative=True)
    for i in range(21):
        people={'a':body(i*10),'b':body(i*10+50),'c':body(i*10+100)}
        rules.step(i/5,{},camera_motion=True,relative_people=people)
    assert not {'following','pushing','group activity','walking together','approaching another person'} & {e['behavior'] for e in rules.finish_all()}


def test_camera_pan_with_stationary_people_does_not_become_carrying_or_walking():
    rules=Rules(5,split_relative=True);objects=ObjectActivities(5);movement=MovementSuggestions(5,mode='trajectory')
    coordinates=TrajectoryCoordinates()
    for i in range(21):
        t=i/5;raw_people,raw_objects=translated(*supported_scene(),i*4)
        transform=np.eye(3);transform[0,2]=4 if i else 0
        inverse=coordinates.update(CameraEstimate('moving',transform))
        people,items=transform_observations(raw_people,raw_objects,inverse)
        rules.step(t,people,False,relative_people=raw_people)
        movement.step(t,people)
        objects.step(t,people,items,relative_people=raw_people,relative_objects=raw_objects)
    results=rules.finish_all()+movement.results()+objects.results()
    assert 'holding' in {e['behavior'] for e in results}
    assert not {'walking','carrying','walking while carrying','following','group activity'} & {e['behavior'] for e in results}


def test_large_appearance_cut_is_not_unknown_stationary_background():
    estimate=estimate_camera(np.zeros((128,128),dtype=np.uint8),np.full((128,128),255,dtype=np.uint8),[],5)
    assert estimate.state=='cut' and estimate.transform is None


def test_scene_boundary_renews_object_alias_without_reusing_ids():
    model,_=fake_model([([[0,0,10,10]],[1],[39]),([[0,0,10,10]],[1],[39])])
    tracker=ObjectTracker(model)
    before=tracker.step(0,None,'cpu',640)
    tracker.reset_tracks()
    after=tracker.step(.5,None,'cpu',640)
    assert set(before).isdisjoint(after)
