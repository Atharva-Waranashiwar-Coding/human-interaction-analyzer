import json
import pytest
from app.database import SessionLocal
from app.models import Video
from app.api import routes
from app.services.local_analysis import AnalysisResult
from test_api import client, reset_database, make_test_video
from test_sequences import complete
from test_spatial import frame


def upload(tmp_path):
    path=tmp_path/'scene.mp4';make_test_video(path,8)
    r=client.post('/api/videos',files={'file':('scene.mp4',path.read_bytes(),'video/mp4')})
    assert r.status_code==200;return r.json()['id']


def test_cached_understanding_regions_search_and_legacy_export(tmp_path,monkeypatch):
    vid=upload(tmp_path);frames=[frame(i*.2,.1+i*.03) for i in range(21)]
    for f in frames:
        for item in f['items']:item['id']='p' if item['class_name']=='person' else 'o'
    monkeypatch.setattr(routes,'run_model_inference',lambda *a:AnalysisResult(complete(),frames))
    assert client.post(f'/api/videos/{vid}/analyze').status_code==200
    rows=client.get(f'/api/videos/{vid}').json()['annotations']
    assert any(a['annotation_kind']=='activity' for a in rows)
    assert any(a['annotation_kind']=='spatial' for a in rows)
    assert client.post(f'/api/videos/{vid}/understanding').json()['model_passes']==0
    search=client.get('/api/activities/search',params={'behavior':'carrying','person_count':1,'object_category':'cardboard box','participant_id':'p','object_id':'o','review_status':'needs_review','video_id':vid}).json()
    assert search['total']==1
    r=client.get('/api/activities/search',params={'sequence':'picking up / lifting > carrying > placing / putting down'}).json()
    assert r['total']==1, client.get(f'/api/videos/{vid}').json()['annotations']
    assert client.get('/api/activities/search?behavior=nonexistent').json()['total']==0
    region=dict(id='table',label='Table',kind='table',box=[.3,.1,.9,.9])
    assert client.put(f'/api/videos/{vid}/regions',json={'regions':[region]}).status_code==200
    assert client.put(f'/api/videos/{vid}/regions',json={'regions':[dict(region,box=[0,0,2,1])]}).status_code==422
    detail=client.get(f'/api/videos/{vid}').json()
    assert detail['scene_regions']==[region]
    assert client.get(f'/api/videos/{vid}/export').json()['scene_regions']==[region]
    assert len({a['id'] for a in detail['annotations']})==len(detail['annotations'])


@pytest.mark.parametrize('outcome,behavior,expected',[('reviewed','carrying','accepted'),('rejected','carrying','rejected'),('reviewed','holding','corrected')])
def test_review_provenance_and_feedback(tmp_path,monkeypatch,outcome,behavior,expected):
    vid=upload(tmp_path)
    monkeypatch.setattr(routes,'run_model_inference',lambda *a:[dict(complete()[1],source='local_model')])
    client.post(f'/api/videos/{vid}/analyze')
    rows=client.get(f'/api/videos/{vid}').json()['annotations'];row=rows[0]
    row.update(behavior=behavior,review_status=outcome,reviewer_feedback='Checked video')
    row['original_suggestion']={'behavior':'tampered'}
    assert client.put(f'/api/videos/{vid}/annotations',json={'annotations':rows}).status_code==200
    saved=client.get(f'/api/videos/{vid}').json()['annotations'][0]
    assert saved['original_suggestion']['behavior']=='carrying'
    assert saved['review_history'][-1]['outcome']==expected
    assert client.get(f'/api/videos/{vid}/feedback').json()['review_log'][-1]['feedback']=='Checked video'
    assert client.put(f'/api/videos/{vid}/annotations',json={'annotations':[saved]}).status_code==200
    assert len(client.get(f'/api/videos/{vid}').json()['annotations'][0]['review_history'])==1
    saved['evaluation_role']='ground_truth'
    assert client.put(f'/api/videos/{vid}/annotations',json={'annotations':[saved]}).status_code==422


def test_evaluation_scopes_and_legacy_search(tmp_path):
    vid=upload(tmp_path);row=dict(start_time=1,end_time=2,behavior='walking',review_status='reviewed',evaluation_role='ground_truth')
    assert client.put(f'/api/videos/{vid}/annotations',json={'annotations':[row]}).status_code==200
    scope=dict(covered_behaviors=['walking'],end_time=8,complete_ground_truth=True,scenario='warehouse')
    r=client.post(f'/api/videos/{vid}/evaluation',json=scope).json()
    assert r['summary']['false_negatives']==1 and r['summary']['precision'] is None
    r=client.post('/api/evaluation',json={vid:scope}).json()
    assert r['scenarios']['warehouse']['false_negatives']==1
    assert client.get('/api/activities/search?behavior=walking').json()['total']==1
    assert client.get('/api/activities/search?start_time=3&end_time=2').status_code==422


def test_audio_opt_in_and_missing_video(tmp_path,monkeypatch):
    vid=upload(tmp_path)
    assert client.get(f'/api/videos/{vid}').json()['audio_metadata']=={}
    r=client.post(f'/api/videos/{vid}/audio');assert r.status_code==200
    assert r.json()['has_audio'] is False
    monkeypatch.setattr('app.api.understanding.probe_audio',lambda *a:pytest.fail('Must use cached metadata'))
    assert client.post(f'/api/videos/{vid}/audio').status_code==200
    assert client.post('/api/videos/missing/audio').status_code==404


def test_deleted_feedback_and_audio_annotations_export(tmp_path,monkeypatch):
    vid=upload(tmp_path)
    monkeypatch.setattr(routes,'run_model_inference',lambda *a:[dict(complete()[1],source='local_model')])
    client.post(f'/api/videos/{vid}/analyze')
    assert client.put(f'/api/videos/{vid}/annotations',json={'annotations':[]}).status_code==200
    log=client.get(f'/api/videos/{vid}/feedback').json()['review_log']
    assert len(log)==1 and log[0]['original']['behavior']=='carrying' and log[0]['outcome']=='rejected'
    audio=dict(start_time=1,end_time=2,behavior='impact sound',annotation_kind='audio',participants=[],review_status='reviewed')
    assert client.put(f'/api/videos/{vid}/annotations',json={'annotations':[audio]}).status_code==200
    assert client.get('/api/activities/search?annotation_kind=audio').json()['total']==1
    assert json.loads(client.get(f'/api/videos/{vid}/export?format=jsonl').text)['annotation_kind']=='audio'


def test_advanced_busy_guard_and_reference_validation(tmp_path):
    vid=upload(tmp_path)
    with SessionLocal() as db:
        v=db.get(Video,vid);v.analysis_status='processing';db.commit()
    assert client.post(f'/api/videos/{vid}/understanding').status_code==409
    assert client.put(f'/api/videos/{vid}/regions',json={'regions':[]}).status_code==409
    assert client.post(f'/api/videos/{vid}/audio').status_code==409
    with SessionLocal() as db:
        v=db.get(Video,vid);v.analysis_status='completed';db.commit()
    bad=dict(start_time=0,end_time=1,behavior='sequence',annotation_kind='activity',details={'action_ids':['missing']})
    assert client.put(f'/api/videos/{vid}/annotations',json={'annotations':[bad]}).status_code==422
