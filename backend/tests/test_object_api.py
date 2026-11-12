import json

import pytest
from sqlalchemy import create_engine,text,inspect

from app.database import initialize_database,SessionLocal
from app.api import routes
from app.models import Annotation
from app.services.local_analysis import AnalysisResult
from app.schemas import AnnotationInput
from test_api import client,make_test_video,reset_database
from test_object_activities import evaluate,manipulation


def upload(tmp_path):
    path=tmp_path/'objects.mp4';make_test_video(path,8)
    response=client.post('/api/videos',files={'file':('objects.mp4',path.read_bytes(),'video/mp4')})
    assert response.status_code==200,response.text
    return response.json()['id']


def test_additive_sqlite_migration_preserves_existing_rows(tmp_path):
    path=tmp_path/'old.db';db=create_engine(f'sqlite:///{path}')
    with db.begin() as c:
        c.execute(text('CREATE TABLE annotations (id VARCHAR PRIMARY KEY, description TEXT)'))
        c.execute(text("INSERT INTO annotations VALUES ('existing','reviewed note')"))
        c.execute(text('CREATE TABLE videos (id VARCHAR PRIMARY KEY)'))
        c.execute(text("INSERT INTO videos VALUES ('video')"))
    initialize_database(db);initialize_database(db)
    with db.connect() as c:
        assert c.execute(text('SELECT id,description,objects,parent_activity_id FROM annotations')).one()==('existing','reviewed note','[]',None)
        assert c.execute(text('SELECT id,detections,analysis_warnings FROM videos')).one()==('video','[]','[]')
    assert path.with_suffix('.pre-object-backup.sqlite3').exists()


def test_object_graph_review_export_and_preservation(tmp_path,monkeypatch):
    vid=upload(tmp_path);events=evaluate(manipulation)
    captured=[{'time':0,'items':[{'id':'o','class_name':'bottle','box':[.1,.2,.2,.3]}]}]
    monkeypatch.setattr(routes,'run_model_inference',lambda *a:AnalysisResult(events,captured,['test warning']))
    client.post(f'/api/videos/{vid}/analyze')
    detail=client.get(f'/api/videos/{vid}').json()
    assert detail['analysis_status']=='completed'
    assert detail['detections']==captured and detail['analysis_warnings']==['test warning']
    rows=detail['annotations'];parent=next(e for e in rows if e['behavior']=='moving objects between locations')
    assert len([e for e in rows if e['parent_activity_id']==parent['id']])==3
    for e in rows:
        AnnotationInput(**e)
        e['review_status']='reviewed'
        e['objects']=[{'id':'corrected_object','class_name':'cup'}]
    assert client.put(f'/api/videos/{vid}/annotations',json={'annotations':rows}).status_code==200
    saved=client.get(f'/api/videos/{vid}/export').json()['annotations']
    assert {e['id'] for e in saved}=={e['id'] for e in rows}
    assert any(e['parent_activity_id']==parent['id'] for e in saved)
    assert all(e['objects']==[{'id':'corrected_object','class_name':'cup'}] for e in saved)
    monkeypatch.setattr(routes,'run_model_inference',lambda *a:[])
    client.post(f'/api/videos/{vid}/analyze?force=true')
    exported=[json.loads(line) for line in client.get(f'/api/videos/{vid}/export?format=jsonl').text.splitlines()]
    assert {e['id'] for e in exported}=={e['id'] for e in rows}
    assert all(e['review_status']=='reviewed' for e in exported)


@pytest.mark.parametrize('parents',[{'a':'a'},{'a':'missing'},{'a':'b','b':'a'}])
def test_invalid_parent_links_are_rejected_before_deletion(tmp_path,parents):
    vid=upload(tmp_path)
    rows=[dict(id=aid,start_time=0,end_time=1,behavior='manual',parent_activity_id=parent) for aid,parent in parents.items()]
    assert client.put(f'/api/videos/{vid}/annotations',json={'annotations':rows}).status_code==422


def test_reanalysis_remaps_duplicate_protected_parent(tmp_path,monkeypatch):
    vid=upload(tmp_path)
    protected=dict(id='saved-parent',start_time=0,end_time=7,behavior='moving objects between locations',participants=['p'],objects=[{'id':'o','class_name':'bottle'}])
    assert client.put(f'/api/videos/{vid}/annotations',json={'annotations':[protected]}).status_code==200
    prediction=dict(protected,id='new-parent',source='local_model',review_status='needs_review')
    child=dict(id='new-child',start_time=1,end_time=2,behavior='carrying',participants=['p'],objects=protected['objects'],parent_activity_id='new-parent',source='local_model')
    monkeypatch.setattr(routes,'run_model_inference',lambda *a:[prediction,child])
    client.post(f'/api/videos/{vid}/analyze?force=true')
    rows=client.get(f'/api/videos/{vid}').json()['annotations']
    assert {e['id'] for e in rows}=={'saved-parent','new-child'}
    assert next(e for e in rows if e['id']=='new-child')['parent_activity_id']=='saved-parent'


def test_legacy_annotation_payload_accepts_default_object_fields(tmp_path):
    vid=upload(tmp_path)
    assert client.put(f'/api/videos/{vid}/annotations',json={'annotations':[dict(start_time=0,end_time=1,behavior='manual')]}).status_code==200
    row=client.get(f'/api/videos/{vid}/export').json()['annotations'][0]
    assert row['objects']==[] and row['parent_activity_id'] is None
