import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import local_analysis as local
from app.services.video_service import create_annotation_records
from app.schemas import AnnotationInput
from app.database import SessionLocal
from app.models import Video, Annotation
from app.api import routes
from test_api import client, make_test_video, reset_database


def person(x=0, raised=False):
    pose = [[0., 0., .9] for _ in range(17)]
    pose[5] = [10., 50., .9]
    pose[7] = [10., 30., .9]
    pose[9] = [10., 10. if raised else 70., .9]
    return dict(center=(x, 100.), height=100., pose=pose)


def test_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(local, '_model', None)
    monkeypatch.setattr(local, 'MODEL_PATH', tmp_path / 'missing.pt')
    with pytest.raises(RuntimeError, match='setup_model'):
        local.get_model()


def test_model_loaded_once(monkeypatch, tmp_path):
    import sys
    weights = tmp_path / 'model.pt'
    weights.touch()
    calls = []
    monkeypatch.setattr(local, '_model', None)
    monkeypatch.setattr(local, 'MODEL_PATH', weights)
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=lambda *a, **k: calls.append(a) or object(), settings={}))
    assert local.get_model() is local.get_model()
    assert len(calls) == 1


def test_raised_hand_smoothing_and_schema():
    rules = local.Rules(5)
    for i in range(8):
        rules.step(i/5, {'person_1': person(raised=True)})
    result = rules.finish_all()
    assert len(result) == 1
    assert result[0]['start_time'] == 0
    assert result[0]['end_time'] == 1.4
    assert AnnotationInput(**result[0]).source == 'local_model'
    assert create_annotation_records('video', result)[0].behavior == 'raising a hand'
    rules = local.Rules(5)
    rules.step(0, {'person_1': person(raised=True)})
    rules.step(.2, {})
    assert rules.finish_all() == []


@pytest.mark.parametrize('behavior,positions', [
    ('standing close together', lambda i: (0, 50)),
    ('approaching another person', lambda i: (i*5, 180-i*5)),
    ('moving apart', lambda i: (-i*5, 50+i*5)),
    ('walking together', lambda i: (i*5, 50+i*5)),
])
def test_pair_rules(behavior, positions):
    rules = local.Rules(5)
    for i in range(9):
        a, b = positions(i)
        rules.step(i/5, {'person_1': person(a), 'person_2': person(b)})
    assert behavior in [r['behavior'] for r in rules.finish_all()]


def test_motion_and_track_loss_suppress_pairs():
    rules = local.Rules(5)
    for i in range(10):
        rules.step(i/5, {'person_1': person(), 'person_2': person(50)}, camera_motion=True)
    assert rules.finish_all() == []
    rules = local.Rules(5)
    for i in range(10):
        rules.step(i/5, {f'person_{i}': person(), f'other_{i}': person(50)})
    assert rules.finish_all() == []


def test_nan_timestamp_rejected():
    with pytest.raises(ValueError):
        AnnotationInput(start_time=float('nan'), end_time=1, behavior='test')


def test_frames_timestamps(tmp_path):
    path = tmp_path / 'clip.mp4'
    make_test_video(path, 1)
    sampled = list(local.frames(path, 1, 5, 320))
    assert [t for t, _ in sampled] == [0, .2, .4, .6, .8]
    assert sampled[0][1].shape == (320, 320, 3)


def test_analysis_preserves_edits_and_recovers(monkeypatch, tmp_path):
    path = tmp_path / 'clip.mp4'
    make_test_video(path, 2)
    response = client.post('/api/videos', files={'file': ('clip.mp4', path.read_bytes(), 'video/mp4')})
    vid = response.json()['id']
    prediction = dict(start_time=0, end_time=1, behavior='raising a hand', participants=['person_1'], source='local_model')
    monkeypatch.setattr(routes, 'run_model_inference', lambda *a: [prediction])
    assert client.post(f'/api/videos/{vid}/analyze').status_code == 200
    annotations = client.get(f'/api/videos/{vid}').json()['annotations']
    annotations[0]['review_status'] = 'rejected'
    assert client.put(f'/api/videos/{vid}/annotations', json={'annotations': annotations}).status_code == 200
    assert client.post(f'/api/videos/{vid}/analyze').json()['status'] == 'manual_review'
    monkeypatch.setattr(routes, 'run_model_inference', lambda *a: [])
    client.post(f'/api/videos/{vid}/analyze?force=true')
    assert client.get(f'/api/videos/{vid}').json()['annotations'][0]['review_status'] == 'rejected'
    def fail(*a):
        raise RuntimeError('model failure')
    monkeypatch.setattr(routes, 'run_model_inference', fail)
    client.post(f'/api/videos/{vid}/analyze?force=true')
    assert client.get(f'/api/videos/{vid}').json()['analysis_status'] == 'failed'
    assert client.get(f'/api/videos/{vid}').json()['annotations']
    monkeypatch.setattr(routes, 'run_model_inference', lambda *a: [])
    client.post(f'/api/videos/{vid}/analyze')
    assert client.get(f'/api/videos/{vid}').json()['analysis_status'] == 'completed'
    assert json.loads(client.get(f'/api/videos/{vid}/export').text)['annotations']
    assert json.loads(client.get(f'/api/videos/{vid}/export?format=jsonl').text)['review_status'] == 'rejected'


def test_duplicate_claim(monkeypatch, tmp_path):
    path = tmp_path / 'clip.mp4'
    make_test_video(path, 1)
    vid = client.post('/api/videos', files={'file': ('clip.mp4', path.read_bytes(), 'video/mp4')}).json()['id']
    with SessionLocal() as db:
        db.query(Video).filter(Video.id == vid).update({'analysis_status': 'processing'})
        db.commit()
    assert client.post(f'/api/videos/{vid}/analyze').status_code == 409
    assert client.put(f'/api/videos/{vid}/annotations', json={'annotations': []}).status_code == 409


def test_model_tracking_adapter_and_video_reset(monkeypatch):
    import numpy as np
    class Tensor:
        def __init__(self, data): self.data = np.array(data)
        def cpu(self): return self
        def numpy(self): return self.data
    class Tracker:
        resets = 0
        def reset(self): self.resets += 1
    tracker = Tracker()
    calls = []
    sequence = [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    def track(frame, **kwargs):
        calls.append(kwargs)
        n = sequence[(len(calls)-1) % len(sequence)]
        boxes = SimpleNamespace(id=Tensor(list(range(n))), xyxy=Tensor([[10+50*i, 10, 50+50*i, 110] for i in range(n)]))
        poses = [[list(p) for p in person(raised=True)['pose']] for _ in range(n)]
        return [SimpleNamespace(boxes=boxes, keypoints=SimpleNamespace(data=Tensor(poses)))]
    model = SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker]), track=track)
    monkeypatch.setattr(local, 'get_model', lambda: model)
    monkeypatch.setattr(local, 'get_object_model', lambda: (None, 'Object detector disabled in this mock test.'))
    monkeypatch.setattr(local, 'device', lambda: 'cpu')
    monkeypatch.setattr(local, 'frames', lambda *a: ((i/5, np.zeros((128,128,3), dtype=np.uint8)) for i in range(10)))
    monkeypatch.setenv('LOCAL_IMAGE_SIZE', '128')
    result = local.analyze(Path('mock.mp4'), {'duration': 2})
    assert result and all(a['source'] == 'local_model' for a in result)
    assert len({p for a in result for p in a['participants']}) == 2
    local.analyze(Path('next.mp4'), {'duration': 2})
    assert tracker.resets == 2
    assert all(c['tracker'] == 'bytetrack.yaml' and c['persist'] for c in calls)


def test_corrupt_frames_fail(tmp_path):
    corrupt = tmp_path / 'bad.mp4'
    corrupt.write_bytes(b'not a video')
    with pytest.raises(RuntimeError, match='decode'):
        list(local.frames(corrupt, 1, 5, 128))


REAL_VIDEOS = os.getenv('REAL_TEST_VIDEOS', os.getenv('REAL_TEST_VIDEO','')).split(os.pathsep)


@pytest.mark.parametrize('real_video_path', REAL_VIDEOS)
@pytest.mark.skipif(not any(REAL_VIDEOS), reason='Set REAL_TEST_VIDEO or REAL_TEST_VIDEOS to real video paths')
def test_real_video_workflow(real_video_path):
    path = Path(real_video_path)
    filename = path.name if path.suffix.lower() in {'.mp4','.mov','.webm'} else path.name+'.mp4'
    uploaded = client.post('/api/videos', files={'file': (filename, path.read_bytes(), 'video/mp4')})
    assert uploaded.status_code == 200, uploaded.text
    vid = uploaded.json()['id']
    assert client.post(f'/api/videos/{vid}/analyze').status_code == 200
    detail = client.get(f'/api/videos/{vid}').json()
    assert detail['analysis_status'] == 'completed', detail
    if os.getenv('REAL_HANDSHAKE_VIDEO') == real_video_path:
        assert any(a['behavior']=='handshaking' for a in detail['annotations']), detail
    if os.getenv('REAL_CARRYING_VIDEO') == real_video_path:
        assert any(a['behavior']=='carrying' and a['objects'] for a in detail['annotations']), detail
    if os.getenv('REAL_HOLDING_VIDEO') == real_video_path:
        assert any(a['behavior']=='holding' and a['objects'] for a in detail['annotations']), detail
    if os.getenv('REAL_BOX_VIDEO') == real_video_path:
        classes={item['class_name'] for frame in detail['detections'] for item in frame['items']}
        assert 'cardboard box' in classes, classes
        assert not ({'bed','refrigerator'} & classes), classes
    for annotation in detail['annotations']:
        assert 0 <= annotation['start_time'] <= annotation['end_time'] <= detail['duration']
        assert annotation['source'] == 'local_model'
    edited = detail['annotations'] + [dict(start_time=0, end_time=min(1, detail['duration']), behavior='manual test', participants=[], source='manual', review_status='reviewed')]
    assert client.put(f'/api/videos/{vid}/annotations', json={'annotations': edited}).status_code == 200
    assert any(a['behavior'] == 'manual test' for a in client.get(f'/api/videos/{vid}/export').json()['annotations'])
    assert client.get(f'/api/videos/{vid}/export?format=jsonl').status_code == 200


def test_manual_annotation_before_first_analysis(monkeypatch, tmp_path):
    path = tmp_path / 'first.mp4'
    make_test_video(path, 1)
    vid = client.post('/api/videos', files={'file': ('first.mp4', path.read_bytes(), 'video/mp4')}).json()['id']
    annotation = dict(start_time=0, end_time=.5, behavior='manual first', source='manual')
    assert client.put(f'/api/videos/{vid}/annotations', json={'annotations': [annotation]}).status_code == 200
    calls = []
    monkeypatch.setattr(routes, 'run_model_inference', lambda *a: calls.append(a) or [])
    assert client.post(f'/api/videos/{vid}/analyze').status_code == 200
    assert len(calls) == 1
    assert client.get(f'/api/videos/{vid}').json()['annotations'][0]['behavior'] == 'manual first'


def test_auto_device_uses_validated_cpu_and_allows_override(monkeypatch):
    monkeypatch.setenv('LOCAL_DEVICE','auto')
    assert local.device()=='cpu'
    monkeypatch.setenv('LOCAL_DEVICE','mps')
    assert local.device()=='mps'
