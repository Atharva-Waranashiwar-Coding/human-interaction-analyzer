"""Exercise all new suggestions through existing persistence/review/export routes."""
import json

import pytest

from app.api import routes
from test_api import client, make_test_video, reset_database
from test_temporal_behaviors import CASES, run


@pytest.mark.parametrize('label,sequence,seconds', CASES)
def test_suggestion_review_and_exports(label,sequence,seconds,monkeypatch,tmp_path):
    path=tmp_path/'actions.mp4'
    make_test_video(path,seconds)
    response=client.post('/api/videos',files={'file':('actions.mp4',path.read_bytes(),'video/mp4')})
    assert response.status_code==200,response.text
    vid=response.json()['id']
    suggestions=run(sequence,seconds)
    monkeypatch.setattr(routes,'run_model_inference',lambda *a:suggestions)
    assert client.post(f'/api/videos/{vid}/analyze').status_code==200
    detail=client.get(f'/api/videos/{vid}').json()
    assert detail['analysis_status']=='completed'
    proposed=next(e for e in detail['annotations'] if e['behavior']==label)
    assert proposed['review_status']=='needs_review'
    assert 'Geometry-only' in proposed['description']
    proposed['review_status']='rejected'
    assert client.put(f'/api/videos/{vid}/annotations',json={'annotations':[proposed]}).status_code==200
    client.post(f'/api/videos/{vid}/analyze?force=true')
    data=client.get(f'/api/videos/{vid}/export').json()['annotations']
    matching=[e for e in data if e['behavior']==label and e['participants']==proposed['participants']]
    assert len(matching)==1
    assert matching[0]['review_status']=='rejected'
    assert matching[0]['description']==proposed['description']
    lines=client.get(f'/api/videos/{vid}/export?format=jsonl').text.splitlines()
    assert any(json.loads(line)['description']==proposed['description'] for line in lines)
