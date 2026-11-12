import json
import subprocess
from types import SimpleNamespace
import pytest
from app.services.audio import probe_audio


def test_real_audio_metadata_and_video_timestamps(tmp_path):
    path=tmp_path/'audio.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=size=160x120:duration=2','-f','lavfi','-i','sine=frequency=440:duration=2','-c:v','libx264','-c:a','aac','-shortest',str(path)],check=True)
    result=probe_audio(path,2)
    assert result['has_audio'] and result['streams'][0]['codec']=='aac'
    assert 0<=result['streams'][0]['start_time']<result['streams'][0]['end_time']<=2


def test_offset_audio_timing_and_fallback(monkeypatch):
    payload=dict(format=dict(start_time='10'),streams=[dict(index=1,codec_type='audio',start_time='11',duration='N/A',sample_rate='16000',channels=1)])
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps(payload)))
    stream=probe_audio('unused',5)['streams'][0]
    assert stream['start_time']==1 and stream['end_time']==5 and 'fallback' in stream['timing_source']


def test_probe_failure(monkeypatch):
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1))
    with pytest.raises(ValueError):probe_audio('unused',5)


def test_audio_uses_video_stream_origin_not_earlier_format_origin(monkeypatch):
    payload=dict(format=dict(start_time='8'),streams=[dict(index=0,codec_type='video',start_time='10'),dict(index=1,codec_type='audio',start_time='11',duration='2')])
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps(payload)))
    stream=probe_audio('unused',5)['streams'][0]
    assert stream['start_time']==1 and stream['end_time']==3
