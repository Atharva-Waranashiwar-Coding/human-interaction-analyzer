"""Opt-in metadata probe only. No audio decoding, models or speech inference."""
import json
import math
import subprocess


def probe_audio(path, video_duration):
    result=subprocess.run(['ffprobe','-v','error','-show_entries',
        'stream=index,codec_type,codec_name,sample_rate,channels,duration,start_time:format=start_time', '-of','json',str(path)],capture_output=True,text=True,timeout=15)
    if result.returncode:raise ValueError('FFprobe could not read audio metadata')
    data=json.loads(result.stdout);streams=[]
    def number(value,default):
        try:
            n=float(value)
            return n if math.isfinite(n) else default
        except (ValueError,TypeError):return default
    origin=number(data.get('format',{}).get('start_time'),0)
    video_stream=next((s for s in data.get('streams',[]) if s.get('codec_type')=='video'),{})
    origin=number(video_stream.get('start_time'),origin)
    for stream in data.get('streams',[]):
        if stream.get('codec_type')!='audio':continue
        offset=number(stream.get('start_time'),origin)-origin
        duration=number(stream.get('duration'),max(0,video_duration-offset))
        start=max(0,min(video_duration,offset));end=max(start,min(video_duration,offset+duration))
        streams.append(dict(stream_index=stream['index'],codec=stream.get('codec_name'),sample_rate=stream.get('sample_rate'),
            channels=stream.get('channels'),start_time=start,end_time=end,timing_source='stream metadata' if stream.get('duration') not in {None,'N/A'} else 'video duration fallback'))
    return dict(enabled=True,has_audio=bool(streams),streams=streams,
                analysis='Metadata only; audio event labels are manual and no speech, identity, emotion or intent is inferred.')
