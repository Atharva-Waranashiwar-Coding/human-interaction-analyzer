"""One-time offline-vocabulary preparation; no CLIP model is used at inference."""
import sys
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from setup_model import download
from app.services.local_analysis import get_model
from app.services.object_tracking import BOX_MODEL_PATH,MODEL_ROOT


def prepare(force=False,keep_setup_cache=False):
    if BOX_MODEL_PATH.is_file() and not force:
        if not keep_setup_cache:
            (MODEL_ROOT/'setup-cache'/'ViT-B-32.pt').unlink(missing_ok=True)
        print(f'Box-capable model already cached: {BOX_MODEL_PATH}')
        return
    get_model()  # Configure local/offline Ultralytics settings.
    from ultralytics import YOLOWorld
    import clip
    source=MODEL_ROOT/'yolov8s-worldv2.pt'
    download(source,'yolov8s-worldv2.pt')
    model=YOLOWorld(str(source))
    from app.services.object_activities import MANIPULABLE
    vocabulary=['person'] + sorted(MANIPULABLE | {'cardboard carton','parcel','padded envelope','envelope'})
    original_load=clip.load
    def local_clip_load(*args,**kwargs):
        kwargs.setdefault('download_root',str(MODEL_ROOT/'setup-cache'))
        return original_load(*args,**kwargs)
    clip.load=local_clip_load
    try:
        model.model.set_classes(vocabulary,cache_clip_model=False)
    finally:
        clip.load=original_load
    model.model.names=dict(enumerate(vocabulary))
    model.model.clip_model=None
    temporary=BOX_MODEL_PATH.with_name('prepared-world-temporary.pt')
    try:
        model.save(str(temporary))
        verified=YOLOWorld(str(temporary))
        assert 'cardboard box' in verified.names.values()
        temporary.replace(BOX_MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    # The setup-only encoder is not needed after vocabulary embeddings are cached.
    if not keep_setup_cache:
        (MODEL_ROOT/'setup-cache'/'ViT-B-32.pt').unlink(missing_ok=True)
    print(f'Box/packet detector ready: {BOX_MODEL_PATH}; runtime text encoder removed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--force',action='store_true',help='Rebuild the cached vocabulary checkpoint')
    parser.add_argument('--keep-setup-cache',action='store_true',help='Retain the one-time encoder download for vocabulary experiments')
    args=parser.parse_args()
    prepare(args.force,args.keep_setup_cache)
