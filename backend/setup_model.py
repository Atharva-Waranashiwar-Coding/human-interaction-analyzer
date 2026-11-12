"""Explicit one-time downloads; inference never downloads missing weights."""
import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app.services.local_analysis import MODEL_PATH,get_model
from app.services.object_tracking import OBJECT_MODEL_PATH,get_object_model


def download(path,filename):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temporary=path.with_suffix('.download')
        try:
            urlretrieve(f'https://github.com/ultralytics/assets/releases/download/v8.3.0/{filename}',temporary)
            temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--objects',action='store_true',help='Also install the optional COCO nano object detector')
    args=parser.parse_args()
    download(MODEL_PATH,'yolo11n-pose.pt');get_model();print(f'Pose model ready: {MODEL_PATH}')
    if args.objects:
        download(OBJECT_MODEL_PATH,'yolo11n.pt')
        model,warning=get_object_model()
        if model is None:raise RuntimeError(warning)
        print(f'Object model ready: {OBJECT_MODEL_PATH}')
