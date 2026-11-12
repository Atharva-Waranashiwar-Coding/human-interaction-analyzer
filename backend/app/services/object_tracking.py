"""Optional nano COCO detector using the installed Ultralytics/ByteTrack stack."""
import math
import os
from pathlib import Path

MODEL_ROOT=Path(__file__).resolve().parents[2]/'data/models'
BOX_MODEL_PATH=MODEL_ROOT/'yolov8s-world-boxes.pt'
OBJECT_MODEL_PATH=Path(os.getenv('LOCAL_OBJECT_MODEL_PATH',str(BOX_MODEL_PATH if BOX_MODEL_PATH.is_file() else MODEL_ROOT/'yolo11n.pt')))
_object_model=None


def get_object_model():
    global _object_model
    if os.getenv('OBJECT_DETECTION_ENABLED','true').lower()!='true':
        return None,'Object detection disabled; person analysis and manual object annotation remain available.'
    if not OBJECT_MODEL_PATH.is_file():
        return None,'Object weights unavailable. Run python backend/setup_model.py --objects. Person analysis and manual object annotation remain available.'
    try:
        os.environ['YOLO_OFFLINE']='true'
        os.environ['YOLO_AUTOINSTALL']='false'
        if _object_model is None:
            from ultralytics import YOLO
            _object_model=YOLO(str(OBJECT_MODEL_PATH),task='detect')
        names=_object_model.names
        labels=set(names.values() if isinstance(names,dict) else names)
        warning=None if 'cardboard box' in labels else 'This object model has no cardboard-box/packet classes. Run python backend/setup_box_model.py, restart the backend, then Reanalyze.'
        return _object_model,warning
    except Exception as exc:
        return None,f'Object model could not initialize ({type(exc).__name__}); person analysis/manual annotations remain available.'


class ObjectTracker:
    def __init__(self,model,fps=2):
        self.model=model;self.fps=fps;self.aliases={};self.last={};self.count=0
        names=getattr(model,'names',None)
        names=names if isinstance(names,dict) else dict(enumerate(names)) if names is not None else None
        self.classes=[int(i) for i,label in names.items() if label!='person'] if names is not None else list(range(1,80))
        self.open_vocabulary=bool(names and 'cardboard box' in names.values())
        self.reset_tracks()

    def reset_tracks(self):
        self.aliases.clear();self.last.clear()
        if self.model.predictor is not None and hasattr(self.model.predictor,'trackers'):
            for tracker in self.model.predictor.trackers:tracker.reset()

    def step(self,t,frame,device,size):
        import numpy as np
        result=self.model.track(frame,persist=True,tracker='bytetrack.yaml',classes=self.classes,agnostic_nms=self.open_vocabulary,
                                conf=float(os.getenv('OBJECT_DETECTION_CONFIDENCE','.3')),device=device,imgsz=size,verbose=False,save=False)[0]
        if result.boxes is None or result.boxes.id is None:return {}
        boxes=result.boxes.xyxy.cpu().numpy();ids=result.boxes.id.cpu().numpy();classes=result.boxes.cls.cpu().numpy()
        output={}
        for box,raw_id,cls in zip(boxes,ids,classes):
            if not np.isfinite(box).all() or not np.isfinite(raw_id) or not np.isfinite(cls) or box[2]<=box[0] or box[3]<=box[1]:continue
            label=result.names[int(cls)]
            if label in {'plastic package','packet','padded envelope','envelope','parcel'}:label='package'
            if label=='cardboard carton':label='cardboard box'
            key=int(raw_id);previous=self.last.get(key)
            if label=='person':continue
            # Ambiguous same-class overlaps do not receive a confident object identity.
            ambiguous=False
            for other,other_id,other_cls in zip(boxes,ids,classes):
                if other_id==raw_id or other_cls!=cls:continue
                intersection=max(0,min(box[2],other[2])-max(box[0],other[0]))*max(0,min(box[3],other[3])-max(box[1],other[1]))
                area=(box[2]-box[0])*(box[3]-box[1]);oa=(other[2]-other[0])*(other[3]-other[1])
                if intersection/max(1,area+oa-intersection)>.5:ambiguous=True
            if ambiguous:
                self.last.pop(key,None);continue
            cx,cy=(float(box[0]+box[2])/2,float(box[1]+box[3])/2)
            if (previous is None or t-previous[0]>1.6/self.fps or previous[1]!=label or
                math.dist((cx,cy),previous[2])>max(40,2*max(box[2]-box[0],box[3]-box[1]))):
                self.count+=1;self.aliases[key]=f'object_{self.count}'
            self.last[key]=(t,label,(cx,cy))
            output[self.aliases[key]]=dict(class_name=label,box=[float(v) for v in box])
        return output
