"""Background-motion measurement and trajectory coordinates; no video decoding."""
from dataclasses import dataclass
import math


@dataclass
class CameraEstimate:
    state: str  # static, moving, unknown, cut
    transform: object = None  # previous frame -> current frame, 3x3

    def __bool__(self):
        # Preserve conservative truth testing for existing callers/tests.
        return self.state != 'static'


def estimate_camera(previous,current,boxes,fps):
    import cv2
    import numpy as np
    original_height,original_width=previous.shape
    scale=min(1.,480/original_width)
    if scale<1:
        shape=(round(original_width*scale),round(original_height*scale))
        previous=cv2.resize(previous,shape);current=cv2.resize(current,shape)
    height,width=previous.shape
    mask=np.full(previous.shape,255,dtype=np.uint8)
    for box in boxes:
        x1,y1,x2,y2=(v*scale for v in box);pad=max(5,round(width*.01))
        mask[max(0,int(y1)-pad):min(height,int(y2)+pad),max(0,int(x1)-pad):min(width,int(x2)+pad)]=0
    def unknown():
        # A large appearance discontinuity without a usable transform breaks all histories.
        difference=cv2.absdiff(previous,current).astype(np.float32)/255
        cut=float(difference.mean())>.22 and float((difference>.2).mean())>.65
        return CameraEstimate('cut' if cut else 'unknown')
    points=cv2.goodFeaturesToTrack(previous,180,.01,max(5,round(width*.015)),mask=mask)
    if points is None or len(points)<6:return unknown()
    opts=dict(winSize=(21,21),maxLevel=3)
    nxt,status,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,**opts)
    if nxt is None or status is None:return unknown()
    back,back_status,_=cv2.calcOpticalFlowPyrLK(current,previous,nxt,None,**opts)
    if back is None or back_status is None:return unknown()
    good=(status.reshape(-1)==1)&(back_status.reshape(-1)==1)&(np.linalg.norm((back-points).reshape(-1,2),axis=1)<1.5)
    if good.sum()<6:return unknown()
    transform,inliers=cv2.estimateAffinePartial2D(points[good],nxt[good],method=cv2.RANSAC,ransacReprojThreshold=2.)
    if transform is None or inliers is None or inliers.sum()<6 or inliers.mean()<.6:return unknown()
    positions=points[good].reshape(-1,2)[inliers.reshape(-1)==1]
    # A tiny corner cluster cannot establish whole-frame camera motion.
    if np.ptp(positions[:,0])<.08*width or np.ptp(positions[:,1])<.08*height:return unknown()
    zoom=math.sqrt(abs(np.linalg.det(transform[:,:2])))
    angle=abs(math.degrees(math.atan2(transform[1,0],transform[0,0])))
    if not .85<=zoom<=1.18 or angle>12:return unknown()
    displacement=positions@transform[:,:2].T+transform[:,2]-positions
    moving=float(np.median(np.linalg.norm(displacement,axis=1)))>width*.015/fps
    matrix=np.eye(3);matrix[:2]=transform;matrix[:2,2]/=scale
    return CameraEstimate('moving' if moving else 'static',matrix)


class TrajectoryCoordinates:
    """Rebase after unknown intervals; never join trajectories across reference changes."""
    def __init__(self):
        import numpy as np
        self.forward=np.eye(3);self.reliable=False

    def update(self,estimate):
        import numpy as np
        # Existing integrations can still mock the legacy bool interface.
        if isinstance(estimate,bool):estimate=CameraEstimate('unknown' if estimate else 'static',None if estimate else np.eye(3))
        if estimate.transform is None:
            self.forward=np.eye(3);self.reliable=False
            return None
        self.forward=estimate.transform@self.forward if self.reliable else np.eye(3)
        self.reliable=True
        if not np.isfinite(self.forward).all() or np.linalg.cond(self.forward[:2,:2])>10:
            self.reliable=False;return None
        return np.linalg.inv(self.forward)


def transform_observations(people,objects,inverse):
    """Separate small copies for behavior analysis; overlays retain raw frame coordinates."""
    import numpy as np
    if inverse is None:return {},{}
    def point(p):return tuple(float(x) for x in (inverse@np.array([p[0],p[1],1.]))[:2])
    def box(b):
        corners=[point(p) for p in ((b[0],b[1]),(b[2],b[1]),(b[2],b[3]),(b[0],b[3]))]
        return [min(p[0] for p in corners),min(p[1] for p in corners),max(p[0] for p in corners),max(p[1] for p in corners)]
    scale=math.sqrt(abs(float(np.linalg.det(inverse[:2,:2]))))
    stable_people={pid:dict(p,center=point(p['center']),height=p['height']*scale,
        pose=[(*point(k),float(k[2])) for k in p['pose']],**({'box':box(p['box'])} if 'box' in p else {})) for pid,p in people.items()}
    stable_objects={oid:dict(o,box=box(o['box'])) for oid,o in objects.items()}
    return stable_people,stable_objects
