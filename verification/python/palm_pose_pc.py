"""Single anonymous palm ROI prediction and OPEN/PINCH/GRIP geometry rules."""
import math
import cv2
import numpy as np

from hand_grasp import pose_features
from zoo_palm_pose import MPPalmDet, MPHandPose, ROOT


def pose_to_palm(points,confidence=1.):
    xy=np.asarray(points,np.float64)[:,:2]
    if xy.shape!=(21,2) or not np.isfinite(xy).all():
        return None
    tracked=xy[[0,5,9,13,17,1,2]]
    if np.linalg.norm(tracked[0]-tracked[2])<8:
        return None
    lo,hi=tracked.min(axis=0),tracked.max(axis=0)
    center=(lo+hi)*.5;size=np.maximum(hi-lo,24.)
    return np.r_[center-size/2,center+size/2,tracked.ravel(),confidence]


class SideAwareHandPose(MPHandPose):
    def _cropAndPadFromPalm(self,image,palm_bbox,for_rotation=False):
        # The second rotated crop is also made from the seven palm landmarks.
        # Padding only the outer 19-value hint leaves that inner crop collapsed.
        box=np.asarray(palm_bbox,np.float64)
        center=box.mean(axis=0);size=box[1]-box[0]
        size=np.maximum(size,max(24.,float(size.max())*.35))
        return super()._cropAndPadFromPalm(image,np.array([center-size/2,center+size/2]),for_rotation)


def tof_foreground_rois(depth_mm,size):
    depth=cv2.resize(np.asarray(depth_mm,np.float32),size,interpolation=cv2.INTER_NEAREST)
    valid=np.isfinite(depth)&(depth>150)&(depth<2000)
    if np.count_nonzero(valid)<100:return []
    regions=[]
    for distance in np.percentile(depth[valid],[10,40,70]):
        mask=(valid&(np.abs(depth-distance)<180)).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        count,_,stats,_=cv2.connectedComponentsWithStats(mask)
        for x,y,w,h,area in stats[1:count]:
            if area<100 or area>depth.size*.4 or min(w,h)<16:continue
            pad=max(16,round(max(w,h)*.2))
            box=(max(0,x-pad),max(0,y-pad),min(size[0],x+w+pad),min(size[1],y+h+pad))
            if box not in [r[1] for r in regions]:regions.append((float(distance),box))
    return [box for _,box in sorted(regions,key=lambda r:r[0])[:2]]


def classify_hand(world, points):
    features = [(pose_features(world,False), .7), (pose_features(points,False), .3)]
    usable = [(f,w) for f,w in features if f is not None]
    if not usable:
        return None, dict(reason='invalid_landmarks')
    scores = np.zeros(3)
    for f,weight in usable:
        angles=np.asarray(f['angles_deg']);extension=np.asarray(f['extension'])
        opened=float(np.mean(.5*np.clip((angles[:,1]-100)/65,0,1)+.5*np.clip((extension-.65)/.3,0,1)))
        grip=float(np.mean(.5*np.clip((145-angles[:,1])/65,0,1)+.3*np.clip((.9-extension)/.45,0,1)+.2*np.clip((150-angles[:,0])/70,0,1)))
        pinch=float(np.clip((.65-f['thumb_tip_distances'][0])/.5,0,1))
        if f['curled']>=3:
            pinch*=.25
        scores+=weight*np.array([opened,pinch,grip])
    scores/=sum(w for _,w in usable)
    dominant=usable[0][0]
    if dominant['straight']==4 and dominant['thumb_tip_distances'][0]>.35:
        state='OPEN'
    elif dominant['thumb_tip_distances'][0]<.3 and dominant['curled']<3:
        state='PINCH'
    elif dominant['curled']>=2:
        state='GRIP'
    else:
        state=('OPEN','PINCH','GRIP')[int(scores.argmax())]
    return state,dict(reason='relative_3d_geometry',scores=dict(zip(('OPEN','PINCH','GRIP'),scores.tolist())),
                      straight_fingers=dominant['straight'],curled_fingers=dominant['curled'],
                      pinch_distance_palm_widths=dominant['thumb_tip_distances'][0],
                      scores_are_probabilities=False)


class PalmRoiTracker:
    def __init__(self,palm_confidence=.45,pose_confidence=.6):
        self.palm_path=ROOT/'models/palm_detection_mediapipe_2023feb.onnx'
        self.pose_path=ROOT/'models/handpose_estimation_mediapipe_2023feb.onnx'
        self.palm=MPPalmDet(str(self.palm_path),scoreThreshold=palm_confidence,nmsThreshold=.3)
        self.pose=SideAwareHandPose(str(self.pose_path),confThreshold=pose_confidence)
        self.roi=None;self.last_seen=None;self.last_time=None;self.last_center=None
        self.kalman=cv2.KalmanFilter(4,2,0,cv2.CV_64F)
        self.kalman.measurementMatrix=np.array([[1,0,0,0],[0,1,0,0]],np.float64)
        self.kalman.measurementNoiseCov=np.eye(2)*4
        self.status='SEARCHING';self.palm_calls=0;self.pose_calls=0;self.predictions=0
        self.last_size=None
        self.lost=0
        self.tof_roi_calls=0

    def _pose(self,image,roi):
        clipped=np.clip(roi[:4],[0,0,0,0],[image.shape[1],image.shape[0]]*2)
        if min(clipped[2:]-clipped[:2])<2:
            return None
        self.pose_calls+=1
        try:
            result=self.pose.infer(image,roi.copy())
        except cv2.error:
            # A predicted crop can leave the image during a fast movement.
            return None
        if result is None or not np.isfinite(result).all():
            return None
        points=result[4:67].reshape(21,3)
        if pose_features(points,False) is None:
            return None
        return result

    def infer(self,image,now,depth_mm=None):
        h,w=image.shape[:2];width=min(640,w);height=round(h*width/w)
        small=cv2.resize(image,(width,height))
        if self.last_size!=(width,height):
            self.roi=None;self.last_seen=None;self.last_size=(width,height);self.lost=0
        dt=1/15 if self.last_time is None else float(np.clip(now-self.last_time,.001,.2))
        self.last_time=now
        self.kalman.transitionMatrix=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],np.float64)
        q=np.array([[dt**4/4,0,dt**3/2,0],[0,dt**4/4,0,dt**3/2],
                    [dt**3/2,0,dt**2,0],[0,dt**3/2,0,dt**2]],np.float64)
        self.kalman.processNoiseCov=q*300**2+np.eye(4)*1e-5
        result=None;source='palm_detection';palm_score=1.
        if self.roi is not None:
            predicted=self.kalman.predict().reshape(-1)[:2]
            delta=predicted-self.last_center
            roi=self.roi.copy();roi[:4]+=np.tile(delta,2);roi[4:18]+=np.tile(delta,7)
            result=self._pose(small,roi);palm_score=float(self.roi[-1]);source='tracked_roi'
        if result is None:
            self.palm_calls+=1
            palms=self.palm.infer(small)
            for palm in sorted(palms,key=lambda p:-p[-1]):
                result=self._pose(small,palm)
                if result is not None:
                    palm_score=float(palm[-1]);break
            source='palm_detection'
        if result is None and depth_mm is not None and self.palm_calls%3==0:
            for x1,y1,x2,y2 in tof_foreground_rois(depth_mm,(width,height)):
                self.tof_roi_calls+=1
                for palm in self.palm.infer(small[y1:y2,x1:x2]):
                    hint=palm.copy();hint[:4]+=np.tile([x1,y1],2);hint[4:18]+=np.tile([x1,y1],7)
                    result=self._pose(small,hint)
                    if result is not None:
                        palm_score=float(palm[-1]);source='tof_palm_detection';break
                if result is not None:break
        if result is None:
            self.lost+=1
            if self.roi is not None and self.lost<3 and now-self.last_seen<=.25:
                self.status='PREDICTING';self.predictions+=1
            else:
                self.roi=None;self.status='SEARCHING'
            return None
        points=result[4:67].reshape(21,3).copy()
        center=(points[0,:2]+points[9,:2])*.5
        if self.roi is None or source in ('palm_detection','tof_palm_detection'):
            self.kalman.statePost=np.array([[center[0]],[center[1]],[0.],[0.]],np.float64)
            self.kalman.errorCovPost=np.diag([4.,4.,10000.,10000.])
        else:
            self.kalman.correct(center.reshape(2,1).astype(np.float64))
        self.roi=pose_to_palm(points,float(result[-1]))
        self.last_center=center;self.last_seen=now;self.lost=0
        self.status='TRACKING' if source=='tracked_roi' and self.roi is not None else 'DETECTED'
        points[:,0]*=w/width;points[:,1]*=h/height;points[:,2]*=w/width
        return dict(points=points,world=result[67:130].reshape(21,3).copy(),
                    confidence=float(min(palm_score,result[-1])),source=source,
                    roi_small=None if self.roi is None else self.roi[:4].tolist())


class SinglePalmState:
    def __init__(self):
        self.xyz=None;self.position_time=None;self.last_seen=None
        self.state=None;self.pending=None;self.count=0;self.pending_since=0

    def update(self,hand,xyz,gesture,now):
        if hand is None:
            if self.last_seen is not None and now-self.last_seen>.3:
                self.state=None;self.pending=None;self.count=0
            return None
        self.last_seen=now
        if self.state is None:self.state=gesture
        if self.pending!=gesture:
            self.pending=gesture;self.count=1;self.pending_since=now
        else:self.count+=1
        if self.count>=3 and now-self.pending_since>=.12:self.state=gesture
        valid=False
        if xyz is not None:
            elapsed=float('inf') if self.position_time is None else now-self.position_time
            jump=0 if self.xyz is None else np.linalg.norm(xyz-self.xyz)
            if elapsed>=.5 or jump<=max(120.,2000*elapsed):
                alpha=1-math.exp(-elapsed/.12) if elapsed<.5 else 1.
                self.xyz=xyz.copy() if self.xyz is None else self.xyz*(1-alpha)+xyz*alpha
                self.position_time=now;valid=True
        return dict(xyz_mm=None if self.xyz is None else np.round(self.xyz,1).tolist(),
                    state=self.state,class_id=('OPEN','PINCH','GRIP').index(self.state),
                    confidence=round(hand['confidence'],4),valid=valid,
                    position_status='VALID' if valid else 'HOLD' if self.xyz is not None else 'NO_DEPTH')
