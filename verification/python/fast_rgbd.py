"""Read-only RGB-D registration using a batch of exact vendor projections."""
import ctypes as c
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from live_ainn_viewer import ScFrame
from top_view import Vector, DepthPoint, Pixel


class Intrinsic(c.Structure):
    _pack_ = 1
    _fields_ = [(name, c.c_double) for name in
                ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')]

class Extrinsic(c.Structure):
    _pack_ = 1
    _fields_ = [('rotation', c.c_double*9), ('translation', c.c_double*3)]

class Calibration:
    @staticmethod
    def read_parameter(sdk, getter, operation, *args):
        # Retry the intermittent PRODUCT_NOT_SUPPORT reported during startup.
        # Retry the actual device query, never substitute cached calibration.
        for delay in (0, .25, .5, 1.):
            if delay:
                time.sleep(delay)
            status = getter(sdk.device, *args)
            sdk.diagnostics.setdefault('calibration_reads', []).append(
                dict(operation=operation, status=int(status)))
            if status != -20:
                break
        sdk._check(status, operation)

    def __init__(self, sdk):
        self.sdk, self.lock = sdk, threading.Lock()
        api = sdk.base
        api.scGetSensorIntrinsicParameters.argtypes = [c.c_void_p,c.c_int,c.POINTER(Intrinsic)]
        api.scGetSensorIntrinsicParameters.restype = c.c_int
        api.scGetSensorExtrinsicParameters.argtypes = [c.c_void_p,c.POINTER(Extrinsic)]
        api.scGetSensorExtrinsicParameters.restype = c.c_int
        p,td,e = Intrinsic(),Intrinsic(),Extrinsic()
        self.read_parameter(sdk, api.scGetSensorIntrinsicParameters, 'RGB intrinsics', 2, c.byref(p))
        self.read_parameter(sdk, api.scGetSensorIntrinsicParameters, 'ToF intrinsics', 1, c.byref(td))
        self.read_parameter(sdk, api.scGetSensorExtrinsicParameters, 'extrinsics', c.byref(e))
        self.K=np.array([[p.fx,0,p.cx],[0,p.fy,p.cy],[0,0,1]],np.float64)
        self.D=np.array([p.k1,p.k2,p.p1,p.p2,p.k3,p.k4,p.k5,p.k6])
        self.tofK=np.array([[td.fx,0,td.cx],[0,td.fy,td.cy],[0,0,1]])
        self.R,self.t=np.array(e.rotation).reshape(3,3),np.array(e.translation)
        self.metadata=dict(rgb_K=self.K.tolist(),rgb_D=self.D.tolist(),tof_K=self.tofK.tolist(),
                           R=self.R.tolist(),translation_mm=self.t.tolist())
        for name in ('scGetTransformColorImgToDepthSensorEnabled','scGetTransformDepthImgToColorSensorEnabled'):
            getter=getattr(api,name)
            getter.argtypes=[c.c_void_p,c.POINTER(c.c_bool)]
            getter.restype=c.c_int
            enabled=c.c_bool()
            status=getter(sdk.device,c.byref(enabled))
            self.metadata[name]=dict(status=status,value=enabled.value if status==0 else None)
        self.maps,self.verified=None,False
        self.batch=c.CDLL(str(Path(__file__).resolve().parents[1]/'bin'/'rgbd_batch.dll'))
        self.batch.project_depth_batch.argtypes=[c.c_void_p,c.c_void_p,c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_int,c.c_void_p,c.c_void_p,c.c_void_p]
        self.batch.project_depth_batch.restype=c.c_int
        api.scConvertDepthFrameToPointCloudVector.argtypes=[c.c_void_p,c.POINTER(ScFrame),c.POINTER(Vector)]
        api.scConvertDepthFrameToPointCloudVector.restype=c.c_int
        api.scTransformDepthPointToColorPoint.argtypes=[c.c_void_p,DepthPoint,Pixel,c.POINTER(Pixel)]
        api.scTransformDepthPointToColorPoint.restype=c.c_int

    def to_tof(self, xyz_rgb_m):
        return (xyz_rgb_m*1000-self.t) @ self.R / 1000

    def project_tof(self, xyz_m):
        # Model joints are continuous 3-D points, not depth-frame pixel indices.
        # The SDK depth-pixel mapper must not receive out-of-frame predictions.
        xyz=np.asarray(xyz_m,np.float64).reshape(-1,3)
        rgb_xyz=xyz*1000 @ self.R.T+self.t
        result=np.full((len(xyz),2),np.nan)
        valid=np.isfinite(rgb_xyz).all(axis=1)&(rgb_xyz[:,2]>0)
        if valid.any():
            uv,_=cv2.projectPoints(rgb_xyz[valid],np.zeros(3),np.zeros(3),self.K,self.D)
            uv=uv.reshape(-1,2)
            width,height=self.rgb_size
            visible=np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
            result[np.flatnonzero(valid)[visible]]=uv[visible]
        return result

    def raw_to_rectified(self, uv):
        return cv2.undistortPoints(np.asarray(uv,np.float64).reshape(-1,1,2),self.K,self.D,P=self.smallK).reshape(-1,2)

    def capture(self, ready, frame, bgr, registration_only=False):
        started=time.perf_counter()
        if not ready.bits & 1: return None
        depth=ScFrame()
        with self.lock:
            self.sdk._check(self.sdk.base.scGetFrame(self.sdk.device,0,c.byref(depth)), 'depth frame')
            if depth.deviceTimestamp!=frame.deviceTimestamp: return None
            if depth.pixelFormat!=0 or depth.dataLen<depth.width*depth.height*2 or not depth.pFrameData:
                raise RuntimeError('Expected uint16 depth frame')
            mm=np.frombuffer(c.string_at(depth.pFrameData,depth.width*depth.height*2),np.uint16).reshape(depth.height,depth.width)
            cloud=np.empty((*mm.shape,3),np.float32)
            self.sdk._check(self.sdk.base.scConvertDepthFrameToPointCloudVector(self.sdk.device,c.byref(depth),
                           cloud.ctypes.data_as(c.POINTER(Vector))), 'point cloud')
            cloud_done=time.perf_counter()
            uv,xyz,indices=np.empty((mm.size,2),np.float32),np.empty((mm.size,3),np.float32),np.empty(mm.size,np.int32)
            count=self.batch.project_depth_batch(self.sdk.device,mm.ctypes.data,cloud.ctypes.data,
                depth.width,depth.height,frame.width,frame.height,uv.ctypes.data,xyz.ctypes.data,indices.ctypes.data)
            uv,xyz,indices=uv[:count],xyz[:count],indices[:count]
            batch_done=time.perf_counter()
            if not self.verified and count>=32:
                errors=[]
                for i in np.linspace(0,count-1,128,dtype=int):
                    y,x=divmod(int(indices[i]),depth.width)
                    q=Pixel()
                    self.sdk._check(self.sdk.base.scTransformDepthPointToColorPoint(self.sdk.device,
                        DepthPoint(x,y,int(mm[y,x])),Pixel(frame.width,frame.height),c.byref(q)), 'projection check')
                    errors.append(float(np.max(np.abs(uv[i]-[q.x,q.y]))))
                if max(errors)>0: raise RuntimeError('Native projection differs from SDK reference')
                self.verified=True
                self.validation=dict(samples=len(errors),max_pixel_error=max(errors),method='exact vendor batch')
        if registration_only:
            return dict(uv=uv, xyz_m=xyz, raw_depth_mm=mm, raw_cloud_mm=cloud)
        if self.maps is None:
            self.rgb_size=(frame.width,frame.height)
            self.small_size=(640,round(bgr.shape[0]*640/bgr.shape[1]))
            self.smallK=self.K.copy()
            self.smallK[:2]*=640/bgr.shape[1]
            self.maps=cv2.initUndistortRectifyMap(self.K,self.D,None,self.smallK,self.small_size,cv2.CV_32FC1)
            gy,gx=np.mgrid[:frame.height,:frame.width]
            self.rectified_lookup=np.rint(self.raw_to_rectified(np.column_stack((gx.ravel(),gy.ravel())))).astype(np.int16).reshape(frame.height,frame.width,2)
        rectified=cv2.remap(bgr,*self.maps,cv2.INTER_LINEAR)
        rgb_xyz=xyz*1000 @ self.R.T+self.t
        w,h=self.small_size
        integer_uv=uv.astype(np.int32)
        pixel=self.rectified_lookup[integer_uv[:,1],integer_uv[:,0]]
        valid=(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)&(rgb_xyz[:,2]>0)
        depth_small=np.full(w*h,np.inf,np.float32)
        np.minimum.at(depth_small,pixel[valid,1]*w+pixel[valid,0],rgb_xyz[valid,2])
        depth_small[~np.isfinite(depth_small)]=0
        return dict(rectified=rectified,depth_mm=depth_small.reshape(h,w),uv=uv,xyz_m=xyz,
                    raw_depth_mm=mm,raw_cloud_mm=cloud,
                    profile=dict(cloud_ms=(cloud_done-started)*1000,batch_ms=(batch_done-cloud_done)*1000,
                                 rest_ms=(time.perf_counter()-batch_done)*1000,points=count),
                    intrinsics=(self.smallK[0,0],self.smallK[1,1],self.smallK[0,2],self.smallK[1,2]))
