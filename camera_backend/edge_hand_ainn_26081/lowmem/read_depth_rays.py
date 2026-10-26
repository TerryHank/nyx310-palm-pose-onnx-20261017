"""Read-only SDK calibration: preserve the manufacturer's exact depth rays."""
import ctypes as c
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'verification'/'python'))
from live_ainn_viewer import ScepterSdk,ScFrame,ScFrameReady
from fast_rgbd import Intrinsic,Extrinsic
from top_view import Vector

def main():
    bin_dir=PROJECT/'verification'/'bin';os.chdir(bin_dir);sdk=ScepterSdk(bin_dir)
    report={}
    try:
        report['session']=sdk.open('192.168.1.101',True,require_ai=False)
        sdk.base.scGetSensorIntrinsicParameters.argtypes=[c.c_void_p,c.c_int,c.POINTER(Intrinsic)]
        sdk.base.scGetSensorExtrinsicParameters.argtypes=[c.c_void_p,c.POINTER(Extrinsic)]
        values=[]
        for sensor in (1,2):
            value=Intrinsic();sdk._check(sdk.base.scGetSensorIntrinsicParameters(sdk.device,sensor,c.byref(value)),'read intrinsics')
            values.append({k:getattr(value,k) for k,_ in Intrinsic._fields_})
        extr=Extrinsic();sdk._check(sdk.base.scGetSensorExtrinsicParameters(sdk.device,c.byref(extr)),'read extrinsics')
        report.update(tof=values[0],color=values[1],rotation=list(extr.rotation),translation_mm=list(extr.translation))
        sdk.base.scConvertDepthFrameToPointCloudVector.argtypes=[c.c_void_p,c.POINTER(ScFrame),c.POINTER(Vector)]
        width,height=640,480
        constant=np.full((height,width),1000,np.uint16);cloud=np.empty((height,width,3),np.float32)
        frame=ScFrame();frame.frameType=0;frame.pixelFormat=0;frame.width=width;frame.height=height
        frame.dataLen=constant.nbytes;frame.pFrameData=constant.ctypes.data_as(c.POINTER(c.c_uint8))
        sdk._check(sdk.base.scConvertDepthFrameToPointCloudVector(sdk.device,c.byref(frame),cloud.ctypes.data_as(c.POINTER(Vector))),'read manufacturer rays')
        np.testing.assert_allclose(cloud[:,:,2],1000,atol=1e-3)
        rays=np.ascontiguousarray(cloud[:,:,:2]/1000,dtype='<f4')
        ready=ScFrameReady();actual=ScFrame()
        for _ in range(30):
            status=sdk.base.scGetFrameReady(sdk.device,100,c.byref(ready))
            if status==0 and ready.bits&1:break
        else:raise RuntimeError('No raw depth frame for calibration verification')
        sdk._check(sdk.base.scGetFrame(sdk.device,0,c.byref(actual)),'read raw depth')
        if (actual.width,actual.height)!=(width,height):raise RuntimeError('Unexpected ToF resolution')
        depth=np.frombuffer(c.string_at(actual.pFrameData,width*height*2),np.uint16).reshape(height,width)
        measured=np.empty_like(cloud)
        sdk._check(sdk.base.scConvertDepthFrameToPointCloudVector(sdk.device,c.byref(actual),measured.ctypes.data_as(c.POINTER(Vector))),'verify depth rays')
        valid=(depth>0)&(depth<65535)
        expected=np.dstack((rays*depth[:,:,None],depth.astype(np.float32)))
        error=float(np.max(np.abs(expected[valid]-measured[valid])))
        if error>.002:raise RuntimeError(f'Depth rays are not linear: {error}mm')
        report.update(width=width,height=height,verified_points=int(valid.sum()),max_xyz_error_mm=error,
                      camera_parameters_modified=False,camera_ai_started=False)
        rays.tofile(ROOT/'depth_rays_xy_float32.bin')
        np.savez_compressed(ROOT/'depth_ray_verification.npz',raw_depth_mm=depth,cloud_mm=measured)
    finally:
        report['close_statuses']=sdk.close()
        (ROOT/'calibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='session'},indent=2))

if __name__=='__main__':main()
