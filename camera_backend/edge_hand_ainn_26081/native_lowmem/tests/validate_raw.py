"""Independent OpenCV projection reference and raw-frame native integration."""
import ctypes as c
import json
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'verification'/'python'))
from palm_position import measure_palm
from validate_native import P,ptr,infer_type,result_type


def dense_reference(depth,rays,K,D,R,t,size):
    xyz=np.dstack((rays.astype(float)*depth[:,:,None],depth.astype(float))).reshape(-1,3)
    valid=((depth>0)&(depth<65535)).ravel()
    xyz=xyz[valid]@R.T+t
    valid=np.isfinite(xyz).all(axis=1)&(xyz[:,2]>50)&(xyz[:,2]<5000)
    xyz=xyz[valid]
    if not len(xyz):return np.zeros((size[1],size[0]),np.float32)
    uv,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,D)
    uv=np.rint(uv.reshape(-1,2)).astype(int)
    w,h=size;valid=(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<w)&(uv[:,1]<h)
    flat=np.full(w*h,np.inf,np.float32)
    np.minimum.at(flat,uv[valid,1]*w+uv[valid,0],xyz[valid,2].astype(np.float32))
    flat[~np.isfinite(flat)]=0
    return flat.reshape(h,w)


def main():
    cv2.setNumThreads(1)
    library=c.CDLL(str(ROOT/'build-host'/'libreplay.so'))
    alg=c.CDLL(str(ROOT/'build-host'/'libalg.so'))
    mock=c.CDLL(str(ROOT/'build-host'/'librknn_mock.so'))
    library.core_measure_raw.argtypes=[P,P,P,c.c_int,c.c_int,c.c_int,c.c_int,P,P,P,P,c.c_int,P]
    library.core_measure_raw.restype=c.c_int
    calibration=json.loads((ROOT.parent/'lowmem'/'calibration.json').read_text())
    rays=np.fromfile(ROOT.parent/'lowmem'/'depth_rays_xy_float32.bin',np.float32).reshape(480,640,2)
    color=calibration['color'];K=np.array([[color['fx']*.4,0,color['cx']*.4],[0,color['fy']*.4,color['cy']*.4],[0,0,1]],float)
    D=np.array([color[n] for n in ('k1','k2','p1','p2','k3','k4','k5','k6')])
    R=np.array(calibration['rotation']).reshape(3,3);t=np.array(calibration['translation_mm'])
    points=np.zeros((21,3),float)
    for i,xy in {0:(320,330),1:(265,300),5:(265,230),9:(320,210),13:(360,230),17:(385,265)}.items():points[i,:2]=xy
    fixtures=[];cases=[]
    for name in ('plane800','near200','plane1200','missing','invalid65535','flying','mixed','timestamp_mismatch','recorded'):
        depth=np.full((480,640),800,np.uint16);matched=name!='timestamp_mismatch'
        if name=='near200':depth[:]=200
        if name=='plane1200':depth[:]=1200
        if name=='missing':depth[:]=0
        if name=='invalid65535':depth[:]=65535
        if name=='flying':depth[::4,::4]=2500
        if name=='mixed':depth[:,::2]=1800
        if name=='recorded':depth=np.load(ROOT.parent/'lowmem'/'depth_ray_verification.npz')['raw_depth_mm']
        aligned=dense_reference(depth,rays,K,D,R,t,(640,480))
        expected,info=measure_palm(points,aligned,K,D,matched)
        output=np.empty(3,float)
        ok=library.core_measure_raw(ptr(points),ptr(depth),ptr(rays),640,480,640,480,ptr(K),ptr(D),ptr(R),ptr(t),int(matched),ptr(output))
        assert bool(ok)==(expected is not None),(name,ok,info)
        error=0. if not ok else float(np.max(np.abs(expected-output)))
        assert error<.002,(name,error,expected,output)
        cases.append(dict(case=name,valid=bool(ok),xyz_error_mm=error,
                          input_kind='recorded_depth_synthetic_palm_roi' if name=='recorded' else 'synthetic_depth'))
        fixtures.append((points.copy(),depth.copy(),K.copy(),D.copy(),R.copy(),t.copy(),matched,expected))
    # Integration uses the actual production API/queue with model execution on PC.
    options=ort.SessionOptions();options.intra_op_num_threads=2
    sessions=[ort.InferenceSession(str(ROOT.parent/'source'/n),sess_options=options,providers=['CPUExecutionProvider']) for n in
              ('palm_detection_mediapipe_2023feb.onnx','handpose_estimation_mediapipe_2023feb.onnx')]
    native_errors=[]
    @infer_type
    def infer(kind,address,size,outputs,counts):
        try:
            image=np.ctypeslib.as_array(c.cast(address,c.POINTER(c.c_uint8)),shape=(size*size*3,)).reshape(1,size,size,3)
            result=sessions[kind].run(None,{sessions[kind].get_inputs()[0].name:image.astype(np.float32)/255})
            for i,value in enumerate(result):np.ctypeslib.as_array(outputs[i],shape=(counts[i],))[:]=value.ravel()
            return 0
        except Exception as error:native_errors.append(str(error));return -1
    received=[];event=threading.Event()
    @result_type
    def receive(stamp,data,length):
        received.append(dict(timestamp=int(stamp),data=json.loads(c.string_at(data,length).rstrip(b'\0'))));event.set()
    mock.mock_set_callback.argtypes=[infer_type];mock.mock_set_callback(infer)
    library.replay_initialize.argtypes=[c.c_char_p,result_type]
    library.replay_push_raw.argtypes=[P,c.c_int,c.c_int,P,c.c_int,c.c_int,c.c_uint64,c.c_uint64,P]
    stage=ROOT/'stage'/'nyx310'/'ai_common_app_install'
    assert library.replay_initialize(str(stage).encode(),receive)==0
    assert alg.algo_ctrl_start()==0
    sample=json.loads((ROOT.parent/'tests'/'samples.json').read_text())[0]
    path=Path(sample['image'].replace('D:\\','/mnt/d/').replace('\\','/'))
    rgb=np.ascontiguousarray(cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB));h,w=rgb.shape[:2]
    depth=np.full((480,640),800,np.uint16)
    lens=np.zeros(20,np.float32)
    tof=calibration['tof'];lens[:4]=[tof[n] for n in ('cx','cy','fx','fy')]
    lens[4:8]=[color[n] for n in ('cx','cy','fx','fy')];lens[8:17]=R.ravel();lens[17:20]=t
    stamp=1000
    def push(matched=True,lens_value=lens):
        nonlocal stamp
        stamp+=70;event.clear()
        assert library.replay_push_raw(ptr(rgb),w,h,ptr(depth),640,480,stamp,stamp if matched else stamp+1,ptr(lens_value))==0
        assert event.wait(10),'raw result timeout'
        assert not native_errors,native_errors
        return received[-1]['data']
    first=push();assert first['hands'] and any(v['valid'] for v in first['hands']),first
    mismatch=push(False);assert all(not v['valid'] for v in mismatch['hands'])
    bad_lens=lens.copy();bad_lens[2]*=2
    invalid=push(True,bad_lens);assert all(not v['valid'] for v in invalid['hands'])
    for _ in range(5):push()
    process=psutil.Process();baseline=process.memory_info().rss;peak=baseline
    for _ in range(60):
        result=push();assert any(v['valid'] for v in result['hands'])
        peak=max(peak,process.memory_info().rss)
    growth=peak-baseline
    assert growth<32*1024*1024,('host RSS continued to grow',growth)
    assert alg.algo_ctrl_stop()==0;assert alg.algo_release()==0;assert mock.mock_context_count()==0
    with (ROOT/'raw_fixtures.bin').open('wb') as file:
        file.write(struct.pack('<I',len(fixtures)))
        file.write(rays.astype('<f4').tobytes())
        for points,depth,K,D,R,t,matched,expected in fixtures:
            file.write(struct.pack('<ii',int(matched),int(expected is not None)))
            for value in (points,K,D,R,t,np.zeros(3) if expected is None else expected):file.write(np.asarray(value,dtype='<f8').tobytes())
            file.write(depth.astype('<u2').tobytes())
    report=dict(passed=True,hardware_test=False,raw_registration_cases=cases,
                full_api_raw_frames=len(received),input_rgb_size=[w,h],input_depth_size=[640,480],
                retained_rgb_size=[640,round(h*640/w)],
                old_two_frame_storage_bytes=2*(w*h*3+w*h*2),
                new_two_frame_storage_bytes=2*(640*round(h*640/w)*3+640*480*2),
                calibrated_ray_table_bytes=rays.nbytes,host_warmed_RSS_growth_bytes=growth,
                callback_example=first,
                memory_limit_note='Host RSS is a bounded-queue regression; not a camera/NPU peak measurement.')
    (ROOT/'raw_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
