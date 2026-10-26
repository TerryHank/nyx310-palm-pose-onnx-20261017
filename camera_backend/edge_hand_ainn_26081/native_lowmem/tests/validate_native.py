"""Offline tests of the production C++ code. Host RKNN transport uses ONNX Runtime.
No camera access and no claim that CPU mock inference is NPU hardware acceptance.
"""
import ctypes as c
import json
import sys
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'verification'/'python'))
from zoo_palm_pose import PalmPose
from hand_grasp import classify_grasp
from palm_position import measure_palm

P=c.c_void_p
infer_type=c.CFUNCTYPE(c.c_int,c.c_int,P,c.c_int,c.POINTER(c.POINTER(c.c_float)),c.POINTER(c.c_uint32))
result_type=c.CFUNCTYPE(None,c.c_uint64,P,c.c_uint32)
def ptr(a):return a.ctypes.data_as(P)


def main():
    cv2.setNumThreads(1)
    lib=c.CDLL(str(ROOT/'build-host'/'libreplay.so'))
    alg=c.CDLL(str(ROOT/'build-host'/'libalg.so'))
    mock=c.CDLL(str(ROOT/'build-host'/'librknn_mock.so'))
    prototypes={
        'core_palm_input':[P,c.c_int,c.c_int,P,P],
        'core_palm_decode':[P,P,c.c_int,c.c_int,P,P],
        'core_pose_input':[P,c.c_int,c.c_int,P,P,P],
        'core_pose_decode':[P,c.c_float,c.c_float,P,P,c.c_double,P],
        'core_classify':[P,P,P],
        'core_measure':[P,P,c.c_int,c.c_int,P,P,c.c_int,P],
        'replay_initialize':[c.c_char_p,result_type],
        'replay_push':[P,P,c.c_int,c.c_int,c.c_uint64,c.c_uint64,P],
    }
    for name,args in prototypes.items():getattr(lib,name).argtypes=args;getattr(lib,name).restype=c.c_int
    for name in ('algo_ctrl_start','algo_ctrl_stop','algo_release'):getattr(alg,name).restype=c.c_int
    mock.mock_set_callback.argtypes=[infer_type]
    mock.mock_fail_next.argtypes=[c.c_int]
    report={'hardware_test':False,'npu_backend':'host-only ONNX Runtime transport; same production C++ API/core',
            'preprocess':[],'classification':[],'depth':[],'lifecycle':[],'callbacks':[]}
    arm_classification=[];arm_depth=[];arm_preprocess=[]
    pipeline=PalmPose()
    options=ort.SessionOptions();options.intra_op_num_threads=2
    sessions=[ort.InferenceSession(str(ROOT.parent/'source'/name),sess_options=options,providers=['CPUExecutionProvider'])
              for name in ('palm_detection_mediapipe_2023feb.onnx','handpose_estimation_mediapipe_2023feb.onnx')]
    samples=json.loads((ROOT.parent/'tests'/'samples.json').read_text())
    images=[]
    def image_path(p):return Path(p.replace('D:\\','/mnt/d/').replace('\\','/'))
    for sample in samples:
        path=image_path(sample['image'])
        if path not in images:images.append(path)
    for path in images[:12]:
        image=cv2.imread(str(path));h,w=image.shape[:2]
        small=cv2.resize(image,(640,round(h*640/w)));rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
        palm_raw=np.empty((192,192,3),np.uint8);bias=np.zeros(2,np.float64)
        assert lib.core_palm_input(ptr(rgb),rgb.shape[1],rgb.shape[0],ptr(palm_raw),ptr(bias))==0
        want,want_bias=pipeline.palm._preprocess(small)
        error=float(np.max(np.abs(palm_raw.astype(float)/255-want[0])))
        assert error<1e-6,(path,'palm input',error)
        np.testing.assert_equal(bias,want_bias)
        output=sessions[0].run(None,{sessions[0].get_inputs()[0].name:want})
        decoded=np.empty((2,19),np.float64)
        n=lib.core_palm_decode(ptr(output[0]),ptr(output[1]),640,small.shape[0],ptr(bias),ptr(decoded))
        assert n>=0
        # True XYXY IoU NMS in native code (the original Zoo wrapper passes XYXY
        # to an XYWH API); compare the top palm which is unaffected by that bug.
        reference=pipeline.palm._postprocess(output,np.array([640,small.shape[0]]),want_bias)
        if len(reference) and n:
            np.testing.assert_allclose(decoded[0],reference[0],atol=5e-4)
        record={'image':str(path),'palm_input_max_abs':error,'native_palms':n,'reference_palms':len(reference),'pose_crops':[]}
        for palm in decoded[:n]:
            pose_raw=np.empty((224,224,3),np.uint8);meta=np.empty(13,np.float64)
            assert lib.core_pose_input(ptr(rgb),640,small.shape[0],ptr(palm),ptr(pose_raw),ptr(meta))==0
            tensor,bbox,angle,rotation,pad=pipeline.pose._preprocess(small,palm.copy())
            want_raw=np.rint(tensor[0]*255).astype(np.uint8)
            diff=np.abs(pose_raw.astype(float)-want_raw.astype(float))
            record['pose_crops'].append(dict(max_abs=float(diff.max()),mean_abs=float(diff.mean())))
            assert diff.mean()<=.5 and diff.max()<=3,(path,'pose input',diff.max(),diff.mean(),meta,bbox,angle,pad)
            arm_preprocess.append((rgb.copy(),palm.copy(),palm_raw.copy(),want_raw.copy()))
            out=sessions[1].run(None,{sessions[1].get_inputs()[0].name:tensor})
            native=np.empty(128,np.float64)
            result=lib.core_pose_decode(ptr(out[0]),float(out[1][0,0]),float(out[2][0,0]),ptr(out[3]),ptr(meta),float(palm[-1]),ptr(native))
            reference_pose=pipeline.pose._postprocess(out,bbox,angle,rotation,pad)
            if reference_pose is None:assert result==-1;continue
            assert result==0
            np.testing.assert_allclose(native[:63],reference_pose[4:67],atol=1e-3)
            np.testing.assert_allclose(native[63:126],reference_pose[67:130],atol=1e-6)
        report['preprocess'].append(record)
    # Stored real model outputs, then arbitrary 3-D rotations, translations,
    # handedness mirrors and scale. This verifies the port, not image robustness.
    rng=np.random.default_rng(42)
    count=0;max_error=0
    for sample in sorted((ROOT.parent/'tests').glob('pose_*.npz')):
        data=np.load(sample);world=data['output_3'].reshape(21,3).astype(float);points=data['output_0'].reshape(21,3).astype(float)
        for i in range(25):
            rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
            if i%2:rotation[:,0]*=-1
            q=np.ascontiguousarray(world@rotation*rng.uniform(.02,500)+rng.normal(size=3)*10)
            p=np.ascontiguousarray(points@rotation*rng.uniform(.02,500)+rng.normal(size=3)*10)
            expected,info=classify_grasp(q,p,.95);scores=np.empty(3,np.float64)
            state=lib.core_classify(ptr(q),ptr(p),ptr(scores));assert 0<=state<3;assert ('OPEN','GRASP','CLOSED')[state]==expected
            error=float(np.max(np.abs(scores-np.array(list(info['scores'].values())))));assert error<1e-9,(error,scores,info['scores'])
            max_error=max(max_error,error);count+=1
            arm_classification.append((q,p,state,scores.copy()))
    report['classification']=dict(cases=count,max_score_abs_error=max_error)
    # Exact depth rejection behavior: zero, sparse fragments, mismatched time,
    # mixed surfaces, invalid/out-of-range samples, and normal depth with outliers.
    points=np.zeros((21,3),np.float64)
    for i,xy in {0:(100,140),1:(65,120),5:(65,65),9:(100,55),13:(125,65),17:(140,85)}.items():points[i,:2]=xy
    K=np.array([[200.,0,100],[0,200.,100],[0,0,1]]);D=np.zeros(8)
    for name in ('valid','missing','sparse','mixed','mismatch','below50','above5000','near200'):
        depth=np.full((200,200),600,np.uint16);matched=name!='mismatch'
        if name=='missing':depth[:]=0
        if name=='sparse':depth[:]=0;depth[90:92,90:98]=600
        if name=='mixed':depth[:,::2]=2000
        if name=='below50':depth[:]=50
        if name=='above5000':depth[:]=5000
        if name=='near200':depth[:]=200
        if name=='valid':depth[::7,::7]=2200;depth[100,100]=0
        expected,info=measure_palm(points,depth,K,D,matched)
        actual=np.empty(3,np.float64);ok=lib.core_measure(ptr(points),ptr(depth),200,200,ptr(K),ptr(D),int(matched),ptr(actual))
        assert bool(ok)==(expected is not None),(name,info,ok)
        if ok:np.testing.assert_allclose(actual,expected,atol=1e-9)
        report['depth'].append(dict(case=name,valid=bool(ok)))
        arm_depth.append((points.copy(),depth.copy(),K.copy(),D.copy(),matched,expected))
    for sample in sorted((PROJECT/'verification'/'captures').glob('nyx310-palm-position-*/valid_depth_sample.npz'),reverse=True)[:3]:
        data=np.load(sample);raw=np.asarray(data['depth_mm']);depth=np.where(np.isfinite(raw)&(raw>0)&(raw<65535),raw,0).astype(np.uint16)
        points=np.asarray(data['points'],np.float64);K=np.asarray(data['K'],np.float64);D=np.asarray(data['D'],np.float64)
        expected,info=measure_palm(points,depth,K,D,True);actual=np.zeros(3,np.float64)
        h,w=depth.shape;ok=lib.core_measure(ptr(points),ptr(depth),w,h,ptr(K),ptr(D),1,ptr(actual))
        assert bool(ok)==(expected is not None),(sample,info)
        if ok:np.testing.assert_allclose(actual,expected,atol=1e-6)
        report['depth'].append(dict(case='recorded_RGBD',sample=str(sample),valid=bool(ok)))
        arm_depth.append((points.copy(),depth.copy(),K.copy(),D.copy(),True,expected))
    # Exercise the actual algo_* C++ API and worker queue with a host NPU seam.
    # Replay saved RGB only, then synthetic aligned depth to isolate pipeline
    # behavior. Synthetic depth is never presented as a measured camera result.
    event=threading.Event();callback_rows=[];infer_calls=[];callback_errors=[]
    @infer_type
    def infer(kind,input_pointer,size,outputs,counts):
        try:
            raw=np.ctypeslib.as_array(c.cast(input_pointer,c.POINTER(c.c_uint8)),shape=(size*size*3,)).reshape(1,size,size,3)
            tensor=raw.astype(np.float32)/255
            result=sessions[kind].run(None,{sessions[kind].get_inputs()[0].name:tensor})
            for i,value in enumerate(result):np.ctypeslib.as_array(outputs[i],shape=(counts[i],))[:]=value.ravel()
            infer_calls.append(kind);return 0
        except Exception as error:callback_errors.append(str(error));return -1
    @result_type
    def receive(timestamp,text,length):
        try:callback_rows.append(dict(timestamp=int(timestamp),result=json.loads(c.string_at(text,length).rstrip(b'\0'))))
        except Exception as error:callback_errors.append(str(error))
        event.set()
    mock.mock_set_callback(infer)
    stage=ROOT/'stage'/'nyx310'/'ai_common_app_install'
    assert lib.replay_initialize(str(stage).encode(),receive)==0
    assert mock.mock_context_count()==2
    assert alg.algo_ctrl_start()==0
    alg.algo_get_param.argtypes=[P,c.c_uint16,c.POINTER(c.c_char_p),c.POINTER(c.c_uint16)]
    alg.algo_set_param.argtypes=[P,c.c_uint16,P,P]
    def get_param(identifier):
        input_value=c.c_uint32(identifier);output_value=c.c_char_p();length=c.c_uint16()
        status=alg.algo_get_param(c.byref(input_value),4,c.byref(output_value),c.byref(length))
        assert status==0
        return c.string_at(output_value,length.value).rstrip(b'\0').decode()
    assert get_param(0)=='26.8.1-hand.2' and get_param(1)=='PalmPoseGraspXYZ'
    for identifier,key,value in ((0x21,'Conf_Thresh',.65),(0x22,'IOU_Thresh',.35)):
        payload=struct.pack('<I',identifier)+json.dumps({key:value}).encode()
        buffer=c.create_string_buffer(payload)
        assert alg.algo_set_param(buffer,len(payload),None,None)==0
        assert json.loads(get_param(identifier))[key]==value
        payload=struct.pack('<I',identifier)+json.dumps({key:.6 if identifier==0x21 else .3}).encode()
        assert alg.algo_set_param(c.create_string_buffer(payload),len(payload),None,None)==0
    assert alg.algo_get_param(None,0,None,None)==-3
    assert alg.algo_set_param(None,0,None,None)==-3
    report['lifecycle'].append(dict(case='parameter_ABI_get_set_and_invalid_pointers',passed=True))
    cfg=json.loads((ROOT/'hand.json').read_text());lens=np.zeros(20,np.float32)
    lens[4:8]=cfg['ExpectedColorIntrinsics'];lens[8:17]=np.eye(3).ravel()
    stamp=1000
    image=cv2.imread(str(images[0]));rgb=np.ascontiguousarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));h,w=rgb.shape[:2]
    depth=np.full((h,w),800,np.uint16)
    def push(rgb_value,depth_value,time_offset=0,lens_value=lens):
        nonlocal stamp
        event.clear();stamp+=70
        ret=lib.replay_push(ptr(rgb_value),ptr(depth_value) if depth_value is not None else None,w,h,stamp,stamp+time_offset,ptr(lens_value))
        assert ret==0,ret
        assert event.wait(10),'native callback timeout'
        assert not callback_errors,callback_errors
        row=callback_rows[-1];assert row['timestamp']==stamp
        for hand in row['result']['hands']:
            assert set(hand)=={'hand_id','xyz_mm','state','confidence','valid'}
            assert hand['state'] in ('OPEN','GRASP','CLOSED')
        return row
    normal=push(rgb,depth)
    assert normal['result']['hands'] and any(v['valid'] for v in normal['result']['hands'])
    for mode in ('zero_depth','timestamp_mismatch','rgb_only','calibration_mismatch'):
        bad_lens=lens.copy();bad_lens[6]=1
        row=push(rgb,np.zeros_like(depth) if mode=='zero_depth' else None if mode=='rgb_only' else depth,
                 1 if mode=='timestamp_mismatch' else 0,bad_lens if mode=='calibration_mismatch' else lens)
        assert all(not v['valid'] for v in row['result']['hands']),mode
        assert any(v['xyz_mm'] is not None for v in row['result']['hands']),mode
        report['lifecycle'].append(dict(case=mode,hold_validated=True))
    for operation in (1,2,3):
        mock.mock_fail_next(operation);row=push(rgb,depth)
        assert row['result'].get('error')=='inference_failed'
        assert all(not v['valid'] for v in row['result']['hands'])
        recovery=push(rgb,depth);assert 'error' not in recovery['result']
        report['lifecycle'].append(dict(case='NPU_API_error_'+str(operation),recovered=True))
    assert lib.replay_push(ptr(rgb),ptr(depth),w,h,stamp,stamp,ptr(lens))!=0
    assert alg.algo_push_frame(None)!=0
    for repeat in range(3):assert alg.algo_ctrl_stop()==0;assert alg.algo_ctrl_start()==0
    assert alg.algo_ctrl_stop()==0
    assert lib.replay_push(ptr(rgb),ptr(depth),w,h,stamp+1,stamp+1,ptr(lens))!=0
    assert alg.algo_release()==0
    assert mock.mock_context_count()==0
    assert alg.algo_ctrl_start()==-8
    report['callbacks']=callback_rows
    report['infer_calls']=dict(palm=infer_calls.count(0),pose=infer_calls.count(1))
    assert report['infer_calls']['palm']>0 and report['infer_calls']['pose']>0
    report['passed']=True
    with (ROOT/'arm_fixtures.bin').open('wb') as f:
        f.write(struct.pack('<III',len(arm_classification),len(arm_depth),len(arm_preprocess)))
        for world,points,state,scores in arm_classification:
            f.write(world.astype('<f8').tobytes());f.write(points.astype('<f8').tobytes());f.write(struct.pack('<i',state));f.write(scores.astype('<f8').tobytes())
        for points,depth,K,D,matched,expected in arm_depth:
            h,w=depth.shape;f.write(struct.pack('<iiii',w,h,int(matched),int(expected is not None)))
            for value in (points,K,D,np.zeros(3) if expected is None else expected):f.write(np.asarray(value,dtype='<f8').tobytes())
            f.write(depth.astype('<u2').tobytes())
        for rgb,palm,palm_raw,pose_raw in arm_preprocess:
            h,w=rgb.shape[:2];f.write(struct.pack('<ii',w,h));f.write(rgb.tobytes());f.write(palm.astype('<f8').tobytes());f.write(palm_raw.tobytes());f.write(pose_raw.tobytes())
    (ROOT/'native_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='callbacks'},indent=2),flush=True)


if __name__=='__main__':main()
