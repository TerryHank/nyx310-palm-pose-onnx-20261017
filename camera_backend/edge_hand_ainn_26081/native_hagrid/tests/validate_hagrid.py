"""Production C++ tests with host-only ONNX Runtime behind the RKNN ABI."""
import ctypes as c
import json
import shutil
import struct
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT.parent/'hagrid'
sys.path.insert(0,str(TASK))
from export_yolo import decode,letterbox

P=c.c_void_p
INFER=c.CFUNCTYPE(c.c_int,c.c_int,P,c.c_int,c.POINTER(c.POINTER(c.c_float)),c.POINTER(c.c_uint32))
RESULT=c.CFUNCTYPE(None,c.c_uint64,P,c.c_uint32)
def ptr(a):return a.ctypes.data_as(P)


def main():
    cv2.setNumThreads(1)
    replay=c.CDLL(str(ROOT/'build-host/libreplay.so'))
    alg=c.CDLL(str(ROOT/'build-host/libalg.so'))
    mock=c.CDLL(str(ROOT/'build-host/librknn_mock.so'))
    replay.core_gesture_decode.argtypes=[c.POINTER(c.POINTER(c.c_float)),c.c_int,c.c_int,P]
    replay.core_gesture_input.argtypes=[P,c.c_int,c.c_int,P]
    cases=[]
    report=dict(hardware_test=False,host_npu_transport='ONNX Runtime, not NPU hardware',preprocess=[],decode=[],integration={})
    for path in sorted((TASK/'fixtures').glob('yolo_*.npz')):
        f=np.load(path);bgr=f['bgr'];h,w=bgr.shape[:2]
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        out=np.empty((480,640,3),np.uint8)
        assert replay.core_gesture_input(ptr(rgb),w,h,ptr(out))==0
        expected,ratio,left,top=letterbox(bgr)
        error=int(np.max(np.abs(out.astype(int)-expected.astype(int))))
        assert error==0,(path,error)
        report['preprocess'].append(dict(sample=path.name,max_pixel_error=error))
        tensors=[np.ascontiguousarray(f[f'out_{i}'],np.float32) for i in range(6)]
        pointers=(c.POINTER(c.c_float)*6)(*[a.ctypes.data_as(c.POINTER(c.c_float)) for a in tensors])
        decoded=np.empty((2,6),np.float64)
        n=replay.core_gesture_decode(pointers,w,h,ptr(decoded))
        reference=decode(tensors,bgr.shape,ratio,left,top)
        assert n==len(reference),(path,n,reference)
        expected_rows=np.array([v['xyxy']+[v['confidence'],v['class_id']] for v in reference],np.float64).reshape(n,6)
        max_error=float(np.max(np.abs(decoded[:n]-expected_rows))) if n else 0
        assert max_error<1e-4,(path,max_error)
        report['decode'].append(dict(sample=path.name,count=n,max_abs_error=max_error))
        cases.append((w,h,expected_rows,tensors))
    # No detections is a first-class result, not a fabricated hand.
    zeros=[np.zeros((1,64 if i%2==0 else 34,60>>(i//2),80>>(i//2)),np.float32) for i in range(6)]
    pointers=(c.POINTER(c.c_float)*6)(*[a.ctypes.data_as(c.POINTER(c.c_float)) for a in zeros])
    assert replay.core_gesture_decode(pointers,800,600,ptr(np.empty((2,6))))==0
    cases.append((800,600,np.empty((0,6)),zeros))
    with (ROOT/'hagrid_fixtures.bin').open('wb') as file:
        file.write(struct.pack('<I',len(cases)))
        for w,h,expected,tensors in cases:
            file.write(struct.pack('<iii',w,h,len(expected)))
            file.write(expected.astype('<f8').tobytes())
            for tensor in tensors:file.write(tensor.astype('<f4').tobytes())

    stage=ROOT/'host-stage'
    (stage/'config').mkdir(parents=True,exist_ok=True);(stage/'model').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'hand.json',stage/'config/hand.json')
    shutil.copy2(ROOT.parent/'lowmem/depth_rays_xy_float32.bin',stage/'config/depth_rays.bin')
    for src,dst in [(TASK/'hagrid_rv1126b_fp16.rknn','hagrid.rknn'),
                    (ROOT.parent/'output/palm_detection_mediapipe_2023feb_rv1126b_fp16.rknn','palm.rknn'),
                    (ROOT.parent/'output/handpose_estimation_mediapipe_2023feb_rv1126b_fp16.rknn','handpose.rknn')]:
        shutil.copy2(src,stage/'model'/dst)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    sessions=[ort.InferenceSession(str(ROOT.parent/'source'/name),sess_options=options,providers=['CPUExecutionProvider'])
              for name in ['palm_detection_mediapipe_2023feb.onnx','handpose_estimation_mediapipe_2023feb.onnx']]
    sessions.append(ort.InferenceSession(str(TASK/'hagrid_ir9.onnx'),sess_options=options,providers=['CPUExecutionProvider']))
    rows=[];errors=[];calls=[];event=threading.Event()
    empty=False
    @INFER
    def infer(kind,input_pointer,size,outputs,counts):
        try:
            height=480 if kind==2 else size
            raw=np.ctypeslib.as_array(c.cast(input_pointer,c.POINTER(c.c_uint8)),shape=(height*size*3,)).reshape(1,height,size,3)
            x=raw.astype(np.float32)/255
            if kind==2:x=x.transpose(0,3,1,2)
            values=sessions[kind].run(None,{sessions[kind].get_inputs()[0].name:x})
            for i,value in enumerate(values):
                target=np.ctypeslib.as_array(outputs[i],shape=(counts[i],))
                target[:]=0 if kind==2 and empty else value.ravel()
            calls.append(kind);return 0
        except Exception as error:errors.append(str(error));return -1
    @RESULT
    def receive(stamp,text,length):
        try:rows.append(dict(timestamp=int(stamp),data=json.loads(c.string_at(text,length).rstrip(b'\0'))))
        except Exception as error:errors.append(str(error))
        event.set()
    mock.mock_set_callback.argtypes=[INFER];mock.mock_set_callback(infer)
    replay.replay_initialize.argtypes=[c.c_char_p,RESULT]
    replay.replay_push_raw.argtypes=[P,c.c_int,c.c_int,P,c.c_int,c.c_int,c.c_uint64,c.c_uint64,P]
    assert replay.replay_initialize(str(stage).encode(),receive)==0
    assert mock.mock_context_count()==3
    assert alg.algo_ctrl_start()==0
    calibration=json.loads((ROOT.parent/'lowmem/calibration.json').read_text())
    color,tof=calibration['color'],calibration['tof']
    lens=np.zeros(20,np.float32)
    lens[:4]=[tof[n] for n in ('cx','cy','fx','fy')]
    lens[4:8]=[color[n] for n in ('cx','cy','fx','fy')]
    # Use the real calibration and synthetic 800mm raw ToF to isolate the port.
    lens[8:17]=np.asarray(calibration['rotation']).ravel()
    lens[17:20]=calibration['translation_mm']
    depth=np.full((480,640),800,np.uint16)
    fixtures=[np.load(TASK/'fixtures'/f'yolo_{i}.npz') for i in range(4)]
    stamp=1000
    def push(index,matched=True,modified_lens=None):
        nonlocal stamp
        bgr=cv2.resize(fixtures[index]['bgr'],(800,600))
        rgb=np.ascontiguousarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        stamp+=70;event.clear()
        assert replay.replay_push_raw(ptr(rgb),800,600,ptr(depth),640,480,stamp,stamp if matched else stamp+1,ptr(lens if modified_lens is None else modified_lens))==0
        assert event.wait(15),'native result timed out'
        assert not errors,errors
        return rows[-1]['data']
    selected=None
    for i in range(4):
        response=push(i)
        if any(h['valid'] for h in response['hands']):selected=i;break
    assert selected is not None,('No replay palm yielded valid synthetic-depth XYZ',rows)
    valid=push(selected)
    assert any(h['valid'] for h in valid['hands'])
    expected_labels=set(json.loads((TASK/'export_report.json').read_text())['classes'].values())
    assert all(h['state'] in expected_labels and h['class_id']>=0 for h in valid['hands'])
    mismatched=push(selected,False)
    assert not any(h['valid'] for h in mismatched['hands']) and any(h['xyz_mm'] is not None for h in mismatched['hands'])
    bad=lens.copy();bad[2]*=2
    assert not any(h['valid'] for h in push(selected,modified_lens=bad)['hands'])
    for _ in range(3):push(selected)
    baseline=psutil.Process().memory_info().rss;peak=baseline
    for _ in range(12):
        push(selected);peak=max(peak,psutil.Process().memory_info().rss)
    assert peak-baseline<32*1024*1024
    empty=True
    held=push(selected)
    assert not any(h['valid'] for h in held['hands']) and any(h['xyz_mm'] is not None for h in held['hands'])
    assert all(h['raw_class_name'] is None for h in held['hands'])
    assert alg.algo_ctrl_stop()==0 and alg.algo_release()==0 and mock.mock_context_count()==0
    report['integration']=dict(frames=len(rows),three_models_called=set(calls)=={0,1,2},
                               sample_index=selected,valid_example=valid,hold_example=held,
                               timestamp_mismatch_rejected=True,bad_calibration_rejected=True,
                               host_warm_rss_growth_bytes=peak-baseline,
                               depth_source='synthetic raw ToF800mm; real RGB and calibration',callbacks=rows)
    report['passed']=True
    (ROOT/'hagrid_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='integration'},indent=2),flush=True)
    print('INTEGRATION_FRAMES',len(rows),'RSS_GROWTH',peak-baseline,flush=True)


if __name__=='__main__':main()
