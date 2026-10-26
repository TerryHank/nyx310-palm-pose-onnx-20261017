"""Convert to manufacturer RKNN Toolkit2 2.3.2 RV1126B and verify real sample tensors."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import onnx
import onnxruntime as ort
from rknn.api import RKNN
from export_yolo import decode

ROOT=Path(__file__).resolve().parent


def main():
    report=dict(target='rv1126b',toolkit='2.3.2',quantized=False,camera_test_executed=False,
                steps={},samples=[])
    model=onnx.load(str(ROOT/'hagrid_raw.onnx'))
    # Newer ONNX exporters tag an IR version older ORT cannot load; the graph
    # only uses opset13 tensors/initializers, validated by checker after lowering.
    model.ir_version=9
    onnx.checker.check_model(model)
    onnx.save(model,str(ROOT/'hagrid_ir9.onnx'))
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(ROOT/'hagrid_ir9.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    rk=RKNN(verbose=False)
    def step(name,status):
        report['steps'][name]=status
        print(name,status,flush=True)
        if status:raise RuntimeError(f'{name}: {status}')
    try:
        step('config',rk.config(target_platform='rv1126b',mean_values=[[0,0,0]],std_values=[[255,255,255]],optimization_level=3))
        step('load',rk.load_onnx(model=str(ROOT/'hagrid_ir9.onnx')))
        step('build',rk.build(do_quantization=False))
        output=ROOT/'hagrid_rv1126b_fp16.rknn'
        step('export',rk.export_rknn(str(output)))
        report['rknn']=dict(size=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        step('simulator',rk.init_runtime())
        for path in sorted(p for p in (ROOT/'fixtures').glob('yolo_*.npz') if '_rknn' not in p.stem):
            f=np.load(path);x=f['rgb'].transpose(2,0,1)[None].astype(np.float32)/255
            reference=session.run(None,{'rgb':x})
            original=[f[f'out_{i}'] for i in range(6)]
            export_errors=[float(np.max(np.abs(a-b))) for a,b in zip(original,reference)]
            if max(export_errors)>.005:raise RuntimeError(f'ONNX tensor error {export_errors}')
            actual=rk.inference(inputs=[f['rgb'][None]],data_format=['nhwc'])
            if actual is None or len(actual)!=6:raise RuntimeError('RKNN output count')
            errors=[dict(max_abs=float(np.max(np.abs(a-b))),mean_abs=float(np.mean(np.abs(a-b))))
                    for a,b in zip(reference,actual)]
            ref=decode(reference,f['bgr'].shape,float(f['ratio']),int(f['left']),int(f['top']))
            got=decode(actual,f['bgr'].shape,float(f['ratio']),int(f['left']),int(f['top']))
            passed=len(ref)==len(got) and [d['class_id'] for d in ref]==[d['class_id'] for d in got]
            box_error=max((float(np.max(np.abs(np.array(a['xyxy'])-b['xyxy']))) for a,b in zip(ref,got)),default=0)
            score_error=max((abs(a['confidence']-b['confidence']) for a,b in zip(ref,got)),default=0)
            passed=passed and box_error<=3 and score_error<=.025
            item=dict(sample=path.name,export_errors=export_errors,rknn_tensor_errors=errors,
                      reference=ref,rknn=got,max_box_error_pixels=box_error,max_score_error=score_error,passed=passed)
            report['samples'].append(item)
            np.savez_compressed(ROOT/'fixtures'/(path.stem+'_rknn.npz'),rgb=f['rgb'],bgr=f['bgr'],
                                ratio=f['ratio'],left=f['left'],top=f['top'],**{f'out_{i}':v for i,v in enumerate(actual)})
            print(path.name,'passed',passed,'box_error',box_error,flush=True)
        report['passed']=bool(report['samples']) and all(s['passed'] for s in report['samples'])
        if not report['passed']:raise RuntimeError('functional simulator regression failed')
    finally:
        rk.release()
        (ROOT/'conversion_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
