"""AINN 26.08.1 RV1126B model conversion and CPU-simulator tensor regression."""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import onnx
import onnxruntime as ort
from rknn.api import RKNN

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'verification'/'python'))
from zoo_palm_pose import PalmPose
from hand_grasp import classify_grasp
MODELS = [('palm','palm_detection_mediapipe_2023feb'),
          ('pose','handpose_estimation_mediapipe_2023feb')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = dict(target='rv1126b', toolkit='2.3.2', quantization=False,
                  camera_test_executed=False, upload_executed=False, models=[])
    postprocessor=PalmPose()
    for kind,name in MODELS:
        source = ROOT/'source'/(name+'.onnx')
        output = ROOT/'output'/(name+'_rv1126b_fp16.rknn')
        model_report = dict(name=name, source_sha256=digest(source), steps={}, replay=[])
        report['models'].append(model_report)
        rknn = RKNN(verbose=False)
        try:
            model = onnx.load(str(source))
            onnx.checker.check_model(model)
            model_report['input'] = [dict(name=x.name, shape=[d.dim_value for d in x.type.tensor_type.shape.dim]) for x in model.graph.input]
            model_report['outputs'] = [dict(name=x.name,shape=[d.dim_value for d in x.type.tensor_type.shape.dim]) for x in model.graph.output]
            # Toolkit interprets ONNX inputs as NCHW. Preserve the original
            # NHWC graph behind an explicit, mathematically exact input adapter.
            old_name=model.graph.input[0].name
            _,height,width,channels=model_report['input'][0]['shape']
            model.graph.input[0].name='rgb_nchw'
            for dim,value in zip(model.graph.input[0].type.tensor_type.shape.dim,
                                 (1,channels,height,width)):
                dim.dim_value=value
            model.graph.node.insert(0,onnx.helper.make_node(
                'Transpose',['rgb_nchw'],[old_name],perm=[0,2,3,1],name='AINN_input_NCHW_to_NHWC'))
            adapted=ROOT/'output'/(name+'_nchw.onnx')
            onnx.checker.check_model(model)
            onnx.save(model,str(adapted))
            model_report.update(adapted_onnx=adapted.name,adapted_sha256=digest(adapted),
                                runtime_input='uint8 RGB NHWC; rknn input fmt NHWC; normalization /255 embedded')
            def step(name,ret):
                model_report['steps'][name] = ret
                print(kind,name,ret,flush=True)
                if ret != 0:
                    raise RuntimeError(f'{kind} {name} returned {ret}')
            step('config',rknn.config(target_platform='rv1126b',mean_values=[[0,0,0]],
                                     std_values=[[255,255,255]],optimization_level=3))
            step('load_onnx',rknn.load_onnx(model=str(adapted)))
            step('build',rknn.build(do_quantization=False))
            step('export',rknn.export_rknn(str(output)))
            model_report.update(output=str(output.name),size=output.stat().st_size,
                                sha256=digest(output))
            step('simulator_init',rknn.init_runtime())
            options=ort.SessionOptions()
            options.intra_op_num_threads=2
            session=ort.InferenceSession(str(source),sess_options=options,
                                         providers=['CPUExecutionProvider'])
            adapted_session=ort.InferenceSession(str(adapted),sess_options=options,
                                                providers=['CPUExecutionProvider'])
            for sample in sorted((ROOT/'tests').glob(kind+'_*.npz')):
                data=np.load(sample)
                expected=session.run(None,{session.get_inputs()[0].name:data['input']})
                adapter_outputs=adapted_session.run(None,{'rgb_nchw':data['input'].transpose(0,3,1,2)})
                adapter_max=max(float(np.max(np.abs(a-b))) for a,b in zip(expected,adapter_outputs))
                if adapter_max > 1e-4:
                    raise RuntimeError('Input layout adaptation changed source ONNX results')
                actual=rknn.inference(inputs=[data['raw_rgb']],data_format=['nhwc'])
                if actual is None or len(actual)!=len(expected):
                    raise RuntimeError('Simulator output count mismatch')
                metrics=[]
                for i,(want,got) in enumerate(zip(expected,actual)):
                    if want.size != got.size:
                        raise RuntimeError('Output element count mismatch')
                    got=np.asarray(got).reshape(want.shape)
                    error=np.abs(want-got)
                    # Compare source ONNX with the current PC/OpenCV implementation too.
                    cv_error=float(np.max(np.abs(want-data[f'output_{i}'])))
                    metrics.append(dict(index=i,shape=list(want.shape),
                        max_abs=float(error.max()),mean_abs=float(error.mean()),
                        opencv_onnx_max_abs=cv_error))
                if kind=='palm':
                    decode=lambda outputs: postprocessor.palm._postprocess(
                        outputs,data['original_shape'],data['pad_bias'])
                    ref,got=decode(expected),decode(actual)
                    function=dict(reference_palms=len(ref),rknn_palms=len(got),
                                  count_match=len(ref)==len(got))
                else:
                    decode=lambda outputs: postprocessor.pose._postprocess(
                        outputs,data['rotated_bbox'],float(data['angle']),
                        data['rotation'],data['pad_bias'])
                    ref,got=decode(expected),decode(actual)
                    if ref is None or got is None:
                        raise RuntimeError('Reference or RKNN pose lost in replay')
                    ref_class,_=classify_grasp(ref[67:130].reshape(21,3),ref[4:67].reshape(21,3),float(ref[-1]))
                    got_class,_=classify_grasp(got[67:130].reshape(21,3),got[4:67].reshape(21,3),float(got[-1]))
                    function=dict(reference_class=ref_class,rknn_class=got_class,
                        class_match=ref_class==got_class,
                        decoded_xy_max_abs_pixels=float(np.max(np.abs(ref[4:67].reshape(21,3)[:,:2]-got[4:67].reshape(21,3)[:,:2]))))
                if kind=='palm':
                    limits=[.5,.1]
                else:
                    limits=[.5,.02,.02,.002]
                passed=all(m['max_abs']<=limit and m['opencv_onnx_max_abs']<=1e-3
                           for m,limit in zip(metrics,limits)) and function.get('class_match',function.get('count_match'))
                model_report['replay'].append(dict(sample=sample.name,outputs=metrics,
                    adapter_max_abs=adapter_max,functional=function,passed=bool(passed)))
                print(kind,'replay',sample.name,'passed',passed,function,flush=True)
            model_report['replay_passed']=all(s['passed'] for s in model_report['replay'])
        except Exception as error:
            model_report['failure']=f'{type(error).__name__}: {error}'
            print(model_report['failure'],flush=True)
        finally:
            rknn.release()
            (ROOT/'output'/'conversion_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if any('failure' in m or not m.get('replay_passed') for m in report['models']):
        raise SystemExit(2)


if __name__=='__main__':
    main()
