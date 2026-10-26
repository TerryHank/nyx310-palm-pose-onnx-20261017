"""Read back exported RKNN files and assemble the model-only delivery."""
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path
from rknn.api import RKNN

ROOT=Path(__file__).resolve().parent


def main():
    report=json.loads((ROOT/'output'/'conversion_report.json').read_text())
    assert len(report['models'])==2
    destination=ROOT.parent/'dist'/'hand_models_ainn_26081'
    destination.mkdir(exist_ok=False)
    for model in report['models']:
        assert model['replay_passed'] and len(model['replay'])==12 and 'failure' not in model
        path=ROOT/'output'/model['output']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==model['sha256']
        verifier=RKNN(verbose=False)
        try:
            model['exported_file_reload_status']=verifier.load_rknn(str(path))
            assert model['exported_file_reload_status']==0
        finally:
            verifier.release()
        shutil.copy2(path,destination/path.name)
    report.update(
        delivery_type='MODEL_RESOURCES_ONLY_NOT_CAMERA_UPLOAD_PACKAGE',
        full_camera_application_implemented=False,
        camera_runtime_compatibility_verified=False,
        hardware_precompile_executed=False,
        environment={name:importlib.metadata.version(name) for name in
                     ('rknn-toolkit2','torch','onnx','onnxruntime','numpy','protobuf','opencv-python')})
    (destination/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    contract=dict(
        target='NYX310 / rv1126b', toolkit='AINN 26.08.1 supplied RKNN Toolkit2 2.3.2',
        precision='FP16 non-quantized',
        runtime_input=dict(type='UINT8',fmt='NHWC',pass_through=0,color='RGB',
                           normalization='(x-0)/255 embedded; do not divide twice'),
        palm=dict(size=[192,192],preprocess='aspect-preserving black padding, official OpenCV Zoo palm wrapper',
                  outputs=[['Identity',[1,2016,18]],['Identity_1',[1,2016,1]]],
                  postprocess='2016 anchors + sigmoid + bounding box/keypoint decode + NMS'),
        pose=dict(size=[224,224],preprocess='palm-driven rotation + square crop, official OpenCV Zoo hand wrapper',
                  outputs=[['Identity',[1,63]],['Identity_1',[1,1]],['Identity_2',[1,1]],['Identity_3',[1,63]]],
                  postprocess='21 relative screen landmarks, presence, handedness, 21 relative world landmarks; inverse crop/rotation'),
        native_runtime='query rknn tensor attributes; request float outputs with want_float=1; preserve named output order',
        not_in_models=['OPEN/GRASP/CLOSED rules','palm ROI depth median','metric RGB-D XYZ','position smoothing','HOLD','AI Lib callbacks'],
        integration='implement dual-model sequencing in custom libalg.so; stock YOLO libalg.so is incompatible')
    (destination/'model_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    readme=ROOT/'README.txt'
    shutil.copy2(readme,destination/readme.name)
    entries=[]
    for path in sorted(destination.iterdir()):
        if path.is_file():
            entries.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.name)
    (destination/'SHA256SUMS.txt').write_text('\n'.join(entries)+'\n',encoding='utf-8')
    print(destination,flush=True)
    print('\n'.join(entries),flush=True)


if __name__=='__main__':
    main()
