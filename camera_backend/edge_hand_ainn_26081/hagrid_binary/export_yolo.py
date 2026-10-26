"""Export the exact HaGRID branch used by the current PC loader; CPU DFL/NMS."""
import copy
import hashlib
import json
import os
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(PROJECT/'verification/local_checks/yoloe-config')
import cv2
import numpy as np


def raw_forward(self, features):
    outputs = []
    for i in range(3):
        outputs.extend([self.cv2[i](features[i]), self.cv3[i](features[i]).sigmoid()])
    return tuple(outputs)


def letterbox(image):
    h,w=image.shape[:2]
    ratio=min(416/w,320/h)
    width,height=round(w*ratio),round(h*ratio)
    resized=cv2.resize(image,(width,height))
    left,top=round((416-width)/2-.1),round((320-height)/2-.1)
    padded=cv2.copyMakeBorder(resized,top,320-height-top,left,416-width-left,
                             cv2.BORDER_CONSTANT,value=(114,114,114))
    return cv2.cvtColor(padded,cv2.COLOR_BGR2RGB),ratio,left,top


def decode(outputs, shape, ratio, left, top):
    boxes, probabilities=[],[]
    for raw,scores in zip(outputs[::2],outputs[1::2]):
        _,_,h,w=raw.shape
        logits=raw.reshape(1,4,16,h,w)
        p=np.exp(logits-logits.max(axis=2,keepdims=True))
        distances=(p/p.sum(axis=2,keepdims=True)*np.arange(16).reshape(1,1,16,1,1)).sum(axis=2)[0]
        gy,gx=np.mgrid[:h,:w]
        box=np.stack([gx+.5-distances[0],gy+.5-distances[1],
                      gx+.5+distances[2],gy+.5+distances[3]],axis=-1)*(416/w)
        boxes.append(box.reshape(-1,4))
        probabilities.append(scores[0].transpose(1,2,0).reshape(-1,34))
    boxes=np.concatenate(boxes);probabilities=np.concatenate(probabilities)
    # The current PC compatibility loader sets end2end=False on this checkpoint.
    labels=probabilities.argmax(axis=1)
    confidence=probabilities[np.arange(len(labels)),labels]
    rank=np.argsort(-confidence,kind='stable')
    result=[]
    selected=[]
    for anchor in rank:
        label=int(labels[anchor]);score=float(confidence[anchor])
        if score < .175:break
        a=boxes[anchor]
        suppressed=False
        for old in selected:
            if old['class_id']!=label:continue
            b=np.asarray(old['raw_box']);inter=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
            union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter
            if union>0 and inter/union>.7:suppressed=True;break
        if suppressed:continue
        selected.append(dict(raw_box=a.tolist(),class_id=label))
        box=(boxes[anchor]-[left,top,left,top])/ratio
        box=np.clip(box,[0,0,0,0],[shape[1],shape[0],shape[1],shape[0]])
        duplicate=False
        for other in result:
            b=np.array(other['xyxy']);inter=np.prod(np.maximum(0,np.minimum(box[2:],b[2:])-np.maximum(box[:2],b[:2])))
            union=np.prod(box[2:]-box[:2])+np.prod(b[2:]-b[:2])-inter
            if union>0 and inter/union>.8:duplicate=True
        if not duplicate:result.append(dict(class_id=label,confidence=score,xyxy=box.tolist()))
        if len(selected)==2:break
    return result


def main():
    import torch
    from ultralytics import YOLO
    ROOT.mkdir(exist_ok=True)
    (ROOT/'fixtures').mkdir(exist_ok=True)
    source=Path(r'C:\Users\user\Downloads\YOLOv10n_gestures.pt')
    original=YOLO(str(source))
    if original.model.model[-1].end2end:
        raise RuntimeError('PC loader branch changed; revalidate export branch')
    raw=copy.deepcopy(original.model).float().cpu().eval()
    raw.model[-1].forward=types.MethodType(raw_forward,raw.model[-1])
    dummy=torch.zeros(1,3,320,416)
    with torch.no_grad():outputs=raw(dummy)
    onnx_path=ROOT/'hagrid_raw.onnx'
    torch.onnx.export(raw,dummy,str(onnx_path),input_names=['rgb'],
                      output_names=[f'{kind}_{stride}' for stride in (8,16,32) for kind in ('dfl','scores')],
                      opset_version=13,dynamic_axes=None,dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sources=[PROJECT/'verification/captures/hands-final-offline/hand_sample_rgb.jpg',
             PROJECT/'verification/captures/hagrid-20261016-live-test/best_rgb.jpg',
             PROJECT/'verification/captures/hagrid-20261016-final-live/best_rgb.jpg']
    sample=PROJECT/'verification/captures/hagrid-20261016-final-live/position_sample.npz'
    report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                classes=original.names,shape=[1,3,320,416],one_to_one=False,
                outputs=[list(v.shape) for v in outputs],export='raw cv2/cv3 DFL logits + class sigmoid; DFL and class-aware NMS on CPU; matches PC end2end=False',
                samples=[])
    images=[cv2.imread(str(p)) for p in sources]
    if sample.exists():images.append(np.load(sample)['bgr'])
    for i,image in enumerate(images):
        if image is None:raise RuntimeError('missing replay image')
        rgb,ratio,left,top=letterbox(image)
        x=rgb.transpose(2,0,1)[None].astype(np.float32)/255
        with torch.no_grad():want=[v.numpy() for v in raw(torch.from_numpy(x))]
        actual=want
        reference=original.predict(image,rect=False,imgsz=(320,416),conf=.175,max_det=2,device='cpu',verbose=False)[0]
        decoded=decode(actual,image.shape,ratio,left,top)
        # Compare raw TopK before our cross-class duplicate suppression.
        expected=reference.boxes.data.cpu().tolist()
        for item in decoded:
            candidates=[row for row in expected if int(row[5])==item['class_id']]
            if not candidates or min(np.max(np.abs(np.array(row[:4])-item['xyxy'])) for row in candidates)>.2:
                raise RuntimeError(f'Raw-head export differs: sample={i}, raw={decoded}, original={expected}, end2end={original.model.model[-1].end2end}')
        np.savez_compressed(ROOT/'fixtures'/f'yolo_{i}.npz',rgb=rgb,bgr=image,
                            ratio=ratio,left=left,top=top,**{f'out_{j}':v for j,v in enumerate(actual)})
        report['samples'].append(dict(sample=i,decoded=decoded,
                                      original_detections=expected))
    report['onnx_sha256']=hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    (ROOT/'export_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
