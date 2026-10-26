"""Capture exact deployed OpenCV preprocessing tensors from saved camera images."""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'verification' / 'python'))
from zoo_palm_pose import PalmPose


def main():
    cv2.setNumThreads(1)
    pipeline = PalmPose()
    captures = PROJECT / 'verification' / 'captures'
    files = sorted(captures.glob('nyx310-palm-position-*/gesture_*.jpg'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    files += sorted(captures.glob('nyx310-palm-position-*/hand_rgb.jpg'),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    used, counts, samples = set(), dict(palm=0, pose=0), []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in used:
            continue
        used.add(digest)
        image = cv2.imread(str(path))
        if image is None:
            continue
        h,w = image.shape[:2]
        small = cv2.resize(image, (640,round(h*640/w)))
        tensor,bias = pipeline.palm._preprocess(small)
        if counts['palm'] < 12:
            pipeline.palm.model.setInput(tensor)
            expected = pipeline.palm.model.forward(pipeline.palm.model.getUnconnectedOutLayersNames())
            name = f'palm_{counts["palm"]:02d}.npz'
            np.savez_compressed(ROOT/'tests'/name, input=tensor,
                raw_rgb=np.rint(tensor*255).astype(np.uint8),
                original_shape=np.array([640,small.shape[0]]),pad_bias=bias,
                **{f'output_{i}':a for i,a in enumerate(expected)})
            samples.append(dict(file=name, image=str(path), image_sha256=digest,
                                outputs=list(pipeline.palm.model.getUnconnectedOutLayersNames())))
            counts['palm'] += 1
        palms = pipeline.palm.infer(small)
        for palm in sorted(palms, key=lambda p:-p[-1])[:1]:
            hand = pipeline.pose.infer(small,palm)
            if hand is None or counts['pose'] >= 12:
                continue
            tensor,bbox,angle,rotation,pad = pipeline.pose._preprocess(small,palm)
            pipeline.pose.model.setInput(tensor)
            expected = pipeline.pose.model.forward(pipeline.pose.model.getUnconnectedOutLayersNames())
            name = f'pose_{counts["pose"]:02d}.npz'
            np.savez_compressed(ROOT/'tests'/name, input=tensor,
                raw_rgb=np.rint(tensor*255).astype(np.uint8),
                rotated_bbox=bbox,angle=angle,rotation=rotation,pad_bias=pad,
                **{f'output_{i}':a for i,a in enumerate(expected)})
            samples.append(dict(file=name, image=str(path), image_sha256=digest,
                                outputs=list(pipeline.pose.model.getUnconnectedOutLayersNames())))
            counts['pose'] += 1
        if min(counts.values()) >= 12:
            break
    (ROOT/'tests'/'samples.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
    print(counts)
    if not all(counts.values()):
        raise RuntimeError('No replay inputs for one of the two models')


if __name__ == '__main__':
    main()
