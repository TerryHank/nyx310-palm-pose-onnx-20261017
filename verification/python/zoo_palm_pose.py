"""Use unmodified OpenCV Zoo Palm/Hand Pose preprocessing and postprocessing."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "vendor" / "opencv_zoo" / "models"
sys.path.insert(0, str(ZOO / "palm_detection_mediapipe"))
sys.path.insert(0, str(ZOO / "handpose_estimation_mediapipe"))
from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose


class PalmPose:
    def __init__(self):
        self.palm_path = ROOT / "models" / "palm_detection_mediapipe_2023feb.onnx"
        self.pose_path = ROOT / "models" / "handpose_estimation_mediapipe_2023feb.onnx"
        self.palm = MPPalmDet(str(self.palm_path), scoreThreshold=.6, nmsThreshold=.3)
        self.pose = MPHandPose(str(self.pose_path), confThreshold=.7)
        self.previous_palms = []

    def infer(self, image):
        h, w = image.shape[:2]
        width = min(w, 640)
        height = round(h * width / w)
        small = cv2.resize(image, (width, height))
        palms = self.palm.infer(small)
        # The palm detector often drops a hand as fingers close around an object.
        # Re-run the pose model on the last palm ROI, not on stale landmarks.
        if len(palms) == 0:
            palms = self.previous_palms
        self.previous_palms = []
        result = []
        for palm in sorted(palms, key=lambda p: -p[-1])[:2]:
            clipped = np.clip(palm[:4], (0, 0, 0, 0), (width, height, width, height))
            if min(clipped[2] - clipped[0], clipped[3] - clipped[1]) < 2:
                continue
            hand = self.pose.infer(small, palm)
            if hand is None:
                continue
            tracked = hand[4:67].reshape(21, 3)[self.pose.PALM_LANDMARK_IDS, :2]
            lo, hi = tracked.min(axis=0), tracked.max(axis=0)
            self.previous_palms.append(np.r_[lo, hi, tracked.ravel(), hand[-1]])
            points = hand[4:67].reshape(21, 3).copy()
            points[:, 0] *= w / width
            points[:, 1] *= h / height
            points[:, 2] *= w / width
            result.append(dict(
                points=points, world=hand[67:130].reshape(21, 3).copy(),
                confidence=float(min(palm[-1], hand[-1])),
                side="Right" if hand[-2] > .5 else "Left",
            ))
        return result
