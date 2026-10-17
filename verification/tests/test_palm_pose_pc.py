import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from palm_pose_pc import classify_hand,PalmRoiTracker,SinglePalmState


def skeleton(curled=False):
    p=np.zeros((21,3),np.float64)
    p[1:5]=[[-.3,.2,0],[-.6,.4,0],[-1.,.6,0],[-1.4,.8,0]]
    for m,x,y in [(5,-.45,1.),(9,0,1.1),(13,.4,1.),(17,.75,.8)]:
        p[m]=[x,y,0]
        direction=p[m]/np.linalg.norm(p[m])
        p[m+1]=p[m]+direction*.45
        if curled:
            p[m+2]=p[m+1]+[0,-.15,-.2]
            p[m+3]=p[m+2]+[0,-.35,0]
        else:
            p[m+2]=p[m+1]+direction*.3
            p[m+3]=p[m+2]+direction*.25
    return p


class PalmPcTests(unittest.TestCase):
    def test_three_rules_under_rotation_translation_and_scale(self):
        opened=skeleton();pinch=opened.copy();pinch[4]=pinch[8]+[.02,0,0]
        examples=[('OPEN',opened),('PINCH',pinch),('GRIP',skeleton(True))]
        rng=np.random.default_rng(17)
        for expected,points in examples:
            for _ in range(10):
                rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
                transformed=points@rotation*rng.uniform(.01,300)+rng.normal(size=3)
                state,info=classify_hand(transformed,transformed)
                self.assertEqual(state,expected,(expected,info))
        self.assertIsNone(classify_hand(np.zeros((21,3)),np.zeros((21,3)))[0])

    def test_short_prediction_then_full_reacquisition_without_predicted_measurements(self):
        tracker=PalmRoiTracker()
        points=skeleton()*50+[320,150,0]
        ids=tracker.pose.PALM_LANDMARK_IDS
        tracked=points[ids,:2]
        roi=np.r_[tracked.min(axis=0),tracked.max(axis=0),tracked.ravel(),.9]
        row=np.r_[100,100,450,400,points.ravel(),skeleton().ravel(),.5,.95]
        tracker.palm.infer=Mock(return_value=np.array([roi]))
        tracker.pose.infer=Mock(return_value=row)
        image=np.zeros((600,800,3),np.uint8)
        self.assertIsNotNone(tracker.infer(image,0))
        self.assertIsNotNone(tracker.infer(image,.066))
        self.assertEqual(tracker.palm_calls,1)
        tracker.pose.infer.return_value=None
        self.assertIsNone(tracker.infer(image,.13))
        self.assertEqual(tracker.status,'PREDICTING')
        self.assertEqual(tracker.palm_calls,1)
        self.assertIsNone(tracker.infer(image,.4))
        self.assertEqual(tracker.status,'SEARCHING')
        self.assertEqual(tracker.palm_calls,2)
        tracker.pose.infer.return_value=row
        self.assertIsNotNone(tracker.infer(image,.466))
        self.assertEqual(tracker.status,'DETECTED')
        self.assertEqual(tracker.palm_calls,3)

    def test_tof_hold_is_invalid_and_no_id_is_created(self):
        state=SinglePalmState();hand=dict(confidence=.9)
        first=state.update(hand,np.array([10.,20.,800.]),'OPEN',0)
        self.assertTrue(first['valid'])
        held=state.update(hand,None,'OPEN',.1)
        self.assertEqual(held['xyz_mm'],first['xyz_mm'])
        self.assertFalse(held['valid'])
        self.assertEqual(held['position_status'],'HOLD')
        self.assertNotIn('hand_id',held)
        self.assertIsNone(state.update(None,None,None,.2))


if __name__=='__main__':unittest.main()
