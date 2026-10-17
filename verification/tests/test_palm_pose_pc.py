import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from palm_pose_pc import classify_hand,PalmRoiTracker,SinglePalmState,pose_to_palm,tof_foreground_rois


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
    def test_tof_foreground_generates_bounded_hints_and_not_a_detection(self):
        depth=np.zeros((480,640),np.float32);depth[120:220,280:320]=750
        boxes=tof_foreground_rois(depth,(640,480))
        self.assertTrue(boxes)
        self.assertLessEqual(len(boxes),2)
        self.assertTrue(all(0<=x1<x2<=640 and 0<=y1<y2<=480 for x1,y1,x2,y2 in boxes))
        self.assertEqual(tof_foreground_rois(np.zeros((480,640)),(640,480)),[])
        tracker=PalmRoiTracker();tracker.palm.infer=Mock(return_value=np.empty((0,19)))
        tracker.pose.infer=Mock(return_value=None)
        for i in range(3):self.assertIsNone(tracker.infer(np.zeros((480,640,3),np.uint8),i/15,depth))
        self.assertGreater(tracker.tof_roi_calls,0)

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
        self.assertEqual(tracker.palm_calls,2)
        self.assertIsNone(tracker.infer(image,.4))
        self.assertEqual(tracker.status,'SEARCHING')
        self.assertEqual(tracker.palm_calls,3)
        tracker.pose.infer.return_value=row
        self.assertIsNotNone(tracker.infer(image,.466))
        self.assertEqual(tracker.status,'DETECTED')
        self.assertEqual(tracker.palm_calls,4)

    def test_side_view_roi_and_inner_crop_keep_minimum_width(self):
        points=skeleton()*50+[320,150,0]
        points[:,0]=320+(points[:,0]-320)*.005
        roi=pose_to_palm(points,.9)
        self.assertIsNotNone(roi)
        self.assertGreaterEqual(roi[2]-roi[0],24)
        tracker=PalmRoiTracker()
        image=np.zeros((480,640,3),np.uint8)
        crop,box,_=tracker.pose._cropAndPadFromPalm(image,np.array([[320,100],[320.2,160]]),False)
        self.assertGreaterEqual(box[1,0]-box[0,0],72)
        self.assertGreater(crop.size,0)
        self.assertIsNone(pose_to_palm(np.zeros((21,3))))

    def test_palm_failure_does_not_gate_a_valid_tracked_side_hand(self):
        tracker=PalmRoiTracker()
        points=skeleton()*50+[320,150,0]
        row=np.r_[100,100,450,400,points.ravel(),skeleton().ravel(),.5,.95]
        tracker.palm.infer=Mock(return_value=np.array([pose_to_palm(points,.9)]))
        tracker.pose.infer=Mock(return_value=row)
        image=np.zeros((600,800,3),np.uint8)
        self.assertIsNotNone(tracker.infer(image,0))
        tracker.palm.infer.return_value=np.empty((0,19))
        for i in range(1,8):self.assertIsNotNone(tracker.infer(image,i/15))
        self.assertEqual(tracker.palm_calls,1)
        self.assertEqual(tracker.status,'TRACKING')

    def test_current_pose_with_degenerate_next_roi_does_not_crash(self):
        tracker=PalmRoiTracker()
        points=skeleton()*50+[320,150,0]
        palm=pose_to_palm(points,.9)
        points[9]=points[0]+[0,4,0]
        row=np.r_[100,100,450,400,points.ravel(),skeleton().ravel(),.5,.95]
        tracker.palm.infer=Mock(return_value=np.array([palm]))
        tracker.pose.infer=Mock(return_value=row)
        hand=tracker.infer(np.zeros((600,800,3),np.uint8),0)
        self.assertIsNotNone(hand)
        self.assertIsNone(hand['roi_small'])
        self.assertEqual(tracker.status,'DETECTED')

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
