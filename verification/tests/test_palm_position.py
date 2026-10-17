import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from palm_position import palm_geometry, measure_palm, PalmTracker


class PalmPositionTests(unittest.TestCase):
    def setUp(self):
        self.points = np.zeros((21, 3), np.float32)
        for i, xy in {0:(100,140),1:(65,120),5:(65,65),9:(100,55),
                      13:(125,65),17:(140,85)}.items():
            self.points[i,:2] = xy
        self.K = np.array([[200.,0,100],[0,200.,100],[0,0,1]])
        self.D = np.zeros(8)

    def test_reference_does_not_follow_fingertip_box(self):
        before, _ = palm_geometry(self.points)
        self.points[[4,8,12,16,20],:2] = [180,10]
        after, _ = palm_geometry(self.points)
        np.testing.assert_equal(before, after)

    def test_roi_median_rejects_flying_pixels_without_center_depth(self):
        depth = np.full((200,200), 600., np.float32)
        ref, polygon = palm_geometry(self.points)
        hull = np.rint(polygon).astype(int)
        mask = np.zeros(depth.shape,np.uint8)
        cv2.fillConvexPoly(mask,hull,1)
        depth[mask==0]=4000
        yy,xx=np.where(mask>0)
        depth[yy[::7],xx[::7]]=2000
        depth[int(ref[1]),int(ref[0])]=0
        xyz, info=measure_palm(self.points,depth,self.K,self.D)
        self.assertEqual(info["reason"],"ok")
        self.assertAlmostEqual(xyz[2],600.)
        self.assertGreater(info["rejected_samples"],0)
        ray=cv2.undistortPoints(ref.reshape(1,1,2).astype(float),self.K,self.D)[0,0]
        np.testing.assert_allclose(xyz,[ray[0]*600,ray[1]*600,600])

    def test_missing_and_mismatched_depth_are_invalid(self):
        depth=np.zeros((200,200),np.float32)
        self.assertIsNone(measure_palm(self.points,depth,self.K,self.D)[0])
        self.assertEqual(measure_palm(self.points,depth,self.K,self.D,False)[1]["reason"],
                         "timestamp_mismatch")

    def test_sparse_background_fragments_are_not_valid_measurements(self):
        depth=np.zeros((200,200),np.float32)
        _,polygon=palm_geometry(self.points)
        mask=np.zeros(depth.shape,np.uint8)
        cv2.fillConvexPoly(mask,np.rint(polygon).astype(int),1)
        mask=cv2.erode(mask,np.ones((3,3),np.uint8))
        yy,xx=np.where(mask>0)
        depth[yy[:15],xx[:15]]=1600
        xyz,info=measure_palm(self.points,depth,self.K,self.D)
        self.assertIsNone(xyz)
        self.assertEqual(info["reason"],"insufficient_spatial_support")

    def test_filter_hold_ids_and_state(self):
        tracker=PalmTracker()
        def observation(xyz,extended=4):
            return dict(reference=np.array([100.,100.]),side="Right",confidence=.95,
                        xyz=xyz,gesture="OPEN" if extended == 4 else "CLOSED",diagnostics={})
        first=tracker.update([observation(np.array([0.,0.,600.]))],0.,(200,200))[0]
        second=tracker.update([observation(np.array([10.,0.,600.]))],.07,(200,200))[0]
        self.assertEqual(first["hand_id"],second["hand_id"])
        self.assertTrue(0<second["xyz_mm"][0]<10)
        third=tracker.update([observation(np.array([10.,0.,600.]))],.14,(200,200))[0]
        self.assertEqual(third["state"],"OPEN")
        lost=tracker.update([],1.,(200,200))[0]
        self.assertFalse(lost["valid"])
        self.assertEqual(lost["xyz_mm"],third["xyz_mm"])
        for t in (1.1,1.2,1.3):
            closed=tracker.update([observation(np.array([0.,0.,600.]),0)],t,(200,200))[0]
        self.assertEqual(closed["state"],"CLOSED")
        self.assertEqual(set(closed),{"hand_id","xyz_mm","state","confidence","valid"})


if __name__ == "__main__":
    unittest.main()
