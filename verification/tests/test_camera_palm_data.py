import ast
import copy
import ctypes as c
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from camera_palm_data import parse_hands, CameraPalmReceiver, HAGRID_LABELS, GESTURE_BINARY_MAP, map_grasp_open
from nyx310_camera_palm_viewer import render_camera, render_json_panel

HAND=dict(hand_id=1,xyz_mm=[125.,-60.,800.],state='GRASP',confidence=.95,valid=True)


class CameraDataTests(unittest.TestCase):
    def test_native_binary_payload_has_no_ids_and_at_most_one_hand(self):
        hand={key:value for key,value in HAND.items() if key!='hand_id'}
        hand['class_id']=0
        packet=dict(hand_present=True,hands=[hand],timing_ms=dict(total=50))
        self.assertEqual(parse_hands(json.dumps(packet).encode(),'HagridBinaryXYZ'),packet)
        self.assertEqual(parse_hands(b'{"hand_present":false,"hands":[]}','HagridBinaryXYZ')['hands'],[])
        for bad in (dict(packet,hand_present=False),dict(packet,hands=[hand,hand]),
                    dict(packet,hands=[dict(hand,hand_id=1)]),dict(packet,hands=[dict(hand,state='fist')])):
            with self.assertRaises(ValueError):parse_hands(json.dumps(bad).encode(),'HagridBinaryXYZ')

    def test_all_camera_hands_map_to_binary_without_mutating_camera_json(self):
        hands=[dict(HAND,hand_id=i+1,state=name,class_id=i,raw_class_name=name)
               for i,name in enumerate(HAGRID_LABELS)]
        payload=dict(hands=hands);before=copy.deepcopy(payload)
        mapped=map_grasp_open(payload)
        self.assertEqual(set(GESTURE_BINARY_MAP),set(HAGRID_LABELS))
        self.assertEqual(len(mapped['hands']),len(hands))
        self.assertTrue(mapped['hand_present'])
        self.assertTrue(all('hand_id' not in h for h in mapped['hands']))
        self.assertTrue(all(h['state'] in ('GRASP','OPEN') for h in mapped['hands']))
        self.assertTrue(all(h['class_id']==(0 if h['state']=='GRASP' else 1) for h in mapped['hands']))
        self.assertEqual(payload,before)
        self.assertEqual(map_grasp_open(dict(hands=[dict(HAND,state='grip',raw_class_name='fist')]))['hands'][0]['state'],'GRASP')
        held=dict(HAND,state='palm',raw_class_name='palm',valid=False,confidence=.6)
        result=map_grasp_open(dict(hands=[held]))['hands'][0]
        self.assertEqual(result['state'],'OPEN')
        self.assertFalse(result['valid'])
        self.assertEqual(result['xyz_mm'],HAND['xyz_mm'])

    def test_anonymous_hands_do_not_inherit_state_by_id_or_keep_expired_tracks(self):
        def packet(state):return dict(hands=[dict(HAND,state=state,raw_class_name=state)])
        first=map_grasp_open(packet('no_gesture'))['hands'][0]
        self.assertEqual(first['state'],'OPEN')
        self.assertEqual(map_grasp_open(packet('fist'))['hands'][0]['state'],'GRASP')
        self.assertEqual(map_grasp_open(packet('no_gesture'))['hands'][0]['state'],'OPEN')
        self.assertEqual(map_grasp_open(packet('stop'))['hands'][0]['state'],'OPEN')
        expired=dict(HAND,state='grip',raw_class_name=None,confidence=0.,valid=False)
        self.assertEqual(map_grasp_open(dict(hands=[expired])),dict(hand_present=False,hands=[]))
        self.assertEqual(map_grasp_open(dict(hands=[])),dict(hand_present=False,hands=[]))

    def test_anonymous_live_payload_renders_without_hand_id(self):
        source=dict(hands=[dict(HAND,state='grip',raw_class_name='grip')])
        data=map_grasp_open(source)
        image=np.zeros((600,800,3),np.uint8)
        view=render_camera(image,data['hands'],True,1.5,15,5,10,None,selected_only=True)
        self.assertEqual(view.shape,(700,1573,3))
        self.assertNotIn('hand_id',json.dumps(data))

    def test_json_panel_keeps_camera_payload_and_waiting_distinct(self):
        image=np.zeros((700,1573,3),np.uint8)
        data=dict(hands=[dict(HAND,hand_id=i+1) for i in range(32)])
        before=copy.deepcopy(data)
        packet=dict(result_index=7,timestamp_ms=1000,data=data)
        rendered,scroll=render_json_panel(image,packet,dict(connected=True,error=None),10,9999)
        self.assertEqual(rendered.shape,(992,1573,3))
        self.assertGreater(scroll,0)
        self.assertEqual(data,before)
        waiting,offset=render_json_panel(image,None,dict(error='AINN -122'),None,999)
        self.assertEqual(offset,0)
        self.assertEqual(waiting.shape,rendered.shape)

    def test_hagrid_native_labels_and_class_ids_are_checked(self):
        for index,name in enumerate(HAGRID_LABELS):
            hand=dict(HAND,state=name,class_id=index,raw_class_name=name)
            payload=json.dumps(dict(hands=[hand])).encode()
            self.assertEqual(parse_hands(payload,'HagridPalmXYZ')['hands'][0],hand)
            with self.assertRaises(ValueError):parse_hands(payload)
        bad=dict(HAND,state='fist',class_id=0)
        with self.assertRaises(ValueError):parse_hands(json.dumps(dict(hands=[bad])).encode(),'HagridPalmXYZ')

    def test_native_json_passes_unchanged_including_hold(self):
        data=dict(hands=[HAND,dict(HAND,hand_id=2,valid=False)])
        self.assertEqual(parse_hands(json.dumps(data).encode()+b'\0'),data)
        self.assertEqual(parse_hands(b'{"hands":[]}'),dict(hands=[]))

    def test_reject_wrong_algorithm_schema_and_bad_values(self):
        cases=[{},dict(hands='x'),dict(hands=[HAND,HAND])]
        for key,value in [('state','UNKNOWN'),('valid',1),('hand_id',True),
                          ('xyz_mm',[float('nan'),0,500]),('confidence',float('inf')),
                          ('xyz_mm',None),('xyz_mm',[0,0,1e300])]:
            cases.append(dict(hands=[dict(HAND,**{key:value})]))
        for data in cases:
            with self.assertRaises(ValueError):parse_hands(json.dumps(data).encode())

    def test_render_does_not_modify_camera_values_on_transport_stale(self):
        image=np.full((480,640,3),120,np.uint8);states=[copy.deepcopy(HAND)];before=copy.deepcopy(states)
        view=render_camera(image,states,False,1.5,15,8,900,None)
        self.assertEqual(states,before)
        self.assertTrue(states[0]['valid']) # raw camera flag retained; UI alone marks STALE
        self.assertEqual(view.ndim,3)
        np.testing.assert_equal(image,np.full((480,640,3),120,np.uint8))

    def test_no_model_classification_or_depth_backend_imported(self):
        forbidden={'zoo_palm_pose','palm_position','hand_grasp','nyx310_rgbd','fast_rgbd','mediapipe','torch','onnxruntime'}
        for name in ('camera_palm_data.py','nyx310_camera_palm_viewer.py'):
            tree=ast.parse((ROOT/'python'/name).read_text(encoding='utf-8'))
            modules={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
            modules.update(alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names)
            self.assertFalse(forbidden & modules)

    def test_receiver_copies_buffer_and_preserves_camera_schema(self):
        receiver=CameraPalmReceiver('192.168.1.101')
        receiver.sdk=Mock();receiver.sdk.module=c.c_void_p()
        def open_module(ip,out):out._obj.value=123;return 0
        receiver.sdk.ai.scOpenAinnModuleByIP.side_effect=open_module
        receiver.sdk.get_ainn_param.side_effect=lambda i:(0,'PalmPoseGraspXYZ' if i==1 else '26.8.1-hand.1')
        def enabled(module,out):out._obj.value=True;return 0
        def mode(module,out):out._obj.value=0;return 0
        receiver.sdk.ai.scAinnModuleGetEnabled.side_effect=enabled
        receiver.sdk.ai.scAinnModuleGetWorkMode.side_effect=mode
        payload=c.create_string_buffer(json.dumps(dict(hands=[HAND])).encode()+b'\0')
        calls=[]
        def get_result(module,wait,out):
            if calls:
                # Vendor memory has been reused: the receiver's saved JSON must not change.
                c.memset(c.addressof(payload),0,c.sizeof(payload));receiver.stop.set();return -112
            calls.append(True);r=out._obj;r.resultIndex=7;r.resultTimestamp=1000
            r.dataLen=c.sizeof(payload);r.pResultData=c.cast(payload,c.POINTER(c.c_uint8));return 0
        receiver.sdk.ai.scAinnModuleGetResult.side_effect=get_result
        receiver.sdk.ai.scCloseAinnModule.return_value=0
        with tempfile.TemporaryDirectory() as folder:
            receiver.output=Path(folder);receiver._results()
            packet,status=receiver.snapshot()
            self.assertEqual(packet['data'],dict(hands=[HAND]))
            self.assertEqual(status['results'],1)
            self.assertEqual(json.loads((receiver.output/'latest.json').read_text()),[HAND])
        receiver.sdk.ai.scAinnModuleSetEnabled.assert_not_called()
        receiver.sdk.ai.scAinnModuleSetWorkMode.assert_not_called()

    def test_foreign_algorithm_is_never_rendered_as_our_hand(self):
        receiver=CameraPalmReceiver('192.168.1.101');receiver.sdk=Mock();receiver.sdk.module=c.c_void_p()
        def open_module(ip,out):out._obj.value=123;return 0
        receiver.sdk.ai.scOpenAinnModuleByIP.side_effect=open_module
        receiver.sdk.get_ainn_param.side_effect=lambda i:(0,'BoxDetection' if i==1 else '26.8.1')
        def close_module(out):receiver.stop.set();return 0
        receiver.sdk.ai.scCloseAinnModule.side_effect=close_module
        with tempfile.TemporaryDirectory() as folder:
            receiver.output=Path(folder);receiver._results()
            packet,status=receiver.snapshot();self.assertIsNone(packet)
            self.assertIn('BoxDetection',status['error'])
            receiver.sdk.ai.scAinnModuleGetResult.assert_not_called()


if __name__=='__main__':unittest.main()
