"""Read camera RGB and AINN hand JSON. No host model or depth computation."""
import ctypes as c
import json
import math
import os
import threading
import time
from pathlib import Path

from live_ainn_viewer import ScepterSdk, ScFrame, ScFrameReady, ScAinnResult, frame_to_bgr
from nyx310_capture_profile import configure_capture

ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = "PalmPoseGraspXYZ"
HAGRID_LABELS = ('grabbing','grip','holy','point','call','three3','timeout','xsign',
    'hand_heart','hand_heart2','little_finger','middle_finger','take_picture','dislike','fist','four',
    'like','mute','ok','one','palm','peace','peace_inverted','rock','stop','stop_inverted',
    'three','three2','two_up','two_up_inverted','three_gun','thumb_index','thumb_index2','no_gesture')
GESTURE_BINARY_MAP = {
    'grabbing':'GRASP', 'grip':'GRASP', 'fist':'GRASP', 'ok':'GRASP',
    'thumb_index':'GRASP', 'thumb_index2':'GRASP', 'hand_heart':'GRASP',
    'hand_heart2':'GRASP', 'point':'GRASP', 'one':'GRASP', 'call':'GRASP',
    'like':'GRASP', 'dislike':'GRASP', 'little_finger':'GRASP',
    'middle_finger':'GRASP', 'mute':'GRASP', 'rock':'GRASP', 'xsign':'GRASP',
    'peace':'GRASP', 'peace_inverted':'GRASP', 'two_up':'GRASP', 'two_up_inverted':'GRASP',
    'palm':'OPEN', 'stop':'OPEN', 'stop_inverted':'OPEN', 'four':'OPEN',
    'three':'OPEN', 'three2':'OPEN', 'three3':'OPEN', 'three_gun':'OPEN',
    'holy':'OPEN', 'timeout':'OPEN', 'take_picture':None, 'no_gesture':None,
}


def map_grasp_open(data):
    """Anonymous current detections; semantic grouping, no ID association/history."""
    hands = []
    for hand in data['hands']:
        # The device keeps old tracks with zero confidence and no current label.
        if hand['confidence'] <= 0 and hand.get('raw_class_name') is None:
            continue
        category = hand.get('raw_class_name') or hand['state']
        state = GESTURE_BINARY_MAP.get(category)
        if state is None:
            state = GESTURE_BINARY_MAP.get(hand['state']) or 'OPEN'
        hands.append(dict(xyz_mm=hand['xyz_mm'], state=state,
                          class_id=0 if state == 'GRASP' else 1,
                          confidence=hand['confidence'], valid=hand['valid']))
    return dict(data, hand_present=bool(hands), hands=hands)


def parse_hands(raw, algorithm=ALGORITHM):
    data = json.loads(raw.rstrip(b"\0").decode("utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("hands"), list):
        raise ValueError("AINN result does not contain hands[]")
    if len(data["hands"]) > 32:
        raise ValueError("Too many hand records")
    native_binary = algorithm == 'HagridBinaryXYZ'
    if native_binary and (type(data.get('hand_present')) is not bool or
                          data['hand_present'] != bool(data['hands']) or len(data['hands']) > 1):
        raise ValueError('Invalid anonymous binary hand presence')
    ids = set()
    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    for hand in data["hands"]:
        if not isinstance(hand, dict):
            raise ValueError("Invalid hand record")
        if native_binary:
            if 'hand_id' in hand:
                raise ValueError('Anonymous camera result must not contain hand_id')
        else:
            identifier = hand.get("hand_id")
            if type(identifier) is not int or identifier < 1 or identifier in ids:
                raise ValueError("Invalid or duplicate hand_id")
            ids.add(identifier)
        allowed = HAGRID_LABELS if algorithm == 'HagridPalmXYZ' else ('GRASP','OPEN') if native_binary else ("OPEN", "GRASP", "CLOSED")
        if hand.get("state") not in allowed:
            raise ValueError("Invalid camera gesture")
        if native_binary:
            if type(hand.get('class_id')) is not int or hand['class_id'] != (0 if hand['state']=='GRASP' else 1):
                raise ValueError('Invalid binary class ID/name')
        if algorithm == 'HagridPalmXYZ':
            category = hand.get('class_id')
            if type(category) is not int or not 0 <= category < len(HAGRID_LABELS) or HAGRID_LABELS[category] != hand['state']:
                raise ValueError('Invalid HaGRID class ID/name')
            if hand.get('raw_class_name') is not None and hand['raw_class_name'] not in HAGRID_LABELS:
                raise ValueError('Invalid raw HaGRID category')
        if type(hand.get("valid")) is not bool:
            raise ValueError("Missing camera valid flag")
        confidence = hand.get("confidence")
        if not number(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid camera confidence")
        xyz = hand.get("xyz_mm")
        if xyz is not None and (not isinstance(xyz, list) or len(xyz) != 3 or not all(number(v) for v in xyz)):
            raise ValueError("Invalid camera XYZ")
        if xyz is not None and any(abs(v) > 100000 for v in xyz):
            raise ValueError("Camera XYZ is outside the display safety bound")
        if hand["valid"] and xyz is None:
            raise ValueError("Camera marks a null position valid")
    return data


class CameraPalmReceiver:
    def __init__(self, ip, algorithm=ALGORITHM, confidence=None):
        self.ip = ip
        self.algorithm = algorithm
        self.confidence = confidence
        self.sdk = None
        self.previous_cwd = None
        self.stop = threading.Event()
        self.thread = None
        self.lock = threading.Lock()
        self.latest = None
        self.info = dict(connected=False, algorithm_name="", algorithm_version="",
                         error="Waiting for camera AINN", results=0, parse_errors=0,
                         timeouts=0, reconnects=0)
        self.close_statuses = []

    def open(self, output):
        self.output = output
        self.previous_cwd = Path.cwd()
        os.chdir(ROOT / "bin")
        self.sdk = ScepterSdk(ROOT / "bin")
        session = self.sdk.open(self.ip, True, require_ai=False, start_stream=False)
        session['capture_profile'] = configure_capture(self.sdk)
        self.sdk._check(self.sdk.base.scStartStream(self.sdk.device), 'scStartStream')
        self.sdk.stream_started = True
        session['stream_start_status'] = 0
        print('CAPTURE_READY: RGB 800x600 @ 15 FPS (readback verified); ToF unchanged', flush=True)
        session.update(pipeline="camera RGB + camera AINN JSON, display only",
                       local_inference=False, host_depth_computation=False,
                       host_classification=False, automatic_host_fallback=False,
                       expected_algorithm=self.algorithm,
                       sync_policy="latest RGB + latest AINN; timestamps and result age displayed, not exact frame pairing")
        self.thread = threading.Thread(target=self._results, name="camera-AINN", daemon=True)
        self.thread.start()
        return session

    def _update(self, **values):
        with self.lock:
            self.info.update(values)

    def snapshot(self):
        with self.lock:
            return self.latest, dict(self.info)

    def get_rgb(self):
        ready, frame = ScFrameReady(), ScFrame()
        status = self.sdk.base.scGetFrameReady(self.sdk.device, 100, c.byref(ready))
        if status == -11:
            return None
        self.sdk._check(status, "scGetFrameReady")
        if not ready.color:
            return None
        self.sdk._check(self.sdk.base.scGetFrame(self.sdk.device, 3, c.byref(frame)), "RGB frame")
        image = frame_to_bgr(frame)
        if image.shape[:2] != (600, 800):
            raise RuntimeError(f'Unexpected RGB frame size: {image.shape[:2]}; expected 600x800')
        return image, int(frame.deviceTimestamp)

    def _close_module(self):
        if self.sdk.module.value:
            status = self.sdk.ai.scCloseAinnModule(c.byref(self.sdk.module))
            self.close_statuses.append(("scCloseAinnModule", int(status)))
            self.sdk.module = c.c_void_p()
        self._update(connected=False)

    def _results(self):
        last_key = None
        errors = 0
        try:
            with (self.output / "camera_results.jsonl").open("x", encoding="utf-8", buffering=1) as raw_log, (
                self.output / "hand_states.jsonl"
            ).open("x", encoding="utf-8", buffering=1) as states_log:
                while not self.stop.is_set():
                    if not self.sdk.module.value:
                        status = self.sdk.ai.scOpenAinnModuleByIP(self.ip.encode("ascii"), c.byref(self.sdk.module))
                        if status:
                            self._update(error=f"AINN open failed: ScStatus({status}); no host fallback")
                            if self.sdk.module.value:
                                self._close_module()
                            self.stop.wait(2)
                            continue
                        name_status, name = self.sdk.get_ainn_param(1)
                        version_status, version = self.sdk.get_ainn_param(0)
                        self._update(algorithm_name=name, algorithm_version=version)
                        if name_status or version_status or name != self.algorithm:
                            self._update(error=f"Expected {self.algorithm}; camera reports {name!r} ({name_status}/{version_status})")
                            self._close_module()
                            self.stop.wait(2)
                            continue
                        if self.confidence is not None:
                            confidence = self.sdk.set_ainn_confidence(self.confidence)
                            self._update(confidence_threshold=confidence['active'])
                            (self.output/'camera_confidence.json').write_text(json.dumps(confidence,indent=2),encoding='utf-8')
                            print(f"CAMERA_CONFIDENCE: {confidence['before']} -> {confidence['active']} (readback verified)",flush=True)
                        enabled, mode = c.c_bool(), c.c_int()
                        enabled_status = self.sdk.ai.scAinnModuleGetEnabled(self.sdk.module, c.byref(enabled))
                        mode_status = self.sdk.ai.scAinnModuleGetWorkMode(self.sdk.module, c.byref(mode))
                        self._update(connected=True, enabled=enabled.value if not enabled_status else None,
                                     work_mode=mode.value if not mode_status else None,
                                     error=None if enabled_status == 0 and enabled.value else "Camera AINN disabled or state unreadable")
                        last_key = None
                        errors = 0
                    result = ScAinnResult()
                    status = self.sdk.ai.scAinnModuleGetResult(self.sdk.module, 100, c.byref(result))
                    if status == -112:
                        with self.lock:
                            self.info["timeouts"] += 1
                        continue
                    if status:
                        errors += 1
                        self._update(error=f"AINN receive failed: ScStatus({status})")
                        if errors >= 3:
                            self._close_module()
                            with self.lock:
                                self.info["reconnects"] += 1
                            self.stop.wait(1)
                        continue
                    errors = 0
                    key = (int(result.resultIndex), int(result.resultTimestamp))
                    if key == last_key:
                        continue
                    if not result.pResultData or not 0 < result.dataLen <= 1024 * 1024:
                        self._update(error="Invalid AINN buffer or payload length")
                        continue
                    # SDK owns its buffer: copy before the next SDK get call.
                    raw = c.string_at(result.pResultData, result.dataLen)
                    try:
                        data = parse_hands(raw, self.algorithm)
                    except (ValueError, UnicodeError) as error:
                        raw_log.write(json.dumps(dict(result_index=key[0],timestamp_ms=key[1],
                            raw_text=raw.rstrip(b"\0").decode("utf-8",errors="replace"),parse_error=str(error))) + "\n")
                        with self.lock:
                            self.info["parse_errors"] += 1
                        self._update(error=f"Invalid camera JSON: {error}")
                        continue
                    packet = dict(result_index=key[0], timestamp_ms=key[1], data=data,
                                  payload_text=raw.rstrip(b'\0').decode('utf-8'),
                                  received_monotonic=time.monotonic())
                    raw_log.write(json.dumps(dict(result_index=key[0], timestamp_ms=key[1], data=data), ensure_ascii=False) + "\n")
                    for hand in data["hands"]:
                        states_log.write(json.dumps(hand, ensure_ascii=False) + "\n")
                    (self.output / "latest.tmp").write_text(json.dumps(data["hands"]), encoding="utf-8")
                    (self.output / "latest.tmp").replace(self.output / "latest.json")
                    with self.lock:
                        self.latest = packet
                        self.info["results"] += 1
                        self.info["error"] = data.get("error")
                    last_key = key
        except Exception as error:
            self._update(error=f"AINN receiver stopped: {type(error).__name__}: {error}", connected=False)
        finally:
            self._close_module()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(5)
            if self.thread.is_alive():
                self.close_statuses.append(("receiver_thread_not_stopped", -1))
                return
        if self.sdk is not None:
            self.close_statuses.extend(self.sdk.close())
            self.sdk = None
        if self.previous_cwd is not None:
            os.chdir(self.previous_cwd)
