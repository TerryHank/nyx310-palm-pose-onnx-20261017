"""NYX310 RGB-D capture using Scepter's timestamp pairing and calibration."""
import ctypes as c
import os
import time
from pathlib import Path

import numpy as np

from fast_rgbd import Calibration
from live_ainn_viewer import (
    ScepterSdk, ScFrame, ScFrameReady, frame_to_bgr, SC_GET_FRAME_READY_TIME_OUT,
)

ROOT = Path(__file__).resolve().parents[1]


class Nyx310RgbdCamera:
    def __init__(self, ip="192.168.1.101", allow_discovery=False, audit=False, capture_profile=False, rgb_exposure_cap_us=None):
        self.ip = ip
        self.allow_discovery = allow_discovery
        self.audit = audit
        self.capture_profile = capture_profile
        self.rgb_exposure_cap_us = rgb_exposure_cap_us
        self._original_exposure_cap = None
        self.sdk = None
        self.calibration = None
        self.close_statuses = []
        self._previous_cwd = None

    def open(self):
        self._previous_cwd = Path.cwd()
        os.chdir(ROOT / "bin")
        self.sdk = ScepterSdk(ROOT / "bin")
        info = self.sdk.open(self.ip, self.allow_discovery, require_ai=False,
                             start_stream=not self.capture_profile)
        product=next((d['product_name'] for d in info.get('discovered_devices',[]) if d['ip']==self.ip),'NYX310')
        if self.capture_profile:
            from nyx310_capture_profile import configure_capture
            info['capture_profile'] = configure_capture(self.sdk,None if 'NYX320' in product.upper() else (800,600))
        if self.rgb_exposure_cap_us is not None:
            mode_getter=self.sdk.base.scGetExposureControlMode
            mode_getter.argtypes=[c.c_void_p,c.c_int,c.POINTER(c.c_int)]
            mode_getter.restype=c.c_int
            getter=self.sdk.base.scGetColorAECMaxExposureTime
            getter.argtypes=[c.c_void_p,c.POINTER(c.c_int32)]
            getter.restype=c.c_int
            setter=self.sdk.base.scSetColorAECMaxExposureTime
            setter.argtypes=[c.c_void_p,c.c_int32]
            setter.restype=c.c_int
            mode=c.c_int();self.sdk._check(mode_getter(self.sdk.device,2,c.byref(mode)),'get RGB exposure mode')
            if mode.value==0:
                before=c.c_int32();self.sdk._check(getter(self.sdk.device,c.byref(before)),'get RGB AEC exposure cap')
                target=min(before.value,self.rgb_exposure_cap_us)
                if target!=before.value:
                    self._original_exposure_cap=before.value
                    self.sdk._check(setter(self.sdk.device,target),'set temporary RGB AEC exposure cap')
                active=c.c_int32();self.sdk._check(getter(self.sdk.device,c.byref(active)),'read RGB AEC exposure cap')
                if active.value!=target:raise RuntimeError(f'RGB exposure cap mismatch: {active.value} vs {target}')
                info['rgb_exposure_cap_us']=dict(before=before.value,active=active.value,restore_on_exit=self._original_exposure_cap is not None)
            else:
                info['rgb_exposure_cap_us']=dict(applied=False,reason='camera in manual exposure mode')
        if self.capture_profile:
            self.sdk._check(self.sdk.base.scStartStream(self.sdk.device), 'scStartStream')
            self.sdk.stream_started = True
            info['stream_start_status'] = 0
        self.calibration = Calibration(self.sdk)
        info["calibration_reads"] = list(self.sdk.diagnostics.get("calibration_reads", []))
        self.color_K = self.calibration.K
        self.color_D = self.calibration.D
        range_min, range_max = c.c_int16(), c.c_int16()
        getter = self.sdk.base.scGetDepthRangeValue
        getter.argtypes = [c.c_void_p, c.POINTER(c.c_int16), c.POINTER(c.c_int16)]
        getter.restype = c.c_int
        status = getter(self.sdk.device, c.byref(range_min), c.byref(range_max))
        info["current_depth_range_mm"] = dict(
            status=status, min=range_min.value if status == 0 else None,
            max=range_max.value if status == 0 else None,
        )
        info.update(
            camera=product, driver="Scepter BaseSDK",
            calibration=self.calibration.metadata,
            sync_policy="exact RGB/depth deviceTimestamp equality",
            depth_registration="manufacturer depth-to-color projection",
        )
        return info

    def get_frame(self):
        ready, frame = ScFrameReady(), ScFrame()
        status = self.sdk.base.scGetFrameReady(self.sdk.device, 100, c.byref(ready))
        if status == SC_GET_FRAME_READY_TIME_OUT:
            return None
        self.sdk._check(status, "scGetFrameReady")
        if not ready.color:
            return None
        self.sdk._check(
            self.sdk.base.scGetFrame(self.sdk.device, 3, c.byref(frame)), "RGB frame"
        )
        image = frame_to_bgr(frame)
        data = self.calibration.capture(ready, frame, image, registration_only=True)
        depth = np.zeros(image.shape[:2], dtype=np.float32)
        matched = data is not None
        if matched:
            uv = data["uv"].astype(np.int32)
            xyz = data["xyz_m"]
            r = self.calibration.R[2]
            rgb_z = (xyz[:, 0] * r[0] + xyz[:, 1] * r[1] + xyz[:, 2] * r[2]) * 1000 + self.calibration.t[2]
            valid = np.isfinite(rgb_z) & (rgb_z > 0)
            flat = np.full(depth.size, np.inf, np.float32)
            indices = uv[valid, 1] * frame.width + uv[valid, 0]
            np.minimum.at(flat, indices, rgb_z[valid])
            flat[~np.isfinite(flat)] = 0
            depth = flat.reshape(depth.shape)
        packet = {
            "bgr": image, "depth_mm": depth,
            "color_timestamp_us": int(frame.deviceTimestamp) * 1000,
            "depth_timestamp_us": int(frame.deviceTimestamp) * 1000 if matched else None,
            "timestamp_match": matched,
            "host_monotonic": time.monotonic(),
        }
        if self.audit and data is not None:
            packet.update(raw_depth_mm=data["raw_depth_mm"],
                          raw_cloud_mm=data["raw_cloud_mm"],
                          registered_uv=data["uv"],
                          registered_tof_xyz_m=data["xyz_m"])
        return packet

    def close(self):
        if self.sdk is not None:
            restore=[]
            if self._original_exposure_cap is not None:
                status=self.sdk.base.scSetColorAECMaxExposureTime(self.sdk.device,self._original_exposure_cap)
                restore.append(('restore_RGB_AEC_cap',int(status)))
                active=c.c_int32()
                status=self.sdk.base.scGetColorAECMaxExposureTime(self.sdk.device,c.byref(active))
                restore.append(('verify_RGB_AEC_cap_restore',int(status) if status else 0 if active.value==self._original_exposure_cap else -1))
                self._original_exposure_cap=None
            self.close_statuses = restore + self.sdk.close()
            self.sdk = None
        if self._previous_cwd is not None:
            os.chdir(self._previous_cwd)
