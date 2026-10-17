#!/usr/bin/env python3
"""Display NYX310 color frames with exact-timestamp AINN results.

The program calls Scepter BaseSDK 26.06.1 directly through ctypes. It never
loads a local YOLO model. It requires an explicit IP. Vendor SDK discovery is
disabled by default and only runs with --allow-sdk-discovery.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import signal
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = ROOT / "bin"
TOOLS_DIR = ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))
from analyze_results import decode_detection_payload, parse_payload  # noqa: E402


SC_OK = 0
SC_GET_FRAME_READY_TIME_OUT = -11
SC_GET_AI_RESULT_TIME_OUT = -112
SC_CMD_ERROR = -104
SC_AINN_OPEN_FAILED = -122
SC_COLOR_FRAME = 3
SC_PIXEL_FORMAT_RGB_888_JPEG = 3
SC_PIXEL_FORMAT_BGR_888_JPEG = 4
SC_PIXEL_FORMAT_RGB_888 = 5
SC_PIXEL_FORMAT_BGR_888 = 6
SC_PIXEL_FORMAT_RGB_565 = 7
SC_PIXEL_FORMAT_BGR_565 = 8
AINN_CONTINUOUS_RUN_MODE = 0
SC_INVALID_PARAMS = -15


class ScFrame(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("frameIndex", ctypes.c_uint32),
        ("frameType", ctypes.c_int),
        ("pixelFormat", ctypes.c_int),
        ("pFrameData", ctypes.POINTER(ctypes.c_uint8)),
        ("dataLen", ctypes.c_uint32),
        ("width", ctypes.c_uint16),
        ("height", ctypes.c_uint16),
        ("deviceTimestamp", ctypes.c_uint64),
    ]


class ScFrameReady(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("bits", ctypes.c_uint32)]

    @property
    def color(self) -> bool:
        return bool((self.bits >> 2) & 1)


class ScAinnResult(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("resultIndex", ctypes.c_uint32),
        ("pResultData", ctypes.POINTER(ctypes.c_uint8)),
        ("dataLen", ctypes.c_uint32),
        ("resultTimestamp", ctypes.c_uint64),
    ]


class ScDeviceInfo(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("productName", ctypes.c_char * 64),
        ("serialNumber", ctypes.c_char * 64),
        ("ip", ctypes.c_char * 17),
        ("status", ctypes.c_int),
    ]


@dataclass
class FrameRecord:
    index: int
    timestamp_ms: int
    pixel_format: int
    image: np.ndarray


@dataclass
class ResultRecord:
    index: int
    timestamp_ms: int
    payload_text: str
    decoded: dict[str, Any]
    received_monotonic: float


@dataclass
class PairedRecord:
    frame: FrameRecord
    result: ResultRecord


class ScepterSdk:
    def __init__(self, bin_dir: Path):
        if os.name != "nt":
            raise RuntimeError("This packaged viewer uses the Windows x64 BaseSDK DLLs.")
        self._dll_directory = os.add_dll_directory(str(bin_dir))
        self.base = ctypes.CDLL(str(bin_dir / "Scepter_api.dll"))
        self.ai = ctypes.CDLL(str(bin_dir / "Scepter_ai_api.dll"))
        self.device = ctypes.c_void_p()
        self.module = ctypes.c_void_p()
        self.ai_backend: str | None = None
        self.diagnostics: dict[str, Any] = {}
        self.initialized = False
        self.stream_started = False
        self._declare_functions()

    def _declare_functions(self) -> None:
        self.base.scInitialize.argtypes = []
        self.base.scInitialize.restype = ctypes.c_int
        self.base.scShutdown.argtypes = []
        self.base.scShutdown.restype = ctypes.c_int
        self.base.scOpenDeviceByIP.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.base.scOpenDeviceByIP.restype = ctypes.c_int
        self.base.scGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
        self.base.scGetDeviceCount.restype = ctypes.c_int
        self.base.scGetDeviceInfoList.argtypes = [ctypes.c_uint32, ctypes.POINTER(ScDeviceInfo)]
        self.base.scGetDeviceInfoList.restype = ctypes.c_int
        self.base.scCloseDevice.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.base.scCloseDevice.restype = ctypes.c_int
        self.base.scStartStream.argtypes = [ctypes.c_void_p]
        self.base.scStartStream.restype = ctypes.c_int
        self.base.scStopStream.argtypes = [ctypes.c_void_p]
        self.base.scStopStream.restype = ctypes.c_int
        self.base.scGetFrameReady.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.POINTER(ScFrameReady)]
        self.base.scGetFrameReady.restype = ctypes.c_int
        self.base.scGetFrame.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ScFrame)]
        self.base.scGetFrame.restype = ctypes.c_int
        self.base.scGetFirmwareVersion.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
        self.base.scGetFirmwareVersion.restype = ctypes.c_int
        self.base.scAIModuleGetEnabled.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
        self.base.scAIModuleGetEnabled.restype = ctypes.c_int
        self.base.scAIModuleGetWorkMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.base.scAIModuleGetWorkMode.restype = ctypes.c_int
        self.base.scAIModuleGetParam.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint16),
        ]
        self.base.scAIModuleGetParam.restype = ctypes.c_int
        self.base.scAIModuleGetResult.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.POINTER(ScAinnResult)]
        self.base.scAIModuleGetResult.restype = ctypes.c_int

        self.ai.scOpenAinnModuleByIP.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.ai.scOpenAinnModuleByIP.restype = ctypes.c_int
        self.ai.scCloseAinnModule.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.ai.scCloseAinnModule.restype = ctypes.c_int
        self.ai.scAinnModuleGetEnabled.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
        self.ai.scAinnModuleGetEnabled.restype = ctypes.c_int
        self.ai.scAinnModuleGetWorkMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.ai.scAinnModuleGetWorkMode.restype = ctypes.c_int
        self.ai.scAinnModuleGetParam.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.ai.scAinnModuleGetParam.restype = ctypes.c_int
        self.ai.scAinnModuleSetParam.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                               ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        self.ai.scAinnModuleSetParam.restype = ctypes.c_int
        self.ai.scAinnModuleGetResult.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.POINTER(ScAinnResult)]
        self.ai.scAinnModuleGetResult.restype = ctypes.c_int

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != SC_OK:
            raise RuntimeError(f"{operation} failed, ScStatus({status})")

    def discover_exact_ip(self, requested_ip: str, scan_ms: int = 3000) -> list[dict[str, Any]]:
        count = ctypes.c_uint32(0)
        self._check(self.base.scGetDeviceCount(ctypes.byref(count), scan_ms), "scGetDeviceCount")
        if count.value == 0:
            raise RuntimeError(f"BaseSDK discovery returned no devices after {scan_ms} ms")
        device_array = (ScDeviceInfo * count.value)()
        self._check(self.base.scGetDeviceInfoList(count.value, device_array), "scGetDeviceInfoList")
        devices = []
        for item in device_array:
            devices.append({
                "product_name": bytes(item.productName).split(b"\0", 1)[0].decode("utf-8", errors="replace"),
                "serial_number": bytes(item.serialNumber).split(b"\0", 1)[0].decode("utf-8", errors="replace"),
                "ip": bytes(item.ip).split(b"\0", 1)[0].decode("ascii", errors="replace"),
                "status": int(item.status),
            })
        if not any(device["ip"] == requested_ip for device in devices):
            discovered = ", ".join(device["ip"] or "<empty>" for device in devices)
            raise RuntimeError(
                f"Requested IP {requested_ip} was not found by BaseSDK discovery; discovered: {discovered}"
            )
        return devices

    def open(
        self,
        ip: str,
        allow_sdk_discovery: bool = False,
        require_ai: bool = True,
        start_stream: bool = True,
    ) -> dict[str, Any]:
        encoded_ip = ip.encode("ascii")
        self.diagnostics = {
            "sdk_version": "26.06.1.0",
            "ip": ip,
            "sdk_discovery_performed": allow_sdk_discovery,
            "local_inference": False,
        }
        self._check(self.base.scInitialize(), "scInitialize")
        self.initialized = True
        discovered_devices: list[dict[str, Any]] = []
        if allow_sdk_discovery:
            discovered_devices = self.discover_exact_ip(ip)
        self.diagnostics["discovered_devices"] = discovered_devices
        open_status = self.base.scOpenDeviceByIP(encoded_ip, ctypes.byref(self.device))
        self.diagnostics["device_open_status"] = open_status
        if open_status == SC_INVALID_PARAMS and not allow_sdk_discovery:
            raise RuntimeError(
                "scOpenDeviceByIP returned SC_INVALID_PARAMS (-15). BaseSDK 26.06.1 keeps an internal "
                "device list and its official NYX310 DeviceConnectByIP sample populates that list with "
                "scGetDeviceCount/scGetDeviceInfoList first. Re-run with --allow-sdk-discovery only if "
                "you authorize the vendor SDK's 3-second device discovery broadcast."
            )
        self._check(open_status, "scOpenDeviceByIP")
        firmware_buffer = ctypes.create_string_buffer(128)
        firmware_status = self.base.scGetFirmwareVersion(self.device, firmware_buffer, len(firmware_buffer))
        self.diagnostics.update({
            "firmware_query_status": firmware_status,
            "firmware": firmware_buffer.value.decode("utf-8", errors="replace"),
        })

        enabled = ctypes.c_bool(False)
        mode = ctypes.c_int(-1)
        enabled_status: int | None = None
        mode_status: int | None = None
        name_status: int | None = None
        version_status: int | None = None
        name = ""
        version = ""

        if require_ai:
            morph_enabled = ctypes.c_bool(False)
            morph_mode = ctypes.c_int(-1)
            morph_enabled_status = self.base.scAIModuleGetEnabled(self.device, ctypes.byref(morph_enabled))
            morph_mode_status = self.base.scAIModuleGetWorkMode(self.device, ctypes.byref(morph_mode))
            self.diagnostics.update({
                "morph_enabled_query_status": morph_enabled_status,
                "morph_mode_query_status": morph_mode_status,
            })

            if morph_enabled_status == SC_OK and morph_mode_status == SC_OK:
                self.ai_backend = "morph"
                enabled = morph_enabled
                mode = morph_mode
                enabled_status = morph_enabled_status
                mode_status = morph_mode_status
                name_status, name = self.get_morph_param(1)
                version_status, version = self.get_morph_param(0)
            else:
                ainn_open_status = self.ai.scOpenAinnModuleByIP(encoded_ip, ctypes.byref(self.module))
                self.diagnostics["ainn_open_status"] = ainn_open_status
                if ainn_open_status != SC_OK:
                    raise RuntimeError(
                        "No readable camera AI backend. Legacy Morph getters returned "
                        f"ScStatus({morph_enabled_status})/ScStatus({morph_mode_status}); "
                        f"scOpenAinnModuleByIP returned ScStatus({ainn_open_status}). "
                        "For BaseSDK 26.06.1, -104 is SC_CMD_ERROR and -122 is "
                        "SC_AINN_OPEN_FAILED. The camera-side AI app/service must be "
                        "installed and running before this viewer can receive results."
                    )
                self.ai_backend = "ainn"
                enabled_status = self.ai.scAinnModuleGetEnabled(self.module, ctypes.byref(enabled))
                mode_status = self.ai.scAinnModuleGetWorkMode(self.module, ctypes.byref(mode))
                name_status, name = self.get_ainn_param(1)
                version_status, version = self.get_ainn_param(0)
        else:
            self.ai_backend = None

        if start_stream:
            self._check(self.base.scStartStream(self.device), "scStartStream")
            self.stream_started = True
        self.diagnostics.update({
            "ai_backend": self.ai_backend,
            "ainn_enabled_query_status": enabled_status,
            "ainn_enabled": bool(enabled.value),
            "ainn_mode_query_status": mode_status,
            "ainn_mode": int(mode.value),
            "ainn_mode_name": "AINN_CONTINUOUS_RUN_MODE" if mode.value == AINN_CONTINUOUS_RUN_MODE else "OTHER",
            "algorithm_name_query_status": name_status,
            "algorithm_name": name,
            "algorithm_version_query_status": version_status,
            "algorithm_version": version,
            "sync_policy": "exact deviceTimestamp == resultTimestamp",
            "stream_start_status": SC_OK if start_stream else None,
        })
        return dict(self.diagnostics)

    def get_ainn_param(self, param_id: int) -> tuple[int, str]:
        input_value = ctypes.c_uint32(param_id)
        output = ctypes.create_string_buffer(256)
        output_size = ctypes.c_uint32(len(output))
        status = self.ai.scAinnModuleGetParam(
            self.module,
            ctypes.byref(input_value),
            ctypes.sizeof(input_value),
            output,
            ctypes.byref(output_size),
        )
        return status, output.value.decode("utf-8", errors="replace") if status == SC_OK else ""

    def set_ainn_confidence(self, value: float) -> dict[str, float]:
        if not 0 < value < 1:
            raise ValueError('confidence must be between 0 and 1')
        status, text = self.get_ainn_param(0x21)
        self._check(status, 'get AINN confidence')
        before = float(json.loads(text)['Conf_Thresh'])
        if abs(before-value) > 1e-6:
            data = (0x21).to_bytes(4, 'little') + json.dumps({'Conf_Thresh': value}).encode('ascii') + b'\0'
            request = ctypes.create_string_buffer(data)
            response = ctypes.create_string_buffer(256)
            length = ctypes.c_uint32(len(response))
            self._check(self.ai.scAinnModuleSetParam(self.module, request, len(data), response,
                                                   ctypes.byref(length)), 'set AINN confidence')
        status, text = self.get_ainn_param(0x21)
        self._check(status, 'read back AINN confidence')
        active = float(json.loads(text)['Conf_Thresh'])
        if abs(active-value) > 1e-6:
            raise RuntimeError(f'AINN confidence readback mismatch: requested {value}, got {active}')
        return dict(before=before, active=active)

    def get_morph_param(self, param_id: int) -> tuple[int, str]:
        output_pointer = ctypes.c_void_p()
        output_size = ctypes.c_uint16(0)
        status = self.base.scAIModuleGetParam(
            self.device,
            param_id,
            ctypes.byref(output_pointer),
            ctypes.byref(output_size),
        )
        if status != SC_OK or not output_pointer.value or output_size.value == 0:
            return status, ""
        raw = ctypes.string_at(output_pointer, output_size.value).rstrip(b"\0")
        return status, raw.decode("utf-8", errors="replace")

    def close(self) -> list[tuple[str, int]]:
        statuses: list[tuple[str, int]] = []
        if self.stream_started:
            statuses.append(("scStopStream", self.base.scStopStream(self.device)))
            self.stream_started = False
        if self.module.value:
            statuses.append(("scCloseAinnModule", self.ai.scCloseAinnModule(ctypes.byref(self.module))))
        if self.device.value:
            statuses.append(("scCloseDevice", self.base.scCloseDevice(ctypes.byref(self.device))))
        if self.initialized:
            statuses.append(("scShutdown", self.base.scShutdown()))
            self.initialized = False
        return statuses


class PairStore:
    def __init__(self, max_items: int = 180):
        self.max_items = max_items
        self.frames: OrderedDict[int, FrameRecord] = OrderedDict()
        self.results: OrderedDict[int, ResultRecord] = OrderedDict()
        self.lock = threading.Lock()
        self.pairs: queue.Queue[PairedRecord] = queue.Queue(maxsize=4)

    def add_frame(self, frame: FrameRecord) -> None:
        with self.lock:
            self.frames[frame.timestamp_ms] = frame
            result = self.results.pop(frame.timestamp_ms, None)
            if result is not None:
                self.frames.pop(frame.timestamp_ms, None)
                self._publish(PairedRecord(frame, result))
            self._trim(self.frames)

    def add_result(self, result: ResultRecord) -> None:
        with self.lock:
            self.results[result.timestamp_ms] = result
            frame = self.frames.pop(result.timestamp_ms, None)
            if frame is not None:
                self.results.pop(result.timestamp_ms, None)
                self._publish(PairedRecord(frame, result))
            self._trim(self.results)

    def latest_frame(self) -> FrameRecord | None:
        with self.lock:
            return next(reversed(self.frames.values()), None) if self.frames else None

    def pending_counts(self) -> tuple[int, int]:
        with self.lock:
            return len(self.frames), len(self.results)

    def _publish(self, pair: PairedRecord) -> None:
        while self.pairs.full():
            try:
                self.pairs.get_nowait()
            except queue.Empty:
                break
        self.pairs.put_nowait(pair)

    def _trim(self, mapping: OrderedDict[int, Any]) -> None:
        while len(mapping) > self.max_items:
            mapping.popitem(last=False)


class SessionStats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frames = 0
        self.results = 0
        self.result_parse_errors = 0
        self.frame_errors = 0
        self.result_errors = 0
        self.frame_timeouts = 0
        self.result_timeouts = 0
        self.pairs_displayed = 0
        self.started = time.monotonic()
        self.result_times: list[float] = []

    def result_hz(self) -> float:
        with self.lock:
            now = time.monotonic()
            recent = [value for value in self.result_times if now - value <= 5.0]
            self.result_times[:] = recent
            if len(recent) < 2:
                return 0.0
            return (len(recent) - 1) / (recent[-1] - recent[0])

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "elapsed_sec": round(time.monotonic() - self.started, 3),
                "color_frames": self.frames,
                "ainn_results": self.results,
                "exact_pairs_displayed": self.pairs_displayed,
                "result_parse_errors": self.result_parse_errors,
                "frame_errors": self.frame_errors,
                "result_errors": self.result_errors,
                "frame_timeouts": self.frame_timeouts,
                "result_timeouts": self.result_timeouts,
            }


def frame_to_bgr(frame: ScFrame) -> np.ndarray:
    if not frame.pFrameData or frame.width == 0 or frame.height == 0:
        raise ValueError("empty color frame")
    if frame.pixelFormat in {
        SC_PIXEL_FORMAT_RGB_888_JPEG,
        SC_PIXEL_FORMAT_BGR_888_JPEG,
        SC_PIXEL_FORMAT_RGB_888,
        SC_PIXEL_FORMAT_BGR_888,
    }:
        needed = int(frame.width) * int(frame.height) * 3
        if frame.dataLen < needed:
            raise ValueError(f"24-bit color frame is short: {frame.dataLen} < {needed}")
        raw = ctypes.string_at(frame.pFrameData, needed)
        image = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width, 3).copy()
        if frame.pixelFormat in {SC_PIXEL_FORMAT_RGB_888_JPEG, SC_PIXEL_FORMAT_RGB_888}:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    if frame.pixelFormat in {SC_PIXEL_FORMAT_RGB_565, SC_PIXEL_FORMAT_BGR_565}:
        needed = int(frame.width) * int(frame.height) * 2
        if frame.dataLen < needed:
            raise ValueError(f"16-bit color frame is short: {frame.dataLen} < {needed}")
        raw = ctypes.string_at(frame.pFrameData, needed)
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width, 2).copy()
        if frame.pixelFormat == SC_PIXEL_FORMAT_RGB_565:
            rgb = cv2.cvtColor(packed, cv2.COLOR_BGR5652RGB)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(packed, cv2.COLOR_BGR5652BGR)
    raise ValueError(f"unsupported color pixel format: {frame.pixelFormat}")


def draw_pair(pair: PairedRecord, result_hz: float) -> np.ndarray:
    image = pair.frame.image.copy()
    decoded = pair.result.decoded
    classes = decoded.get("classes", [])
    boxes = decoded.get("boxes", [])
    centers = decoded.get("world_centers", [])
    for index, box in enumerate(boxes):
        left, top, right, bottom = [int(round(value)) for value in box]
        left = max(0, min(image.shape[1] - 1, left))
        right = max(0, min(image.shape[1] - 1, right))
        top = max(0, min(image.shape[0] - 1, top))
        bottom = max(0, min(image.shape[0] - 1, bottom))
        label = classes[index] if index < len(classes) else "object"
        if index < len(centers) and centers[index]:
            label = f"{label}  xyz={centers[index]}"
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_top = max(0, top - text_height - 8)
        cv2.rectangle(image, (left, label_top), (left + text_width + 6, top), (0, 128, 0), -1)
        cv2.putText(image, label, (left + 3, max(text_height + 1, top - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    header = (
        f"EXACT SYNC  frame={pair.frame.index} result={pair.result.index} "
        f"ts={pair.frame.timestamp_ms} count={decoded.get('declared_count')} "
        f"result~{result_hz:.1f}Hz confidence=N/A"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return image


def draw_waiting(frame: FrameRecord, pending: tuple[int, int]) -> np.ndarray:
    image = frame.image.copy()
    text = f"LIVE COLOR frame={frame.index} ts={frame.timestamp_ms} | waiting exact AI pair F={pending[0]} R={pending[1]}"
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    return image


def draw_color_only(frame: FrameRecord) -> np.ndarray:
    image = frame.image.copy()
    text = f"COLOR-ONLY DIAGNOSTIC frame={frame.index} ts={frame.timestamp_ms} | AI interface not opened"
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    return image


def validate_ipv4(value: str) -> str:
    pieces = value.split(".")
    if len(pieces) != 4 or any(not piece.isdigit() or str(int(piece)) != piece or not 0 <= int(piece) <= 255 for piece in pieces):
        raise argparse.ArgumentTypeError("a canonical explicit IPv4 address is required")
    return value


def capture_worker(sdk: ScepterSdk, store: PairStore, stats: SessionStats, stop: threading.Event) -> None:
    while not stop.is_set():
        ready = ScFrameReady()
        status = sdk.base.scGetFrameReady(sdk.device, 100, ctypes.byref(ready))
        if status == SC_GET_FRAME_READY_TIME_OUT:
            with stats.lock:
                stats.frame_timeouts += 1
            continue
        if status != SC_OK:
            with stats.lock:
                stats.frame_errors += 1
            continue
        if not ready.color:
            continue
        frame = ScFrame()
        status = sdk.base.scGetFrame(sdk.device, SC_COLOR_FRAME, ctypes.byref(frame))
        if status != SC_OK:
            with stats.lock:
                stats.frame_errors += 1
            continue
        try:
            image = frame_to_bgr(frame)
        except ValueError as error:
            print(f"Color frame rejected: {error}", file=sys.stderr)
            with stats.lock:
                stats.frame_errors += 1
            continue
        store.add_frame(FrameRecord(int(frame.frameIndex), int(frame.deviceTimestamp), int(frame.pixelFormat), image))
        with stats.lock:
            stats.frames += 1


def result_worker(
    sdk: ScepterSdk,
    store: PairStore,
    stats: SessionStats,
    stop: threading.Event,
    jsonl_stream: Any,
) -> None:
    while not stop.is_set():
        result = ScAinnResult()
        if sdk.ai_backend == "morph":
            status = sdk.base.scAIModuleGetResult(sdk.device, 100, ctypes.byref(result))
        elif sdk.ai_backend == "ainn":
            status = sdk.ai.scAinnModuleGetResult(sdk.module, 100, ctypes.byref(result))
        else:
            return
        if status == SC_GET_AI_RESULT_TIME_OUT:
            with stats.lock:
                stats.result_timeouts += 1
            continue
        if status != SC_OK:
            with stats.lock:
                stats.result_errors += 1
            continue
        payload = ctypes.string_at(result.pResultData, result.dataLen) if result.pResultData and result.dataLen else b""
        payload_text = payload.rstrip(b"\0").decode("utf-8", errors="replace")
        received = time.monotonic()
        try:
            decoded = decode_detection_payload(parse_payload(payload_text))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            decoded = {
                "declared_count": None,
                "classes": [],
                "world_centers": [],
                "boxes": [],
                "parse_error": str(error),
                "confidence_available": False,
            }
            with stats.lock:
                stats.result_parse_errors += 1
        record = ResultRecord(int(result.resultIndex), int(result.resultTimestamp), payload_text, decoded, received)
        store.add_result(record)
        jsonl_stream.write(json.dumps({
            "host_received_unix": time.time(),
            "result_index": record.index,
            "result_timestamp_ms": record.timestamp_ms,
            "data_len": int(result.dataLen),
            "payload_text": payload_text,
        }, ensure_ascii=False, separators=(",", ":")) + "\n")
        jsonl_stream.flush()
        with stats.lock:
            stats.results += 1
            stats.result_times.append(received)


def run_self_test(output: Path) -> int:
    if (ctypes.sizeof(ScFrame) != 36 or ctypes.sizeof(ScFrameReady) != 4
            or ctypes.sizeof(ScAinnResult) != 24 or ctypes.sizeof(ScDeviceInfo) != 149):
        raise RuntimeError(
            f"ctypes ABI mismatch: ScFrame={ctypes.sizeof(ScFrame)}, "
            f"ScFrameReady={ctypes.sizeof(ScFrameReady)}, ScAinnResult={ctypes.sizeof(ScAinnResult)}, "
            f"ScDeviceInfo={ctypes.sizeof(ScDeviceInfo)}"
        )
    sdk = ScepterSdk(BIN_DIR)
    del sdk
    image = np.full((480, 640, 3), 32, dtype=np.uint8)
    frame = FrameRecord(10, 123456, SC_PIXEL_FORMAT_BGR_888, image)
    result = ResultRecord(
        7,
        123456,
        "",
        {
            "declared_count": 1,
            "classes": ["person"],
            "world_centers": ["1, 2, 3"],
            "boxes": [[120, 80, 420, 400]],
            "confidence_available": False,
        },
        time.monotonic(),
    )
    rendered = draw_pair(PairedRecord(frame, result), 10.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), rendered):
        raise RuntimeError(f"failed to write self-test image: {output}")
    print(f"SELF_TEST=PASS\nABI=ScFrame:36,ScFrameReady:4,ScAinnResult:24,ScDeviceInfo:149\nOUTPUT={output.resolve()}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", type=validate_ipv4, help="explicit NYX310 IPv4 address; no default")
    parser.add_argument("--output", type=Path, default=ROOT / "captures" / "live")
    parser.add_argument("--duration-sec", type=int, default=0, help="0 means run until Q/Esc/Ctrl+C")
    parser.add_argument(
        "--allow-sdk-discovery",
        action="store_true",
        help="allow the vendor SDK's 3-second device discovery broadcast; exact requested IP is still enforced",
    )
    parser.add_argument(
        "--color-only-diagnostic",
        action="store_true",
        help="open and display only the camera color stream; do not open or read either AI backend",
    )
    parser.add_argument("--self-test", action="store_true", help="render a synthetic overlay without initializing or connecting a camera")
    parser.add_argument("--self-test-output", type=Path, default=ROOT / "local_checks" / "python_viewer" / "self_test.jpg")
    args = parser.parse_args()
    if args.duration_sec < 0:
        parser.error("--duration-sec must be >= 0")
    if not args.self_test and not args.ip:
        parser.error("--ip is required unless --self-test is used")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test(args.self_test_output)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    screenshots = output / "screenshots"
    screenshots.mkdir(exist_ok=True)
    stop = threading.Event()

    def request_stop(*_: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)

    old_cwd = Path.cwd()
    os.chdir(BIN_DIR)
    sdk = ScepterSdk(BIN_DIR)
    store = PairStore()
    stats = SessionStats()
    session: dict[str, Any] = {}
    failure: str | None = None
    results_stream = None
    threads: list[threading.Thread] = []
    latest_rendered: np.ndarray | None = None
    started = time.monotonic()
    try:
        session = sdk.open(
            args.ip,
            allow_sdk_discovery=args.allow_sdk_discovery,
            require_ai=not args.color_only_diagnostic,
        )
        session["started_unix"] = time.time()
        session["viewer"] = str(Path(__file__).resolve())
        (output / "session.json").write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(session, ensure_ascii=False, indent=2))
        if sdk.ai_backend is not None and not session.get("ainn_enabled"):
            print("WARNING: AINN reports disabled. This viewer will not enable it.", file=sys.stderr)
        if sdk.ai_backend is not None and session.get("ainn_mode") != AINN_CONTINUOUS_RUN_MODE:
            print("WARNING: AINN is not in continuous mode. This viewer will not change it.", file=sys.stderr)

        threads = [
            threading.Thread(target=capture_worker, name="camera-frames", args=(sdk, store, stats, stop), daemon=True)
        ]
        if sdk.ai_backend is not None:
            results_stream = (output / "results.jsonl").open("a", encoding="utf-8", buffering=1)
            threads.append(
                threading.Thread(
                    target=result_worker,
                    name=f"{sdk.ai_backend}-results",
                    args=(sdk, store, stats, stop, results_stream),
                    daemon=True,
                )
            )
        for thread in threads:
            thread.start()

        cv2.namedWindow("NYX310 AINN - exact synchronized result", cv2.WINDOW_NORMAL)
        while not stop.is_set():
            if args.duration_sec and time.monotonic() - started >= args.duration_sec:
                break
            try:
                pair = store.pairs.get(timeout=0.03)
                latest_rendered = draw_pair(pair, stats.result_hz())
                with stats.lock:
                    stats.pairs_displayed += 1
            except queue.Empty:
                frame = store.latest_frame()
                if frame is not None:
                    if args.color_only_diagnostic:
                        latest_rendered = draw_color_only(frame)
                    else:
                        latest_rendered = draw_waiting(frame, store.pending_counts())
            if latest_rendered is not None:
                cv2.imshow("NYX310 AINN - exact synchronized result", latest_rendered)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")) and latest_rendered is not None:
                target = screenshots / f"annotated_{int(time.time() * 1000)}.jpg"
                cv2.imwrite(str(target), latest_rendered)
                print(f"Saved {target}")
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        print(f"ERROR: {failure}", file=sys.stderr)
        if sdk.diagnostics:
            session = dict(sdk.diagnostics)
            session["failure"] = failure
            (output / "session.json").write_text(
                json.dumps(session, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=2.0)
        if results_stream is not None:
            results_stream.close()
        close_statuses = sdk.close()
        cv2.destroyAllWindows()
        os.chdir(old_cwd)
        if latest_rendered is not None:
            cv2.imwrite(str(output / "last_display.jpg"), latest_rendered)
        summary = stats.snapshot()
        summary["close_statuses"] = [{"operation": name, "status": status} for name, status in close_statuses]
        summary["camera_test_executed"] = bool(session.get("stream_start_status") == SC_OK)
        summary["failure"] = failure
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
