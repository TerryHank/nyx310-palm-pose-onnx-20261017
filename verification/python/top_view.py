"""Read-only SDK depth registration and camera X/Z target map (metres)."""
import ctypes as c
import time

import cv2
import numpy as np

from live_ainn_viewer import ScFrame


class DepthPoint(c.Structure):
    _pack_ = 1
    _fields_ = [("x", c.c_int32), ("y", c.c_int32), ("z", c.c_uint16)]


class Pixel(c.Structure):
    _pack_ = 1
    _fields_ = [("x", c.c_uint16), ("y", c.c_uint16)]


class Vector(c.Structure):
    _pack_ = 1
    _fields_ = [("x", c.c_float), ("y", c.c_float), ("z", c.c_float)]


COLORS = [(80, 210, 255), (255, 175, 95), (160, 240, 100), (220, 140, 240)]


class DepthMapper:
    def __init__(self, sdk, sample_step=6):
        self.sdk = sdk
        self.sample_step = sample_step
        sdk.base.scConvertDepthFrameToPointCloudVector.argtypes = [c.c_void_p, c.POINTER(ScFrame), c.POINTER(Vector)]
        sdk.base.scConvertDepthFrameToPointCloudVector.restype = c.c_int
        sdk.base.scTransformDepthPointToColorPoint.argtypes = [c.c_void_p, DepthPoint, Pixel, c.POINTER(Pixel)]
        sdk.base.scTransformDepthPointToColorPoint.restype = c.c_int

    def capture(self, ready, color_frame):
        start = time.perf_counter()
        info = {"status": "depth_not_ready", "matched": False, "registered_points": 0}
        empty = (np.empty((0, 2)), np.empty((0, 3)))
        if not ready.bits & 1:
            return (*empty, info)
        depth = ScFrame()
        status = self.sdk.base.scGetFrame(self.sdk.device, 0, c.byref(depth))
        info.update(sdk_status=status, depth_timestamp_ms=int(depth.deviceTimestamp))
        if status != 0:
            info["status"] = "depth_read_failed"
            return (*empty, info)
        if depth.deviceTimestamp != color_frame.deviceTimestamp:
            info["status"] = "timestamp_mismatch"
            return (*empty, info)
        if depth.pixelFormat != 0 or not depth.pFrameData or depth.dataLen < depth.width * depth.height * 2:
            info["status"] = "invalid_depth_format"
            return (*empty, info)
        info.update(matched=True, depth_size=[int(depth.width), int(depth.height)])
        mm = np.frombuffer(c.string_at(depth.pFrameData, depth.width * depth.height * 2), dtype=np.uint16).reshape(depth.height, depth.width)
        cloud = np.empty((depth.height, depth.width, 3), dtype=np.float32)
        status = self.sdk.base.scConvertDepthFrameToPointCloudVector(self.sdk.device, c.byref(depth), cloud.ctypes.data_as(c.POINTER(Vector)))
        if status != 0:
            info.update(status="point_cloud_failed", sdk_status=status)
            return (*empty, info)
        # Sparse registration uses the manufacturer's calibration, distortion and
        # depth-to-colour transform. Never scale RGB pixels to depth pixels.
        yy, xx = np.mgrid[0:depth.height:self.sample_step, 0:depth.width:self.sample_step]
        zz = mm[yy, xx]
        valid = (zz > 0) & (zz < 65535)
        xs, ys, zs = xx[valid], yy[valid], zz[valid]
        uv, xyz = [], []
        size = Pixel(color_frame.width, color_frame.height)
        mapping_failures = 0
        for x, y, z in zip(xs, ys, zs):
            point = Pixel()
            st = self.sdk.base.scTransformDepthPointToColorPoint(
                self.sdk.device, DepthPoint(int(x), int(y), int(z)), size, c.byref(point))
            if st != 0:
                mapping_failures += 1
                continue
            p = cloud[y, x]
            if point.x < size.x and point.y < size.y and np.isfinite(p).all() and p[2] > 0:
                uv.append((point.x, point.y))
                xyz.append(p / 1000.0)
        info.update(status="ok", registered_points=len(uv), mapping_failures=mapping_failures,
                    registration_ms=round((time.perf_counter() - start) * 1000, 2))
        return np.asarray(uv).reshape(-1, 2), np.asarray(xyz).reshape(-1, 3), info


def fit_plane_map(uv, xyz):
    """Fit an RGB-pixel -> camera X/Z map from the visible near-horizontal plane.

    This is a fallback for RGB pixels outside the ToF/color overlap.  The
    returned map is an estimate and must never be reported as a measured depth.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(uv) != len(xyz) or len(uv) < 40:
        return None
    valid = (np.isfinite(uv).all(axis=1) & np.isfinite(xyz).all(axis=1)
             & (xyz[:, 2] > 0))
    if not valid.any():
        return None
    uv, xyz = uv[valid], xyz[valid]
    lower = uv[:, 1] >= np.percentile(uv[:, 1], 65)
    candidate = np.flatnonzero(lower)
    if len(candidate) < 30:
        candidate = np.arange(len(uv))
    points = xyz[candidate]
    if len(points) < 30:
        return None
    rng = np.random.default_rng(17)
    best = None
    for _ in range(160):
        sample = rng.choice(len(points), 3, replace=False)
        a, b, c = points[sample]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        if abs(normal[1]) < .65:
            continue
        offset = -float(normal @ a)
        distance = np.abs(points @ normal + offset)
        inliers = distance < .03
        count = int(inliers.sum())
        if best is None or count > best[0]:
            best = count, normal, offset, inliers
    if best is None or best[0] < 30:
        return None
    inlier_indices = candidate[best[3]]
    homography, _ = cv2.findHomography(
        uv[inlier_indices].astype(np.float32),
        xyz[inlier_indices][:, [0, 2]].astype(np.float32),
        cv2.RANSAC,
        .03,
    )
    if homography is None or not np.isfinite(homography).all():
        return None
    return {
        "homography": homography,
        "plane": (best[1], best[2]),
        "inliers": int(best[0]),
    }


def locate_target(box, uv, xyz, plane_map=None):
    """Estimate the visible surface from the central 40% of a detection box."""
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    keep = ((uv[:, 0] >= left + w * .3) & (uv[:, 0] <= right - w * .3)
            & (uv[:, 1] >= top + h * .3) & (uv[:, 1] <= bottom - h * .3))
    points = xyz[keep]
    missing = {"status": "insufficient_depth", "xyz_m": None, "map_xz_m": None, "samples": len(points)}
    if len(points) >= 12:
        z = points[:, 2]
        median_z = np.median(z)
        spread = float(np.percentile(z, 75) - np.percentile(z, 25))
        if spread <= max(.35, .25 * median_z):
            inliers = points[np.abs(z - median_z) <= max(.1, .08 * median_z)]
            if len(inliers) >= 12:
                position = np.median(inliers, axis=0)
                return {"status": "ok", "xyz_m": [round(float(v), 3) for v in position],
                        "map_xz_m": [round(float(v), 3) for v in position[[0, 2]]],
                        "samples": len(inliers), "depth_iqr_m": round(spread, 3),
                        "method": "central_roi_surface_median"}
        else:
            missing = dict(missing, status="mixed_depth", depth_iqr_m=round(spread, 3))

    if plane_map is not None:
        left, top, right, bottom = box
        center = np.array([[[((left + right) * .5), ((top + bottom) * .5)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(center, plane_map["homography"])[0, 0]
        if np.isfinite(mapped).all():
            x, z = map(float, mapped)
            normal, offset = plane_map["plane"]
            y = float(-(normal[0] * x + normal[2] * z + offset) / normal[1])
            return {
                "status": "rgb_plane_estimate",
                "xyz_m": [round(x, 3), round(y, 3), round(z, 3)],
                "map_xz_m": [round(x, 3), round(z, 3)],
                "samples": 0,
                "method": "rgb_pixel_to_visible_plane_homography",
                "plane_inliers": plane_map["inliers"],
            }
    return missing


def compose_view(rgb, detections, depth_info, extent=8.0, draw_rgb_labels=True, show_target_ids=True):
    """Equal-scale X/Z map with camera at bottom centre; no floor calibration."""
    panel = np.full((700, 640, 3), (27, 24, 20), dtype=np.uint8)
    def text(value, xy, color=(205, 210, 215), scale=.48):
        cv2.putText(panel, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    text("TARGET MAP  /  camera X-Z (metres)", (20, 26), scale=.62)
    text("Up = farther   Right = camera right", (20, 50))
    text("Floor tilt NOT calibrated; visible-surface estimates", (20, 73), (90, 180, 245), .43)
    small_scale = extent <= 2.0
    text("Minor grid 0.01m; scale is visual, not 1cm accuracy", (20, 96),
         (90, 180, 245), .40 if small_scale else .36)
    origin = (320, 620)
    scale = 500 / extent
    def pixel(x, z):
        return int(origin[0] + x * scale), int(origin[1] - z * scale)
    major_step = .1 if small_scale else 1.0
    minor_step = .01 if small_scale else .1
    for v in np.arange(0, extent + minor_step * .5, minor_step):
        a, b = pixel(-extent/2, v), pixel(extent/2, v)
        major = abs((v / major_step) - round(v / major_step)) < 1e-6
        cv2.line(panel, a, b, (55, 55, 55) if major else (38, 38, 38), 1)
        if major:
            text(f"{v:.2f}m" if small_scale else f"{v:g}m", (25, a[1] + 4), scale=.4)
    for v in np.arange(-extent/2, extent/2 + minor_step * .5, minor_step):
        a, b = pixel(v, 0), pixel(v, extent)
        major = abs((v / major_step) - round(v / major_step)) < 1e-6
        cv2.line(panel, a, b, (55, 55, 55) if major else (38, 38, 38), 1)
        if major:
            text(f"{v:.2f}" if small_scale else f"{v:g}", (a[0] - 12, 641), scale=.4)
    cv2.drawMarker(panel, origin, (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 18, 2)
    text("CAMERA", (288, 660), scale=.45)
    valid_count, estimate_count, outside = 0, 0, 0
    for i, detection in enumerate(detections):
        location = detection["position"]
        if location["status"] not in (
            "ok", "rgb_plane_estimate", "skeleton_mean", "held"
        ):
            continue
        valid_count += 1
        estimate_count += int(location["status"] == "rgb_plane_estimate")
        x, z = location["map_xz_m"]
        if not (-extent/2 <= x <= extent/2 and 0 <= z <= extent):
            outside += 1
            continue
        px, py = pixel(x, z)
        if location["status"] == "rgb_plane_estimate":
            color = (0, 165, 255)
        elif location["status"] == "skeleton_mean":
            color = (255, 210, 0)
        elif location["status"] == "held":
            color = (130, 130, 130)
        else:
            color = COLORS[i % len(COLORS)] if show_target_ids else COLORS[0]
        cv2.circle(panel, (px, py), 7, color, -1, cv2.LINE_AA)
        source = (
            "RGB EST" if location["status"] == "rgb_plane_estimate"
            else "SKELETON" if location["status"] == "skeleton_mean"
            else "HOLD" if location["status"] == "held"
            else "ToF"
        )
        label = f"{('#'+str(i+1)+' ') if show_target_ids else ''}{source} ({x:.2f}, {z:.2f})m"
        # Stack labels above the map so near-coincident targets remain readable.
        text(label, (82, 112 + 17 * i), color, .4)
        if show_target_ids:
            text(str(i+1), (px+9, py-6), color, .45)
    text(f"Located {valid_count}/{len(detections)} | RGB estimate {estimate_count} | out of range {outside} | depth: {depth_info['status']}", (15, 688), scale=.4)
    preview = rgb.copy()
    for i, detection in enumerate(detections):
        if detection.get("rgb_visible") is False:
            continue
        x, y, r, b = [int(v) for v in detection["xyxy"]]
        location = detection["position"]
        label = f"#{i+1} depth unavailable"
        if location["status"] == "ok":
            mx, mz = location["map_xz_m"]
            label = f"#{i+1} X={mx:.2f}m Z={mz:.2f}m"
        if detection.get("position_pixel") is not None:
            cx, cy = [int(v) for v in detection["position_pixel"]]
        else:
            cx, cy = int((x+r)/2), int((y+b)/2)
        cv2.circle(preview, (cx, cy), 7, COLORS[i % len(COLORS)], -1)
        if draw_rgb_labels:
            cv2.putText(preview, label, (max(0, cx-100), cy+25), cv2.FONT_HERSHEY_SIMPLEX, .6,
                        COLORS[i % len(COLORS)], 2, cv2.LINE_AA)
    width = int(preview.shape[1] * 700 / preview.shape[0])
    return np.hstack((cv2.resize(preview, (width, 700)), panel))
