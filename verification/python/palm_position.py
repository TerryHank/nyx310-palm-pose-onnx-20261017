"""Palm reference geometry, robust ToF depth, hand IDs and temporal filtering."""
import math

import cv2
import numpy as np


def palm_geometry(points):
    points = np.asarray(points, np.float32)
    reference = (points[0, :2] + points[9, :2]) * .5
    hull = cv2.convexHull(points[[0, 1, 5, 9, 13, 17], :2]).reshape(-1, 2)
    center = hull.mean(axis=0)
    interior = center + (hull - center) * .70
    return reference, interior


def measure_palm(points, depth_mm, K, distortion, matched=True):
    reference, polygon = palm_geometry(points)
    info = dict(reason="depth_unavailable", reference_uv=reference.tolist(),
                roi_polygon=polygon.tolist(), samples=0, raw_xyz_mm=None)
    if not matched:
        return None, dict(info, reason="timestamp_mismatch")
    h, w = depth_mm.shape
    if not np.isfinite(points).all() or not np.isfinite(reference).all():
        return None, dict(info, reason="invalid_landmarks")
    if not (0 <= reference[0] < w and 0 <= reference[1] < h):
        return None, dict(info, reason="reference_outside_image")
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    # Shrinkage excludes the contour; erosion removes a further RGB-pixel rim.
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    area = int(mask.sum())
    if area < 25:
        return None, dict(info, reason="palm_roi_too_small", roi_pixels=area)
    yy, xx = np.where(mask > 0)
    values = depth_mm[yy, xx].astype(np.float64)
    valid = np.isfinite(values) & (values > 50) & (values < 5000)
    values, xx, yy = values[valid], xx[valid], yy[valid]
    info.update(roi_pixels=area, valid_samples=int(len(values)))
    if len(values) < 12:
        return None, dict(info, reason="insufficient_depth")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    cutoff = float(np.clip(3 * 1.4826 * mad, 20, 80))
    keep = np.abs(values - median) <= cutoff
    inliers = values[keep]
    info.update(depth_mad_mm=mad, samples=int(len(inliers)),
                rejected_samples=int(len(values) - len(inliers)))
    if len(inliers) < 12 or len(inliers) < .6 * len(values):
        return None, dict(info, reason="mixed_depth")
    center = polygon.mean(axis=0)
    quadrants = (xx[keep] >= center[0]).astype(int) + 2 * (yy[keep] >= center[1]).astype(int)
    coverage = len(inliers) / area
    info.update(depth_coverage=coverage, covered_quadrants=int(len(np.unique(quadrants))))
    spatial_support=len(np.unique(quadrants))>=3
    _,singular,axes=np.linalg.svd(polygon-center,full_matrices=False)
    if singular[0]>3*max(singular[1],1e-6):
        # A diagonal, edge-on palm can occupy only two IMAGE quadrants even
        # when depths span the whole palm. Check support along its own axis.
        roi_axis=(polygon-center)@axes[0]
        samples_axis=(np.column_stack((xx[keep],yy[keep]))-center)@axes[0]
        span=float(np.ptp(roi_axis))
        bins=np.clip(((samples_axis-roi_axis.min())/max(span,1e-6)*4).astype(int),0,3)
        axis_coverage=float(np.ptp(samples_axis))/max(span,1e-6)
        spatial_support=len(np.unique(bins))>=3 and axis_coverage>=.5
        info.update(spatial_support_method='palm_major_axis',axis_bins=int(len(np.unique(bins))),axis_coverage=axis_coverage)
    if coverage < .01 or not spatial_support:
        return None, dict(info, reason="insufficient_spatial_support")
    depth = float(np.median(inliers))
    ray = cv2.undistortPoints(reference.reshape(1, 1, 2).astype(np.float64),
                             K, distortion)[0, 0]
    xyz = np.array([ray[0] * depth, ray[1] * depth, depth], np.float64)
    return xyz, dict(info, reason="ok", median_depth_mm=depth, raw_xyz_mm=xyz.tolist())


def open_fingers(world):
    """Count extended fingers from image-plane joint angles and extension ratios."""
    world = np.asarray(world, np.float64)
    count = 0
    for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12),
                               (13, 14, 15, 16), (17, 18, 19, 20)):
        a, b = world[mcp] - world[pip], world[dip] - world[pip]
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator < 1e-12:
            continue
        angle = math.degrees(math.acos(float(np.clip(np.dot(a, b) / denominator, -1, 1))))
        longer = np.linalg.norm(world[tip] - world[0]) > np.linalg.norm(world[pip] - world[0]) * 1.08
        count += int(angle >= 150 and longer)
    return count


class PalmTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, observations, now, image_size):
        available = set(self.tracks)
        assignments = {}
        distances = []
        diagonal = float(np.hypot(*image_size))
        for i, observation in enumerate(observations):
            for key in available:
                track = self.tracks[key]
                distance = np.linalg.norm(observation["reference"] - track["reference"])
                side_penalty = .05 * diagonal if observation["side"] != track["side"] else 0
                distances.append((distance + side_penalty, distance, i, key))
        for _, distance, i, key in sorted(distances):
            if i not in assignments and key in available and distance < .20 * diagonal:
                assignments[i] = key
                available.remove(key)
        active = set()
        for i, observation in enumerate(observations):
            key = assignments.get(i)
            if key is None:
                key = self.next_id
                self.next_id += 1
                self.tracks[key] = dict(
                    xyz=None, time=None, state=observation["gesture"], pending=None,
                    state_count=0, confidence=0., valid=False, reference=observation["reference"],
                )
            track = self.tracks[key]
            active.add(key)
            candidate = observation["gesture"]
            if now - track.get("seen", now) > .3:
                track.update(pending=None, state_count=0)
            track["seen"] = now
            if candidate != track.get("pending"):
                track["pending"], track["state_count"] = candidate, 1
                track["pending_since"] = now
            else:
                track["state_count"] += 1
            if track["state_count"] >= 3 and now - track["pending_since"] >= .12:
                track["state"] = candidate
            track.update(reference=observation["reference"], side=observation["side"],
                         confidence=observation["confidence"], valid=False)
            raw = observation["xyz"]
            if raw is not None:
                elapsed = now - track["time"] if track["time"] is not None else float("inf")
                jump = np.linalg.norm(raw - track["xyz"]) if track["xyz"] is not None else 0
                if elapsed < .5 and jump > max(120., 2000 * elapsed):
                    observation["diagnostics"]["reason"] = "temporal_jump_rejected"
                else:
                    alpha = 1 - math.exp(-elapsed / .12) if elapsed < .5 else 1.
                    track["xyz"] = raw.copy() if track["xyz"] is None else (
                        track["xyz"] * (1 - alpha) + raw * alpha
                    )
                    track["time"], track["valid"] = now, True
            observation["hand_id"] = key
        for key, track in self.tracks.items():
            if key not in active:
                track.update(valid=False, confidence=0., pending=None, state_count=0)
        return [dict(
            hand_id=key, xyz_mm=[round(float(v), 1) for v in track["xyz"]]
            if track["xyz"] is not None else None,
            state=track["state"], confidence=round(float(track["confidence"]), 4),
            valid=bool(track["valid"]),
        ) for key, track in self.tracks.items()]
