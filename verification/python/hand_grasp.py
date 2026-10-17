"""Rotation/scale invariant hand-pose heuristics, not object-contact detection."""
import numpy as np


def joint_angle(a, b, c):
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(
        np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def pose_features(points, check_geometry=True):
    p = np.asarray(points, np.float64)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return None
    width = max(np.linalg.norm(p[5] - p[17]), .7*np.linalg.norm(p[9]-p[0]))
    if width < 1e-8:
        return None
    p = (p - p[0]) / width
    if check_geometry and not .4 < np.linalg.norm(p[9]) < 3:
        return None
    # Reject collapsed/implausible skeletons before dividing by bone lengths.
    lengths = [np.linalg.norm(p[b] - p[a]) for chain in
               ((0,1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20))
               for a,b in zip(chain,chain[1:])]
    if min(lengths) < 1e-8 or (check_geometry and (min(lengths) < .035 or max(lengths) > 2)):
        return None
    angles, extension = [], []
    for m in (5,9,13,17):
        angles.append([joint_angle(p[0],p[m],p[m+1]),
                       joint_angle(p[m],p[m+1],p[m+2]),
                       joint_angle(p[m+1],p[m+2],p[m+3])])
        extension.append(float(np.linalg.norm(p[m+3]-p[m]) / sum(
            np.linalg.norm(p[j+1]-p[j]) for j in range(m,m+3))))
    angles = np.asarray(angles)
    straight = ((angles[:,0] >= 145) & (angles[:,1] >= 140) &
                (angles[:,2] >= 140) & (np.asarray(extension) > .88))
    curled = (angles[:,1] < 115) | (np.asarray(extension) < .65)
    bent = (angles[:,0] < 135) | (angles[:,1] < 140) | (angles[:,2] < 135)
    # Thumb opposition is necessary to distinguish a grasp from bent fingers.
    opposition = [float(np.linalg.norm(p[4]-p[i])) for i in (8,12,16,20)]
    return dict(angles_deg=angles.tolist(), extension=extension,
                straight=int(straight.sum()), curled=int(curled.sum()),
                bent=int(bent.sum()), thumb_tip_distances=opposition,
                pinch=min(opposition[:2]), opposition=min(opposition))


def classify_grasp(world, image_points, confidence):
    """Both inputs are MODEL-predicted relative 3D, never measured ToF XYZ."""
    world_features = pose_features(world, check_geometry=False)
    image_features = pose_features(image_points, check_geometry=False)
    info = dict(source="model_relative_3d", world=world_features, image=image_features)
    def scores(f):
        a = np.asarray(f["angles_deg"])
        clip = lambda x: np.clip(x, 0, 1)
        extension = np.asarray(f["extension"])
        opened = float(np.mean(.5*clip((a[:,0]-90)/70) +
                               .3*clip((a[:,1]-70)/100) + .2*clip((a[:,2]-70)/100)))
        opened = .8*opened + .2*float(clip((f["opposition"]-.35)/1.0))
        closed = float(.55*np.mean(clip((155-a[:,1])/80)) +
                       .25*np.mean(clip((.95-extension)/.5)) +
                       .20*np.mean(clip((150-a[:,0])/70)))
        grasp = float(.5*np.mean(clip((155-a[:,0])/65)) +
                      .5*clip((1.5-f["pinch"])/1.2))
        grasp = max(grasp, float(.95*clip((.75-f["pinch"])/.5)))
        return np.array([opened, grasp, closed])
    available = [(f,weight) for f,weight in ((world_features,.65),(image_features,.35))
                 if f is not None]
    if not available:
        return None, dict(info, reason="no_usable_landmarks")
    combined = sum(scores(f)*weight for f,weight in available) / sum(w for _,w in available)
    labels = ("OPEN", "GRASP", "CLOSED")
    ordered = np.sort(combined)
    return labels[int(np.argmax(combined))], dict(
        info, reason="closest_geometric_score", model_confidence=float(confidence),
        scores=dict(zip(labels,combined.tolist())), margin=float(ordered[-1]-ordered[-2]),
        score_definition="heuristic similarity, not calibrated probabilities")
