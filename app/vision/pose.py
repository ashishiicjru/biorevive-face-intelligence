"""Head pose, eye-closure (EAR), mouth opening (MAR) and gaze proxy from face landmarks.

Landmark indices follow the MediaPipe Face Mesh topology (468 points + 10 iris points = 478),
which is what the browser's FaceLandmarker returns. Input: array (N,3) of normalised x,y,z.
"""
import math

import cv2
import numpy as np

# Verified against MediaPipe's region sets. NB: MediaPipe calls these RIGHT_EYE / LEFT_EYE from the SUBJECT's
# point of view; here L_EYE/R_EYE mean the eye on the image-left / image-right (mirror-free naming).
L_EYE = [33, 160, 158, 133, 153, 144]       # image-left eye:  outer, top1, top2, inner, bottom2, bottom1
R_EYE = [362, 385, 387, 263, 373, 380]      # image-right eye
MOUTH = {"top": 13, "bottom": 14, "left": 61, "right": 291}
IRIS_L, IRIS_R = 468, 473
# Generic 3D face model, right-handed like the camera: x right, y down, z away from the camera.
# The nose tip is the origin; eyes/mouth lie further from the camera than the nose. (mm, approx.)
_MODEL = {1: (0.0, 0.0, 0.0), 152: (0.0, 63.6, 12.5), 33: (-43.3, -32.7, 26.0),
          263: (43.3, -32.7, 26.0), 61: (-28.9, 28.9, 24.1), 291: (28.9, 28.9, 24.1)}


def _p(lm, i, W, H):
    return np.array([lm[i][0] * W, lm[i][1] * H])


def eye_aspect_ratio(lm, W=1, H=1):
    def ear(ix):
        p = [_p(lm, i, W, H) for i in ix]
        return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]) + 1e-9)
    l, r = ear(L_EYE), ear(R_EYE)
    return float((l + r) / 2), float(l), float(r)


def mouth_aspect_ratio(lm, W=1, H=1):
    t, b = _p(lm, MOUTH["top"], W, H), _p(lm, MOUTH["bottom"], W, H)
    l, r = _p(lm, MOUTH["left"], W, H), _p(lm, MOUTH["right"], W, H)
    return float(np.linalg.norm(t - b) / (np.linalg.norm(l - r) + 1e-9))


def head_pose(lm, W, H):
    """-> dict(yaw, pitch, roll) in degrees, 0/0/0 = facing the camera.
    yaw   + = face turned toward the IMAGE RIGHT (mirrored selfie view: toward the viewer's right)
    pitch + = looking DOWN        roll + = head tilted clockwise in the image"""
    ids = list(_MODEL)
    obj = np.array([_MODEL[i] for i in ids], np.float64)
    img = np.array([[lm[i][0] * W, lm[i][1] * H] for i in ids], np.float64)
    f = float(W)
    cam = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)                      # R = Rz(roll) @ Ry(theta) @ Rx(pitch)
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    theta = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    yaw = -theta                                    # +theta turns the nose toward image LEFT
    return {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1)}


def gaze_ratio(lm, W=1, H=1):
    """Horizontal iris position within the eye (0.5 = centred) averaged over both eyes; None if no iris points."""
    if len(lm) < 478:
        return None
    def ratio(outer, inner, iris):
        o, i, c = _p(lm, outer, W, H), _p(lm, inner, W, H), _p(lm, iris, W, H)
        v = i - o
        return float(np.dot(c - o, v) / (np.dot(v, v) + 1e-9))
    return round((ratio(33, 133, IRIS_L) + ratio(362, 263, IRIS_R)) / 2, 3)


def features(lm, W, H):
    lm = np.asarray(lm, np.float64)
    ear, el, er = eye_aspect_ratio(lm, W, H)
    return {"ear": round(ear, 3), "ear_left": round(el, 3), "ear_right": round(er, 3),
            "mar": round(mouth_aspect_ratio(lm, W, H), 3), "pose": head_pose(lm, W, H), "gaze_h": gaze_ratio(lm, W, H)}
