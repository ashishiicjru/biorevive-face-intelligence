"""Face detection helpers shared by every vision module.

detect_faces(): SSD ResNet-10 when installed (optionally tiled so small faces in group photos are
found), otherwise OpenCV's bundled Haar cascade. Boxes are (x, y, w, h) in pixels.
"""
import threading

import cv2
import numpy as np

from .. import model_registry as mr

SSD_MODEL = "face_ssd_res10"
SSD_CONF = 0.6
_lock = threading.Lock()
_ssd = None
_haar = None


class NoFaceError(RuntimeError):
    """Image problem (no / multiple faces, unreadable image) -> HTTP 422 for API callers."""


def has_caffe():
    return hasattr(cv2.dnn, "readNetFromCaffe")


def ssd_ready():
    return has_caffe() and mr.is_installed(SSD_MODEL)


def _run_ssd(bgr, conf):
    global _ssd
    h, w = bgr.shape[:2]
    with _lock:
        if _ssd is None:
            _ssd = cv2.dnn.readNetFromCaffe(str(mr.path_of("face_ssd_deploy.prototxt")),
                                            str(mr.path_of("res10_300x300_ssd_iter_140000.caffemodel")))
        _ssd.setInput(cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)))
        det = _ssd.forward()[0, 0]
    out = []
    for d in det:
        if d[2] < conf:
            continue
        x0, y0 = max(0, int(d[3] * w)), max(0, int(d[4] * h))
        x1, y1 = min(w, int(d[5] * w)), min(h, int(d[6] * h))
        if x1 - x0 >= 40 and y1 - y0 >= 40:
            out.append(((x0, y0, x1 - x0, y1 - y0), float(d[2])))
    return out


def _nms(boxes_scores, iou=0.4):
    if not boxes_scores:
        return []
    boxes = [list(map(int, b)) for b, _ in boxes_scores]
    scores = [float(s) for _, s in boxes_scores]
    idx = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou)
    return [boxes_scores[i] for i in np.array(idx).flatten()]


def detect_ssd(bgr, tiles=False, conf=SSD_CONF):
    res = _run_ssd(bgr, conf)
    if tiles:
        H, W = bgr.shape[:2]
        g, ov = 3, 0.3
        sh, sw = H / g, W / g
        for r in range(g):
            for c in range(g):
                x0, y0 = int(max(0, c * sw - ov * sw)), int(max(0, r * sh - ov * sh))
                x1, y1 = int(min(W, (c + 1) * sw + ov * sw)), int(min(H, (r + 1) * sh + ov * sh))
                for (x, y, w, h), s in _run_ssd(bgr[y0:y1, x0:x1], conf):
                    res.append(((x + x0, y + y0, w, h), s))
        res = _nms(res)
    return res


def detect_haar(bgr):
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    h, w = gray.shape
    m = max(48, int(min(h, w) * 0.1))
    return [(tuple(int(v) for v in f), 0.5) for f in
            _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(m, m))]


def detect_faces(bgr, tiles=False):
    """-> list of dicts {box:(x,y,w,h), score, detector}, largest first."""
    res, name = [], "haar_cascade"
    if ssd_ready():
        try:
            res, name = detect_ssd(bgr, tiles), "ssd_res10"
        except Exception:      # noqa: BLE001 - optional detector must never break analysis
            res = []
    if not res:
        res, name = detect_haar(bgr), "haar_cascade"
    res.sort(key=lambda t: t[0][2] * t[0][3], reverse=True)
    return [{"box": b, "score": s, "detector": name} for b, s in res]


def primary_face(bgr):
    faces = detect_faces(bgr)
    if not faces:
        raise NoFaceError("No clear frontal face found. Use a well-lit, front-facing photo.")
    if len(faces) > 1:
        a0 = faces[0]["box"][2] * faces[0]["box"][3]
        a1 = faces[1]["box"][2] * faces[1]["box"][3]
        if a1 > 0.35 * a0:
            raise NoFaceError("More than one face found. Use a photo with exactly one person.")
    return faces[0]


def crop(bgr, box, pad=0.15):
    x, y, w, h = box
    p = int(pad * max(w, h))
    H, W = bgr.shape[:2]
    return bgr[max(0, y - p):min(H, y + h + p), max(0, x - p):min(W, x + w + p)]


def box_from_landmarks(lm, W, H, grow=1.15):
    """lm: (N,>=2) normalised landmarks -> pixel box (x,y,w,h), slightly enlarged."""
    xs, ys = lm[:, 0] * W, lm[:, 1] * H
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * grow, (y1 - y0) * grow
    x, y = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    return (x, y, int(min(W - x, w)), int(min(H - y, h)))


def decode_image(raw):
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise NoFaceError("The file is not a readable image.")
    return arr
