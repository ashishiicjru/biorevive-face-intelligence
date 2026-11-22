"""Mobile-phone / object detection with YOLOv4-tiny (COCO, via OpenCV DNN).

'cell phone' is COCO class 67. The tiny model is fast but weak on small objects: a phone held near
the camera is found at the default size; a small phone in a wide group photo needs size=832..1024
(slower, more false positives). 'remote' is a frequent look-alike and is reported as `possible_phone`.
"""
import threading

import cv2
import numpy as np

from .. import model_registry as mr

MODEL = "phone_yolov4_tiny"
PHONE_CLASS = "cell phone"
LOOKALIKES = {"remote"}
_lock = threading.Lock()
_model = None
_names = None
_size = None


def available():
    return mr.is_installed(MODEL)


def _load(size):
    global _model, _names, _size
    if _model is None or _size != size:
        net = cv2.dnn.readNetFromDarknet(str(mr.path_of("yolov4-tiny.cfg")), str(mr.path_of("yolov4-tiny.weights")))
        m = cv2.dnn_DetectionModel(net)
        m.setInputParams(size=(size, size), scale=1 / 255.0, swapRB=True)
        _model, _size = m, size
        _names = mr.path_of("coco.names").read_text().strip().split("\n")
    return _model


def detect(bgr, conf=0.3, size=608, nms=0.4, all_objects=False):
    """-> list of {label, confidence, box:[x,y,w,h], is_phone, possible_phone}.
    By default returns only phones (and look-alikes); all_objects=True returns every COCO class."""
    if not available():
        raise RuntimeError("Phone detector not installed. Run: python -m app.download_models phone_yolov4_tiny")
    size = int(max(320, min(1024, round(size / 32) * 32)))
    with _lock:
        m = _load(size)
        classes, scores, boxes = m.detect(bgr, conf, nms)
    out = []
    for c, s, b in zip(np.array(classes).flatten(), np.array(scores).flatten(), boxes):
        name = _names[int(c)]
        is_phone, maybe = name == PHONE_CLASS, name in LOOKALIKES
        if all_objects or is_phone or maybe:
            out.append({"label": name, "confidence": round(float(s), 3), "box": [int(v) for v in b],
                        "is_phone": is_phone, "possible_phone": maybe})
    out.sort(key=lambda d: d["confidence"], reverse=True)
    return out


def phone_present(dets, min_conf=0.3):
    return any(d["is_phone"] and d["confidence"] >= min_conf for d in dets)
