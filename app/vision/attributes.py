"""Apparent age and (optional) apparent gender from a face crop - Levi & Hassner (2015) CNNs.

Both are classifiers trained on Adience web photos. They are approximate, can be biased across
age, skin tone and appearance, and gender output is a binary *appearance* guess that cannot
represent gender identity. Gender is therefore opt-in everywhere in this app.
"""
import threading

import cv2
import numpy as np

from .. import model_registry as mr

AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
AGE_MID = np.array([1.0, 5.0, 10.0, 17.5, 28.5, 40.5, 50.5, 70.0])
MEAN_BGR = (78.4263377603, 87.7689143744, 114.895847746)
_lock = threading.Lock()
_nets = {}


def _net(kind):
    if kind not in _nets:
        if not hasattr(cv2.dnn, "readNetFromCaffe"):
            raise RuntimeError("This OpenCV build has no Caffe importer. Install opencv-python>=4.10,<5.")
        proto, weights = (("age_deploy.prototxt", "age_net.caffemodel") if kind == "age"
                          else ("gender_deploy.prototxt", "gender_net.caffemodel"))
        _nets[kind] = cv2.dnn.readNetFromCaffe(str(mr.path_of(proto)), str(mr.path_of(weights)))
    return _nets[kind]


def _ten_crop_probs(face_bgr, kind):
    face = cv2.resize(face_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    s, o = 227, 29
    crops = []
    for cx, cy in [(0, 0), (o, 0), (0, o), (o, o), (o // 2, o // 2)]:
        c = face[cy:cy + s, cx:cx + s]
        crops += [c, c[:, ::-1]]
    blob = cv2.dnn.blobFromImages(crops, 1.0, (s, s), MEAN_BGR, swapRB=False)
    with _lock:
        n = _net(kind)
        n.setInput(blob)
        p = n.forward().mean(axis=0)
    return p / p.sum()


def age_available():
    return mr.is_installed("age_levi_hassner")


def gender_available():
    return mr.is_installed("gender_levi_hassner")


def predict_age(face_bgr):
    p = _ten_crop_probs(face_bgr, "age")
    top = int(p.argmax())
    lo, hi = max(0, top - 1), min(len(p), top + 2)       # neighbourhood mean: robust to stray far modes
    est = float(np.dot(p[lo:hi], AGE_MID[lo:hi]) / p[lo:hi].sum())
    return {"apparent_age": round(est, 1), "age_bucket": AGE_BUCKETS[top], "bucket_probability": round(float(p[top]), 3),
            "bucket_probabilities": {b: round(float(v), 3) for b, v in zip(AGE_BUCKETS, p)},
            "model": "Levi-Hassner CNN (10-crop)",
            "uncertainty_note": "Bucket classifier, typical error several years; no calibrated interval."}


def predict_gender(face_bgr):
    p = _ten_crop_probs(face_bgr, "gender")
    i = int(p.argmax())
    return {"label": ["male-presenting", "female-presenting"][i], "probability": round(float(p[i]), 3),
            "probs": {"male-presenting": round(float(p[0]), 3), "female-presenting": round(float(p[1]), 3)},
            "model": "Levi-Hassner CNN (10-crop)",
            "caveat": "Appearance-based binary guess; error-prone (it misjudged some of the author's own test photos) "
                      "and not a statement about anyone's gender identity."}
