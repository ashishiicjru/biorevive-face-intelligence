"""Facial-expression recognition (EfficientNet-B0 trained on AffectNet, via OpenCV DNN).

IMPORTANT: this classifies the *visible facial configuration* (e.g. a smile), not inner feelings.
Posed, neutral-serious or culturally different expressions are often misread (a stern neutral
face is commonly called 'anger'). Treat the output as a rough signal, never as a diagnosis of mood.
"""
import threading

import cv2
import numpy as np

from .. import model_registry as mr

MODEL = "expression_effnet_b0"
FILE = "enet_b0_8_best_afew.onnx"
CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
# friendly grouping used by the dashboards
MOOD = {"Happiness": "happy", "Surprise": "surprised", "Neutral": "neutral", "Sadness": "sad",
        "Anger": "angry", "Fear": "fearful", "Disgust": "disgusted", "Contempt": "contempt"}
NEGATIVE = {"Anger", "Disgust", "Fear", "Sadness", "Contempt"}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_lock = threading.Lock()
_net = None


def available():
    return mr.is_installed(MODEL)


def _load():
    global _net
    if _net is None:
        _net = cv2.dnn.readNetFromONNX(str(mr.path_of(FILE)))
    return _net


def predict(face_bgr):
    """face_bgr: cropped face. Returns dict(label, mood, confidence, probs{class: p})."""
    if not available():
        raise RuntimeError("Expression model not installed. Run: python -m app.download_models expression_effnet_b0")
    rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
    with _lock:
        net = _load()
        net.setInput(np.ascontiguousarray(x, dtype=np.float32))
        logits = net.forward()[0].astype(np.float64)
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    i = int(p.argmax())
    probs = {c: round(float(v), 4) for c, v in zip(CLASSES, p)}
    return {"label": CLASSES[i], "mood": MOOD[CLASSES[i]], "confidence": round(float(p[i]), 4), "probs": probs,
            "negative_affect": round(float(sum(p[j] for j, c in enumerate(CLASSES) if c in NEGATIVE)), 4),
            "note": "Expression of the face, not a measurement of feelings."}


def smooth(prev, cur, alpha=0.6):
    """Exponential smoothing of probability dicts (for live video)."""
    if not prev:
        return cur
    return {k: alpha * cur[k] + (1 - alpha) * prev.get(k, cur[k]) for k in cur}
