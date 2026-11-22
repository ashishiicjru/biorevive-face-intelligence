"""Facial apparent-age estimation (file-based API used by /api/age/apparent).

Backends (AGE_BACKEND env: auto | levi | deepface):
  levi      Levi-Hassner CNN via OpenCV DNN (light, default). Models come from the registry.
  deepface  optional heavier regression model (pip install -r requirements-age.txt)
Face finding and the CNN live in app/vision/faces.py and app/vision/attributes.py.
"""
import os

from . import model_registry as mr
from .vision import attributes, faces
from .vision.faces import NoFaceError       # re-exported: main.py maps it to HTTP 422

AGE_MODEL = "age_levi_hassner"
DET_MODEL = "face_ssd_res10"


def _deepface_available():
    try:
        import deepface  # noqa: F401
        return True
    except Exception:
        return False


def weights_ready():
    return mr.is_installed(AGE_MODEL)


def detector_ready():
    return mr.is_installed(DET_MODEL)


def _choose_backend():
    pref = os.getenv("AGE_BACKEND", "auto").lower()
    if pref in ("deepface", "levi"):
        return pref
    return "deepface" if _deepface_available() else "levi"


def status():
    return {
        "levi_hassner": {"available": faces.has_caffe(), "weights_downloaded": weights_ready(),
                          "weights_path": str(mr.path_of("age_net.caffemodel"))},
        "face_detector": "ssd_res10" if faces.ssd_ready() else "haar_cascade (fallback)",
        "deepface": {"available": _deepface_available()},
        "active_backend": _choose_backend(),
    }


def install_weights(progress=None):
    """Install the age net (required) and the SSD face detector (optional: Haar is the fallback)."""
    mr.install(AGE_MODEL, progress)
    try:
        mr.install(DET_MODEL)
    except Exception:       # noqa: BLE001
        pass
    return str(mr.path_of("age_net.caffemodel"))


def _levi_predict(path):
    bgr = _load_image(path)
    f = faces.primary_face(bgr)
    if not weights_ready():
        install_weights()
    face = faces.crop(bgr, f["box"], pad=0.15 if f["detector"] == "ssd_res10" else 0.25)
    r = attributes.predict_age(face)
    x, y, w, h = f["box"]
    r.update({"face_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}, "face_detector": f["detector"],
              "interval": None, "model": "Levi-Hassner CNN (OpenCV DNN, 10-crop)"})
    return r


def _load_image(path):
    from PIL import Image, ImageOps
    import numpy as np
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception as exc:
        raise NoFaceError("The file is not a readable image.") from exc
    if max(img.size) > 1600:
        img.thumbnail((1600, 1600))
    return np.array(img)[:, :, ::-1].copy()


def _deepface_predict(path):
    try:
        from deepface import DeepFace
    except Exception as exc:
        raise RuntimeError("DeepFace is not installed. pip install -r requirements-age.txt") from exc
    try:
        res = DeepFace.analyze(img_path=path, actions=["age"], enforce_detection=True,
                               detector_backend="opencv", silent=True)
    except ValueError as exc:
        raise NoFaceError("DeepFace could not find a face: " + str(exc)) from exc
    item = res[0] if isinstance(res, list) else res
    return {"apparent_age": round(float(item["age"]), 1), "model": "DeepFace pretrained age estimator",
            "age_bucket": None, "interval": None,
            "uncertainty_note": "No calibrated prediction interval is provided; treat as approximate."}


def predict_apparent_age(image_path):
    try:
        return _deepface_predict(image_path) if _choose_backend() == "deepface" else _levi_predict(image_path)
    except (NoFaceError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError("Age inference failed: " + str(exc)) from exc
