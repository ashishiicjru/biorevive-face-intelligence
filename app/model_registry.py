"""Computer-vision model registry + downloader.

* Built-in registry of models the app can use (checksummed, resumable downloads).
* Extra models can be added by the user (python -m app.download_models --add ...), they are stored
  in models/custom_models.json and downloaded/verified the same way (they are files for your own
  code; the app itself only uses the built-in models).
* Used by the CLI (app/download_models.py), the REST API and the web UI.
"""
import hashlib
import json
import os
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
CUSTOM_FILE = MODEL_DIR / "custom_models.json"

AGE_PROTO = """name: "CNN_age"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 227
input_dim: 227
layer { name: "conv1" type: "Convolution" bottom: "data" top: "conv1" convolution_param { num_output: 96 kernel_size: 7 stride: 4 } }
layer { name: "relu1" type: "ReLU" bottom: "conv1" top: "conv1" }
layer { name: "pool1" type: "Pooling" bottom: "conv1" top: "pool1" pooling_param { pool: MAX kernel_size: 3 stride: 2 } }
layer { name: "norm1" type: "LRN" bottom: "pool1" top: "norm1" lrn_param { local_size: 5 alpha: 0.0001 beta: 0.75 } }
layer { name: "conv2" type: "Convolution" bottom: "norm1" top: "conv2" convolution_param { num_output: 256 pad: 2 kernel_size: 5 } }
layer { name: "relu2" type: "ReLU" bottom: "conv2" top: "conv2" }
layer { name: "pool2" type: "Pooling" bottom: "conv2" top: "pool2" pooling_param { pool: MAX kernel_size: 3 stride: 2 } }
layer { name: "norm2" type: "LRN" bottom: "pool2" top: "norm2" lrn_param { local_size: 5 alpha: 0.0001 beta: 0.75 } }
layer { name: "conv3" type: "Convolution" bottom: "norm2" top: "conv3" convolution_param { num_output: 384 pad: 1 kernel_size: 3 } }
layer { name: "relu3" type: "ReLU" bottom: "conv3" top: "conv3" }
layer { name: "pool5" type: "Pooling" bottom: "conv3" top: "pool5" pooling_param { pool: MAX kernel_size: 3 stride: 2 } }
layer { name: "fc6" type: "InnerProduct" bottom: "pool5" top: "fc6" inner_product_param { num_output: 512 } }
layer { name: "relu6" type: "ReLU" bottom: "fc6" top: "fc6" }
layer { name: "drop6" type: "Dropout" bottom: "fc6" top: "fc6" dropout_param { dropout_ratio: 0.5 } }
layer { name: "fc7" type: "InnerProduct" bottom: "fc6" top: "fc7" inner_product_param { num_output: 512 } }
layer { name: "relu7" type: "ReLU" bottom: "fc7" top: "fc7" }
layer { name: "drop7" type: "Dropout" bottom: "fc7" top: "fc7" dropout_param { dropout_ratio: 0.5 } }
layer { name: "fc8" type: "InnerProduct" bottom: "fc7" top: "fc8" inner_product_param { num_output: 8 } }
layer { name: "prob" type: "Softmax" bottom: "fc8" top: "prob" }
"""

GENDER_PROTO = AGE_PROTO.replace('name: "CNN_age"', 'name: "CNN_gender"').replace("num_output: 8", "num_output: 2")

BUILTIN = {
    "age_levi_hassner": {
        "title": "Apparent-age CNN (Levi-Hassner)", "task": "age estimation", "recommended": True,
        "description": "8-bucket age classifier used by the apparent-age model. Runs via OpenCV DNN.",
        "files": [{"name": "age_net.caffemodel",
                   "url": os.getenv("LEVI_WEIGHTS_URL", "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel"),
                   "size": 45661480,
                   "sha256": "6dde5d07df5ca1d66ff39e525693f05ccfb9d2c437e188fdd1a10d42e57fabd6"}],
        "generated": {"age_deploy.prototxt": AGE_PROTO},
    },
    "expression_effnet_b0": {
        "title": "Facial-expression CNN (EfficientNet-B0, AffectNet 8-class)", "task": "expression recognition", "recommended": True,
        "description": "Anger / Contempt / Disgust / Fear / Happiness / Neutral / Sadness / Surprise. From EmotiEffLib. Trained on AffectNet (non-commercial research data).",
        "files": [{"name": "enet_b0_8_best_afew.onnx", "size": 16039595,
                   "url": "https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx",
                   "sha256": "7aa2ea31c1311f4f8aa9d3fdb085d418dd4e7a48c4b9ed41df8c044f91d0213f"}],
    },
    "phone_yolov4_tiny": {
        "title": "Object detector (YOLOv4-tiny, COCO) - mobile phone detection", "task": "phone / object detection", "recommended": True,
        "description": "80 COCO classes incl. 'cell phone'. Larger input sizes find smaller phones.",
        "files": [
            {"name": "yolov4-tiny.weights", "size": 24251276,
             "url": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
             "sha256": "cf9fbfd0f6d4869b35762f56100f50ed05268084078805f0e7989efe5bb8ca87"},
            {"name": "yolov4-tiny.cfg", "size": 3231,
             "url": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
             "sha256": "f858e3724962eedf3ac44e3b6cb3f0c3d9ed067c306bb831f539c578b924c90e"},
            {"name": "coco.names", "size": 625,
             "url": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
             "sha256": "634a1132eb33f8091d60f2c346ababe8b905ae08387037aed883953b7329af84"},
        ],
    },
    "gender_levi_hassner": {
        "title": "Apparent-gender CNN (Levi-Hassner) - optional", "task": "gender classification", "recommended": False,
        "description": "Binary appearance classifier trained on Adience. Error-prone and cannot represent gender identity; opt-in only.",
        "files": [{"name": "gender_net.caffemodel", "size": 45649168,
                   "url": "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/gender_net.caffemodel",
                   "sha256": "ac7571b281ae078817764b645a20541bd6aa1babeac20a45e6d8de7d61ba0e50"}],
        "generated": {"gender_deploy.prototxt": GENDER_PROTO},
    },
    "face_landmarker_mediapipe": {
        "title": "MediaPipe Face Landmarker (478 landmarks + blendshapes) - for the browser", "task": "face landmarks", "recommended": False,
        "description": "Served locally to the browser so the live pages work offline. The app falls back to Google's CDN copy if not installed. (Checksum not pinned.)",
        "files": [{"name": "face_landmarker.task", "size": None,
                   "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                   "sha256": None}],
    },
    "face_ssd_res10": {
        "title": "Face detector (OpenCV SSD ResNet-10)", "task": "face detection", "recommended": True,
        "description": "More robust than the built-in Haar cascade (angles, lighting). Used before age estimation.",
        "files": [
            {"name": "res10_300x300_ssd_iter_140000.caffemodel", "size": 10666211,
             "url": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
             "sha256": "2a56a11a57a4a295956b0660b4a3d76bbdca2206c4961cea8efe7d95c7cb2f2d"},
            {"name": "face_ssd_deploy.prototxt", "size": 28104,
             "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
             "sha256": "dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9"},
        ],
    },
}


# ---------------------------------------------------------------- registry
def load_custom():
    try:
        return json.loads(CUSTOM_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def registry():
    reg = {k: dict(v, builtin=True) for k, v in BUILTIN.items()}
    for k, v in load_custom().items():
        if k not in reg:
            reg[k] = dict(v, builtin=False, recommended=False)
    return reg


def add_custom(name, url, filename=None, sha256=None, title=None, description=""):
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Model name may only contain letters, digits, - and _.")
    if name in BUILTIN:
        raise ValueError("That name is reserved by a built-in model.")
    fname = filename or Path(url.split("?")[0]).name
    if not fname or "/" in fname or "\\" in fname or fname.startswith("."):
        raise ValueError("Could not derive a safe file name; pass --filename.")
    data = load_custom()
    data[name] = {"title": title or name, "task": "custom", "description": description,
                  "files": [{"name": fname, "url": url, "size": None, "sha256": sha256}]}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data[name]


# ---------------------------------------------------------------- hashing / status
_hash_cache = {}


def sha256_file(path):
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _hash_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _hash_cache[key] = h.hexdigest()
    return _hash_cache[key]


def file_state(f):
    """missing | corrupt | ok | present (present = exists, no checksum known)"""
    p = MODEL_DIR / f["name"]
    if not p.exists():
        return "missing"
    if f.get("size") and p.stat().st_size != f["size"]:
        return "corrupt"
    if f.get("sha256"):
        return "ok" if sha256_file(p) == f["sha256"] else "corrupt"
    return "present"


def model_state(name):
    m = registry()[name]
    states = [file_state(f) for f in m["files"]]
    for g in m.get("generated", {}):
        if not (MODEL_DIR / g).exists():
            states.append("missing")
    if all(s in ("ok", "present") for s in states):
        return "installed"
    if any(s == "corrupt" for s in states):
        return "corrupt"
    return "missing"


def is_installed(name):
    return model_state(name) == "installed"


def path_of(filename):
    return MODEL_DIR / filename


# ---------------------------------------------------------------- downloading
class DownloadError(RuntimeError):
    pass


def download_file(url, dest, size=None, sha256=None, progress=None, retries=3, timeout=60):
    """Resumable, checksum-verified download. Writes dest atomically."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "BioRevive/2.1"}
            if have:
                headers["Range"] = f"bytes={have}-"
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                if have and r.status != 206:      # server ignored Range -> start over
                    have = 0
                mode = "ab" if have else "wb"
                done = have
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 18)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, size)
            if size and part.stat().st_size != size:
                raise DownloadError(f"size {part.stat().st_size} != expected {size}")
            if sha256:
                h = sha256_file(part)
                if h != sha256:
                    part.unlink()
                    raise DownloadError("checksum mismatch (file corrupted or changed upstream)")
            os.replace(part, dest)
            return dest
        except (urllib.error.URLError, OSError, DownloadError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (403, 404):
                break
            time.sleep(min(2 * attempt, 6))
    raise DownloadError(f"{dest.name}: {last}. Download it manually from {url} into {MODEL_DIR}")


_jobs = {}
_jobs_lock = threading.Lock()


def install(name, progress=None, force=False):
    """Install one model (blocking). Re-downloads corrupt or forced files."""
    reg = registry()
    if name not in reg:
        raise KeyError(f"Unknown model '{name}'.")
    m = reg[name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    files = m["files"]
    total = sum(f.get("size") or 0 for f in files) or None
    base = 0
    for f in files:
        dest = MODEL_DIR / f["name"]
        if force or file_state(f) in ("missing", "corrupt"):
            if dest.exists():
                dest.unlink()
            cb = (lambda d, s, b=base: progress(b + d, total)) if progress else None
            download_file(f["url"], dest, f.get("size"), f.get("sha256"), cb)
        base += f.get("size") or 0
    for gname, text in m.get("generated", {}).items():
        (MODEL_DIR / gname).write_text(text, encoding="utf-8")
    return model_state(name)


def install_async(name, force=False):
    """Start a background install; poll with job_status()/overview()."""
    with _jobs_lock:
        j = _jobs.get(name)
        if j and j["state"] == "running":
            return j
        job = {"state": "running", "done": 0, "total": None, "error": None}
        _jobs[name] = job

    def run():
        def prog(d, t):
            job["done"], job["total"] = d, t
        try:
            install(name, prog, force)
            job["state"] = "done"
        except Exception as exc:       # noqa: BLE001 - reported to the UI
            job["state"], job["error"] = "error", str(exc)
    threading.Thread(target=run, daemon=True).start()
    return job


def overview():
    out = []
    for name, m in registry().items():
        total = sum(f.get("size") or 0 for f in m["files"])
        j = _jobs.get(name)
        out.append({"name": name, "title": m["title"], "task": m["task"], "description": m["description"],
                    "recommended": bool(m.get("recommended")), "builtin": m.get("builtin", False),
                    "size_mb": round(total / 1e6, 1) if total else None, "state": model_state(name),
                    "job": dict(j) if j else None, "files": [f["name"] for f in m["files"]]})
    return out
