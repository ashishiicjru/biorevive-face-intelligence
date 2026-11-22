"""One-call frame analysis used by the live pages and the REST API.

analyze(bgr, landmarks=None, modules={...}, session_id=None, ...) ->
  faces (boxes + expression / age / gender / texture per face), phone detections, head pose / eyes
  from landmarks, and the study-focus state for the session. Models that are not installed are
  reported under `unavailable` instead of raising, so the UI can offer to install them.
"""
import threading
import time

import numpy as np

from . import attributes, engagement, expression, faces, phone, pose, texture

ALL_MODULES = {"expression", "phone", "age", "gender", "texture", "engagement"}
INSTALL = {"expression": "expression_effnet_b0", "phone": "phone_yolov4_tiny", "age": "age_levi_hassner",
           "gender": "gender_levi_hassner"}
_sessions = {}
_lock = threading.Lock()
MAX_SESSIONS, IDLE_S = 50, 2 * 3600


def _check(module):
    ok = {"expression": expression.available, "phone": phone.available, "age": attributes.age_available,
          "gender": attributes.gender_available}.get(module, lambda: True)()
    return ok


def get_session(session_id):
    now = time.monotonic()
    with _lock:
        for k in [k for k, v in _sessions.items() if now - v["last"] > IDLE_S]:
            del _sessions[k]
        if session_id not in _sessions:
            if len(_sessions) >= MAX_SESSIONS:
                oldest = min(_sessions, key=lambda k: _sessions[k]["last"])
                del _sessions[oldest]
            _sessions[session_id] = {"study": engagement.StudySession(), "emo": None, "last": now}
        _sessions[session_id]["last"] = now
        return _sessions[session_id]


def end_session(session_id):
    with _lock:
        s = _sessions.pop(session_id, None)
    return s["study"].summary() if s else None


def session_summary(session_id):
    s = _sessions.get(session_id)
    return s["study"].summary() if s else None


def _norm(box, W, H):
    x, y, w, h = box
    return [round(x / W, 4), round(y / H, 4), round(w / W, 4), round(h / H, 4)]


def analyze(bgr, landmarks=None, modules=None, session_id=None, phone_size=608, phone_conf=0.4,
            tiles=False, max_faces=5, now=None, all_objects=False):
    t0 = time.perf_counter()
    mods = set(modules) if modules else {"expression", "phone", "engagement"}
    mods &= ALL_MODULES
    H, W = bgr.shape[:2]
    timings, unavailable = {}, []
    for m in list(mods):
        if m in INSTALL and not _check(m):
            unavailable.append({"module": m, "model": INSTALL[m],
                                "install": f"python -m app.download_models {INSTALL[m]}"})
            mods.discard(m)

    # ---- faces
    lm = None
    if landmarks is not None and len(landmarks) >= 468:
        lm = np.asarray(landmarks, np.float64)[:, :3]
        found = [{"box": faces.box_from_landmarks(lm, W, H), "score": 1.0, "detector": "landmarks"}]
    else:
        t = time.perf_counter()
        found = faces.detect_faces(bgr, tiles=tiles)
        timings["face_detect_ms"] = round((time.perf_counter() - t) * 1000)
    found = found[:max_faces]
    sess = get_session(session_id) if session_id else None

    out_faces = []
    for idx, f in enumerate(found):
        crop = faces.crop(bgr, f["box"], pad=0.1 if f["detector"] == "landmarks" else 0.15)
        item = {"box": _norm(f["box"], W, H), "box_px": [int(v) for v in f["box"]], "detector": f["detector"]}
        if crop.size == 0 or min(crop.shape[:2]) < 24:
            out_faces.append(item)
            continue
        if "expression" in mods:
            t = time.perf_counter()
            e = expression.predict(crop)
            if sess is not None and idx == 0:          # temporal smoothing on the primary face
                sm = expression.smooth(sess["emo"], e["probs"])
                sess["emo"] = sm
                top = max(sm, key=sm.get)
                e = dict(e, label=top, mood=expression.MOOD[top], confidence=round(sm[top], 4), probs={k: round(v, 4) for k, v in sm.items()},
                         negative_affect=round(sum(v for k, v in sm.items() if k in expression.NEGATIVE), 4))
            item["expression"] = e
            timings["expression_ms"] = round((time.perf_counter() - t) * 1000)
        if "age" in mods:
            item["age"] = attributes.predict_age(crop)
        if "gender" in mods:
            item["gender"] = attributes.predict_gender(crop)
        if "texture" in mods:
            item["texture"] = texture.analyze(crop, with_images=False)
        out_faces.append(item)

    # ---- phone
    phones = []
    if "phone" in mods:
        t = time.perf_counter()
        phones = phone.detect(bgr, conf=phone_conf, size=phone_size, all_objects=all_objects)
        timings["phone_ms"] = round((time.perf_counter() - t) * 1000)
        for d in phones:
            d["box_norm"] = _norm(d["box"], W, H)

    # ---- pose / eyes (needs landmarks)
    feats = pose.features(lm, W, H) if lm is not None else None

    # ---- engagement
    study = None
    if "engagement" in mods and sess is not None:
        primary = out_faces[0] if out_faces else None
        pc = max([d["confidence"] for d in phones if d["is_phone"]], default=0.0)
        study = sess["study"].update(
            now if now is not None else time.monotonic(), face_present=primary is not None,
            pose=feats["pose"] if feats else None, ear=feats["ear"] if feats else None,
            mar=feats["mar"] if feats else None, gaze_h=feats["gaze_h"] if feats else None,
            expression=(primary or {}).get("expression"), phone_conf=pc)
        if len(out_faces) > 1:
            study["notes"] = ["More than one face in view; only the largest is scored."]
        if study["limited_mode"]:
            study.setdefault("notes", []).append("Limited mode: no landmarks, so head and eye cues are unavailable.")

    return {"image": {"width": W, "height": H}, "faces": out_faces, "faces_count": len(out_faces),
            "phones": phones, "phone_detected": any(d["is_phone"] for d in phones), "pose": feats, "engagement": study,
            "unavailable": unavailable, "timings": timings, "total_ms": round((time.perf_counter() - t0) * 1000),
            "modules": sorted(mods)}
