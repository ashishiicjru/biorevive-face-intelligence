"""REST API for the computer-vision features: analyze / phone / texture / study sessions /
snapshots / dashboard. Mounted by app/main.py."""
import base64
import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlmodel import Session, select, func

from . import model_registry as mr
from .database import get_session
from .models import Scan, PhenotypicAssessment, Snapshot, StudyRecord, User
from .vision import attributes, expression, faces, phone, pipeline, texture, texture_ml

router = APIRouter()
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
SNAP_DIR = DATA_DIR / "snapshots"
MAX_UPLOAD = 10 * 1024 * 1024


async def _read_image(upload: UploadFile):
    raw = await upload.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Image too large (max 10 MB).")
    try:
        img = faces.decode_image(raw)
    except faces.NoFaceError as e:
        raise HTTPException(422, str(e))
    if max(img.shape[:2]) > 1920:
        s = 1920 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def _jpeg_b64(img, q=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@router.get("/api/vision/capabilities")
def capabilities():
    return {"expression": expression.available(), "phone": phone.available(), "age": attributes.age_available(),
            "gender": attributes.gender_available(), "face_detector": "ssd_res10" if faces.ssd_ready() else "haar_cascade",
            "landmarker_local": mr.is_installed("face_landmarker_mediapipe"),
            "texture_age_model": texture_ml.available("age"), "texture_gender_model": texture_ml.available("gender"),
            "caffe_importer": faces.has_caffe()}


@router.post("/api/vision/analyze")
async def analyze(image: UploadFile = File(...), landmarks: str | None = Form(None), modules: str = Form("expression,phone,engagement"),
                  session_id: str | None = Form(None), phone_size: int = Form(608), phone_conf: float = Form(0.4),
                  tiles: bool = Form(False), all_objects: bool = Form(False)):
    img = await _read_image(image)
    lm = None
    if landmarks:
        try:
            lm = json.loads(landmarks)
            if not isinstance(lm, list) or len(lm) < 468 or any(len(p) < 2 for p in lm[:5]):
                raise ValueError
        except ValueError:
            raise HTTPException(422, "landmarks must be a JSON list of >=468 [x,y,z] points.")
    mods = {m.strip() for m in modules.split(",") if m.strip()}
    if session_id and len(session_id) > 64:
        raise HTTPException(422, "session_id too long.")
    try:
        return await run_in_threadpool(pipeline.analyze, img, lm, mods, session_id, phone_size, phone_conf, tiles, 12 if tiles else 5, None, all_objects)
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@router.post("/api/vision/phone")
async def phone_detect(image: UploadFile = File(...), conf: float = Form(0.3), size: int = Form(832),
                       all_objects: bool = Form(False), annotate: bool = Form(True)):
    img = await _read_image(image)
    if not phone.available():
        raise HTTPException(503, "Phone detector not installed. Run: python -m app.download_models phone_yolov4_tiny")
    dets = await run_in_threadpool(phone.detect, img, conf, size, 0.4, all_objects)
    res = {"phone_detected": phone.phone_present(dets, conf), "detections": dets, "size": size,
           "note": "YOLOv4-tiny (COCO). Small phones in wide shots may need size 832-1024; 'remote' is a common look-alike."}
    if annotate:
        out = img.copy()
        for d in dets:
            x, y, w, h = d["box"]
            col = (0, 0, 255) if d["is_phone"] else ((0, 165, 255) if d["possible_phone"] else (90, 220, 90))
            cv2.rectangle(out, (x, y), (x + w, y + h), col, max(2, img.shape[1] // 400))
            cv2.putText(out, f'{d["label"]} {d["confidence"]:.2f}', (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, img.shape[1] / 1400), col, 2)
        res["annotated"] = _jpeg_b64(out)
    return res


@router.post("/api/vision/texture")
async def texture_report(image: UploadFile = File(...), include_gender: bool = Form(False)):
    img = await _read_image(image)
    try:
        f = await run_in_threadpool(faces.primary_face, img)
    except faces.NoFaceError as e:
        raise HTTPException(422, str(e))
    crop = faces.crop(img, f["box"], pad=0.15)
    rep = await run_in_threadpool(texture.analyze, crop, True)
    res = {"face_box": list(f["box"]), "detector": f["detector"], "texture": rep}
    if attributes.age_available():
        res["age_cnn"] = await run_in_threadpool(attributes.predict_age, crop)
    if texture_ml.available("age"):
        res["age_texture_model"] = await run_in_threadpool(texture_ml.predict_age, crop)
    if include_gender:
        if attributes.gender_available():
            res["gender_cnn"] = await run_in_threadpool(attributes.predict_gender, crop)
        if texture_ml.available("gender"):
            res["gender_texture_model"] = await run_in_threadpool(texture_ml.predict_gender, crop)
    return res


# ------------------------------------------------------------------ study sessions
@router.post("/api/study/{session_id}/end")
def end_study(session_id: str, user_code: str | None = None, save: bool = True, s: Session = Depends(get_session)):
    summ = pipeline.end_session(session_id)
    if summ is None:
        raise HTTPException(404, "Unknown or already ended session.")
    if save and summ["duration_s"] >= 5:
        s.add(StudyRecord(session_code=f"STUDY-{uuid.uuid4().hex[:10].upper()}", user_code=user_code,
                          duration_s=summ["duration_s"], average_score=summ["average_attention_score"], summary=summ))
        s.commit()
    return summ


@router.get("/api/study/{session_id}/summary")
def study_summary(session_id: str):
    summ = pipeline.session_summary(session_id)
    if summ is None:
        raise HTTPException(404, "Unknown session.")
    return summ


@router.get("/api/study/sessions")
def study_sessions(limit: int = 30, s: Session = Depends(get_session)):
    rows = s.exec(select(StudyRecord).order_by(StudyRecord.started_at.desc()).limit(min(limit, 200))).all()
    return [{"session_code": r.session_code, "started_at": r.started_at.isoformat(), "duration_s": r.duration_s,
             "average_score": r.average_score, "state_percent": r.summary.get("state_percent"),
             "distraction_events": r.summary.get("distraction_events")} for r in rows]


# ------------------------------------------------------------------ snapshots (opt-in, local disk)
@router.post("/api/snapshots")
async def save_snapshot(image: UploadFile = File(...), caption: str = Form(""), source: str = Form("live"),
                        analysis: str = Form("{}"), user_code: str | None = Form(None), s: Session = Depends(get_session)):
    img = await _read_image(image)
    if max(img.shape[:2]) > 1280:
        k = 1280 / max(img.shape[:2]); img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    try:
        info = json.loads(analysis)
        assert isinstance(info, dict) and len(analysis) < 50_000
    except (ValueError, AssertionError):
        info = {}
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    code = "SNAP-" + uuid.uuid4().hex[:10].upper()
    fname = code + ".jpg"
    cv2.imwrite(str(SNAP_DIR / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 88])      # re-encoded: no EXIF/metadata kept
    s.add(Snapshot(snapshot_code=code, user_code=user_code, caption=caption[:200], source=source[:20], filename=fname, analysis=info))
    s.commit()
    return {"snapshot_code": code, "url": f"/api/snapshots/{code}/image"}


@router.get("/api/snapshots")
def list_snapshots(limit: int = 60, s: Session = Depends(get_session)):
    rows = s.exec(select(Snapshot).order_by(Snapshot.created_at.desc()).limit(min(limit, 200))).all()
    return [{"snapshot_code": r.snapshot_code, "caption": r.caption, "source": r.source, "created_at": r.created_at.isoformat(),
             "url": f"/api/snapshots/{r.snapshot_code}/image", "analysis": r.analysis} for r in rows]


@router.get("/api/snapshots/{code}/image")
def snapshot_image(code: str, s: Session = Depends(get_session)):
    r = s.exec(select(Snapshot).where(Snapshot.snapshot_code == code)).first()
    p = SNAP_DIR / (r.filename if r else "missing")
    if not r or not p.exists():
        raise HTTPException(404, "Snapshot not found.")
    return FileResponse(p, media_type="image/jpeg")


@router.delete("/api/snapshots/{code}")
def delete_snapshot(code: str, s: Session = Depends(get_session)):
    r = s.exec(select(Snapshot).where(Snapshot.snapshot_code == code)).first()
    if not r:
        raise HTTPException(404, "Snapshot not found.")
    (SNAP_DIR / r.filename).unlink(missing_ok=True)
    s.delete(r); s.commit()
    return {"deleted": code}


# ------------------------------------------------------------------ dashboard
@router.get("/api/dashboard")
def dashboard(s: Session = Depends(get_session)):
    cnt = lambda m: s.exec(select(func.count()).select_from(m)).one()     # noqa: E731
    sessions = s.exec(select(StudyRecord).order_by(StudyRecord.started_at.desc()).limit(60)).all()
    state_t, emo_t, tot = {}, {}, 0.0
    for r in sessions:
        d = r.duration_s or 0
        tot += d
        for k, v in (r.summary.get("state_percent") or {}).items():
            state_t[k] = state_t.get(k, 0.0) + v * d / 100
        for k, v in (r.summary.get("expression_percent") or {}).items():
            emo_t[k] = emo_t.get(k, 0.0) + v * d / 100
    pct = lambda dct: {k: round(100 * v / tot, 1) for k, v in dct.items()} if tot else {}   # noqa: E731
    bio = s.exec(select(PhenotypicAssessment).order_by(PhenotypicAssessment.assessed_at.desc()).limit(12)).all()
    return {"counts": {"users": cnt(User), "scans": cnt(Scan), "assessments": cnt(PhenotypicAssessment),
                       "study_sessions": cnt(StudyRecord), "snapshots": cnt(Snapshot)},
            "study": {"total_minutes": round(tot / 60, 1),
                      "average_score": round(sum(r.average_score * r.duration_s for r in sessions) / tot, 1) if tot else None,
                      "state_percent": pct(state_t), "expression_percent": pct(emo_t),
                      "recent": [{"t": r.started_at.isoformat(), "score": r.average_score, "minutes": round(r.duration_s / 60, 1)}
                                 for r in reversed(sessions)]},
            "bio_age": [{"t": a.assessed_at.isoformat(), "chronological": a.chronological_age, "phenotypic": a.biological_age}
                        for a in reversed(bio)],
            "capabilities": capabilities()}
