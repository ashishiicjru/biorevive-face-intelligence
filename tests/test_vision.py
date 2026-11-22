import io, json, math, os, tempfile
import cv2
import numpy as np
import pytest
os.environ.setdefault("DATABASE_URL", "sqlite:///" + os.path.join(tempfile.mkdtemp(), "v.db"))
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
from app import model_registry as mr
from app.vision import attributes, engagement, expression, faces, phone, pipeline, pose, texture, texture_ml


# ---------------------------------------------------------------- synthetic landmarks
def make_landmarks(yaw=0, pitch=0, roll=0, ear=0.30, mar=0.1, W=640, H=480):
    """478 normalised landmarks: a physically rotated head model + eyes/mouth with a given EAR / MAR."""
    cam_f, C = float(W), np.array([0, 0, 60.0])
    th, ph, ps = math.radians(-yaw), math.radians(pitch), math.radians(roll)
    Rx = np.array([[1, 0, 0], [0, math.cos(ph), -math.sin(ph)], [0, math.sin(ph), math.cos(ph)]])
    Ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    Rz = np.array([[math.cos(ps), -math.sin(ps), 0], [math.sin(ps), math.cos(ps), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    lm = np.zeros((478, 3))
    ang = np.linspace(0, 2 * np.pi, 478, endpoint=False)
    lm[:, 0], lm[:, 1] = 0.5 + 0.13 * np.cos(ang), 0.5 + 0.2 * np.sin(ang)          # face-oval filler
    for i, p in pose._MODEL.items():
        q = C + R @ (np.array(p) - C) + np.array([0, 0, 540.0])
        lm[i] = [(cam_f * q[0] / q[2] + W / 2) / W, (cam_f * q[1] / q[2] + H / 2) / H, 0]
    for ix in (pose.L_EYE, pose.R_EYE):                                              # eye geometry -> chosen EAR
        o, i = lm[ix[0]], lm[ix[3]]
        w = np.linalg.norm((i - o)[:2] * [W, H]); half = ear * w / 2 / H
        mid = (o + i) / 2
        lm[ix[1]] = lm[ix[2]] = mid + [0, -half, 0]; lm[ix[5]] = lm[ix[4]] = mid + [0, half, 0]
        lm[ix[1]][0] -= .004; lm[ix[2]][0] += .004; lm[ix[5]][0] -= .004; lm[ix[4]][0] += .004
    lm[468] = (lm[33] + lm[133]) / 2; lm[473] = (lm[362] + lm[263]) / 2          # irises centred in the eyes
    cx = (lm[61] + lm[291]) / 2; wm = np.linalg.norm((lm[291] - lm[61])[:2] * [W, H]); hm = mar * wm / H / 2
    lm[13], lm[14] = cx + [0, -hm, 0], cx + [0, hm, 0]
    return lm


# ---------------------------------------------------------------- pose / eyes
@pytest.mark.parametrize("ang", [(0, 0, 0), (20, 0, 0), (-25, 0, 0), (0, 15, 0), (0, -20, 0), (0, 0, 10), (15, 10, -5), (35, -10, 8)])
def test_head_pose_recovers_known_rotation(ang):
    got = pose.head_pose(make_landmarks(*ang), 640, 480)
    assert (got["yaw"], got["pitch"], got["roll"]) == pytest.approx(ang, abs=0.3)


def test_pose_sign_convention_is_physical():
    right, left = make_landmarks(yaw=25), make_landmarks(yaw=-25)
    nose_off = lambda lm: lm[1][0] - (lm[33][0] + lm[263][0]) / 2
    assert nose_off(right) > 0 > nose_off(left)             # yaw>0 = nose toward image right
    down = make_landmarks(pitch=20)
    assert (down[1][1] - (down[33][1] + down[263][1]) / 2) > (make_landmarks(pitch=-20)[1][1] - (make_landmarks(pitch=-20)[33][1] + make_landmarks(pitch=-20)[263][1]) / 2)


def test_ear_and_mar():
    assert pose.features(make_landmarks(ear=0.30), 640, 480)["ear"] == pytest.approx(0.30, abs=0.02)
    assert pose.features(make_landmarks(ear=0.05), 640, 480)["ear"] < 0.1
    assert pose.features(make_landmarks(mar=0.8), 640, 480)["mar"] == pytest.approx(0.8, abs=0.05)


# ---------------------------------------------------------------- engagement engine
def run(fn, n, fps=10):
    s, t, o = engagement.StudySession(), 0.0, None
    for i in range(n):
        o = s.update(t, **fn(i, t)); t += 1 / fps
    return s, o

OK = lambda i, t: dict(face_present=True, pose={"yaw": 3, "pitch": 2, "roll": 0}, ear=.30, mar=.1, gaze_h=.5, expression={"label": "Neutral", "negative_affect": .05})


def test_engagement_states():
    assert run(OK, 300)[1]["state"] == "Focused"
    assert run(lambda i, t: dict(OK(i, t), pose={"yaw": 45, "pitch": 0, "roll": 0}), 150)[1]["state"] == "Distracted"
    assert run(lambda i, t: dict(OK(i, t), pose={"yaw": 0, "pitch": 35, "roll": 0}), 150)[1]["state"] == "Distracted"
    assert run(lambda i, t: dict(OK(i, t), ear=.05), 40)[1]["state"] == "Drowsy"
    assert run(lambda i, t: dict(OK(i, t), phone_conf=.6), 60)[1]["state"] == "Phone in use"
    assert run(lambda i, t: dict(face_present=False), 60)[1]["state"] == "Away"


def test_blinks_yawns_and_limited_mode():
    blink = lambda i, t: dict(OK(i, t), ear=.05 if .5 <= t % 2 < .7 else .30)
    assert run(blink, 100)[1]["blinks_per_min"] == 5
    assert run(lambda i, t: dict(OK(i, t), mar=.8), 30)[1]["yawns"] == 1
    _, o = run(lambda i, t: dict(face_present=True, expression={"label": "Neutral", "negative_affect": .05}), 100)
    assert o["limited_mode"] and o["state"] != "Focused"          # no head evidence -> never 'Focused'


def test_time_attribution_for_absence():
    s = engagement.StudySession(); t = 0.0
    for _ in range(60): s.update(t, **OK(0, 0)); t += .1
    for _ in range(40): s.update(t, face_present=False); t += .1
    sp = s.summary()["state_percent"]
    assert sp["Away"] == pytest.approx(40, abs=3) and sp["Focused"] == pytest.approx(60, abs=3)


# ---------------------------------------------------------------- texture
def textured(seed=0, amp=6):
    rng = np.random.RandomState(seed); yy, xx = np.mgrid[0:128, 0:128]
    b = np.stack([150 + .2 * (xx - 64)] * 3, -1) + np.array([-8, 0, 10]) + rng.randn(128, 128, 1) * amp
    return np.clip(b, 0, 255).astype(np.uint8)


def test_texture_report_and_vector():
    img = textured()
    v = texture.feature_vector(img)
    assert v.shape == (texture.FEATURE_DIM,) and np.isfinite(v).all()
    r = texture.analyze(img, with_images=True)
    assert set(r["regions"]) == set(texture.REGIONS) and r["skin_tone"]["ita_category"]
    assert all(k in r["images"] for k in ("face", "skin_mask", "lbp_map", "edge_map"))
    assert "not a medical" in r["disclaimer"].lower() or "not medical" in r["disclaimer"].lower() or "not a medical" in r["disclaimer"]
    smooth, rough = texture.analyze(textured(amp=1), False)["smoothness_index"], texture.analyze(textured(amp=14), False)["smoothness_index"]
    assert smooth > rough                                     # more fine noise -> lower smoothness


def test_texture_ml_learns_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "MODEL_DIR", tmp_path)
    X = np.array([texture.feature_vector(textured(i, amp=1 + (i % 12))) for i in range(90)]); y = np.array([20 + 5 * (i % 12) for i in range(90)], float)
    model, m = texture_ml.train("age", X, y)
    assert m["mae_years"] < m["baseline_mae_years"]
    texture_ml.save("age", model, m)
    assert texture_ml.available("age") and 1 <= texture_ml.predict_age(textured(3, 4))["age"] <= 100
    with pytest.raises(ValueError):
        texture_ml.train("age", X[:10], y[:10])


# ---------------------------------------------------------------- models on a real face (skip if not installed)
@pytest.fixture(scope="module")
def face_img():
    pytest.importorskip("skimage")
    from skimage import data
    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


needs = lambda *n: pytest.mark.skipif(not all(mr.is_installed(x) for x in n), reason=f"models not installed: {n}")


@needs("expression_effnet_b0", "face_ssd_res10")
def test_expression_on_real_smile(face_img):
    f = faces.primary_face(face_img)
    e = expression.predict(faces.crop(face_img, f["box"]))
    assert e["label"] == "Happiness" and sum(e["probs"].values()) == pytest.approx(1, abs=1e-3)


@needs("phone_yolov4_tiny")
def test_phone_detector_runs_and_finds_objects(face_img):
    d = phone.detect(face_img, conf=.2, size=608, all_objects=True)
    assert any(x["label"] == "person" for x in d)
    assert phone.detect(face_img, conf=.4, size=416) == []     # a portrait has no phone


@needs("age_levi_hassner", "face_ssd_res10")
def test_age_in_plausible_range(face_img):
    a = attributes.predict_age(faces.crop(face_img, faces.primary_face(face_img)["box"]))
    assert 15 <= a["apparent_age"] <= 60 and a["age_bucket"] in attributes.AGE_BUCKETS


@needs("expression_effnet_b0", "face_ssd_res10")
def test_pipeline_with_landmarks_and_session(face_img):
    lm = make_landmarks()
    sid = "t-" + os.urandom(3).hex(); t = 0.0
    for _ in range(30):
        r = pipeline.analyze(face_img, landmarks=lm.tolist(), modules={"expression", "engagement"}, session_id=sid, now=t); t += .5
    assert r["engagement"]["state"] == "Focused" and r["pose"]["pose"]["yaw"] == pytest.approx(0, abs=1)
    for _ in range(25):
        r = pipeline.analyze(face_img, landmarks=make_landmarks(yaw=45).tolist(), modules={"engagement"}, session_id=sid, now=t); t += .5
    assert r["engagement"]["state"] == "Distracted"
    s = pipeline.end_session(sid)
    assert s["distraction_events"] >= 1 and pipeline.end_session(sid) is None


def test_missing_models_reported_not_raised(face_img, tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "MODEL_DIR", tmp_path)
    r = pipeline.analyze(face_img, modules={"expression", "phone", "age", "gender"})
    assert {u["module"] for u in r["unavailable"]} == {"expression", "phone", "age", "gender"}


# ---------------------------------------------------------------- REST API
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def jpg(img):
    ok, buf = cv2.imencode(".jpg", img); return io.BytesIO(buf.tobytes())


def test_api_capabilities_and_pages(client):
    c = client.get("/api/vision/capabilities").json()
    assert {"expression", "phone", "age", "gender", "face_detector"} <= set(c)
    for p in ["/", "/expression", "/phone", "/study", "/attributes", "/biosuite", "/dashboard", "/models", "/about"]:
        assert client.get(p).status_code == 200, p
    assert "not a medical" in client.get("/about").text.lower() or "not a medical device" in client.get("/about").text.lower()


def test_api_input_validation(client):
    bad = client.post("/api/vision/analyze", files={"image": ("x.jpg", io.BytesIO(b"not an image"), "image/jpeg")})
    assert bad.status_code == 422
    img = ("f.jpg", jpg(np.full((200, 200, 3), 90, np.uint8)), "image/jpeg")
    r = client.post("/api/vision/analyze", files={"image": img}, data={"landmarks": "[1,2,3]"})
    assert r.status_code == 422


@needs("expression_effnet_b0", "face_ssd_res10", "age_levi_hassner", "phone_yolov4_tiny")
def test_api_analyze_texture_phone_snapshots_dashboard(client, face_img):
    r = client.post("/api/vision/analyze", files={"image": ("f.jpg", jpg(face_img), "image/jpeg")}, data={"modules": "expression,phone,age"})
    assert r.status_code == 200 and r.json()["faces"][0]["expression"]["label"] and r.json()["phone_detected"] is False
    t = client.post("/api/vision/texture", files={"image": ("f.jpg", jpg(face_img), "image/jpeg")})
    assert t.status_code == 200 and "age_cnn" in t.json() and "gender_cnn" not in t.json()      # gender is opt-in
    p = client.post("/api/vision/phone", files={"image": ("f.jpg", jpg(face_img), "image/jpeg")}, data={"annotate": "true"})
    assert p.status_code == 200 and p.json()["annotated"].startswith("data:image/jpeg")
    blank = client.post("/api/vision/texture", files={"image": ("b.jpg", jpg(np.full((300, 300, 3), 128, np.uint8)), "image/jpeg")})
    assert blank.status_code == 422
    s = client.post("/api/snapshots", files={"image": ("s.jpg", jpg(face_img), "image/jpeg")}, data={"caption": "test", "analysis": '{"a":1}'})
    assert s.status_code == 200
    code = s.json()["snapshot_code"]
    assert client.get(s.json()["url"]).headers["content-type"] == "image/jpeg"
    assert any(x["snapshot_code"] == code for x in client.get("/api/snapshots").json())
    assert client.delete(f"/api/snapshots/{code}").status_code == 200 and client.delete(f"/api/snapshots/{code}").status_code == 404
    sid = "api-" + os.urandom(3).hex()
    for _ in range(3):
        client.post("/api/vision/analyze", files={"image": ("f.jpg", jpg(face_img), "image/jpeg")}, data={"modules": "expression,engagement", "session_id": sid, "landmarks": json.dumps(make_landmarks().tolist())})
    assert client.get(f"/api/study/{sid}/summary").status_code == 200
    assert client.post(f"/api/study/{sid}/end?save=false").status_code == 200 and client.post(f"/api/study/{sid}/end").status_code == 404
    d = client.get("/api/dashboard").json()
    assert {"counts", "study", "bio_age", "capabilities"} <= set(d)


def test_api_models_custom_validation(client):
    assert client.post("/api/models/custom", json={"name": "x", "url": "http://insecure/x.bin"}).status_code == 422
    assert client.post("/api/models/custom", json={"name": "x", "url": "https://h/x.bin", "sha256": "abc"}).status_code == 422
    assert client.post("/api/models/custom", json={"name": "../bad", "url": "https://h/x.bin"}).status_code == 422
