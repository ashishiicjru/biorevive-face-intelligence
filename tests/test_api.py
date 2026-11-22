import io, os, tempfile
import pytest
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "t.db")
from fastapi.testclient import TestClient
from app.main import app
from app.biology import phenotypic_age

HEALTHY_CONV = dict(albumin=4.4, creatinine=0.9, glucose=90, crp=0.1, lymphocyte_percent=30,
                    mcv_fl=90, rdw_percent=13, alp_u_l=70, wbc_10e3_ul=6)

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_equation_reference_and_unit_equivalence():
    a = phenotypic_age(50, 4.4, 0.9, 90, 0.1, 30, 90, 13, 70, 6)["phenotypic_age"]
    b = phenotypic_age(50, 44, 79.56, 90 / 18.016, 1.0, 30, 90, 13, 70, 6, unit_system="si")["phenotypic_age"]
    assert 42 < a < 46 and abs(a - b) < 0.05

def test_equation_monotonic():
    base = phenotypic_age(50, 4.4, 0.9, 90, 0.1, 30, 90, 13, 70, 6)["phenotypic_age"]
    assert phenotypic_age(50, 4.4, 0.9, 140, 0.1, 30, 90, 13, 70, 6)["phenotypic_age"] > base   # glucose up
    assert phenotypic_age(50, 4.4, 0.9, 90, 2.0, 30, 90, 13, 70, 6)["phenotypic_age"] > base    # CRP up
    assert phenotypic_age(50, 3.5, 0.9, 90, 0.1, 30, 90, 13, 70, 6)["phenotypic_age"] > base    # albumin down

def test_unit_mixup_rejected():
    with pytest.raises(ValueError):
        phenotypic_age(50, 44, 0.9, 90, 0.1, 30, 90, 13, 70, 6)   # SI albumin, conventional system

def test_full_flow(client):
    r = client.post("/api/users", json={"user_code": "T-001", "full_name": "Test", "date_of_birth": "1976-01-01"})
    assert r.status_code == 200
    r = client.post("/api/age/phenotypic", json={"user_code": "T-001", **HEALTHY_CONV})
    assert r.status_code == 200, r.text
    d = r.json(); assert 30 < d["phenotypic_age"] < 70 and "mortality_score" in d
    si = dict(unit_system="si", albumin=44, creatinine=79.56, glucose=5.0, crp=1.0, lymphocyte_percent=30,
              mcv_fl=90, rdw_percent=13, alp_u_l=70, wbc_10e3_ul=6)
    r2 = client.post("/api/age/phenotypic", json={"user_code": "T-001", **si})
    assert r2.status_code == 200 and abs(r2.json()["phenotypic_age"] - d["phenotypic_age"]) < 0.1
    bad = client.post("/api/age/phenotypic", json={"user_code": "T-001", **{**HEALTHY_CONV, "albumin": 44}})
    assert bad.status_code == 422
    assert client.post("/api/age/phenotypic", json={"user_code": "T-001", "chronological_age": 20, **HEALTHY_CONV}).status_code == 400
    c = client.get("/api/users/T-001/comparison").json()
    assert c["biological_age"] is not None
    pdf = client.get("/api/reports/T-001/latest")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"

def test_apparent_age(client):
    skimage = pytest.importorskip("skimage")
    from PIL import Image
    from skimage import data
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2.dnn, "readNetFromCaffe"):
        pytest.skip("OpenCV >=5 has no Caffe importer")
    buf = io.BytesIO(); Image.fromarray(data.astronaut()).save(buf, "JPEG"); buf.seek(0)
    r = client.post("/api/age/apparent?user_code=T-001", files={"image": ("f.jpg", buf, "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json(); assert 0 < d["apparent_age"] < 100 and d["age_bucket"]
    blank = io.BytesIO(); Image.new("RGB", (400, 400), "gray").save(blank, "JPEG"); blank.seek(0)
    r = client.post("/api/age/apparent?user_code=T-001", files={"image": ("b.jpg", blank, "image/jpeg")})
    assert r.status_code == 422
