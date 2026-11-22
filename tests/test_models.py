import hashlib, os, tempfile, time
import pytest
os.environ.setdefault("DATABASE_URL", "sqlite:///" + os.path.join(tempfile.mkdtemp(), "m.db"))
from app import model_registry as mr, download_models as cli


@pytest.fixture()
def mdir(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(mr, "CUSTOM_FILE", tmp_path / "models" / "custom_models.json")
    return tmp_path


def _src(tmp_path, data=b"model-bytes" * 1000):
    p = tmp_path / "src.bin"; p.write_bytes(data)
    return p, hashlib.sha256(data).hexdigest()


def test_builtin_registry_shape():
    reg = mr.registry()
    assert {"age_levi_hassner", "face_ssd_res10", "expression_effnet_b0", "phone_yolov4_tiny",
            "gender_levi_hassner"} <= set(reg)
    for name, m in mr.BUILTIN.items():
        for f in m["files"]:
            assert f["url"].startswith("https://")
            if name != "face_landmarker_mediapipe":      # that one is deliberately unpinned
                assert len(f["sha256"]) == 64 and f["size"] > 0


def test_download_verify_and_repair(mdir):
    src, h = _src(mdir)
    mr.add_custom("demo", src.as_uri(), sha256=h)
    assert mr.model_state("demo") == "missing"
    assert mr.install("demo") == "installed"
    (mr.MODEL_DIR / "src.bin").write_bytes(b"tampered")
    assert mr.model_state("demo") == "corrupt"
    assert mr.install("demo") == "installed"          # auto-repair


def test_checksum_mismatch_rejected(mdir):
    src, _ = _src(mdir)
    mr.add_custom("bad", src.as_uri(), sha256="0" * 64)
    with pytest.raises(mr.DownloadError):
        mr.install("bad")
    assert not (mr.MODEL_DIR / "src.bin").exists()      # nothing half-written is left behind


def test_unsafe_names_rejected(mdir):
    with pytest.raises(ValueError):
        mr.add_custom("../evil", "file:///x")
    with pytest.raises(ValueError):
        mr.add_custom("ok", "https://h/x", filename="../../etc/passwd")
    with pytest.raises(ValueError):
        mr.add_custom("age_levi_hassner", "https://h/x.bin")


def test_cli_list_and_unknown(mdir, capsys):
    assert cli.main(["--list", "--dir", str(mdir / "models")]) == 0
    assert "face_ssd_res10" in capsys.readouterr().out
    assert cli.main(["nonexistent", "--dir", str(mdir / "models")]) == 2


def test_api_models_and_async_install(mdir):
    from fastapi.testclient import TestClient
    from app.main import app
    src, h = _src(mdir)
    mr.add_custom("demo2", src.as_uri(), sha256=h)
    with TestClient(app) as c:
        names = [m["name"] for m in c.get("/api/models").json()["models"]]
        assert "demo2" in names
        assert c.post("/api/models/install", json={"names": ["nope"]}).status_code == 404
        assert c.post("/api/models/install", json={"names": ["demo2"]}).status_code == 200
        for _ in range(50):
            st = {m["name"]: m for m in c.get("/api/models").json()["models"]}["demo2"]
            if st["state"] == "installed":
                break
            time.sleep(0.1)
        assert st["state"] == "installed"
