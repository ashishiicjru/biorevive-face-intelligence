import os,uuid,tempfile
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI,Depends,HTTPException,UploadFile,File
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse,StreamingResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session,select
from .database import init_db,get_session
from .models import User,Scan,PhenotypicAssessment
from .schemas import UserCreate,ScanCreate,PhenotypicRequest
from .biology import phenotypic_age
from . import age_model, model_registry
from pydantic import BaseModel
from .age_model import predict_apparent_age,NoFaceError
from .reports import make_report
from .vision_api import router as vision_router, DATA_DIR
BASE=Path(__file__).resolve().parent
STATIC=BASE/"static"
@asynccontextmanager
async def lifespan(_):
    init_db()
    if os.getenv("PRELOAD_MODELS","0")=="1":
        for n,v in model_registry.registry().items():
            if v.get("recommended"): model_registry.install_async(n)
    yield
app=FastAPI(title="Bio-Revive Face Intelligence",version="3.0.0",lifespan=lifespan)
STATIC.mkdir(exist_ok=True)
app.include_router(vision_router)
app.mount("/static",StaticFiles(directory=str(STATIC)),name="static")
PAGES={"":"home.html","expression":"expression.html","phone":"phone.html","study":"study.html","attributes":"attributes.html","biosuite":"biosuite.html","dashboard":"dashboard.html","models":"models.html","about":"about.html"}
def _page(name):
    def view(): return HTMLResponse((STATIC/name).read_text(encoding="utf-8"))
    return view
for _slug,_file in PAGES.items(): app.add_api_route("/"+_slug,_page(_file),methods=["GET"],response_class=HTMLResponse,include_in_schema=False)
@app.get("/models/face_landmarker.task",include_in_schema=False)
def landmarker_file():
    p=model_registry.path_of("face_landmarker.task")
    if not p.exists(): raise HTTPException(404,"Landmarker model not installed locally.")
    return FileResponse(p,media_type="application/octet-stream")
@app.get("/api/health")
def health(): return {"status":"ok","service":"Bio-Revive Face Intelligence","version":"3.0.0","models":age_model.status(),"biological_age":"PhenotypicAge (Levine 2018)"}
class InstallRequest(BaseModel):
    names: list[str] | None = None      # default: recommended models
    force: bool = False
@app.get("/api/models/status")
def models_status(): return age_model.status()
class CustomModel(BaseModel):
    name:str
    url:str
    sha256:str|None=None
    filename:str|None=None
@app.post("/api/models/custom")
def models_custom(m:CustomModel):
    if not m.url.startswith("https://"): raise HTTPException(422,"Only https:// URLs are accepted.")
    if m.sha256 and len(m.sha256)!=64: raise HTTPException(422,"SHA-256 must be 64 hex characters.")
    try: model_registry.add_custom(m.name,m.url,m.filename,m.sha256)
    except ValueError as e: raise HTTPException(422,str(e))
    model_registry.install_async(m.name)
    return {"registered":m.name}
@app.get("/api/models")
def models_list(): return {"model_dir":str(model_registry.MODEL_DIR),"models":model_registry.overview()}
@app.post("/api/models/install")
def models_install(req:InstallRequest|None=None):
    """Starts background downloads and returns immediately; poll GET /api/models for progress."""
    reg=model_registry.registry()
    names=(req.names if req and req.names else [k for k,v in reg.items() if v.get("recommended")])
    bad=[n for n in names if n not in reg]
    if bad: raise HTTPException(404,"Unknown model(s): "+", ".join(bad))
    for n in names: model_registry.install_async(n,bool(req and req.force))
    return {"started":names,"models":model_registry.overview()}
@app.post("/api/users")
def create_user(p:UserCreate,s:Session=Depends(get_session)):
    if p.date_of_birth>=date.today(): raise HTTPException(400,"Date of birth must be in the past.")
    if s.exec(select(User).where(User.user_code==p.user_code)).first(): raise HTTPException(409,"User code already exists.")
    u=User(user_code=p.user_code,full_name=p.full_name,date_of_birth=p.date_of_birth.isoformat());s.add(u);s.commit();s.refresh(u);return u
@app.get("/api/users/{code}")
def get_user(code:str,s:Session=Depends(get_session)):
    u=s.exec(select(User).where(User.user_code==code)).first()
    if not u: raise HTTPException(404,"User not found.")
    return u
@app.post("/api/scans")
def save_scan(p:ScanCreate,s:Session=Depends(get_session)):
    if not s.exec(select(User).where(User.user_code==p.user_code)).first(): raise HTTPException(404,"Register user first.")
    if s.exec(select(Scan).where(Scan.scan_code==p.scan_code)).first(): raise HTTPException(409,"Scan code exists.")
    x=Scan(**p.model_dump(),landmark_count=len(p.landmarks));s.add(x);s.commit();s.refresh(x);return {"saved":True,"scan":x}
@app.get("/api/scans/{code}")
def get_scan(code:str,s:Session=Depends(get_session)):
    x=s.exec(select(Scan).where(Scan.scan_code==code)).first()
    if not x: raise HTTPException(404,"Scan not found.")
    return x
@app.get("/api/users/{code}/scans")
def scans(code:str,s:Session=Depends(get_session)):
    return s.exec(select(Scan).where(Scan.user_code==code).order_by(Scan.captured_at.desc())).all()
@app.post("/api/age/apparent")
async def apparent_age(user_code:str,image:UploadFile=File(...),scan_code:str|None=None,s:Session=Depends(get_session)):
    if not s.exec(select(User).where(User.user_code==user_code)).first(): raise HTTPException(404,"User not found.")
    if image.content_type not in ("image/jpeg","image/png","image/webp"): raise HTTPException(400,"Upload JPEG, PNG or WebP.")
    raw=await image.read()
    if len(raw)>8*1024*1024: raise HTTPException(413,"Image limit is 8 MB.")
    path=None
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".jpg") as f:f.write(raw);path=f.name
        result=await run_in_threadpool(predict_apparent_age,path)
        if scan_code:
            x=s.exec(select(Scan).where(Scan.scan_code==scan_code,Scan.user_code==user_code)).first()
            if x:x.apparent_age=result["apparent_age"];x.age_model=result["model"];s.add(x);s.commit()
        return result
    except NoFaceError as e: raise HTTPException(422,str(e))
    except RuntimeError as e: raise HTTPException(503,str(e))
    finally:
        if path and os.path.exists(path):os.remove(path)
@app.post("/api/age/phenotypic")
def bio_age(p:PhenotypicRequest,s:Session=Depends(get_session)):
    u=s.exec(select(User).where(User.user_code==p.user_code)).first()
    if not u:raise HTTPException(404,"User not found.")
    actual=(date.today()-date.fromisoformat(u.date_of_birth)).days/365.2425   # age comes from DOB, never the client
    if p.chronological_age is not None and abs(actual-p.chronological_age)>1:raise HTTPException(400,f"Chronological age does not match profile (approximately {actual:.1f}).")
    if p.scan_code and not s.exec(select(Scan).where(Scan.scan_code==p.scan_code,Scan.user_code==p.user_code)).first():raise HTTPException(404,"Scan not found for this user.")
    try:res=phenotypic_age(actual,p.albumin,p.creatinine,p.glucose,p.crp,p.lymphocyte_percent,p.mcv_fl,p.rdw_percent,p.alp_u_l,p.wbc_10e3_ul,unit_system=p.unit_system)
    except (ValueError,OverflowError) as e:raise HTTPException(422,str(e))
    r=res["phenotypic_age"];actual=round(actual,2)
    code="PHENO-"+uuid.uuid4().hex[:10].upper()
    biomarkers={k:v for k,v in p.model_dump().items() if k not in ("user_code","chronological_age","scan_code")}
    biomarkers["equation_inputs"]=res["equation_inputs"];biomarkers["mortality_score"]=res["mortality_score"]
    a=PhenotypicAssessment(assessment_code=code,user_code=p.user_code,scan_code=p.scan_code,chronological_age=actual,biological_age=r,age_difference=round(r-actual,2),biomarkers=biomarkers)
    s.add(a);s.commit()
    return {"assessment_code":code,"chronological_age":actual,"phenotypic_age":r,"age_difference":a.age_difference,"mortality_score":res["mortality_score"],"notes":res["notes"],"algorithm":"PhenotypicAge (Levine 2018)","status":"calculated","interpretation":"Research estimate only; not a diagnosis."}
@app.get("/api/users/{code}/comparison")
def comparison(code:str,s:Session=Depends(get_session)):
    u=s.exec(select(User).where(User.user_code==code)).first()
    if not u:raise HTTPException(404,"User not found.")
    age=round((date.today()-date.fromisoformat(u.date_of_birth)).days/365.2425,2)
    scans=s.exec(select(Scan).where(Scan.user_code==code).order_by(Scan.captured_at.desc())).all()
    bios=s.exec(select(PhenotypicAssessment).where(PhenotypicAssessment.user_code==code).order_by(PhenotypicAssessment.assessed_at.desc())).all()
    fs=next((x for x in scans if x.apparent_age is not None),None);b=bios[0] if bios else None
    return {"chronological_age":age,"facial_apparent_age":fs.apparent_age if fs else None,"biological_age":b.biological_age if b else None,"facial_age_difference":round(fs.apparent_age-age,2) if fs else None,"biological_age_difference":b.age_difference if b else None}
@app.get("/api/reports/{code}/latest")
def report(code:str,s:Session=Depends(get_session)):
    u=s.exec(select(User).where(User.user_code==code)).first()
    if not u:raise HTTPException(404,"User not found.")
    scans=s.exec(select(Scan).where(Scan.user_code==code).order_by(Scan.captured_at.desc())).all()
    bios=s.exec(select(PhenotypicAssessment).where(PhenotypicAssessment.user_code==code).order_by(PhenotypicAssessment.assessed_at.desc())).all()
    pdf=make_report(u,scans,bios)
    return StreamingResponse(pdf,media_type="application/pdf",headers={"Content-Disposition":f'attachment; filename="BioRevive_{code}.pdf"'})
