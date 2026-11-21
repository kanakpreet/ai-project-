import json
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from .database import Base, engine, get_db
from .models import Job
from .schemas import JobCreate, SearchRequest, RecommendRequest, AgentRequest
from .embeddings import embed
from .matching import job_text, rank_jobs
from .agent import answer_question

load_dotenv()
Base.metadata.create_all(bind=engine)
app = FastAPI(title="HireSense AI", description="Semantic job search and AI-powered job matching")
ROOT = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def seed_jobs(db: Session):
    if db.query(Job).count():
        return
    data = json.loads((ROOT / "data" / "sample_jobs.json").read_text())
    for item in data:
        job = Job(**item, skills=", ".join(item["skills"]))
        job.embedding = json.dumps(embed(job_text(job)))
        db.add(job)
    db.commit()


@app.on_event("startup")
def startup():
    from .database import SessionLocal
    db = SessionLocal()
    try:
        seed_jobs(db)
    finally:
        db.close()


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "HireSense AI"}


@app.get("/jobs")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Job).all()
    return [{"id": j.id, "title": j.title, "company": j.company, "location": j.location, "skills": j.skills.split(", "), "description": j.description} for j in jobs]


@app.post("/jobs")
def create_job(payload: JobCreate, db: Session = Depends(get_db)):
    job = Job(title=payload.title, company=payload.company, location=payload.location,
              skills=", ".join(payload.skills), description=payload.description)
    job.embedding = json.dumps(embed(job_text(job)))
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"message": "Job indexed successfully", "job_id": job.id}


@app.post("/search")
def search(payload: SearchRequest, db: Session = Depends(get_db)):
    jobs = db.query(Job).all()
    if payload.location:
        jobs = [j for j in jobs if payload.location.lower() in j.location.lower()]
    return {"results": rank_jobs(jobs, payload.query, payload.skills)[:payload.limit]}


@app.post("/recommend")
def recommend(payload: RecommendRequest, db: Session = Depends(get_db)):
    if len(payload.resume_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Please provide more resume text for meaningful recommendations.")
    jobs = db.query(Job).all()
    if payload.location:
        jobs = [j for j in jobs if payload.location.lower() in j.location.lower()]
    return {"recommendations": rank_jobs(jobs, payload.resume_text, [])[:payload.limit]}


@app.post("/agent/ask")
def ask_agent(payload: AgentRequest, db: Session = Depends(get_db)):
    return answer_question(db, payload.question, payload.resume_text)
