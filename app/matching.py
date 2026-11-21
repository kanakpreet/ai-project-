import json
from .embeddings import embed, similarity


def job_text(job) -> str:
    return f"{job.title}. {job.company}. {job.location}. Skills: {job.skills}. {job.description}"


def rank_jobs(jobs, query: str, requested_skills: list[str] | None = None):
    query_vector = embed(query)
    requested = {s.strip().lower() for s in (requested_skills or []) if s.strip()}
    results = []
    for job in jobs:
        vector = json.loads(job.embedding) if job.embedding else embed(job_text(job))
        semantic = similarity(query_vector, vector)
        job_skills = {s.strip().lower() for s in job.skills.split(",") if s.strip()}
        skill_score = len(requested & job_skills) / len(requested) if requested else 0.0
        # Blend semantic relevance and explicit skill overlap when skills are supplied.
        score = (0.7 * semantic + 0.3 * skill_score) if requested else semantic
        results.append({
            "id": job.id, "title": job.title, "company": job.company,
            "location": job.location, "skills": sorted(job_skills),
            "description": job.description,
            "match_score": round(max(0.0, min(score, 1.0)) * 100, 1),
            "matched_skills": sorted(requested & job_skills),
            "missing_skills": sorted(requested - job_skills),
        })
    return sorted(results, key=lambda item: item["match_score"], reverse=True)
