from types import SimpleNamespace
from app.matching import rank_jobs


def test_rank_jobs_returns_match_fields():
    jobs = [SimpleNamespace(id=1, title="Python Backend Intern", company="Demo", location="India", skills="Python, SQL", description="Build APIs", embedding=None)]
    result = rank_jobs(jobs, "Python backend APIs", ["Python", "SQL"])
    assert len(result) == 1
    assert "match_score" in result[0]
    assert result[0]["matched_skills"] == ["python", "sql"]
