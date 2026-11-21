import os
from openai import OpenAI
from sqlalchemy.orm import Session
from .models import Job
from .matching import rank_jobs


def find_jobs_tool(db: Session, query: str, skills: list[str] | None = None, location: str | None = None):
    jobs = db.query(Job).all()
    if location:
        jobs = [j for j in jobs if location.lower() in j.location.lower()]
    return rank_jobs(jobs, query, skills or [])[:5]


def answer_question(db: Session, question: str, resume_text: str = "") -> dict:
    # Retrieve evidence first: the response is grounded in the current jobs index.
    retrieved = find_jobs_tool(db, resume_text or question)
    if not retrieved:
        return {"answer": "I couldn't find matching jobs in the current demo index. Try adding more jobs or changing your search.", "sources": []}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        top = retrieved[0]
        return {
            "answer": f"Your strongest match in the current sample index is {top['title']} at {top['company']} ({top['match_score']}% relevance). Matched skills: {', '.join(top['matched_skills']) or 'No explicit skill overlap detected'}. Missing skills to consider: {', '.join(top['missing_skills']) or 'None identified from the supplied skills'}.",
            "sources": retrieved[:3], "mode": "retrieval-only (add OPENAI_API_KEY for LLM answers)"
        }

    client = OpenAI(api_key=api_key)
    tools = [{
        "type": "function", "function": {
            "name": "search_jobs",
            "description": "Search and rank jobs in the current indexed job database.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "skills": {"type": "array", "items": {"type": "string"}},
                "location": {"type": "string"}
            }, "required": ["query"], "additionalProperties": False}
        }
    }]
    messages = [
        {"role": "system", "content": "You are HireSense, a career matching assistant. Use the search_jobs tool for job-specific claims. Explain match strengths and gaps clearly. Never invent jobs, salaries, or requirements. Return concise, structured advice."},
        {"role": "user", "content": f"Candidate resume/context: {resume_text[:6000]}\nQuestion: {question}"}
    ]
    first = client.chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=messages, tools=tools, tool_choice="auto")
    msg = first.choices[0].message
    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            if call.function.name == "search_jobs":
                import json
                args = json.loads(call.function.arguments)
                found = find_jobs_tool(db, args.get("query", question), args.get("skills"), args.get("location"))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": str(found[:5])})
        second = client.chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=messages)
        answer = second.choices[0].message.content or "I couldn't generate an answer."
    else:
        answer = msg.content or "I couldn't generate an answer."
    return {"answer": answer, "sources": retrieved[:3], "mode": "LLM + retrieval + tool calling"}
