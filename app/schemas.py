from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    title: str
    company: str
    location: str = "India"
    skills: list[str] = Field(default_factory=list)
    description: str


class SearchRequest(BaseModel):
    query: str
    skills: list[str] = Field(default_factory=list)
    location: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


class RecommendRequest(BaseModel):
    resume_text: str
    location: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


class AgentRequest(BaseModel):
    question: str
    resume_text: str = ""
