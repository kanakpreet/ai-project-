from sqlalchemy import Column, Integer, String, Text
from .database import Base


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    company = Column(String(200), nullable=False)
    location = Column(String(200), default="India")
    skills = Column(Text, nullable=False)  # comma-separated skills for this MVP
    description = Column(Text, nullable=False)
    embedding = Column(Text, nullable=True)  # JSON-encoded vector
