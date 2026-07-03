from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    paper_id: int
    paper_title: str
    page_number: int
    chunk_id: int
    score: float
    excerpt: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    chat_id: int | None = None
    top_k: int = Field(default=5, ge=1, le=12)


class AskResponse(BaseModel):
    chat_id: int
    answer: str
    citations: list[Citation]


class CompareRequest(BaseModel):
    paper_ids: list[int] = Field(min_length=2)


class RelatedRequest(BaseModel):
    paper_id: int
    top_k: int = Field(default=5, ge=1, le=10)
