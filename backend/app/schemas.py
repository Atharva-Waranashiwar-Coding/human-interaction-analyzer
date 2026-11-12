from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ObjectReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., min_length=1, max_length=128)
    class_name: str = Field(..., min_length=1, max_length=128)


class AnnotationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str | None = None
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., ge=0)
    behavior: str = Field(..., min_length=1)
    participants: list[str] = Field(default_factory=list)
    objects: list[ObjectReference] = Field(default_factory=list)
    parent_activity_id: str | None = None
    description: str = ""
    source: str = "manual"
    confidence: Optional[float] = None
    review_status: str = "needs_review"

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        return self


class AnnotationBulkUpdate(BaseModel):
    annotations: list[AnnotationInput]


class VideoCreateResponse(BaseModel):
    id: str
    original_name: str
    status: str
    analysis_status: str
    duration: float
    width: int
    height: int
    file_size_bytes: int


class VideoSummary(BaseModel):
    id: str
    original_name: str
    duration: float
    width: int
    height: int
    status: str
    analysis_status: str
    message: str | None = None


class VideoDetail(VideoSummary):
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class ExportJSONRequest(BaseModel):
    format: Literal["json", "jsonl"] = "json"
