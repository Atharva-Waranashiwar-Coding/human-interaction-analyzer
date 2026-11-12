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

    annotation_kind: Literal["behavior", "spatial", "activity", "audio"] = "behavior"
    details: dict[str, Any] = Field(default_factory=dict)
    original_suggestion: dict[str, Any] = Field(default_factory=dict)
    review_history: list[dict[str, Any]] = Field(default_factory=list)
    reviewer_feedback: str = ""
    rule_version: str | None = None
    evaluation_role: Literal["unspecified", "ground_truth"] = "unspecified"

    @model_validator(mode="after")
    def validate_time_range(self):
        import json
        try:
            json.dumps(self.details, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("Details must contain finite JSON values") from exc
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


class SceneRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=128)
    kind: Literal["table", "shelf", "container", "workspace", "floor", "other"] = "workspace"
    box: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def check_box(self):
        a,b,c,d = self.box
        if not (0 <= a < c <= 1 and 0 <= b < d <= 1):
            raise ValueError("Region boxes must be ordered normalized letterbox coordinates in 0..1")
        return self


class RegionUpdate(BaseModel):
    regions: list[SceneRegion] = Field(max_length=50)

    @model_validator(mode="after")
    def unique_regions(self):
        if len({r.id for r in self.regions}) != len(self.regions):
            raise ValueError("Region IDs must be unique")
        return self


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    covered_behaviors: list[str] = Field(min_length=1)
    start_time: float = Field(default=0, ge=0)
    end_time: float = Field(gt=0)
    complete_ground_truth: bool = False
    temporal_iou: float = Field(default=.5, gt=0, le=1)
    scenario: str = "unspecified"
