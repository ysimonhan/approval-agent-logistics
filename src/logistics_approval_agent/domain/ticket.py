from __future__ import annotations

from typing import TypedDict


class Repair(TypedDict, total=False):
    code: str
    description: str


class MediaItem(TypedDict, total=False):
    filename: str
    type: str
    repair_code_association: str | None
    uploaded_timestamp: str
    yolo_summary: str


class Ticket(TypedDict, total=False):
    ticket_id: str
    container_id: str
    company: str
    container_age: int
    total_cost_estimate: float
    repairs: list[Repair]
    media: list[MediaItem]
    other_notes: str
    status: str
    ai_decision: str | None
    ai_confidence: float
    ai_reasoning: str | None
    ai_agent_type: str
    ai_missing_data_request: str | None
    ai_processed_date: str
    ai_chat: list[dict]
