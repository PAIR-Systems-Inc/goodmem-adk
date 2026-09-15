# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
# SPDX-License-Identifier: Apache-2.0

"""Compact results shared by ADK tools and callbacks."""

from typing import Any

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """One matching chunk, with its original memory metadata."""

    memory_id: str
    chunk_id: str
    space_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: int | None = None


class RetrievalStatus(BaseModel):
    """A server notice or client-side retrieval error."""

    code: str
    message: str
    details: dict[str, str] = Field(default_factory=dict)


class GoodmemFetchResponse(BaseModel):
    """Search results; partial means the search may be incomplete."""

    success: bool
    memories: list[MemoryItem] = Field(default_factory=list)
    count: int = 0
    statuses: list[RetrievalStatus] = Field(default_factory=list)
    partial: bool = False
    message: str


class AcceptedMemory(BaseModel):
    """An accepted write. Acceptance does not imply indexing has completed."""

    memory_id: str
    processing_status: str
    filename: str | None = None


class SaveFailure(BaseModel):
    """A text or attachment write that was not confirmed by the server."""

    message: str
    filename: str | None = None


class GoodmemSaveResponse(BaseModel):
    """Write outcome, including accepted IDs even when another write fails."""

    success: bool
    memory_id: str | None = None
    attachments_saved: int = 0
    accepted: list[AcceptedMemory] = Field(default_factory=list)
    errors: list[SaveFailure] = Field(default_factory=list)
    partial: bool = False
    message: str


class GoodmemSaveError(RuntimeError):
    """Automatic persistence failed; response retains all confirmed write IDs."""

    def __init__(self, response: GoodmemSaveResponse) -> None:
        self.response = response
        super().__init__(response.model_dump_json())
