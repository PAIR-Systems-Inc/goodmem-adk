# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
# SPDX-License-Identifier: Apache-2.0

"""Internal ADK MemoryService; not part of the package's supported public API.

Session ingestion preserves the existing paired/split-turn behavior. The bounded
in-process record of accepted writes supports retries within this service's
lifetime; it is not a durable ingestion ledger across restarts.
"""

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from goodmem import AsyncGoodmem
from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.genai import types
from pydantic import BaseModel, Field

from ._attachments import attachments_from_content, write_attachment
from ._backend import Backend, text_from_content
from ._results import SaveFailure

_SESSION_CACHE_SIZE = 1024


@dataclass
class _SessionIngestion:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    accepted: set[tuple[int, str]] = field(default_factory=set)
    users: int = 0


class GoodmemMemoryServiceConfig(BaseModel):
    """Configuration for the internal session memory service."""

    top_k: int = Field(default=5, ge=1, le=100)
    timeout: float = Field(default=30.0, gt=0)
    split_turn: bool = False


class GoodmemMemoryService(BaseMemoryService):
    """Persist completed session turns with the asynchronous SDK.

    SDK write exceptions propagate. A failed write does not advance the accepted
    record, so retrying ingestion does not skip it or repeat confirmed writes.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        embedder_id: str | None = None,
        space_id: str | None = None,
        space_name: str | None = None,
        config: GoodmemMemoryServiceConfig | None = None,
        top_k: int = 5,
        timeout: float = 30.0,
        split_turn: bool = False,
        *,
        client: AsyncGoodmem | None = None,
    ) -> None:
        self._config = config or GoodmemMemoryServiceConfig(
            top_k=top_k,
            timeout=timeout,
            split_turn=split_turn,
        )
        self._backend = Backend(
            client=client,
            base_url=base_url,
            api_key=api_key,
            embedder_id=embedder_id,
            space_id=space_id,
            space_name=space_name,
            timeout=self._config.timeout,
        )
        self._sessions: OrderedDict[tuple[str, str, str], _SessionIngestion] = OrderedDict()

    @asynccontextmanager
    async def _session_ingestion(
        self, key: tuple[str, str, str]
    ) -> AsyncIterator[set[tuple[int, str]]]:
        state = self._sessions.setdefault(key, _SessionIngestion())
        self._sessions.move_to_end(key)
        # Count queued callers too: eviction must not replace their lock or ledger.
        state.users += 1
        try:
            async with state.lock:
                yield state.accepted
        finally:
            state.users -= 1
            for old_key in list(self._sessions):
                if len(self._sessions) <= _SESSION_CACHE_SIZE:
                    break
                if not self._sessions[old_key].users:
                    del self._sessions[old_key]

    async def add_session_to_memory(self, session: Session) -> None:
        """Persist user attachments and complete turns, recording each accepted write."""
        session_key = (session.app_name, session.user_id, session.id)
        async with (
            self._session_ingestion(session_key) as accepted,
            self._backend.connect() as client,
        ):
            space_id = await self._backend.resolve_space(
                client,
                f"adk_memory_{session.app_name}_{session.user_id}",
            )
            metadata = {
                "app_name": session.app_name,
                "user_id": session.user_id,
                "session_id": session.id,
                "source": "adk_session",
            }
            user_text = ""
            for index, event in enumerate(session.events):
                if event.partial:
                    continue
                text = text_from_content(event.content)
                if event.author == "user":
                    for attachment in attachments_from_content(event.content):
                        if isinstance(attachment, SaveFailure):
                            raise ValueError(
                                f"{attachment.filename or 'Attachment'}: {attachment.message}"
                            )
                        key = (index, f"attachment_{attachment.part_index}")
                        if key not in accepted:
                            await write_attachment(
                                client.memories,
                                attachment,
                                space_id=space_id,
                                metadata={**metadata, "role": "user"},
                            )
                            accepted.add(key)
                    if text:
                        user_text = text
                    continue
                if event.author in ("tool", "system") or not text:
                    continue
                if user_text and self._config.split_turn:
                    writes = [(f"User: {user_text}", "user"), (f"LLM: {text}", "model")]
                elif user_text:
                    writes = [(f"User: {user_text}\nLLM: {text}", "conversation")]
                else:
                    writes = [(f"LLM: {text}", "model")]
                for value, role in writes:
                    key = (index, role)
                    if key not in accepted:
                        await client.memories.create(
                            space_id=space_id,
                            original_content=value,
                            content_type="text/plain",
                            metadata={**metadata, "role": role},
                        )
                        accepted.add(key)
                user_text = ""

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Return matching chunks; surface incomplete retrieval to the caller."""
        result = await self._backend.fetch(
            default_name=f"adk_memory_{app_name}_{user_id}",
            query=query,
            top_k=self._config.top_k,
        )
        # ADK's native response has no field for partial results or statuses.
        if result.partial:
            raise RuntimeError(result.model_dump_json())
        return SearchMemoryResponse(
            memories=[
                MemoryEntry(
                    id=item.chunk_id,
                    content=types.Content(parts=[types.Part(text=item.content)]),
                    author=str(item.metadata.get("role", "conversation")),
                    timestamp=datetime.fromtimestamp(
                        item.updated_at / 1000,
                        tz=timezone.utc,
                    ).isoformat()
                    if item.updated_at
                    else None,
                )
                for item in result.memories
            ]
        )

    async def close(self) -> None:
        """Owned SDKs close after each operation; an injected SDK belongs to its caller."""


def format_memory_block_for_prompt(response: SearchMemoryResponse) -> str:
    """Format the internal service's retrieved entries for a prompt."""
    return "\n".join(text_from_content(memory.content) for memory in response.memories)
