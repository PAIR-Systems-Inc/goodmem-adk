# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
# SPDX-License-Identifier: Apache-2.0

"""ADK scope and content handling on top of the official asynchronous SDK."""

import os
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol, cast

import httpx
from goodmem import AsyncGoodmem
from goodmem.api.embedders import AsyncEmbeddersAPI
from goodmem.api.memories import AsyncMemoriesAPI
from goodmem.api.spaces import AsyncSpacesAPI
from goodmem.errors import ConflictError, GoodMemError
from goodmem.models.memory import Memory
from goodmem.models.space import Space
from goodmem.models.space_embedder_config import SpaceEmbedderConfig
from google.genai import types

from ._attachments import Attachment, attachments_from_content, write_attachment
from ._results import (
    AcceptedMemory,
    GoodmemFetchResponse,
    GoodmemSaveResponse,
    MemoryItem,
    RetrievalStatus,
    SaveFailure,
)

_INFORMATIONAL = {"FEATURE_DISABLED", "LLM_CAPABILITY_INFERRED"}
_REQUEST_ERRORS = (GoodMemError, httpx.RequestError)


class _SdkApis(Protocol):
    # The SDK attaches these public API groups dynamically. Describe their types
    # here until its generator emits annotations; no runtime adapter is involved.
    spaces: AsyncSpacesAPI
    memories: AsyncMemoriesAPI
    embedders: AsyncEmbeddersAPI


def text_from_content(content: types.Content | None) -> str:
    """Extract visible text, excluding model thoughts and tool payloads."""
    return (
        "\n".join(part.text for part in (content.parts or []) if part.text and not part.thought)
        if content
        else ""
    )


class Backend:
    """Resolve ADK scopes and adapt SDK results without implementing HTTP.

    A supplied SDK belongs to its caller. Otherwise each operation owns a short
    lived async SDK, so standalone ADK FunctionTools need no shutdown hook.
    Scope caches belong to this instance, never to shared ADK session state.
    """

    def __init__(
        self,
        *,
        client: AsyncGoodmem | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        space_id: str | None = None,
        space_name: str | None = None,
        embedder_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if client is not None and not isinstance(client, AsyncGoodmem):
            raise TypeError("client must be an AsyncGoodmem SDK client")
        if client is not None and (base_url is not None or api_key is not None):
            raise ValueError("Pass either client or base_url/api_key, not both")
        self._client = client
        self._base_url = base_url if base_url is not None else os.getenv("GOODMEM_BASE_URL")
        self._api_key = api_key if api_key is not None else os.getenv("GOODMEM_API_KEY")
        if client is None and (not self._base_url or not self._api_key):
            raise ValueError("GOODMEM_BASE_URL and GOODMEM_API_KEY are required without client")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._timeout = timeout
        # Explicit scope settings take precedence over ALL environment scope settings.
        if space_id is None and space_name is None:
            space_id = os.getenv("GOODMEM_SPACE_ID")
            space_name = os.getenv("GOODMEM_SPACE_NAME")
        if space_id == "" or space_name == "" or embedder_id == "":
            raise ValueError("Scope and embedder settings must not be empty strings")
        self.space_id = space_id
        self.space_name = space_name
        self.embedder_id = embedder_id or os.getenv("GOODMEM_EMBEDDER_ID")
        self._spaces: OrderedDict[str, str] = OrderedDict()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[_SdkApis]:
        """Use the caller's SDK, or close our own SDK after the operation."""
        if self._client is not None:
            yield cast(_SdkApis, self._client)
        else:
            assert self._base_url is not None
            async with AsyncGoodmem(
                base_url=self._base_url, api_key=self._api_key, timeout=self._timeout
            ) as client:
                yield cast(_SdkApis, client)

    async def resolve_space(self, client: _SdkApis, default_name: str) -> str:
        """Resolve a configured space, or create a named scope on first use.

        Args:
            client: SDK used for this operation.
            default_name: Scope name derived from the ADK context by the caller.

        Returns:
            The validated space ID for this component and connection.
        """
        name = self.space_name or default_name
        cache_key = self.space_id or name
        if cache_key in self._spaces:
            self._spaces.move_to_end(cache_key)
            return self._spaces[cache_key]
        space: Space | None
        if self.space_id:
            space = await client.spaces.get(id=self.space_id)
            if self.space_name and space.name != self.space_name:
                raise ValueError("space_id does not match the configured space_name")
        else:
            space = await self._find_space(client, name)
            if space is None:
                embedder_id = self.embedder_id
                if embedder_id is None:
                    embedders = await client.embedders.list()
                    if not embedders:
                        raise ValueError(
                            "No embedder is configured on GoodMem. Create one with the SDK "
                            "or GoodMem console, then set GOODMEM_EMBEDDER_ID."
                        )
                    embedder_id = embedders[0].embedder_id
                try:
                    space = await client.spaces.create(
                        name=name,
                        space_embedders=[SpaceEmbedderConfig(embedder_id=embedder_id)],
                    )
                except ConflictError:
                    # A concurrent worker may have created this scope after our lookup.
                    space = await self._find_space(client, name)
                    if space is None:
                        raise
        assert space is not None
        if self.embedder_id and self.embedder_id not in {
            item.embedder_id for item in space.space_embedders
        }:
            raise ValueError("The configured space does not use the configured embedder_id")
        self._spaces[cache_key] = space.space_id
        if len(self._spaces) > 1024:
            self._spaces.popitem(last=False)
        return space.space_id

    @staticmethod
    async def _find_space(client: _SdkApis, name: str) -> Space | None:
        matches = []
        async for space in await client.spaces.list(name_filter=name):
            if space.name == name:
                matches.append(space)
        if len(matches) > 1:
            raise ValueError("Multiple accessible spaces have this name; configure space_id")
        return matches[0] if matches else None

    async def save(
        self,
        *,
        default_name: str,
        text: str,
        content: types.Content | None,
        metadata: dict[str, Any],
    ) -> GoodmemSaveResponse:
        """Save text and inline attachments, retaining every accepted write ID."""
        result = GoodmemSaveResponse(success=False, message="No content to save")
        prepared = list(attachments_from_content(content))
        attachments = [item for item in prepared if isinstance(item, Attachment)]
        result.errors.extend(item for item in prepared if isinstance(item, SaveFailure))
        if not text.strip() and not attachments:
            return result
        try:
            async with self.connect() as client:
                space_id = await self.resolve_space(client, default_name)
                if text.strip():
                    try:
                        memory = await client.memories.create(
                            space_id=space_id,
                            original_content=text,
                            content_type="text/plain",
                            metadata=metadata,
                        )
                        result.memory_id = memory.memory_id
                        result.accepted.append(self._accepted(memory))
                    except _REQUEST_ERRORS as error:
                        result.errors.append(SaveFailure(message=str(error)))
                for attachment in attachments:
                    try:
                        memory = await write_attachment(
                            client.memories,
                            attachment,
                            space_id=space_id,
                            metadata=metadata,
                        )
                        result.accepted.append(self._accepted(memory, attachment.filename))
                        result.attachments_saved += 1
                    except _REQUEST_ERRORS as error:
                        result.errors.append(
                            SaveFailure(filename=attachment.filename, message=str(error))
                        )
        except (*_REQUEST_ERRORS, ValueError) as error:
            result.errors.append(SaveFailure(message=str(error)))
        result.success = bool(result.accepted) and not result.errors
        result.partial = bool(result.accepted) and bool(result.errors)
        result.message = (
            f"Accepted {len(result.accepted)} write(s); indexing may still be pending."
            if result.accepted
            else "No writes were confirmed."
        )
        if result.errors:
            result.message += " Some writes failed. Keep the accepted IDs; do not re-upload them."
        return result

    @staticmethod
    def _accepted(memory: Memory, filename: str | None = None) -> AcceptedMemory:
        return AcceptedMemory(
            memory_id=memory.memory_id,
            processing_status=memory.processing_status or "UNKNOWN",
            filename=filename,
        )

    async def fetch(self, *, default_name: str, query: str, top_k: int) -> GoodmemFetchResponse:
        """Join typed memory definitions to all matching chunks in one retrieval."""
        statuses: list[RetrievalStatus] = []
        definitions: dict[str, Memory] = {}
        chunks = []
        space_id = ""
        try:
            if not query.strip() or not 1 <= top_k <= 100:
                raise ValueError("query must be nonempty and top_k must be between 1 and 100")
            async with self.connect() as client:
                space_id = await self.resolve_space(client, default_name)
                stream = await client.memories.retrieve(
                    message=query,
                    space_ids=[space_id],
                    requested_size=top_k,
                    fetch_memory=True,
                    fetch_memory_content=False,
                )
                assert not isinstance(stream, list)
                async with stream:
                    async for event in stream:
                        if event.status:
                            statuses.append(
                                RetrievalStatus(
                                    code=event.status.code or "UNKNOWN",
                                    message=event.status.message,
                                    details=event.status.details or {},
                                )
                            )
                        if event.memory_definition:
                            memory = event.memory_definition
                            definitions[memory.memory_id] = memory
                        if event.retrieved_item:
                            item = event.retrieved_item
                            if item.memory:
                                definitions[item.memory.memory_id] = item.memory
                            if item.chunk:
                                chunks.append(item.chunk.chunk)
        except (*_REQUEST_ERRORS, ValueError) as error:
            statuses.append(RetrievalStatus(code="RETRIEVAL_ERROR", message=str(error)))
        memories = []
        seen = set()
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            definition = definitions.get(chunk.memory_id)
            memories.append(
                MemoryItem(
                    memory_id=chunk.memory_id,
                    chunk_id=chunk.chunk_id,
                    space_id=definition.space_id if definition else space_id,
                    content=chunk.chunk_text,
                    metadata=(definition.metadata or {}) if definition else {},
                    chunk_metadata=chunk.metadata or {},
                    updated_at=chunk.updated_at,
                )
            )
        partial = any(status.code not in _INFORMATIONAL for status in statuses)
        message = f"Retrieved {len(memories)} matching chunk(s)."
        if partial:
            message += (
                " Search may be incomplete; inspect statuses before concluding nothing exists."
            )
        return GoodmemFetchResponse(
            success=bool(memories) or not partial,
            memories=memories,
            count=len(memories),
            statuses=statuses,
            partial=partial,
            message=message,
        )
