# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
# SPDX-License-Identifier: Apache-2.0

"""Native ADK FunctionTools for explicit persistent memory."""

from goodmem import AsyncGoodmem
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

from ._backend import Backend
from ._results import GoodmemFetchResponse, GoodmemSaveResponse


class GoodmemSaveTool(FunctionTool):
    """Save text and the current user's inline attachments to a configured scope.

    Args:
        base_url: GoodMem server URL; defaults to GOODMEM_BASE_URL.
        api_key: GoodMem credential; defaults to GOODMEM_API_KEY.
        embedder_id: Embedder for new spaces; otherwise uses an existing embedder.
        space_id: Explicit space ID. If a name is also supplied it must match.
        space_name: Explicit shared space; otherwise defaults to adk_tool_{user_id}.
        client: Caller-owned asynchronous GoodMem SDK, instead of connection settings.
        timeout: Request timeout when the tool creates its own SDK.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        embedder_id: str | None = None,
        space_id: str | None = None,
        space_name: str | None = None,
        *,
        client: AsyncGoodmem | None = None,
        timeout: float = 30.0,
    ) -> None:
        backend = Backend(
            client=client,
            base_url=base_url,
            api_key=api_key,
            embedder_id=embedder_id,
            space_id=space_id,
            space_name=space_name,
            timeout=timeout,
        )

        async def goodmem_save(content: str, tool_context: ToolContext) -> GoodmemSaveResponse:
            """Remember useful information and the current user's inline attachments.

            Check success, partial, errors, and accepted IDs before confirming a save.
            Accepted writes may still be indexing. Do not re-upload accepted memories
            when another attachment fails or an immediate search returns nothing.

            Args:
                content: Text to remember, or a description of the uploaded files.
                tool_context: Context supplied by ADK.
            """
            session = tool_context.session
            return await backend.save(
                default_name=f"adk_tool_{tool_context.user_id}",
                text=content,
                content=tool_context.user_content,
                metadata={
                    "app_name": session.app_name,
                    "user_id": session.user_id,
                    "session_id": session.id,
                    "role": "user",
                    "source": "adk_tool",
                },
            )

        super().__init__(goodmem_save)


class GoodmemFetchTool(FunctionTool):
    """Search a configured memory scope and return every matching chunk.

    Args:
        base_url: GoodMem server URL; defaults to GOODMEM_BASE_URL.
        api_key: GoodMem credential; defaults to GOODMEM_API_KEY.
        embedder_id: Embedder for new spaces.
        space_id: Explicit space ID. If a name is also supplied it must match.
        space_name: Explicit shared space; otherwise defaults to adk_tool_{user_id}.
        top_k: Default number of chunks, between 1 and 100.
        client: Caller-owned asynchronous SDK, instead of connection settings.
        timeout: Request timeout when the tool creates its own SDK.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        embedder_id: str | None = None,
        space_id: str | None = None,
        space_name: str | None = None,
        top_k: int = 5,
        *,
        client: AsyncGoodmem | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        backend = Backend(
            client=client,
            base_url=base_url,
            api_key=api_key,
            embedder_id=embedder_id,
            space_id=space_id,
            space_name=space_name,
            timeout=timeout,
        )
        default_top_k = top_k

        async def goodmem_fetch(
            query: str,
            tool_context: ToolContext,
            top_k: int = default_top_k,
        ) -> GoodmemFetchResponse:
            """Retrieve saved facts about this user across conversations.

            Use this before saying you do not know or cannot access a previously saved fact.
            It searches persistent storage even when the current conversation has no history.
            Results contain matching chunks and original source metadata. When partial
            is true, use available chunks but explain limitations shown in statuses.
            Empty results do not establish that an earlier write failed.

            Args:
                query: What to look for in persistent memory.
                top_k: Maximum matching chunks to request, between 1 and 100.
                tool_context: Context supplied by ADK.
            """
            return await backend.fetch(
                default_name=f"adk_tool_{tool_context.user_id}",
                query=query,
                top_k=top_k,
            )

        super().__init__(goodmem_fetch)
