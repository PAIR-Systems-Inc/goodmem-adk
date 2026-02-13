# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for GoodmemMemoryService including space resolution."""

# pylint: disable=protected-access,unused-argument,too-many-public-methods
# pylint: disable=redefined-outer-name

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from google.adk.events.event import Event
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from goodmem_adk.client import GoodmemClient
from goodmem_adk.memory import (
    format_memory_block_for_prompt,
    GoodmemMemoryService,
    GoodmemMemoryServiceConfig,
)

def _wire_ensure_embedder(mock_client: MagicMock) -> None:
    """Wire ensure_embedder on a mock so it delegates to the real impl."""
    mock_client.ensure_embedder = (
        lambda **kw: GoodmemClient.ensure_embedder(mock_client, **kw)
    )
    mock_client._auto_create_google_embedder = (
        lambda **kw: GoodmemClient._auto_create_google_embedder(
            mock_client, **kw
        )
    )


# Mock constants
MOCK_BASE_URL = "https://api.goodmem.ai/v1"
MOCK_API_KEY = "test-api-key"
MOCK_EMBEDDER_ID = "test-embedder-id"
MOCK_SPACE_ID = "test-space-id"
MOCK_SPACE_NAME = "adk_memory_test-app_test-user"
MOCK_APP_NAME = "test-app"
MOCK_USER_ID = "test-user"
MOCK_SESSION_ID = "test-session"
MOCK_MEMORY_ID = "test-memory-id"

MOCK_SESSION = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id=MOCK_SESSION_ID,
    last_update_time=1000,
    events=[
        Event(
            id="event-1",
            invocation_id="inv-1",
            author="user",
            timestamp=12345,
            content=types.Content(
                parts=[types.Part(text="Hello, I like Python.")]
            ),
        ),
        Event(
            id="event-2",
            invocation_id="inv-2",
            author="model",
            timestamp=12346,
            content=types.Content(
                parts=[
                    types.Part(text="Python is a great programming language.")
                ]
            ),
        ),
        # Empty event, should be ignored
        Event(
            id="event-3",
            invocation_id="inv-3",
            author="user",
            timestamp=12347,
        ),
        # Function call event, should be ignored
        Event(
            id="event-4",
            invocation_id="inv-4",
            author="agent",
            timestamp=12348,
            content=types.Content(
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name="test_function")
                    )
                ]
            ),
        ),
    ],
)

MOCK_SESSION_WITH_EMPTY_EVENTS = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id=MOCK_SESSION_ID,
    last_update_time=1000,
)

_CLIENT_PATCH = "goodmem_adk.memory.GoodmemClient"


# ---------------------------------------------------------------------------
# GoodmemMemoryServiceConfig
# ---------------------------------------------------------------------------


class TestGoodmemMemoryServiceConfig:
    """Tests for GoodmemMemoryServiceConfig."""

    def test_default_config(self) -> None:
        config = GoodmemMemoryServiceConfig()
        assert config.top_k == 5
        assert config.timeout == 30.0
        assert config.split_turn is False

    def test_custom_config(self) -> None:
        config = GoodmemMemoryServiceConfig(
            top_k=20, timeout=10.0, split_turn=True,
        )
        assert config.top_k == 20
        assert config.timeout == 10.0
        assert config.split_turn is True

    def test_config_validation_top_k(self) -> None:
        with pytest.raises(Exception):
            GoodmemMemoryServiceConfig(top_k=0)
        with pytest.raises(Exception):
            GoodmemMemoryServiceConfig(top_k=101)


# ---------------------------------------------------------------------------
# GoodmemMemoryService
# ---------------------------------------------------------------------------


class TestGoodmemMemoryService:
    """Tests for GoodmemMemoryService."""

    @pytest.fixture
    def mock_goodmem_client(self) -> Generator[MagicMock, None, None]:
        with patch(_CLIENT_PATCH) as mock_cls:
            client = MagicMock()
            client.list_embedders.return_value = [
                {"embedderId": MOCK_EMBEDDER_ID, "name": "Test Embedder"}
            ]
            client.list_spaces.return_value = []
            client.create_space.return_value = {"spaceId": MOCK_SPACE_ID}
            client.insert_memory.return_value = {
                "memoryId": MOCK_MEMORY_ID,
                "processingStatus": "COMPLETED",
            }
            client.insert_memory_binary.return_value = {
                "memoryId": MOCK_MEMORY_ID,
                "processingStatus": "PROCESSING",
            }
            client.retrieve_memories.return_value = []
            _wire_ensure_embedder(client)
            mock_cls.return_value = client
            yield client

    @pytest.fixture
    def memory_service(
        self, mock_goodmem_client: MagicMock
    ) -> GoodmemMemoryService:
        return GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
        )

    @pytest.fixture
    def memory_service_with_config(
        self, mock_goodmem_client: MagicMock
    ) -> GoodmemMemoryService:
        config = GoodmemMemoryServiceConfig(top_k=5, timeout=10.0)
        return GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            config=config,
        )

    # -- constructor / lazy init ------------------------------------------------

    def test_service_initialization_no_network_call(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
        )
        mock_goodmem_client.list_embedders.assert_not_called()
        mock_goodmem_client.list_spaces.assert_not_called()

    def test_service_initialization_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key is required"):
            GoodmemMemoryService(base_url=MOCK_BASE_URL, api_key="")

    # -- embedder resolution ----------------------------------------------------

    def test_embedder_resolved_on_first_space_creation(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
        )
        assert service._resolved_embedder_id is None
        service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        assert service._resolved_embedder_id == MOCK_EMBEDDER_ID

    def test_embedder_uses_first_available(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.list_embedders.return_value = [
            {"embedderId": "first-emb", "name": "First"},
            {"embedderId": "second-emb", "name": "Second"},
        ]
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY,
        )
        service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        assert service._resolved_embedder_id == "first-emb"

    def test_no_embedders_and_no_api_key_fails(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """When no embedders exist and GOOGLE_API_KEY is unset, raises."""
        mock_goodmem_client.list_embedders.return_value = []
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY,
        )
        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""},
            clear=False,
        ):
            with pytest.raises(ValueError, match="No embedders available"):
                service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

    # -- space management -------------------------------------------------------

    def test_ensure_space_creates_new_space(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        space_id = memory_service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

        mock_goodmem_client.list_spaces.assert_called_once_with(
            name=MOCK_SPACE_NAME
        )
        mock_goodmem_client.create_space.assert_called_once_with(
            MOCK_SPACE_NAME, MOCK_EMBEDDER_ID
        )
        assert space_id == MOCK_SPACE_ID

    def test_ensure_space_uses_existing_space(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "existing-space-id", "name": MOCK_SPACE_NAME}
        ]
        space_id = memory_service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        mock_goodmem_client.create_space.assert_not_called()
        assert space_id == "existing-space-id"

    def test_ensure_space_uses_cache(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        cache_key = f"{MOCK_APP_NAME}:{MOCK_USER_ID}"
        memory_service._space_cache[cache_key] = "cached-space-id"
        space_id = memory_service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        mock_goodmem_client.list_spaces.assert_not_called()
        assert space_id == "cached-space-id"

    # -- add_session_to_memory --------------------------------------------------

    @pytest.mark.asyncio
    async def test_add_session_to_memory_success(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        await memory_service.add_session_to_memory(MOCK_SESSION)
        mock_goodmem_client.insert_memory.assert_called_once()
        call_kw = mock_goodmem_client.insert_memory.call_args.kwargs
        assert "User: Hello, I like Python." in call_kw["content"]
        assert (
            "LLM: Python is a great programming language."
            in call_kw["content"]
        )

    @pytest.mark.asyncio
    async def test_add_session_filters_empty_events(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        await memory_service.add_session_to_memory(
            MOCK_SESSION_WITH_EMPTY_EVENTS
        )
        mock_goodmem_client.insert_memory.assert_not_called()

    # -- search_memory ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_search_memory_success(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        mock_goodmem_client.retrieve_memories.return_value = [
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkText": (
                                "User: What is Python?\n"
                                "LLM: Python is great"
                            ),
                            "memoryId": "mem-1",
                        }
                    }
                }
            },
        ]

        result = await memory_service.search_memory(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            query="Python programming",
        )

        mock_goodmem_client.retrieve_memories.assert_called_once_with(
            query="Python programming",
            space_ids=[MOCK_SPACE_ID],
            request_size=5,
        )
        assert len(result.memories) == 1
        assert "Python is great" in result.memories[0].content.parts[0].text

    @pytest.mark.asyncio
    async def test_search_memory_error_handling(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        mock_goodmem_client.retrieve_memories.side_effect = Exception(
            "API Error"
        )
        result = await memory_service.search_memory(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            query="test query",
        )
        assert len(result.memories) == 0

    # -- close ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_close_calls_client_close(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        await memory_service.close()
        mock_goodmem_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Space ID / Space Name / Env Var Resolution (memory service)
# ---------------------------------------------------------------------------


class TestMemoryServiceSpaceResolution:
    """Tests for space_id / space_name override in memory service."""

    @pytest.fixture
    def mock_goodmem_client(self) -> Generator[MagicMock, None, None]:
        with patch(_CLIENT_PATCH) as mock_cls:
            client = MagicMock()
            client.list_embedders.return_value = [
                {"embedderId": MOCK_EMBEDDER_ID, "name": "Test Embedder"}
            ]
            client.list_spaces.return_value = []
            client.create_space.return_value = {"spaceId": MOCK_SPACE_ID}
            client.insert_memory.return_value = {
                "memoryId": MOCK_MEMORY_ID,
            }
            client.retrieve_memories.return_value = []
            _wire_ensure_embedder(client)
            mock_cls.return_value = client
            yield client

    def test_space_id_exists(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """space_id set and space exists → used directly, no create."""
        mock_goodmem_client.get_space.return_value = {
            "spaceId": "direct-id", "name": "some-space"
        }
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_id="direct-id",
        )
        result = service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        assert result == "direct-id"
        mock_goodmem_client.get_space.assert_called_once_with("direct-id")
        mock_goodmem_client.list_spaces.assert_not_called()
        mock_goodmem_client.create_space.assert_not_called()

    def test_space_id_not_found_raises(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """space_id set but doesn't exist → ValueError."""
        mock_goodmem_client.get_space.return_value = None
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_id="nonexistent-id",
        )
        with pytest.raises(ValueError, match="not found"):
            service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        mock_goodmem_client.get_space.assert_called_once_with("nonexistent-id")
        mock_goodmem_client.create_space.assert_not_called()

    def test_space_id_env_var(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.get_space.return_value = {
            "spaceId": "env-id", "name": "env-space"
        }
        with patch.dict(
            "os.environ", {"GOODMEM_SPACE_ID": "env-id"}, clear=False
        ):
            service = GoodmemMemoryService(
                base_url=MOCK_BASE_URL,
                api_key=MOCK_API_KEY,
                embedder_id=MOCK_EMBEDDER_ID,
            )
        result = service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        assert result == "env-id"
        mock_goodmem_client.get_space.assert_called_once_with("env-id")
        mock_goodmem_client.list_spaces.assert_not_called()

    def test_space_name_overrides_default(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "custom-id", "name": "custom_name"}
        ]
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_name="custom_name",
        )
        result = service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        assert result == "custom-id"
        mock_goodmem_client.list_spaces.assert_called_once_with(
            name="custom_name"
        )

    def test_space_name_env_var(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "env-name-id", "name": "env_name"}
        ]
        with patch.dict(
            "os.environ", {"GOODMEM_SPACE_NAME": "env_name"}, clear=False
        ):
            service = GoodmemMemoryService(
                base_url=MOCK_BASE_URL,
                api_key=MOCK_API_KEY,
                embedder_id=MOCK_EMBEDDER_ID,
            )
        result = service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        assert result == "env-name-id"

    def test_space_id_and_name_matching(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "same-id", "name": "my_space"}
        ]
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_id="same-id",
            space_name="my_space",
        )
        result = service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        assert result == "same-id"

    def test_space_id_and_name_mismatch_raises(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "other-id", "name": "my_space"}
        ]
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_id="wrong-id",
            space_name="my_space",
        )
        with pytest.raises(ValueError, match="refer to different spaces"):
            service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

    def test_space_id_and_name_not_found_raises(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = []
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_id="some-id",
            space_name="nonexistent",
        )
        with pytest.raises(
            ValueError, match="does not match any existing space"
        ):
            service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

    def test_space_id_validation_runs_once(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "same-id", "name": "my_space"}
        ]
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_id="same-id",
            space_name="my_space",
        )
        service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        mock_goodmem_client.list_spaces.assert_called_once()

    def test_space_id_param_overrides_env(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.get_space.return_value = {
            "spaceId": "param-id", "name": "param-space"
        }
        with patch.dict(
            "os.environ", {"GOODMEM_SPACE_ID": "env-id"}, clear=False
        ):
            service = GoodmemMemoryService(
                base_url=MOCK_BASE_URL,
                api_key=MOCK_API_KEY,
                embedder_id=MOCK_EMBEDDER_ID,
                space_id="param-id",
            )
        result = service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        assert result == "param-id"
        mock_goodmem_client.get_space.assert_called_once_with("param-id")

    def test_space_name_auto_creates_if_not_exists(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """space_name auto-creates the space when it doesn't exist."""
        mock_goodmem_client.list_spaces.return_value = []
        mock_goodmem_client.create_space.return_value = {
            "spaceId": "new-custom-id"
        }
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_name="my_custom_space",
        )
        result = service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

        assert result == "new-custom-id"
        mock_goodmem_client.list_spaces.assert_called_once_with(
            name="my_custom_space"
        )
        mock_goodmem_client.create_space.assert_called_once_with(
            "my_custom_space", MOCK_EMBEDDER_ID
        )


# ---------------------------------------------------------------------------
# Embedder Priority (memory service)
# ---------------------------------------------------------------------------


class TestMemoryServiceEmbedderPriority:
    """Tests for embedder resolution priority in the memory service."""

    @pytest.fixture
    def mock_goodmem_client(self) -> Generator[MagicMock, None, None]:
        with patch(_CLIENT_PATCH) as mock_cls:
            client = MagicMock()
            client.list_embedders.return_value = [
                {"embedderId": MOCK_EMBEDDER_ID, "name": "Test Embedder"}
            ]
            client.list_spaces.return_value = []
            client.create_space.return_value = {"spaceId": MOCK_SPACE_ID}
            client.insert_memory.return_value = {"memoryId": MOCK_MEMORY_ID}
            client.retrieve_memories.return_value = []
            _wire_ensure_embedder(client)
            mock_cls.return_value = client
            yield client

    def test_embedder_id_specified_and_valid(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """GOODMEM_EMBEDDER_ID if set and valid → used."""
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
        )
        service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        assert service._resolved_embedder_id == MOCK_EMBEDDER_ID
        mock_goodmem_client.create_space.assert_called_once_with(
            MOCK_SPACE_NAME, MOCK_EMBEDDER_ID
        )

    def test_embedder_id_specified_invalid_raises(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """GOODMEM_EMBEDDER_ID if set but not found → ValueError."""
        mock_goodmem_client.list_embedders.return_value = [
            {"embedderId": "other-emb", "name": "Other"}
        ]
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id="nonexistent-emb",
        )
        with pytest.raises(ValueError, match="not found"):
            service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)
        mock_goodmem_client.create_embedder.assert_not_called()

    def test_auto_create_embedder_with_google_api_key(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """No embedders + GOOGLE_API_KEY → auto-create gemini embedder."""
        mock_goodmem_client.list_embedders.return_value = []
        mock_goodmem_client.create_embedder.return_value = {
            "embedderId": "auto-created-emb"
        }
        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "test-google-key"},
            clear=False,
        ):
            service = GoodmemMemoryService(
                base_url=MOCK_BASE_URL,
                api_key=MOCK_API_KEY,
            )
            service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

        mock_goodmem_client.create_embedder.assert_called_once()
        mock_goodmem_client.create_space.assert_called_once_with(
            MOCK_SPACE_NAME, "auto-created-emb"
        )


# ---------------------------------------------------------------------------
# format_memory_block_for_prompt
# ---------------------------------------------------------------------------


class TestFormatMemoryBlockForPrompt:
    """Tests for format_memory_block_for_prompt."""

    def test_empty_response(self) -> None:
        response = SearchMemoryResponse(memories=[])
        block = format_memory_block_for_prompt(response)
        assert "BEGIN MEMORY" in block
        assert "END MEMORY" in block

    def test_one_chunk_with_timestamp(self) -> None:
        entry = MemoryEntry(
            id="mem-123",
            content=types.Content(
                parts=[
                    types.Part(
                        text="User: My favorite color is blue.\nLLM: I'll remember."
                    )
                ]
            ),
            timestamp="2025-02-05 14:30",
        )
        response = SearchMemoryResponse(memories=[entry])
        block = format_memory_block_for_prompt(response)
        assert "- id: mem-123" in block
        assert "My favorite color is blue." in block

    def test_chunk_without_timestamp(self) -> None:
        entry = MemoryEntry(
            id="mem-456",
            content=types.Content(parts=[types.Part(text="User: Hello.")]),
            timestamp=None,
        )
        response = SearchMemoryResponse(memories=[entry])
        block = format_memory_block_for_prompt(response)
        assert "- id: mem-456" in block
        assert "User: Hello." in block
