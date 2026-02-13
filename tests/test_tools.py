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

"""Unit tests for Goodmem tools (save/fetch) including space resolution."""

from unittest.mock import MagicMock, call, patch

import pytest

from goodmem_adk.client import GoodmemClient
from goodmem_adk import tools as goodmem_tools
from goodmem_adk.tools import goodmem_save, goodmem_fetch
from goodmem_adk.tools import (
    _format_debug_table,
    _format_timestamp_for_table,
    _wrap_content,
    _get_or_create_space,
    GoodmemSaveTool,
    GoodmemFetchTool,
)

CLIENT_PATCH = "goodmem_adk.tools.GoodmemClient"


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


# ---------------------------------------------------------------------------
# goodmem_save
# ---------------------------------------------------------------------------


class TestGoodmemSave:
    """Test cases for goodmem_save function."""

    @pytest.fixture(autouse=True)
    def clear_client_cache(self):
        goodmem_tools._client_cache.clear()
        yield
        goodmem_tools._client_cache.clear()

    @pytest.fixture
    def mock_config(self):
        return {
            "base_url": "http://localhost:8080",
            "api_key": "test-api-key",
        }

    @pytest.fixture
    def mock_tool_context(self):
        context = MagicMock()
        context.user_id = "test-user"
        context.session = MagicMock()
        context.session.id = "test-session"
        context.state = {}
        return context

    @pytest.mark.asyncio
    async def test_save_success(self, mock_config, mock_tool_context):
        with patch(CLIENT_PATCH) as MockClient:
            mock_client = MockClient.return_value
            mock_client.insert_memory.return_value = {"memoryId": "memory-123"}
            mock_client.list_spaces.return_value = [
                {"spaceId": "existing-space-123", "name": "adk_tool_test-user"}
            ]

            response = await goodmem_save(
                content="Test content",
                tool_context=mock_tool_context,
                base_url=mock_config["base_url"],
                api_key=mock_config["api_key"],
            )

            assert response.success is True
            assert response.memory_id == "memory-123"
            assert "Successfully wrote" in response.message
            mock_client.list_spaces.assert_called_once_with(
                name="adk_tool_test-user"
            )
            assert (
                mock_tool_context.state["_goodmem_space_id"]
                == "existing-space-123"
            )

    @pytest.mark.asyncio
    async def test_save_missing_base_url(self, mock_tool_context):
        response = await goodmem_save(
            content="Test content",
            tool_context=mock_tool_context,
            base_url=None,
            api_key="test-api-key",
        )
        assert response.success is False
        assert "base_url" in response.message.lower()

    @pytest.mark.asyncio
    async def test_save_missing_api_key(self, mock_tool_context):
        response = await goodmem_save(
            content="Test content",
            tool_context=mock_tool_context,
            base_url="http://localhost:8080",
            api_key=None,
        )
        assert response.success is False
        assert "api_key" in response.message.lower()

    @pytest.mark.asyncio
    async def test_save_without_tool_context(self, mock_config):
        response = await goodmem_save(
            content="Test content",
            base_url=mock_config["base_url"],
            api_key=mock_config["api_key"],
        )
        assert response.success is False
        assert "tool_context is required" in response.message

    @pytest.mark.asyncio
    async def test_save_creates_space_if_not_exists(
        self, mock_config, mock_tool_context
    ):
        with patch(CLIENT_PATCH) as MockClient:
            mock_client = MockClient.return_value
            _wire_ensure_embedder(mock_client)
            mock_client.list_spaces.return_value = []
            mock_client.list_embedders.return_value = [
                {"embedderId": "embedder-1", "name": "Test Embedder"}
            ]
            mock_client.create_space.return_value = {"spaceId": "new-space-123"}
            mock_client.insert_memory.return_value = {"memoryId": "memory-123"}

            response = await goodmem_save(
                content="Test content",
                tool_context=mock_tool_context,
                base_url=mock_config["base_url"],
                api_key=mock_config["api_key"],
            )

            assert response.success is True
            mock_client.create_space.assert_called_once_with(
                "adk_tool_test-user", "embedder-1"
            )
            assert (
                mock_tool_context.state["_goodmem_space_id"] == "new-space-123"
            )

    @pytest.mark.asyncio
    async def test_save_uses_cached_space_id(
        self, mock_config, mock_tool_context
    ):
        mock_tool_context.state["_goodmem_space_id"] = "cached-space-123"

        with patch(CLIENT_PATCH) as MockClient:
            mock_client = MockClient.return_value
            mock_client.insert_memory.return_value = {"memoryId": "memory-123"}

            response = await goodmem_save(
                content="Test content",
                tool_context=mock_tool_context,
                base_url=mock_config["base_url"],
                api_key=mock_config["api_key"],
            )

            assert response.success is True
            mock_client.list_spaces.assert_not_called()


# ---------------------------------------------------------------------------
# goodmem_fetch
# ---------------------------------------------------------------------------


class TestGoodmemFetch:
    """Test cases for goodmem_fetch function."""

    @pytest.fixture(autouse=True)
    def clear_client_cache(self):
        goodmem_tools._client_cache.clear()
        yield
        goodmem_tools._client_cache.clear()

    @pytest.fixture
    def mock_config(self):
        return {
            "base_url": "http://localhost:8080",
            "api_key": "test-api-key",
        }

    @pytest.fixture
    def mock_tool_context(self):
        context = MagicMock()
        context.user_id = "test-user"
        context.session = MagicMock()
        context.session.id = "test-session"
        context.state = {}
        return context

    @pytest.mark.asyncio
    async def test_fetch_success(self, mock_config, mock_tool_context):
        with patch(CLIENT_PATCH) as MockClient:
            mock_client = MockClient.return_value
            mock_client.list_spaces.return_value = [
                {"spaceId": "existing-space-123", "name": "adk_tool_test-user"}
            ]
            mock_client.retrieve_memories.return_value = [
                {
                    "retrievedItem": {
                        "chunk": {
                            "chunk": {
                                "memoryId": "memory-123",
                                "chunkText": "Test memory content",
                                "updatedAt": 1234567890,
                            }
                        }
                    }
                }
            ]
            mock_client.get_memory_by_id.return_value = {
                "metadata": {"user_id": "test-user"}
            }

            response = await goodmem_fetch(
                query="test query",
                top_k=5,
                tool_context=mock_tool_context,
                base_url=mock_config["base_url"],
                api_key=mock_config["api_key"],
            )

            assert response.success is True
            assert response.count == 1
            assert len(response.memories) == 1
            assert response.memories[0].memory_id == "memory-123"

    @pytest.mark.asyncio
    async def test_fetch_no_results(self, mock_config, mock_tool_context):
        with patch(CLIENT_PATCH) as MockClient:
            mock_client = MockClient.return_value
            mock_client.list_spaces.return_value = [
                {"spaceId": "existing-space-123", "name": "adk_tool_test-user"}
            ]
            mock_client.retrieve_memories.return_value = []

            response = await goodmem_fetch(
                query="test query",
                tool_context=mock_tool_context,
                base_url=mock_config["base_url"],
                api_key=mock_config["api_key"],
            )

            assert response.success is True
            assert response.count == 0
            assert "No memories found" in response.message

    @pytest.mark.asyncio
    async def test_fetch_top_k_capped(self, mock_config, mock_tool_context):
        with patch(CLIENT_PATCH) as MockClient:
            mock_client = MockClient.return_value
            mock_client.list_spaces.return_value = [
                {"spaceId": "existing-space-123", "name": "adk_tool_test-user"}
            ]
            mock_client.retrieve_memories.return_value = []

            await goodmem_fetch(
                query="test",
                top_k=25,
                tool_context=mock_tool_context,
                base_url=mock_config["base_url"],
                api_key=mock_config["api_key"],
            )
            mock_client.retrieve_memories.assert_called_with(
                query="test",
                space_ids=["existing-space-123"],
                request_size=20,
            )


# ---------------------------------------------------------------------------
# _get_or_create_space — space_id / space_name / env var resolution
# ---------------------------------------------------------------------------


class TestGetOrCreateSpaceResolution:
    """Tests for space_id / space_name / env var resolution in tools."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        _wire_ensure_embedder(client)
        return client

    @pytest.fixture
    def mock_tool_context(self):
        context = MagicMock()
        context.user_id = "test-user"
        context.state = {}
        return context

    def test_default_space_name(self, mock_client, mock_tool_context):
        """Without overrides, uses adk_tool_{user_id}."""
        mock_client.list_spaces.return_value = [
            {"spaceId": "found-id", "name": "adk_tool_test-user"}
        ]

        space_id, error = _get_or_create_space(
            mock_client, mock_tool_context
        )

        assert error is None
        assert space_id == "found-id"
        mock_client.list_spaces.assert_called_once_with(
            name="adk_tool_test-user"
        )

    def test_space_name_override(self, mock_client, mock_tool_context):
        """space_name overrides the default naming convention."""
        mock_client.list_spaces.return_value = [
            {"spaceId": "custom-id", "name": "my_custom_space"}
        ]

        space_id, error = _get_or_create_space(
            mock_client, mock_tool_context, space_name="my_custom_space"
        )

        assert error is None
        assert space_id == "custom-id"
        mock_client.list_spaces.assert_called_once_with(
            name="my_custom_space"
        )

    def test_space_name_creates_if_not_exists(
        self, mock_client, mock_tool_context
    ):
        """space_name auto-creates the space if it doesn't exist."""
        mock_client.list_spaces.return_value = []
        mock_client.list_embedders.return_value = [
            {"embedderId": "emb-1", "name": "Embedder"}
        ]
        mock_client.create_space.return_value = {"spaceId": "new-id"}

        space_id, error = _get_or_create_space(
            mock_client, mock_tool_context, space_name="my_new_space"
        )

        assert error is None
        assert space_id == "new-id"
        mock_client.create_space.assert_called_once_with(
            "my_new_space", "emb-1"
        )

    def test_space_id_exists(
        self, mock_client, mock_tool_context
    ):
        """space_id set and space exists → used directly, no create."""
        mock_client.get_space.return_value = {
            "spaceId": "direct-space-id", "name": "some-space"
        }
        space_id, error = _get_or_create_space(
            mock_client, mock_tool_context, space_id="direct-space-id"
        )

        assert error is None
        assert space_id == "direct-space-id"
        mock_client.get_space.assert_called_once_with("direct-space-id")
        mock_client.list_spaces.assert_not_called()
        mock_client.create_space.assert_not_called()
        assert (
            mock_tool_context.state["_goodmem_space_id"] == "direct-space-id"
        )

    def test_space_id_not_found_returns_error(
        self, mock_client, mock_tool_context
    ):
        """space_id set but doesn't exist → error returned."""
        mock_client.get_space.return_value = None
        space_id, error = _get_or_create_space(
            mock_client, mock_tool_context, space_id="nonexistent-id"
        )

        assert space_id is None
        assert "not found" in error
        mock_client.get_space.assert_called_once_with("nonexistent-id")
        mock_client.create_space.assert_not_called()

    def test_space_id_and_space_name_matching(
        self, mock_client, mock_tool_context
    ):
        """When both match, no error."""
        mock_client.list_spaces.return_value = [
            {"spaceId": "correct-id", "name": "my_space"}
        ]

        space_id, error = _get_or_create_space(
            mock_client,
            mock_tool_context,
            space_id="correct-id",
            space_name="my_space",
        )

        assert error is None
        assert space_id == "correct-id"

    def test_space_id_and_space_name_mismatch(
        self, mock_client, mock_tool_context
    ):
        """When both set but refer to different spaces, returns error."""
        mock_client.list_spaces.return_value = [
            {"spaceId": "other-id", "name": "my_space"}
        ]

        space_id, error = _get_or_create_space(
            mock_client,
            mock_tool_context,
            space_id="wrong-id",
            space_name="my_space",
        )

        assert space_id is None
        assert "refer to different spaces" in error

    def test_space_id_and_space_name_not_found(
        self, mock_client, mock_tool_context
    ):
        """When space_name doesn't match any space, returns error."""
        mock_client.list_spaces.return_value = []

        space_id, error = _get_or_create_space(
            mock_client,
            mock_tool_context,
            space_id="some-id",
            space_name="nonexistent_space",
        )

        assert space_id is None
        assert "does not match any existing space" in error


# ---------------------------------------------------------------------------
# Embedder priority in _get_or_create_space
# ---------------------------------------------------------------------------


class TestGetOrCreateSpaceEmbedderPriority:
    """Tests for embedder resolution priority when creating spaces via tools."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.list_spaces.return_value = []  # force space creation
        _wire_ensure_embedder(client)
        return client

    @pytest.fixture
    def mock_tool_context(self):
        context = MagicMock()
        context.user_id = "test-user"
        context.state = {}
        return context

    def test_embedder_id_specified_and_valid(
        self, mock_client, mock_tool_context
    ):
        """GOODMEM_EMBEDDER_ID if set and valid → used."""
        mock_client.list_embedders.return_value = [
            {"embedderId": "custom-emb", "name": "Custom Embedder"}
        ]
        mock_client.create_space.return_value = {"spaceId": "new-space"}

        space_id, error = _get_or_create_space(
            mock_client, mock_tool_context, embedder_id="custom-emb"
        )

        assert error is None
        assert space_id == "new-space"
        mock_client.create_space.assert_called_once_with(
            "adk_tool_test-user", "custom-emb"
        )

    def test_embedder_id_specified_invalid_returns_error(
        self, mock_client, mock_tool_context
    ):
        """GOODMEM_EMBEDDER_ID if set but not found → error returned."""
        mock_client.list_embedders.return_value = [
            {"embedderId": "valid-emb", "name": "Valid Embedder"}
        ]

        space_id, error = _get_or_create_space(
            mock_client, mock_tool_context, embedder_id="nonexistent-emb"
        )

        assert space_id is None
        assert "not found" in error
        mock_client.create_embedder.assert_not_called()
        mock_client.create_space.assert_not_called()

    def test_first_available_embedder_used(
        self, mock_client, mock_tool_context
    ):
        """When no embedder_id, first available embedder is used."""
        mock_client.list_embedders.return_value = [
            {"embedderId": "first-emb", "name": "First"},
            {"embedderId": "second-emb", "name": "Second"},
        ]
        mock_client.create_space.return_value = {"spaceId": "new-space"}

        space_id, error = _get_or_create_space(
            mock_client, mock_tool_context
        )

        assert error is None
        mock_client.create_space.assert_called_once_with(
            "adk_tool_test-user", "first-emb"
        )

    def test_auto_create_embedder_with_google_api_key(
        self, mock_client, mock_tool_context
    ):
        """No embedders + GOOGLE_API_KEY set → auto-create gemini embedder."""
        mock_client.list_embedders.return_value = []
        mock_client.create_embedder.return_value = {
            "embedderId": "auto-created-emb"
        }
        mock_client.create_space.return_value = {"spaceId": "new-space"}

        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "test-google-key"},
            clear=False,
        ):
            space_id, error = _get_or_create_space(
                mock_client, mock_tool_context
            )

        assert error is None
        assert space_id == "new-space"
        mock_client.create_embedder.assert_called_once()
        mock_client.create_space.assert_called_once_with(
            "adk_tool_test-user", "auto-created-emb"
        )

    def test_no_embedders_no_api_key_fails(
        self, mock_client, mock_tool_context
    ):
        """No embedders + no GOOGLE_API_KEY → error returned."""
        mock_client.list_embedders.return_value = []

        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""},
            clear=False,
        ):
            space_id, error = _get_or_create_space(
                mock_client, mock_tool_context
            )

        assert space_id is None
        assert "No embedders available" in error


class TestToolClassesEnvVarFallback:
    """Test that GoodmemSaveTool / GoodmemFetchTool read env vars."""

    def test_save_tool_reads_space_id_env(self):
        with patch.dict(
            "os.environ",
            {"GOODMEM_SPACE_ID": "env-space-id"},
            clear=False,
        ):
            tool = GoodmemSaveTool(
                base_url="http://localhost", api_key="key"
            )
            assert tool._space_id == "env-space-id"

    def test_save_tool_reads_space_name_env(self):
        with patch.dict(
            "os.environ",
            {"GOODMEM_SPACE_NAME": "env-space-name"},
            clear=False,
        ):
            tool = GoodmemSaveTool(
                base_url="http://localhost", api_key="key"
            )
            assert tool._space_name == "env-space-name"

    def test_save_tool_param_overrides_env(self):
        with patch.dict(
            "os.environ",
            {
                "GOODMEM_SPACE_ID": "env-id",
                "GOODMEM_SPACE_NAME": "env-name",
            },
            clear=False,
        ):
            tool = GoodmemSaveTool(
                base_url="http://localhost",
                api_key="key",
                space_id="param-id",
                space_name="param-name",
            )
            assert tool._space_id == "param-id"
            assert tool._space_name == "param-name"

    def test_fetch_tool_reads_space_id_env(self):
        with patch.dict(
            "os.environ",
            {"GOODMEM_SPACE_ID": "env-space-id"},
            clear=False,
        ):
            tool = GoodmemFetchTool(
                base_url="http://localhost", api_key="key"
            )
            assert tool._space_id == "env-space-id"

    def test_fetch_tool_reads_space_name_env(self):
        with patch.dict(
            "os.environ",
            {"GOODMEM_SPACE_NAME": "env-space-name"},
            clear=False,
        ):
            tool = GoodmemFetchTool(
                base_url="http://localhost", api_key="key"
            )
            assert tool._space_name == "env-space-name"


# ---------------------------------------------------------------------------
# Debug table formatting
# ---------------------------------------------------------------------------


class TestDebugTableFormatting:
    """Test cases for debug table formatting functions."""

    def test_format_timestamp_for_table(self):
        timestamp_ms = 1234567890000
        result = _format_timestamp_for_table(timestamp_ms)
        assert result == "2009-02-13 23:31"

        assert _format_timestamp_for_table(None) == ""

    def test_wrap_content(self):
        assert _wrap_content("Short", max_width=55) == ["Short"]

        long = (
            "This is a very long content that should definitely wrap because "
            "it exceeds the maximum width of 55 characters"
        )
        result = _wrap_content(long, max_width=55)
        assert len(result) > 1
        assert all(len(line) <= 55 for line in result)

    def test_format_debug_table_empty(self):
        assert _format_debug_table([]) == ""

    def test_format_debug_table_with_records(self):
        records = [
            {
                "memory_id": "mem-1",
                "timestamp_ms": 1234567890000,
                "role": "user",
                "content": "hello",
            }
        ]
        result = _format_debug_table(records)
        assert "mem-1" in result
        assert "user" in result
        assert "hello" in result
