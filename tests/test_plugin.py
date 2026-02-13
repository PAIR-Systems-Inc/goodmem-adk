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

"""Unit tests for GoodmemPlugin including space resolution."""

import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from google.genai import types

from goodmem_adk.client import GoodmemClient
from goodmem_adk.plugin import GoodmemPlugin

CLIENT_PATCH = "goodmem_adk.plugin.GoodmemClient"


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
MOCK_BASE_URL = "https://api.goodmem.ai"
MOCK_API_KEY = "test-api-key"
MOCK_EMBEDDER_ID = "test-embedder-id"
MOCK_SPACE_ID = "test-space-id"
MOCK_SPACE_NAME = "adk_chat_test_user"
MOCK_USER_ID = "test_user"
MOCK_SESSION_ID = "test_session"
MOCK_MEMORY_ID = "test-memory-id"


class TestGoodmemPlugin:
    """Tests for GoodmemPlugin."""

    @pytest.fixture
    def mock_goodmem_client(self) -> MagicMock:
        with patch(CLIENT_PATCH) as mock_client_class:
            mock_client = MagicMock()
            mock_client.list_embedders.return_value = [
                {"embedderId": MOCK_EMBEDDER_ID, "name": "Test Embedder"}
            ]
            mock_client.list_spaces.return_value = []
            mock_client.create_space.return_value = {"spaceId": MOCK_SPACE_ID}
            mock_client.insert_memory.return_value = {
                "memoryId": MOCK_MEMORY_ID,
                "processingStatus": "COMPLETED",
            }
            mock_client.insert_memory_binary.return_value = {
                "memoryId": MOCK_MEMORY_ID,
                "processingStatus": "COMPLETED",
            }
            mock_client.retrieve_memories.return_value = []
            mock_client.get_memory_by_id.return_value = {
                "memoryId": MOCK_MEMORY_ID,
                "metadata": {"user_id": MOCK_USER_ID, "role": "user"},
            }
            mock_client.get_memories_batch.return_value = [
                {
                    "memoryId": MOCK_MEMORY_ID,
                    "metadata": {"user_id": MOCK_USER_ID, "role": "user"},
                }
            ]
            _wire_ensure_embedder(mock_client)
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def chat_plugin(self, mock_goodmem_client: MagicMock) -> GoodmemPlugin:
        return GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            top_k=5,
            debug=False,
        )

    # -- initialization ---------------------------------------------------------

    def test_plugin_initialization(self, chat_plugin: GoodmemPlugin) -> None:
        assert chat_plugin.name == "GoodmemPlugin"
        assert chat_plugin.top_k == 5

    def test_plugin_initialization_no_embedder_id(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY, top_k=5
        )
        assert plugin._embedder_id is None

    def test_plugin_initialization_no_network_call(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
        )
        mock_goodmem_client.list_embedders.assert_not_called()

    def test_plugin_initialization_requires_base_url(self) -> None:
        with pytest.raises(ValueError):
            GoodmemPlugin(base_url=None, api_key=MOCK_API_KEY)

    def test_plugin_initialization_requires_api_key(self) -> None:
        with pytest.raises(ValueError):
            GoodmemPlugin(base_url=MOCK_BASE_URL, api_key=None)

    # -- _get_space_id ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_space_id_creates_new_space(
        self,
        chat_plugin: GoodmemPlugin,
        mock_goodmem_client: MagicMock,
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = []
        mock_context = MagicMock()
        mock_context.user_id = MOCK_USER_ID
        mock_context.state = {}

        space_id = chat_plugin._get_space_id(mock_context)

        mock_goodmem_client.list_spaces.assert_called_once_with(
            name=MOCK_SPACE_NAME
        )
        mock_goodmem_client.create_space.assert_called_once_with(
            MOCK_SPACE_NAME, MOCK_EMBEDDER_ID
        )
        assert space_id == MOCK_SPACE_ID
        assert mock_context.state["_goodmem_space_id"] == MOCK_SPACE_ID

    @pytest.mark.asyncio
    async def test_get_space_id_uses_existing_space(
        self,
        chat_plugin: GoodmemPlugin,
        mock_goodmem_client: MagicMock,
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "existing-space-id", "name": MOCK_SPACE_NAME}
        ]
        mock_context = MagicMock()
        mock_context.user_id = MOCK_USER_ID
        mock_context.state = {}

        space_id = chat_plugin._get_space_id(mock_context)

        mock_goodmem_client.create_space.assert_not_called()
        assert space_id == "existing-space-id"

    @pytest.mark.asyncio
    async def test_get_space_id_uses_cache(
        self,
        chat_plugin: GoodmemPlugin,
        mock_goodmem_client: MagicMock,
    ) -> None:
        mock_context = MagicMock()
        mock_context.user_id = MOCK_USER_ID
        mock_context.state = {"_goodmem_space_id": "cached-space-id"}

        space_id = chat_plugin._get_space_id(mock_context)

        mock_goodmem_client.list_spaces.assert_not_called()
        assert space_id == "cached-space-id"

    # -- callbacks --------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_on_user_message_logs_text(
        self,
        chat_plugin: GoodmemPlugin,
        mock_goodmem_client: MagicMock,
    ) -> None:
        state_dict = {"_goodmem_space_id": MOCK_SPACE_ID}

        class MockSession:
            id = MOCK_SESSION_ID
            state = state_dict

        mock_context = MagicMock(spec=["user_id", "session"])
        mock_context.user_id = MOCK_USER_ID
        mock_context.session = MockSession()

        user_message = types.Content(
            role="user", parts=[types.Part(text="Hello, how are you?")]
        )

        await chat_plugin.on_user_message_callback(
            invocation_context=mock_context, user_message=user_message
        )

        mock_goodmem_client.insert_memory.assert_called_once()
        call_args = mock_goodmem_client.insert_memory.call_args
        assert "User: Hello, how are you?" in str(call_args)

    @pytest.mark.asyncio
    async def test_before_model_callback_augments_request(
        self,
        chat_plugin: GoodmemPlugin,
        mock_goodmem_client: MagicMock,
    ) -> None:
        mock_context = MagicMock()
        mock_context.user_id = MOCK_USER_ID

        mock_goodmem_client.retrieve_memories.return_value = [
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "chunk1",
                            "memoryId": "mem1",
                            "chunkText": "User: Previous conversation",
                            "updatedAt": 1768694400000,
                        }
                    }
                }
            }
        ]
        mock_goodmem_client.get_memories_batch.return_value = [
            {"memoryId": "mem1", "metadata": {"role": "user"}}
        ]

        mock_request = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Current user query"
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_request.contents = [mock_content]

        result = await chat_plugin.before_model_callback(
            callback_context=mock_context, llm_request=mock_request
        )

        mock_goodmem_client.get_memories_batch.assert_called_once()
        assert "BEGIN MEMORY" in mock_part.text
        assert "Previous conversation" in mock_part.text
        assert result is None

    @pytest.mark.asyncio
    async def test_after_model_callback_logs_response(
        self,
        chat_plugin: GoodmemPlugin,
        mock_goodmem_client: MagicMock,
    ) -> None:
        mock_context = MagicMock()
        mock_context.user_id = MOCK_USER_ID
        mock_context.session = MagicMock()
        mock_context.session.id = MOCK_SESSION_ID
        mock_context.state = {"_goodmem_space_id": MOCK_SPACE_ID}

        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "This is the LLM response"
        mock_response.content = mock_content

        await chat_plugin.after_model_callback(
            callback_context=mock_context, llm_response=mock_response
        )

        mock_goodmem_client.insert_memory.assert_called()
        assert "LLM: This is the LLM response" in str(
            mock_goodmem_client.insert_memory.call_args
        )

    @pytest.mark.asyncio
    async def test_multi_user_isolation(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
        )

        def list_spaces_side_effect(*, name=None, **kwargs):
            if name == "adk_chat_alice":
                return [{"name": "adk_chat_alice", "spaceId": "space_alice"}]
            if name == "adk_chat_bob":
                return [{"name": "adk_chat_bob", "spaceId": "space_bob"}]
            return []

        mock_goodmem_client.list_spaces.side_effect = list_spaces_side_effect

        alice_context = MagicMock()
        alice_context.user_id = "alice"
        alice_context.session = MagicMock(id="session_alice")
        alice_context.state = {}

        bob_context = MagicMock()
        bob_context.user_id = "bob"
        bob_context.session = MagicMock(id="session_bob")
        bob_context.state = {}

        alice_response = MagicMock()
        alice_response.content = MagicMock(text="Alice secret")

        bob_response = MagicMock()
        bob_response.content = MagicMock(text="Bob secret")

        await plugin.after_model_callback(
            callback_context=alice_context, llm_response=alice_response
        )
        calls = mock_goodmem_client.insert_memory.call_args_list
        assert calls[-1][0][0] == "space_alice"

        await plugin.after_model_callback(
            callback_context=bob_context, llm_response=bob_response
        )
        calls = mock_goodmem_client.insert_memory.call_args_list
        assert calls[-1][0][0] == "space_bob"


# ---------------------------------------------------------------------------
# Space ID / Space Name / Env Var Resolution
# ---------------------------------------------------------------------------


class TestPluginSpaceResolution:
    """Tests for space_id / space_name override in the plugin."""

    @pytest.fixture
    def mock_goodmem_client(self) -> MagicMock:
        with patch(CLIENT_PATCH) as mock_client_class:
            mock_client = MagicMock()
            mock_client.list_embedders.return_value = [
                {"embedderId": MOCK_EMBEDDER_ID, "name": "Test Embedder"}
            ]
            mock_client.list_spaces.return_value = []
            mock_client.create_space.return_value = {"spaceId": MOCK_SPACE_ID}
            mock_client.insert_memory.return_value = {"memoryId": MOCK_MEMORY_ID}
            mock_client.retrieve_memories.return_value = []
            mock_client.get_memories_batch.return_value = []
            _wire_ensure_embedder(mock_client)
            mock_client_class.return_value = mock_client
            yield mock_client

    def _make_context(self, user_id: str = MOCK_USER_ID) -> MagicMock:
        ctx = MagicMock()
        ctx.user_id = user_id
        ctx.state = {}
        return ctx

    def test_space_id_param_exists(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """space_id set and space exists → used directly, no create."""
        mock_goodmem_client.get_space.return_value = {
            "spaceId": "explicit-id", "name": "some-space"
        }
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_id="explicit-id",
        )
        ctx = self._make_context()
        sid = plugin._get_space_id(ctx)

        assert sid == "explicit-id"
        mock_goodmem_client.get_space.assert_called_once_with("explicit-id")
        mock_goodmem_client.list_spaces.assert_not_called()
        mock_goodmem_client.create_space.assert_not_called()

    def test_space_id_not_found_raises(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """space_id set but doesn't exist → ValueError."""
        mock_goodmem_client.get_space.return_value = None
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_id="nonexistent-id",
        )
        ctx = self._make_context()
        with pytest.raises(ValueError, match="not found"):
            plugin._get_space_id(ctx)
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
            plugin = GoodmemPlugin(
                base_url=MOCK_BASE_URL,
                api_key=MOCK_API_KEY,
                embedder_id=MOCK_EMBEDDER_ID,
            )
        ctx = self._make_context()
        sid = plugin._get_space_id(ctx)

        assert sid == "env-id"
        mock_goodmem_client.get_space.assert_called_once_with("env-id")
        mock_goodmem_client.list_spaces.assert_not_called()

    def test_space_name_param_overrides_default(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "custom-id", "name": "custom_space_name"}
        ]
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_name="custom_space_name",
        )
        ctx = self._make_context()
        sid = plugin._get_space_id(ctx)

        assert sid == "custom-id"
        mock_goodmem_client.list_spaces.assert_called_once_with(
            name="custom_space_name"
        )

    def test_space_name_env_var(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "env-name-id", "name": "env_space_name"}
        ]
        with patch.dict(
            "os.environ",
            {"GOODMEM_SPACE_NAME": "env_space_name"},
            clear=False,
        ):
            plugin = GoodmemPlugin(
                base_url=MOCK_BASE_URL,
                api_key=MOCK_API_KEY,
                embedder_id=MOCK_EMBEDDER_ID,
            )
        ctx = self._make_context()
        sid = plugin._get_space_id(ctx)

        assert sid == "env-name-id"
        mock_goodmem_client.list_spaces.assert_called_once_with(
            name="env_space_name"
        )

    def test_space_id_and_name_matching(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """No error when both refer to the same space."""
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "same-id", "name": "my_space"}
        ]
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_id="same-id",
            space_name="my_space",
        )
        ctx = self._make_context()
        sid = plugin._get_space_id(ctx)
        assert sid == "same-id"

    def test_space_id_and_name_mismatch_raises(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "other-id", "name": "my_space"}
        ]
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_id="wrong-id",
            space_name="my_space",
        )
        ctx = self._make_context()
        with pytest.raises(ValueError, match="refer to different spaces"):
            plugin._get_space_id(ctx)

    def test_space_id_and_name_not_found_raises(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = []
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_id="some-id",
            space_name="nonexistent",
        )
        ctx = self._make_context()
        with pytest.raises(
            ValueError, match="does not match any existing space"
        ):
            plugin._get_space_id(ctx)

    def test_space_id_validation_runs_once(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """Validation between space_id and space_name runs only on first call."""
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "same-id", "name": "my_space"}
        ]
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_id="same-id",
            space_name="my_space",
        )
        ctx = self._make_context()

        plugin._get_space_id(ctx)
        plugin._get_space_id(ctx)

        # list_spaces called only once for validation, not on second call
        mock_goodmem_client.list_spaces.assert_called_once()

    def test_space_id_param_overrides_env(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.get_space.return_value = {
            "spaceId": "param-id", "name": "param-space"
        }
        with patch.dict(
            "os.environ",
            {"GOODMEM_SPACE_ID": "env-id"},
            clear=False,
        ):
            plugin = GoodmemPlugin(
                base_url=MOCK_BASE_URL,
                api_key=MOCK_API_KEY,
                embedder_id=MOCK_EMBEDDER_ID,
                space_id="param-id",
            )
        ctx = self._make_context()
        sid = plugin._get_space_id(ctx)
        assert sid == "param-id"
        mock_goodmem_client.get_space.assert_called_once_with("param-id")

    def test_space_name_param_overrides_env(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "param-name-id", "name": "param_space_name"}
        ]
        with patch.dict(
            "os.environ",
            {"GOODMEM_SPACE_NAME": "env_space_name"},
            clear=False,
        ):
            plugin = GoodmemPlugin(
                base_url=MOCK_BASE_URL,
                api_key=MOCK_API_KEY,
                embedder_id=MOCK_EMBEDDER_ID,
                space_name="param_space_name",
            )
        ctx = self._make_context()
        sid = plugin._get_space_id(ctx)
        assert sid == "param-name-id"
        mock_goodmem_client.list_spaces.assert_called_once_with(
            name="param_space_name"
        )

    def test_space_name_auto_creates_if_not_exists(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """space_name auto-creates the space when it doesn't exist."""
        mock_goodmem_client.list_spaces.return_value = []
        mock_goodmem_client.create_space.return_value = {
            "spaceId": "new-custom-id"
        }
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            space_name="my_custom_space",
        )
        ctx = self._make_context()
        sid = plugin._get_space_id(ctx)

        assert sid == "new-custom-id"
        mock_goodmem_client.list_spaces.assert_called_once_with(
            name="my_custom_space"
        )
        mock_goodmem_client.create_space.assert_called_once_with(
            "my_custom_space", MOCK_EMBEDDER_ID
        )


# ---------------------------------------------------------------------------
# Embedder Priority (plugin)
# ---------------------------------------------------------------------------


class TestPluginEmbedderPriority:
    """Tests for embedder resolution priority in the plugin."""

    @pytest.fixture
    def mock_goodmem_client(self) -> MagicMock:
        with patch(CLIENT_PATCH) as mock_client_class:
            mock_client = MagicMock()
            mock_client.list_embedders.return_value = [
                {"embedderId": MOCK_EMBEDDER_ID, "name": "Test Embedder"}
            ]
            mock_client.list_spaces.return_value = []
            mock_client.create_space.return_value = {"spaceId": MOCK_SPACE_ID}
            mock_client.insert_memory.return_value = {"memoryId": MOCK_MEMORY_ID}
            mock_client.retrieve_memories.return_value = []
            mock_client.get_memories_batch.return_value = []
            _wire_ensure_embedder(mock_client)
            mock_client_class.return_value = mock_client
            yield mock_client

    def _make_context(self, user_id: str = MOCK_USER_ID) -> MagicMock:
        ctx = MagicMock()
        ctx.user_id = user_id
        ctx.state = {}
        return ctx

    def test_embedder_id_specified_and_valid(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """GOODMEM_EMBEDDER_ID if set and valid → used."""
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
        )
        ctx = self._make_context()
        plugin._get_space_id(ctx)

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
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id="nonexistent-emb",
        )
        with pytest.raises(ValueError, match="not found"):
            plugin._get_embedder_id()
        mock_goodmem_client.create_embedder.assert_not_called()

    def test_first_available_embedder_used(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """When no embedder_id given, first available embedder is used."""
        mock_goodmem_client.list_embedders.return_value = [
            {"embedderId": "first-emb", "name": "First"},
            {"embedderId": "second-emb", "name": "Second"},
        ]
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
        )
        ctx = self._make_context()
        plugin._get_space_id(ctx)

        mock_goodmem_client.create_space.assert_called_once_with(
            MOCK_SPACE_NAME, "first-emb"
        )

    def test_auto_create_embedder_with_google_api_key(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """No embedders + GOOGLE_API_KEY set → auto-create gemini embedder."""
        mock_goodmem_client.list_embedders.return_value = []
        mock_goodmem_client.create_embedder.return_value = {
            "embedderId": "auto-created-emb"
        }
        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "test-google-key"},
            clear=False,
        ):
            plugin = GoodmemPlugin(
                base_url=MOCK_BASE_URL,
                api_key=MOCK_API_KEY,
            )
            ctx = self._make_context()
            plugin._get_space_id(ctx)

        mock_goodmem_client.create_embedder.assert_called_once()
        mock_goodmem_client.create_space.assert_called_once_with(
            MOCK_SPACE_NAME, "auto-created-emb"
        )

    def test_no_embedders_no_api_key_raises(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """No embedders + no GOOGLE_API_KEY → ValueError."""
        mock_goodmem_client.list_embedders.return_value = []
        plugin = GoodmemPlugin(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
        )
        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""},
            clear=False,
        ):
            with pytest.raises(ValueError, match="No embedders available"):
                plugin._get_embedder_id()
