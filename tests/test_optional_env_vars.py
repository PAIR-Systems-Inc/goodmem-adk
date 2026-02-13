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

"""Integration tests for optional env var / constructor parameter combinations.

These tests verify that all meaningful combinations of space_id, space_name,
and embedder_id work correctly — including cross-session scenarios where
different sessions use different configurations pointing to the same space.

See docs/env-var-test-plan.md for the full test plan.

Run with::

    GOODMEM_BASE_URL=http://localhost:8080 \
    GOODMEM_API_KEY=<key> \
    GOOGLE_API_KEY=<key> \
    pytest tests/test_optional_env_vars.py -v

Either GOOGLE_API_KEY or GEMINI_API_KEY can be used for Gemini authentication.
Optional env vars (GOODMEM_SPACE_ID, GOODMEM_SPACE_NAME, GOODMEM_EMBEDDER_ID)
should be **unset** — the tests manage them internally via constructor params
and monkeypatch.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import List

import pytest
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.runners import InMemoryRunner
from google.genai import types

from goodmem_adk import (
    GoodmemClient,
    GoodmemFetchTool,
    GoodmemPlugin,
    GoodmemSaveTool,
)

# ---------------------------------------------------------------------------
# Skip the entire module when required env vars are missing
# ---------------------------------------------------------------------------

_HAS_GOOGLE_KEY = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not (os.getenv("GOODMEM_BASE_URL") and os.getenv("GOODMEM_API_KEY")
             and _HAS_GOOGLE_KEY),
        reason=(
            "Integration tests require GOODMEM_BASE_URL, GOODMEM_API_KEY, "
            "and GOOGLE_API_KEY (or GEMINI_API_KEY) environment variables"
        ),
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_URL = os.getenv("GOODMEM_BASE_URL", "http://localhost:8080")
_API_KEY = os.getenv("GOODMEM_API_KEY", "")
_INDEX_WAIT = 5
_GOLDFISH_KEYWORDS = ["water", "goldfish", "fish", "aquatic", "aquarium"]


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _extract_final_response(events: list) -> str:
    text_parts: List[str] = []
    for event in events:
        content = getattr(event, "content", None)
        if content is None:
            continue
        if getattr(event, "author", None) == "user":
            continue
        parts = getattr(content, "parts", None)
        if not parts:
            continue
        for part in parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
    return " ".join(text_parts)


def _find_space_id(client: GoodmemClient, space_name: str) -> str | None:
    spaces = client.list_spaces(name=space_name)
    for space in spaces:
        if space.get("name") == space_name:
            return space.get("spaceId")
    return None


def _get_first_embedder_id(client: GoodmemClient) -> str | None:
    embedders = client.list_embedders()
    if embedders:
        return embedders[0].get("embedderId")
    return None


def _pre_create_space(
    client: GoodmemClient, space_name: str
) -> str:
    """Create a space and return its space_id.  Resolves embedder automatically."""
    embedder_id = client.ensure_embedder(debug=False)
    result = client.create_space(space_name, embedder_id)
    return result["spaceId"]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cleanup_spaces():
    """Collect space IDs during the test and delete them in teardown."""
    space_ids: List[str] = []
    yield space_ids
    client = GoodmemClient(_BASE_URL, _API_KEY)
    for sid in space_ids:
        try:
            client.delete_space(sid)
        except Exception:
            pass
    client.close()


@pytest.fixture()
def gm_client():
    """Provide a GoodmemClient for test setup, closed after the test."""
    client = GoodmemClient(_BASE_URL, _API_KEY)
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Goldfish session helpers
# ---------------------------------------------------------------------------


async def _run_goldfish_session1_plugin(runner, app_name, user_id="goldfish_user"):
    """Session 1: tell the agent 'I am a goldfish' via plugin."""
    session = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id
    )
    print(f"\n{'=' * 72}")
    print(f"[ENVVAR] SESSION 1  (id={session.id})")
    print(f"{'=' * 72}")

    msg = types.Content(
        role="user", parts=[types.Part(text="I am a goldfish")]
    )
    events = []
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=msg,
    ):
        events.append(event)

    response = _extract_final_response(events)
    print(f"[ENVVAR] Session 1 response: {response}")
    assert response, "Session 1 should produce a model response"
    return response


async def _run_goldfish_session2_plugin(runner, app_name, user_id="goldfish_user"):
    """Session 2: ask 'Do I live in water?' via plugin."""
    session = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id
    )
    print(f"\n{'=' * 72}")
    print(f"[ENVVAR] SESSION 2  (id={session.id})")
    print(f"{'=' * 72}")

    msg = types.Content(
        role="user", parts=[types.Part(text="Do I live in water?")]
    )
    events = []
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=msg,
    ):
        events.append(event)

    response = _extract_final_response(events)
    print(f"[ENVVAR] Session 2 response: {response}")
    return response


async def _run_goldfish_session2_tools(runner, app_name, user_id="goldfish_user"):
    """Session 2: ask 'Do I live in water? Check your memory.' via tools."""
    session = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id
    )
    print(f"\n{'=' * 72}")
    print(f"[ENVVAR] SESSION 2  (id={session.id})")
    print(f"{'=' * 72}")

    msg = types.Content(
        role="user",
        parts=[types.Part(text="Do I live in water? Check your memory.")],
    )
    events = []
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=msg,
    ):
        events.append(event)

    response = _extract_final_response(events)
    print(f"[ENVVAR] Session 2 response: {response}")
    return response


def _assert_goldfish_recalled(response: str) -> None:
    response_lower = response.lower()
    assert any(kw in response_lower for kw in _GOLDFISH_KEYWORDS), (
        f"Expected the LLM to recall the goldfish fact. Got: {response}"
    )


def _wait_for_indexing() -> None:
    print(f"\n{'- ' * 36}")
    print(f"[ENVVAR] Waiting {_INDEX_WAIT}s for Goodmem indexing...")
    print(f"{'- ' * 36}")
    time.sleep(_INDEX_WAIT)


# ===================================================================
# Group A: Space Resolution — Happy Paths
# ===================================================================


class TestSpaceResolutionHappy:

    async def test_a2_space_id_only_pre_created(
        self, cleanup_spaces: List[str], gm_client: GoodmemClient
    ) -> None:
        """A2: Pin a pre-existing space by ID. Both sessions use space_id."""
        space_name = _unique_name("envvar_a2")
        space_id = _pre_create_space(gm_client, space_name)
        cleanup_spaces.append(space_id)

        plugin = GoodmemPlugin(
            base_url=_BASE_URL, api_key=_API_KEY,
            space_id=space_id, top_k=5, debug=True,
        )
        agent = LlmAgent(
            model="gemini-2.5-flash", name="envvar_a2_agent",
            instruction="Answer questions based on what you know.",
        )
        app = App(name="envvar_a2_app", root_agent=agent, plugins=[plugin])
        runner = InMemoryRunner(app=app)

        await _run_goldfish_session1_plugin(runner, app.name)
        _wait_for_indexing()
        response2 = await _run_goldfish_session2_plugin(runner, app.name)
        _assert_goldfish_recalled(response2)

    async def test_a3_space_id_and_name_both_match(
        self, cleanup_spaces: List[str], gm_client: GoodmemClient
    ) -> None:
        """A3: Both space_id and space_name set, matching. Consistency check passes."""
        space_name = _unique_name("envvar_a3")
        space_id = _pre_create_space(gm_client, space_name)
        cleanup_spaces.append(space_id)

        plugin = GoodmemPlugin(
            base_url=_BASE_URL, api_key=_API_KEY,
            space_id=space_id, space_name=space_name,
            top_k=5, debug=True,
        )
        agent = LlmAgent(
            model="gemini-2.5-flash", name="envvar_a3_agent",
            instruction="Answer questions based on what you know.",
        )
        app = App(name="envvar_a3_app", root_agent=agent, plugins=[plugin])
        runner = InMemoryRunner(app=app)

        await _run_goldfish_session1_plugin(runner, app.name)
        _wait_for_indexing()
        response2 = await _run_goldfish_session2_plugin(runner, app.name)
        _assert_goldfish_recalled(response2)


# ===================================================================
# Group B: Cross-Session Config Changes
# ===================================================================


class TestCrossSessionConfig:

    async def test_b1_name_then_id(
        self, cleanup_spaces: List[str]
    ) -> None:
        """B1: Session 1 uses space_name, session 2 uses the resulting space_id."""
        space_name = _unique_name("envvar_b1")

        # -- Session 1: plugin with space_name ---------------------------------
        plugin1 = GoodmemPlugin(
            base_url=_BASE_URL, api_key=_API_KEY,
            space_name=space_name, top_k=5, debug=True,
        )
        agent1 = LlmAgent(
            model="gemini-2.5-flash", name="envvar_b1_agent1",
            instruction="Answer questions based on what you know.",
        )
        app1 = App(name="envvar_b1_app1", root_agent=agent1, plugins=[plugin1])
        runner1 = InMemoryRunner(app=app1)

        await _run_goldfish_session1_plugin(runner1, app1.name)
        _wait_for_indexing()

        # Discover the space_id created in session 1
        client = GoodmemClient(_BASE_URL, _API_KEY)
        space_id = _find_space_id(client, space_name)
        client.close()
        assert space_id is not None, f"Space '{space_name}' should have been auto-created"
        cleanup_spaces.append(space_id)

        # -- Session 2: plugin with space_id only ------------------------------
        plugin2 = GoodmemPlugin(
            base_url=_BASE_URL, api_key=_API_KEY,
            space_id=space_id, top_k=5, debug=True,
        )
        agent2 = LlmAgent(
            model="gemini-2.5-flash", name="envvar_b1_agent2",
            instruction="Answer questions based on what you know.",
        )
        app2 = App(name="envvar_b1_app2", root_agent=agent2, plugins=[plugin2])
        runner2 = InMemoryRunner(app=app2)

        response2 = await _run_goldfish_session2_plugin(runner2, app2.name)
        _assert_goldfish_recalled(response2)

    async def test_b2_plugin_writes_tools_read_same_name(
        self, cleanup_spaces: List[str]
    ) -> None:
        """B2: Plugin writes in session 1, tools read in session 2, same space_name."""
        space_name = _unique_name("envvar_b2")

        # -- Session 1: plugin -------------------------------------------------
        plugin = GoodmemPlugin(
            base_url=_BASE_URL, api_key=_API_KEY,
            space_name=space_name, top_k=5, debug=True,
        )
        agent1 = LlmAgent(
            model="gemini-2.5-flash", name="envvar_b2_agent1",
            instruction="Answer questions based on what you know.",
        )
        app1 = App(name="envvar_b2_app1", root_agent=agent1, plugins=[plugin])
        runner1 = InMemoryRunner(app=app1)

        await _run_goldfish_session1_plugin(runner1, app1.name)
        _wait_for_indexing()

        # Register space for cleanup
        client = GoodmemClient(_BASE_URL, _API_KEY)
        space_id = _find_space_id(client, space_name)
        client.close()
        if space_id:
            cleanup_spaces.append(space_id)

        # -- Session 2: tools --------------------------------------------------
        fetch_tool = GoodmemFetchTool(
            base_url=_BASE_URL, api_key=_API_KEY,
            space_name=space_name, top_k=5, debug=True,
        )
        agent2 = LlmAgent(
            model="gemini-2.5-flash", name="envvar_b2_agent2",
            instruction=(
                "You have access to memory tools. "
                "When the user asks a question, ALWAYS use the goodmem_fetch "
                "tool first to check what you know."
            ),
            tools=[fetch_tool],
        )
        runner2 = InMemoryRunner(agent=agent2, app_name="envvar_b2_app2")

        response2 = await _run_goldfish_session2_tools(runner2, "envvar_b2_app2")
        _assert_goldfish_recalled(response2)

    async def test_b3_plugin_writes_tools_read_via_id(
        self, cleanup_spaces: List[str]
    ) -> None:
        """B3: Plugin writes in session 1, tools read in session 2 via space_id."""
        space_name = _unique_name("envvar_b3")

        # -- Session 1: plugin with space_name ---------------------------------
        plugin = GoodmemPlugin(
            base_url=_BASE_URL, api_key=_API_KEY,
            space_name=space_name, top_k=5, debug=True,
        )
        agent1 = LlmAgent(
            model="gemini-2.5-flash", name="envvar_b3_agent1",
            instruction="Answer questions based on what you know.",
        )
        app1 = App(name="envvar_b3_app1", root_agent=agent1, plugins=[plugin])
        runner1 = InMemoryRunner(app=app1)

        await _run_goldfish_session1_plugin(runner1, app1.name)
        _wait_for_indexing()

        # Discover space_id
        client = GoodmemClient(_BASE_URL, _API_KEY)
        space_id = _find_space_id(client, space_name)
        client.close()
        assert space_id is not None, f"Space '{space_name}' should have been auto-created"
        cleanup_spaces.append(space_id)

        # -- Session 2: tools with space_id ------------------------------------
        fetch_tool = GoodmemFetchTool(
            base_url=_BASE_URL, api_key=_API_KEY,
            space_id=space_id, top_k=5, debug=True,
        )
        agent2 = LlmAgent(
            model="gemini-2.5-flash", name="envvar_b3_agent2",
            instruction=(
                "You have access to memory tools. "
                "When the user asks a question, ALWAYS use the goodmem_fetch "
                "tool first to check what you know."
            ),
            tools=[fetch_tool],
        )
        runner2 = InMemoryRunner(agent=agent2, app_name="envvar_b3_app2")

        response2 = await _run_goldfish_session2_tools(runner2, "envvar_b3_app2")
        _assert_goldfish_recalled(response2)


# ===================================================================
# Group C: Space Resolution — Error Paths
# ===================================================================


class TestSpaceResolutionErrors:

    async def test_c1_space_id_nonexistent(self) -> None:
        """C1: space_id that does not exist raises ValueError."""
        bogus_id = "00000000-0000-0000-0000-000000000000"

        plugin = GoodmemPlugin(
            base_url=_BASE_URL, api_key=_API_KEY,
            space_id=bogus_id, top_k=5, debug=True,
        )
        agent = LlmAgent(
            model="gemini-2.5-flash", name="envvar_c1_agent",
            instruction="Answer questions.",
        )
        app = App(name="envvar_c1_app", root_agent=agent, plugins=[plugin])
        runner = InMemoryRunner(app=app)

        session = await runner.session_service.create_session(
            app_name=app.name, user_id="error_user"
        )
        msg = types.Content(
            role="user", parts=[types.Part(text="Hello")]
        )

        # ADK's PluginManager wraps plugin exceptions in RuntimeError
        with pytest.raises(RuntimeError, match="not found"):
            async for _ in runner.run_async(
                user_id="error_user", session_id=session.id, new_message=msg,
            ):
                pass

    async def test_c2_space_id_and_name_mismatch(
        self, cleanup_spaces: List[str], gm_client: GoodmemClient
    ) -> None:
        """C2: space_id exists but space_name doesn't match -> ValueError."""
        real_name = _unique_name("envvar_c2_alpha")
        space_id = _pre_create_space(gm_client, real_name)
        cleanup_spaces.append(space_id)

        plugin = GoodmemPlugin(
            base_url=_BASE_URL, api_key=_API_KEY,
            space_id=space_id, space_name="wrong_name_that_does_not_match",
            top_k=5, debug=True,
        )
        agent = LlmAgent(
            model="gemini-2.5-flash", name="envvar_c2_agent",
            instruction="Answer questions.",
        )
        app = App(name="envvar_c2_app", root_agent=agent, plugins=[plugin])
        runner = InMemoryRunner(app=app)

        session = await runner.session_service.create_session(
            app_name=app.name, user_id="error_user"
        )
        msg = types.Content(
            role="user", parts=[types.Part(text="Hello")]
        )

        # ADK's PluginManager wraps plugin exceptions in RuntimeError
        with pytest.raises(RuntimeError, match="does not match|different spaces"):
            async for _ in runner.run_async(
                user_id="error_user", session_id=session.id, new_message=msg,
            ):
                pass


# ===================================================================
# Group D: Embedder Resolution
# ===================================================================


class TestEmbedderResolution:

    async def test_d1_embedder_id_valid(
        self, cleanup_spaces: List[str], gm_client: GoodmemClient
    ) -> None:
        """D1: Pin a valid embedder_id. Memory works normally."""
        embedder_id = _get_first_embedder_id(gm_client)
        if embedder_id is None:
            # No embedders exist yet — let ensure_embedder create one
            embedder_id = gm_client.ensure_embedder(debug=False)

        space_name = _unique_name("envvar_d1")

        plugin = GoodmemPlugin(
            base_url=_BASE_URL, api_key=_API_KEY,
            embedder_id=embedder_id, space_name=space_name,
            top_k=5, debug=True,
        )
        agent = LlmAgent(
            model="gemini-2.5-flash", name="envvar_d1_agent",
            instruction="Answer questions based on what you know.",
        )
        app = App(name="envvar_d1_app", root_agent=agent, plugins=[plugin])
        runner = InMemoryRunner(app=app)

        await _run_goldfish_session1_plugin(runner, app.name)
        _wait_for_indexing()
        response2 = await _run_goldfish_session2_plugin(runner, app.name)
        _assert_goldfish_recalled(response2)

        # Cleanup
        space_id = _find_space_id(gm_client, space_name)
        if space_id:
            cleanup_spaces.append(space_id)

    async def test_d2_embedder_id_nonexistent(self) -> None:
        """D2: embedder_id that does not exist raises ValueError."""
        bogus_id = "00000000-0000-0000-0000-000000000000"

        plugin = GoodmemPlugin(
            base_url=_BASE_URL, api_key=_API_KEY,
            embedder_id=bogus_id, space_name=_unique_name("envvar_d2"),
            top_k=5, debug=True,
        )
        agent = LlmAgent(
            model="gemini-2.5-flash", name="envvar_d2_agent",
            instruction="Answer questions.",
        )
        app = App(name="envvar_d2_app", root_agent=agent, plugins=[plugin])
        runner = InMemoryRunner(app=app)

        session = await runner.session_service.create_session(
            app_name=app.name, user_id="error_user"
        )
        msg = types.Content(
            role="user", parts=[types.Part(text="Hello")]
        )

        # ADK's PluginManager wraps plugin exceptions in RuntimeError
        with pytest.raises(RuntimeError, match="not found|not valid"):
            async for _ in runner.run_async(
                user_id="error_user", session_id=session.id, new_message=msg,
            ):
                pass


# ===================================================================
# Group E: Env Var Fallback — Smoke Tests
# ===================================================================


class TestEnvVarFallback:

    async def test_e1_space_name_env_var(
        self, cleanup_spaces: List[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E1: GOODMEM_SPACE_NAME env var is picked up when no constructor param."""
        space_name = _unique_name("envvar_e1")
        monkeypatch.setenv("GOODMEM_SPACE_NAME", space_name)

        # Construct plugin with NO space_id or space_name params
        plugin = GoodmemPlugin(
            base_url=_BASE_URL, api_key=_API_KEY,
            top_k=5, debug=True,
        )
        agent = LlmAgent(
            model="gemini-2.5-flash", name="envvar_e1_agent",
            instruction="Answer questions based on what you know.",
        )
        app = App(name="envvar_e1_app", root_agent=agent, plugins=[plugin])
        runner = InMemoryRunner(app=app)

        await _run_goldfish_session1_plugin(runner, app.name)
        _wait_for_indexing()
        response2 = await _run_goldfish_session2_plugin(runner, app.name)
        _assert_goldfish_recalled(response2)

        # Verify the space was created with the env var name
        client = GoodmemClient(_BASE_URL, _API_KEY)
        space_id = _find_space_id(client, space_name)
        client.close()
        assert space_id is not None, (
            f"Expected space '{space_name}' to be created via GOODMEM_SPACE_NAME env var"
        )
        cleanup_spaces.append(space_id)

    async def test_e2_space_id_env_var(
        self, cleanup_spaces: List[str], gm_client: GoodmemClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """E2: GOODMEM_SPACE_ID env var is picked up when no constructor param."""
        space_name = _unique_name("envvar_e2")
        space_id = _pre_create_space(gm_client, space_name)
        cleanup_spaces.append(space_id)

        monkeypatch.setenv("GOODMEM_SPACE_ID", space_id)

        # Construct plugin with NO space_id or space_name params
        plugin = GoodmemPlugin(
            base_url=_BASE_URL, api_key=_API_KEY,
            top_k=5, debug=True,
        )
        agent = LlmAgent(
            model="gemini-2.5-flash", name="envvar_e2_agent",
            instruction="Answer questions based on what you know.",
        )
        app = App(name="envvar_e2_app", root_agent=agent, plugins=[plugin])
        runner = InMemoryRunner(app=app)

        await _run_goldfish_session1_plugin(runner, app.name)
        _wait_for_indexing()
        response2 = await _run_goldfish_session2_plugin(runner, app.name)
        _assert_goldfish_recalled(response2)
