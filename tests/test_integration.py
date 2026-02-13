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

"""Integration tests for GoodmemPlugin and GoodmemSaveTool/GoodmemFetchTool.

These tests require a live Goodmem server and valid API keys.  They are
skipped automatically when the required environment variables are not set.

Run with::

    GOODMEM_BASE_URL=http://localhost:8080 \
    GOODMEM_API_KEY=<key> \
    GOOGLE_API_KEY=<key> \
    pytest tests/test_integration.py -v

Either GOOGLE_API_KEY or GEMINI_API_KEY can be used for Gemini authentication.
"""

from __future__ import annotations

import asyncio
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

# Time to wait (seconds) for Goodmem to index the embedding after insert.
_INDEX_WAIT = 5
# PDF text-extraction + embedding takes longer than plain text.
_INDEX_WAIT_PDF = 8


def _unique_name(prefix: str) -> str:
    """Return a collision-free space name for this test run."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _extract_final_response(events: list) -> str:
    """Walk runner events and return the last model text response."""
    text_parts: List[str] = []
    for event in events:
        content = getattr(event, "content", None)
        if content is None:
            continue
        author = getattr(event, "author", None)
        # Only look at model / agent responses, skip user echoes
        if author == "user":
            continue
        parts = getattr(content, "parts", None)
        if not parts:
            continue
        for part in parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
    return " ".join(text_parts)


def _find_space_id(client: GoodmemClient, space_name: str) -> str | None:
    """Look up a space by name and return its ID."""
    spaces = client.list_spaces(name=space_name)
    for space in spaces:
        if space.get("name") == space_name:
            return space.get("spaceId")
    return None


# ---------------------------------------------------------------------------
# Shared cleanup fixture
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


# ---------------------------------------------------------------------------
# Plugin integration test
# ---------------------------------------------------------------------------


class TestPluginIntegration:
    """End-to-end test: GoodmemPlugin stores and recalls a fact across
    two separate ADK sessions using a real Goodmem backend and Gemini LLM.
    """

    async def test_goldfish_memory_across_sessions(
        self, cleanup_spaces: List[str]
    ) -> None:
        space_name = _unique_name("integ_plugin")

        # -- build the agent + plugin + runner ---------------------------------
        agent = LlmAgent(
            model="gemini-2.5-flash",
            name="integ_plugin_agent",
            description="A helpful assistant.",
            instruction=(
                "You are a helpful assistant. Answer questions about the user "
                "based on what you know."
            ),
        )

        plugin = GoodmemPlugin(
            base_url=_BASE_URL,
            api_key=_API_KEY,
            space_name=space_name,
            top_k=5,
            debug=True,
        )

        app = App(
            name="integ_plugin_app",
            root_agent=agent,
            plugins=[plugin],
        )

        runner = InMemoryRunner(app=app)

        # -- Session 1: tell the agent a fact ----------------------------------
        session1 = await runner.session_service.create_session(
            app_name=app.name, user_id="goldfish_user"
        )

        print(f"\n{'=' * 72}")
        print(f"[INTEG] SESSION 1  (id={session1.id})")
        print(f"{'=' * 72}")

        msg1 = types.Content(
            role="user",
            parts=[types.Part(text="I am a goldfish")],
        )
        events1 = []
        async for event in runner.run_async(
            user_id="goldfish_user",
            session_id=session1.id,
            new_message=msg1,
        ):
            events1.append(event)

        response1 = _extract_final_response(events1)
        print(f"[INTEG] Session 1 response: {response1}")
        assert response1, "Session 1 should produce a model response"

        # -- Wait for Goodmem indexing -----------------------------------------
        print(f"\n{'- ' * 36}")
        print(f"[INTEG] Waiting {_INDEX_WAIT}s for Goodmem indexing...")
        print(f"{'- ' * 36}")
        time.sleep(_INDEX_WAIT)

        # -- Session 2: ask a question that requires the goldfish memory -------
        session2 = await runner.session_service.create_session(
            app_name=app.name, user_id="goldfish_user"
        )

        print(f"\n{'=' * 72}")
        print(f"[INTEG] SESSION 2  (id={session2.id})")
        print(f"{'=' * 72}")

        msg2 = types.Content(
            role="user",
            parts=[types.Part(text="Do I live in water?")],
        )
        events2 = []
        async for event in runner.run_async(
            user_id="goldfish_user",
            session_id=session2.id,
            new_message=msg2,
        ):
            events2.append(event)

        response2 = _extract_final_response(events2)
        print(f"[INTEG] Session 2 response: {response2}")

        # -- Assertions --------------------------------------------------------
        response_lower = response2.lower()
        recall_keywords = ["water", "goldfish", "fish", "aquatic", "aquarium"]
        assert any(kw in response_lower for kw in recall_keywords), (
            f"Expected the LLM to recall the goldfish fact. "
            f"Got: {response2}"
        )

        # -- Also verify retrieval directly ------------------------------------
        client = GoodmemClient(_BASE_URL, _API_KEY)
        space_id = _find_space_id(client, space_name)
        assert space_id is not None, (
            f"Space '{space_name}' should have been auto-created"
        )
        cleanup_spaces.append(space_id)

        chunks = client.retrieve_memories(
            "Do I live in water?", [space_id], request_size=5
        )
        chunk_texts = []
        for item in chunks:
            try:
                text = (
                    item["retrievedItem"]["chunk"]["chunk"]["chunkText"]
                )
                chunk_texts.append(text)
            except (KeyError, TypeError):
                pass

        print(f"[INTEG] Retrieved chunks: {chunk_texts}")
        assert any("goldfish" in t.lower() for t in chunk_texts), (
            f"Expected to retrieve a chunk mentioning 'goldfish'. "
            f"Got: {chunk_texts}"
        )
        client.close()

    async def test_pdf_receipt_memory_across_sessions(
        self, cleanup_spaces: List[str], mock_receipt_pdf: bytes
    ) -> None:
        """Upload a PDF receipt in session 1, then verify that session 2 can
        recall details (Acme's address) that were *only* in the PDF and never
        mentioned in the conversation text.
        """
        space_name = _unique_name("integ_plugin_pdf")

        # -- build the agent + plugin + runner ---------------------------------
        agent = LlmAgent(
            model="gemini-2.5-flash",
            name="integ_plugin_pdf_agent",
            description="A helpful assistant.",
            instruction=(
                "You are a helpful assistant. Answer questions based on what "
                "you know, including any documents or memories you have access to."
            ),
        )

        plugin = GoodmemPlugin(
            base_url=_BASE_URL,
            api_key=_API_KEY,
            space_name=space_name,
            top_k=5,
            debug=True,
        )

        app = App(
            name="integ_plugin_pdf_app",
            root_agent=agent,
            plugins=[plugin],
        )

        runner = InMemoryRunner(app=app)

        # -- Session 1: send PDF + ask about total -----------------------------
        session1 = await runner.session_service.create_session(
            app_name=app.name, user_id="receipt_user"
        )

        print(f"\n{'=' * 72}")
        print(f"[INTEG-PDF] SESSION 1  (id={session1.id})")
        print(f"{'=' * 72}")

        msg1 = types.Content(
            role="user",
            parts=[
                types.Part(text=(
                    "In the attached receipt, how much did GoodMind pay Acme?"
                )),
                types.Part(
                    inline_data=types.Blob(
                        data=mock_receipt_pdf,
                        mime_type="application/pdf",
                    )
                ),
            ],
        )
        events1 = []
        async for event in runner.run_async(
            user_id="receipt_user",
            session_id=session1.id,
            new_message=msg1,
        ):
            events1.append(event)

        response1 = _extract_final_response(events1)
        print(f"[INTEG-PDF] Session 1 response: {response1}")
        assert response1, "Session 1 should produce a model response"

        # Verify the LLM answered with the correct total
        assert "4,225.50" in response1 or "4225.50" in response1 or "4225" in response1, (
            f"Expected session 1 response to mention the receipt total. "
            f"Got: {response1}"
        )

        # -- Wait for Goodmem PDF indexing -------------------------------------
        print(f"\n{'- ' * 36}")
        print(f"[INTEG-PDF] Waiting {_INDEX_WAIT_PDF}s for Goodmem PDF indexing...")
        print(f"{'- ' * 36}")
        time.sleep(_INDEX_WAIT_PDF)

        # -- Session 2: ask about Acme's address (never mentioned in session 1)
        session2 = await runner.session_service.create_session(
            app_name=app.name, user_id="receipt_user"
        )

        print(f"\n{'=' * 72}")
        print(f"[INTEG-PDF] SESSION 2  (id={session2.id})")
        print(f"{'=' * 72}")

        msg2 = types.Content(
            role="user",
            parts=[types.Part(text="What's the address of Acme?")],
        )
        events2 = []
        async for event in runner.run_async(
            user_id="receipt_user",
            session_id=session2.id,
            new_message=msg2,
        ):
            events2.append(event)

        response2 = _extract_final_response(events2)
        print(f"[INTEG-PDF] Session 2 response: {response2}")

        # -- Assertions: session 2 should recall Acme's address from PDF -------
        response_lower = response2.lower()
        address_keywords = [
            "innovation drive", "san francisco", "94105", "123",
        ]
        assert any(kw in response_lower for kw in address_keywords), (
            f"Expected the LLM to recall Acme's address from the PDF receipt. "
            f"Got: {response2}"
        )

        # -- Direct retrieval check --------------------------------------------
        client = GoodmemClient(_BASE_URL, _API_KEY)
        space_id = _find_space_id(client, space_name)
        assert space_id is not None, (
            f"Space '{space_name}' should have been auto-created"
        )
        cleanup_spaces.append(space_id)

        chunks = client.retrieve_memories(
            "Acme address", [space_id], request_size=5
        )
        chunk_texts = []
        for item in chunks:
            try:
                text = (
                    item["retrievedItem"]["chunk"]["chunk"]["chunkText"]
                )
                chunk_texts.append(text)
            except (KeyError, TypeError):
                pass

        print(f"[INTEG-PDF] Retrieved chunks: {chunk_texts}")
        assert any(
            "acme" in t.lower() or "innovation" in t.lower()
            for t in chunk_texts
        ), (
            f"Expected to retrieve a chunk mentioning 'Acme' or 'Innovation'. "
            f"Got: {chunk_texts}"
        )
        client.close()


# ---------------------------------------------------------------------------
# Tools integration test
# ---------------------------------------------------------------------------


class TestToolsIntegration:
    """End-to-end test: GoodmemSaveTool/GoodmemFetchTool store and recall
    a fact across two separate ADK sessions.
    """

    async def test_goldfish_memory_via_tools(
        self, cleanup_spaces: List[str]
    ) -> None:
        space_name = _unique_name("integ_tools")

        # -- build the agent + tools + runner ----------------------------------
        save_tool = GoodmemSaveTool(
            base_url=_BASE_URL,
            api_key=_API_KEY,
            space_name=space_name,
            debug=True,
        )
        fetch_tool = GoodmemFetchTool(
            base_url=_BASE_URL,
            api_key=_API_KEY,
            space_name=space_name,
            top_k=5,
            debug=True,
        )

        agent = LlmAgent(
            model="gemini-2.5-flash",
            name="integ_tools_agent",
            description="A helpful assistant with memory tools.",
            instruction=(
                "You have access to memory tools. "
                "When the user tells you something about themselves, ALWAYS "
                "save it using the goodmem_save tool. "
                "When the user asks a question about themselves, ALWAYS use "
                "the goodmem_fetch tool first to check what you know."
            ),
            tools=[save_tool, fetch_tool],
        )

        runner = InMemoryRunner(
            agent=agent,
            app_name="integ_tools_app",
        )

        # -- Session 1: tell the agent a fact ----------------------------------
        session1 = await runner.session_service.create_session(
            app_name="integ_tools_app", user_id="goldfish_user"
        )

        print(f"\n{'=' * 72}")
        print(f"[INTEG] SESSION 1  (id={session1.id})")
        print(f"{'=' * 72}")

        msg1 = types.Content(
            role="user",
            parts=[types.Part(text="Remember this: I am a goldfish")],
        )
        events1 = []
        async for event in runner.run_async(
            user_id="goldfish_user",
            session_id=session1.id,
            new_message=msg1,
        ):
            events1.append(event)

        response1 = _extract_final_response(events1)
        print(f"[INTEG] Session 1 response: {response1}")

        # Verify the save tool was invoked
        save_called = False
        for event in events1:
            content = getattr(event, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None) == "goodmem_save":
                    save_called = True
                    break
        assert save_called, (
            "Expected the LLM to call goodmem_save in session 1"
        )

        # -- Wait for Goodmem indexing -----------------------------------------
        print(f"\n{'- ' * 36}")
        print(f"[INTEG] Waiting {_INDEX_WAIT}s for Goodmem indexing...")
        print(f"{'- ' * 36}")
        time.sleep(_INDEX_WAIT)

        # -- Session 2: ask a question that requires the goldfish memory -------
        session2 = await runner.session_service.create_session(
            app_name="integ_tools_app", user_id="goldfish_user"
        )

        print(f"\n{'=' * 72}")
        print(f"[INTEG] SESSION 2  (id={session2.id})")
        print(f"{'=' * 72}")

        msg2 = types.Content(
            role="user",
            parts=[types.Part(text="Do I live in water?")],
        )
        events2 = []
        async for event in runner.run_async(
            user_id="goldfish_user",
            session_id=session2.id,
            new_message=msg2,
        ):
            events2.append(event)

        response2 = _extract_final_response(events2)
        print(f"[INTEG] Session 2 response: {response2}")

        # Verify the fetch tool was invoked
        fetch_called = False
        for event in events2:
            content = getattr(event, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None) == "goodmem_fetch":
                    fetch_called = True
                    break
        assert fetch_called, (
            "Expected the LLM to call goodmem_fetch in session 2"
        )

        # -- Assertions --------------------------------------------------------
        response_lower = response2.lower()
        recall_keywords = ["water", "goldfish", "fish", "aquatic", "aquarium"]
        assert any(kw in response_lower for kw in recall_keywords), (
            f"Expected the LLM to recall the goldfish fact. "
            f"Got: {response2}"
        )

        # -- Cleanup: find and register the space for deletion -----------------
        client = GoodmemClient(_BASE_URL, _API_KEY)
        space_id = _find_space_id(client, space_name)
        if space_id:
            cleanup_spaces.append(space_id)
        client.close()

    async def test_pdf_receipt_memory_via_tools(
        self, cleanup_spaces: List[str], mock_receipt_pdf: bytes
    ) -> None:
        """Save a PDF receipt via the goodmem_save tool in session 1, then
        verify that session 2 can fetch Acme's address (only present in the
        PDF, never mentioned in conversation) via goodmem_fetch.
        """
        space_name = _unique_name("integ_tools_pdf")

        # -- build the agent + tools + runner ----------------------------------
        save_tool = GoodmemSaveTool(
            base_url=_BASE_URL,
            api_key=_API_KEY,
            space_name=space_name,
            debug=True,
        )
        fetch_tool = GoodmemFetchTool(
            base_url=_BASE_URL,
            api_key=_API_KEY,
            space_name=space_name,
            top_k=5,
            debug=True,
        )

        agent = LlmAgent(
            model="gemini-2.5-flash",
            name="integ_tools_pdf_agent",
            description="A helpful assistant with memory tools.",
            instruction=(
                "You have access to memory tools. "
                "When the user asks you to save something, ALWAYS use the "
                "goodmem_save tool. "
                "When the user asks a question and says to check memory, "
                "ALWAYS use the goodmem_fetch tool first."
            ),
            tools=[save_tool, fetch_tool],
        )

        runner = InMemoryRunner(
            agent=agent,
            app_name="integ_tools_pdf_app",
        )

        # -- Session 1: send PDF + ask to save it -----------------------------
        session1 = await runner.session_service.create_session(
            app_name="integ_tools_pdf_app", user_id="receipt_user"
        )

        print(f"\n{'=' * 72}")
        print(f"[INTEG-PDF] SESSION 1  (id={session1.id})")
        print(f"{'=' * 72}")

        msg1 = types.Content(
            role="user",
            parts=[
                types.Part(text=(
                    "Save this receipt to memory and tell me the total amount."
                )),
                types.Part(
                    inline_data=types.Blob(
                        data=mock_receipt_pdf,
                        mime_type="application/pdf",
                    )
                ),
            ],
        )
        events1 = []
        async for event in runner.run_async(
            user_id="receipt_user",
            session_id=session1.id,
            new_message=msg1,
        ):
            events1.append(event)

        response1 = _extract_final_response(events1)
        print(f"[INTEG-PDF] Session 1 response: {response1}")

        # Verify the save tool was invoked
        save_called = False
        for event in events1:
            content = getattr(event, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None) == "goodmem_save":
                    save_called = True
                    break
        assert save_called, (
            "Expected the LLM to call goodmem_save in session 1"
        )

        # -- Wait for Goodmem PDF indexing -------------------------------------
        print(f"\n{'- ' * 36}")
        print(f"[INTEG-PDF] Waiting {_INDEX_WAIT_PDF}s for Goodmem PDF indexing...")
        print(f"{'- ' * 36}")
        time.sleep(_INDEX_WAIT_PDF)

        # -- Session 2: ask about Acme's address (never mentioned in session 1)
        session2 = await runner.session_service.create_session(
            app_name="integ_tools_pdf_app", user_id="receipt_user"
        )

        print(f"\n{'=' * 72}")
        print(f"[INTEG-PDF] SESSION 2  (id={session2.id})")
        print(f"{'=' * 72}")

        msg2 = types.Content(
            role="user",
            parts=[types.Part(text=(
                "What's the address of Acme? Check your memory."
            ))],
        )
        events2 = []
        async for event in runner.run_async(
            user_id="receipt_user",
            session_id=session2.id,
            new_message=msg2,
        ):
            events2.append(event)

        response2 = _extract_final_response(events2)
        print(f"[INTEG-PDF] Session 2 response: {response2}")

        # Verify the fetch tool was invoked
        fetch_called = False
        for event in events2:
            content = getattr(event, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None) == "goodmem_fetch":
                    fetch_called = True
                    break
        assert fetch_called, (
            "Expected the LLM to call goodmem_fetch in session 2"
        )

        # -- Assertions: session 2 should recall Acme's address from PDF -------
        response_lower = response2.lower()
        address_keywords = [
            "innovation drive", "san francisco", "94105", "123",
        ]
        assert any(kw in response_lower for kw in address_keywords), (
            f"Expected the LLM to recall Acme's address from the PDF receipt. "
            f"Got: {response2}"
        )

        # -- Cleanup: find and register the space for deletion -----------------
        client = GoodmemClient(_BASE_URL, _API_KEY)
        space_id = _find_space_id(client, space_name)
        if space_id:
            cleanup_spaces.append(space_id)
        client.close()
