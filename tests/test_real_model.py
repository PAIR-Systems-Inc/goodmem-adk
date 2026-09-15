"""Real provider decisions, actual ADK runners, and live GoodMem retrieval."""

import asyncio
import base64
import os

import pytest
from fpdf import FPDF
from goodmem import AsyncGoodmem
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner

from goodmem_adk import GoodmemFetchTool, GoodmemPlugin, GoodmemSaveTool
from tests.support import all_text, append_evidence, responses, turn, unique

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.model]


def model():
    identifier = os.environ.get("GOODMEM_TEST_MODEL")
    if not identifier:
        pytest.skip("GOODMEM_TEST_MODEL must select an authenticated real model")
    if "/" in identifier:
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(model=identifier, temperature=0, max_tokens=512)
    return identifier


def final_text(events):
    return " ".join(
        part.text
        for event in events
        if event.is_final_response() and event.content
        for part in (event.content.parts or [])
        if part.text
    )


@pytest.mark.parametrize("path", ["plugin", "tools"])
async def test_provider_recalls_exact_random_fact_in_new_session(live, path):
    plugin = GoodmemPlugin(**live.config) if path == "plugin" else None
    tool_list = (
        [GoodmemSaveTool(**live.config), GoodmemFetchTool(**live.config)] if path == "tools" else []
    )
    agent = LlmAgent(
        name="memory_audit_agent",
        model=model(),
        tools=tool_list,
        instruction=(
            "Help the user remember their personal project facts. "
            "When memory tools are available, use goodmem_save when asked to remember "
            "a fact. When asked about a saved fact, call goodmem_fetch BEFORE answering, "
            "even when the current conversation has no history. Do not say you lack "
            "access when these tools are available. Otherwise use the supplied "
            "memory context. Report errors honestly. Never invent a missing identifier. "
            "Keep replies brief and reproduce identifiers exactly."
        ),
    )
    app = App(name=unique("realapp"), root_agent=agent, plugins=[plugin] if plugin else [])
    runner = InMemoryRunner(app=app)
    user, fact = unique("realuser"), unique("orchid")
    evidence = {"model": os.environ["GOODMEM_TEST_MODEL"], "path": path, "fact": fact}
    try:
        first, events = await turn(
            runner, user, f"Please remember this exact fact: my collection label is {fact}."
        )
        evidence["save_response"] = final_text(events)
        evidence["save_tools"] = responses(events, "goodmem_save")
        if path == "tools":
            assert evidence["save_tools"] and evidence["save_tools"][0]["success"], evidence
        await live.wait_for_writes()
        second, events = await turn(
            runner,
            user,
            "What is my collection label? Consult memory and give the exact identifier.",
        )
        assert first != second
        evidence["recall_response"] = final_text(events)
        evidence["fetch_tools"] = responses(events, "goodmem_fetch")
        if path == "tools":
            assert evidence["fetch_tools"] and fact in all_text(evidence["fetch_tools"]), evidence
        assert fact in evidence["recall_response"], evidence
        _, events = await turn(
            runner,
            unique("unrelateduser"),
            "What is my collection label? Consult memory and give the exact identifier.",
        )
        evidence["other_user_response"] = final_text(events)
        assert fact not in evidence["other_user_response"], evidence
    finally:
        await runner.close()
        target = os.environ.get("GOODMEM_TEST_MODEL_EVIDENCE")
        if target:
            await asyncio.to_thread(append_evidence, target, evidence)


@pytest.mark.parametrize("path", ["plugin", "tools"])
async def test_provider_answers_from_pdf_extracted_by_goodmem(live, path):
    """The provider receives retrieved text; GoodMem receives the binary PDF."""
    sid, fact = live.new_space(), unique("dock")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, f"Receiving instructions. The assigned loading dock identifier is {fact}.")
    async with AsyncGoodmem(base_url=live.base_url, api_key=live.api_key) as client:
        await client.memories.create(
            space_id=sid,
            original_content_b64=base64.b64encode(bytes(pdf.output())).decode("ascii"),
            content_type="application/pdf",
        )
    await live.wait_for_writes()
    plugin = GoodmemPlugin(**live.config, space_id=sid) if path == "plugin" else None
    agent = LlmAgent(
        name="pdf_audit_agent",
        model=model(),
        tools=[GoodmemFetchTool(**live.config, space_id=sid)] if path == "tools" else [],
        instruction="Answer from memory. Use goodmem_fetch if available. Copy identifiers exactly; never guess.",
    )
    runner = InMemoryRunner(
        app=App(name=unique("pdfapp"), root_agent=agent, plugins=[plugin] if plugin else [])
    )
    evidence = {
        "model": os.environ["GOODMEM_TEST_MODEL"],
        "path": path,
        "case": "PDF extracted by GoodMem",
        "fact": fact,
    }
    try:
        _, events = await turn(
            runner,
            unique("pdfuser"),
            "What is the assigned loading dock identifier? Look in memory.",
        )
        evidence["response"] = final_text(events)
        evidence["fetch_tools"] = responses(events, "goodmem_fetch")
        assert fact in evidence["response"], evidence
        if path == "tools":
            assert evidence["fetch_tools"] and fact in all_text(evidence["fetch_tools"]), evidence
    finally:
        await runner.close()
        target = os.environ.get("GOODMEM_TEST_MODEL_EVIDENCE")
        if target:
            await asyncio.to_thread(append_evidence, target, evidence)
