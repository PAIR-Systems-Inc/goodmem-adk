"""Real ADK + published plugin + live GoodMem; model boundary is deterministic."""

import pytest
from fpdf import FPDF
from goodmem import AsyncGoodmem
from google.adk.events import Event
from google.adk.sessions import Session
from google.adk.tools.load_memory_tool import load_memory_tool
from google.genai import types

from goodmem_adk import GoodmemFetchTool, GoodmemPlugin, GoodmemSaveTool
from goodmem_adk.memory import GoodmemMemoryService
from tests.support import all_text, responses, runner_for, text_content, turn, unique

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_plugin_text_recall_in_fresh_session_and_other_user_isolation(live):
    plugin = GoodmemPlugin(**live.config)
    runner, model = runner_for(plugin=plugin)
    user = unique("user")
    token = unique("orchid")
    try:
        await turn(runner, user, f"My project recovery phrase is {token}.")
        await live.wait_for_writes()
        await turn(runner, user, "What is my project recovery phrase?")
        assert token in all_text(model.requests[-1].contents)
        await turn(runner, unique("otheruser"), "What is my project recovery phrase?")
        assert token not in all_text(model.requests[-1].contents)
    finally:
        await runner.close()


async def test_tools_save_fetch_in_fresh_session_and_other_user_isolation(live):
    runner, _ = runner_for(tools=[GoodmemSaveTool(**live.config), GoodmemFetchTool(**live.config)])
    user = unique("user")
    token = unique("willow")
    try:
        _, saved_events = await turn(runner, user, f"SAVE: My parcel password is {token}.")
        saved = responses(saved_events, "goodmem_save")
        assert saved and saved[0]["success"] and saved[0]["memory_id"]
        await live.wait_for_writes()
        _, fetched_events = await turn(runner, user, "FETCH: What is my parcel password?")
        fetched = responses(fetched_events, "goodmem_fetch")
        assert fetched and fetched[0]["success"] and token in all_text(fetched)
        _, other_events = await turn(
            runner, unique("otheruser"), "FETCH: What is my parcel password?"
        )
        assert token not in all_text(responses(other_events, "goodmem_fetch"))
    finally:
        await runner.close()


@pytest.mark.parametrize("path", ["plugin", "tools"])
async def test_pdf_fact_reaches_fresh_session_without_original_attachment(live, path):
    token = unique("warehouse")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, f"Parcel receipt. Warehouse access code: {token}. Total: 712.34 dollars.")
    plugin = GoodmemPlugin(**live.config) if path == "plugin" else None
    tool_list = (
        [GoodmemSaveTool(**live.config), GoodmemFetchTool(**live.config)] if path == "tools" else []
    )
    runner, model = runner_for(plugin=plugin, tools=tool_list)
    user = unique("pdfuser")
    try:
        message = types.Content(
            role="user",
            parts=[
                types.Part(
                    text="SAVE: Store this receipt." if path == "tools" else "Store this receipt."
                ),
                types.Part(
                    inline_data=types.Blob(mime_type="application/pdf", data=bytes(pdf.output()))
                ),
            ],
        )
        _, events = await turn(runner, user, message)
        if path == "tools":
            assert responses(events, "goodmem_save")[0]["attachments_saved"] == 1
        assert any(w["contentType"] == "application/pdf" for w in live.writes)
        await live.wait_for_writes()
        _, events = await turn(
            runner,
            user,
            ("FETCH: " if path == "tools" else "") + "What is the warehouse access code?",
        )
        observed = (
            responses(events, "goodmem_fetch") if path == "tools" else model.requests[-1].contents
        )
        assert token in all_text(observed)
    finally:
        await runner.close()


async def test_plugin_and_tools_respect_separate_configured_space_names(live):
    plugin_name = unique("pluginspace")
    tool_name = unique("toolspace")
    plugin = GoodmemPlugin(**live.config, space_name=plugin_name)
    runner, _ = runner_for(
        plugin=plugin, tools=[GoodmemSaveTool(**live.config, space_name=tool_name)]
    )
    try:
        _, events = await turn(
            runner, unique("user"), "SAVE: Put this note in the configured tools space."
        )
        assert responses(events, "goodmem_save")[0]["success"]
        assert tool_name in live.spaces.values(), (
            "Tool reused the plugin's cached space instead of its configured space_name",
            live.spaces,
        )
    finally:
        await runner.close()


async def test_plugin_preserves_attachment_filename_in_model_context(live):
    sid = live.new_space()
    filename = unique("audit_source") + ".txt"
    response = live.http.post(
        "/v1/memories",
        json={
            "spaceId": sid,
            "originalContent": "The audit source describes orchid delivery.",
            "contentType": "text/plain",
            "metadata": {"filename": filename, "role": "user"},
        },
    )
    response.raise_for_status()
    await live.wait_for_writes()
    plugin = GoodmemPlugin(**live.config, space_id=sid)
    runner, model = runner_for(plugin=plugin)
    try:
        await turn(runner, unique("user"), "What does the audit source describe?")
        assert "orchid delivery" in all_text(model.requests[-1].contents)
        assert filename in all_text(model.requests[-1].contents)
    finally:
        await runner.close()


async def test_tools_keep_multiple_relevant_chunks_from_one_document(live):
    sid = live.new_space(chunk_size=100)
    token_a, token_b = unique("alpha"), unique("beta")
    content = (
        f"The first required delivery password is {token_a}.\n\n"
        + "Background filler. " * 14
        + f"\n\nThe second required delivery password is {token_b}."
    )
    response = live.http.post(
        "/v1/memories",
        json={
            "spaceId": sid,
            "originalContent": content,
            "contentType": "text/plain",
        },
    )
    response.raise_for_status()
    await live.wait_for_writes()
    async with AsyncGoodmem(base_url=live.base_url, api_key=live.api_key) as client:
        chunks = await client.memories.retrieve(
            message="required delivery passwords", space_ids=[sid], requested_size=20, stream=False
        )
        assert token_a in all_text(chunks) and token_b in all_text(chunks)
    runner, _ = runner_for(tools=[GoodmemFetchTool(**live.config, space_id=sid, top_k=20)])
    try:
        _, events = await turn(runner, unique("user"), "FETCH: required delivery passwords")
        fetched = responses(events, "goodmem_fetch")
        assert token_a in all_text(fetched) and token_b in all_text(fetched), fetched
    finally:
        await runner.close()


async def test_internal_memory_service_with_native_load_memory_tool(live):
    service = GoodmemMemoryService(**live.config)
    runner, _ = runner_for(tools=[load_memory_tool])
    runner.memory_service = service
    user, token = unique("user"), unique("cedar")
    session = Session(
        app_name=runner.app_name,
        user_id=user,
        id=unique("session"),
        events=[
            Event(author="user", content=text_content(f"My reservation code is {token}.")),
            Event(
                author="audit_agent",
                content=types.Content(role="model", parts=[types.Part(text="Acknowledged.")]),
            ),
        ],
    )
    try:
        await service.add_session_to_memory(session)
        count = len(live.writes)
        await service.add_session_to_memory(session)
        assert len(live.writes) == count, "Repeated session ingestion duplicated existing events"
        await live.wait_for_writes()
        _, events = await turn(runner, user, "LOAD: What is my reservation code?")
        assert token in all_text(responses(events, "load_memory"))
        other = await service.search_memory(
            app_name=runner.app_name, user_id=unique("other"), query="reservation code"
        )
        assert token not in all_text(other)
        other_app = await service.search_memory(
            app_name=unique("otherapp"), user_id=user, query="reservation code"
        )
        assert token not in all_text(other_app)
    finally:
        await runner.close()
        await service.close()


@pytest.mark.parametrize("scope", ["id", "name", "matching", "env-id", "env-name"])
async def test_plugin_write_and_tool_read_share_only_the_configured_scope(live, monkeypatch, scope):
    name = unique("shared")
    options = {}
    if scope in {"id", "matching", "env-id"}:
        sid = live.new_space(name=name)
        if scope == "env-id":
            monkeypatch.setenv("GOODMEM_SPACE_ID", sid)
        else:
            options["space_id"] = sid
    if scope in {"name", "matching"}:
        options["space_name"] = name
    if scope == "env-name":
        monkeypatch.setenv("GOODMEM_SPACE_NAME", name)
    # Omit embedder_id to exercise selection of an existing server embedder.
    config = {"base_url": live.base_url, "api_key": live.api_key, **options}
    fact = unique("shared_fact")
    plugin_runner, _ = runner_for(plugin=GoodmemPlugin(**config))
    tool_runner, _ = runner_for(tools=[GoodmemFetchTool(**config)])
    try:
        await turn(plugin_runner, "writer", f"The project launch code is {fact}.")
        await live.wait_for_writes()
        _, events = await turn(tool_runner, "reader", "FETCH: project launch code")
        result = responses(events, "goodmem_fetch")[0]
        assert result["success"] and fact in all_text(result)
        assert list(live.spaces.values()).count(name) == 1
    finally:
        await plugin_runner.close()
        await tool_runner.close()


async def test_invalid_explicit_space_returns_an_error_without_creating_another(live):
    runner, _ = runner_for(
        tools=[
            GoodmemFetchTool(
                **live.config,
                space_id="00000000-0000-0000-0000-000000000000",
            )
        ]
    )
    try:
        _, events = await turn(runner, unique("user"), "FETCH: missing space")
        result = responses(events, "goodmem_fetch")[0]
        assert not result["success"] and "404" in all_text(result["statuses"])
        assert not live.spaces and not live.writes
    finally:
        await runner.close()
