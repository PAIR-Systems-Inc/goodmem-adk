"""User-visible ADK behavior through the actual SDK over a fake HTTP transport."""

import asyncio
import json
import uuid

import httpx
import pytest
import pytest_asyncio
from goodmem import AsyncGoodmem, Goodmem
from google.adk.events import Event
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import Session
from google.genai import types

from goodmem_adk import GoodmemFetchTool, GoodmemPlugin, GoodmemSaveError, GoodmemSaveTool
from goodmem_adk.memory import GoodmemMemoryService, GoodmemMemoryServiceConfig
from tests.support import (
    RecordingModel,
    all_text,
    responses,
    runner_for,
    text_content,
    turn,
)
from tests.wire import CHUNK_ID, EMBEDDER_ID, MEMORY_ID, SPACE_ID, chunk, memory, space

pytestmark = pytest.mark.asyncio


class WireServer:
    def __init__(self):
        self.spaces = {SPACE_ID: space()}
        self.events = []
        self.requests = []
        self.writes = []
        self.fail_write = None
        self.conflict_once = False
        self.content_error = None
        self.bad_stream = False
        self.empty_embedders = False

    async def handle(self, request):
        data = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, request.url.path, data, dict(request.url.params)))
        path = request.url.path
        if path == "/v1/spaces" and request.method == "GET":
            name = request.url.params.get("name_filter")
            return httpx.Response(
                200,
                json={
                    "spaces": [
                        value for value in self.spaces.values() if not name or name in value["name"]
                    ]
                },
            )
        if path == "/v1/spaces":
            value = space(spaceId=str(uuid.uuid4()), name=data["name"])
            self.spaces[value["spaceId"]] = value
            if self.conflict_once:
                self.conflict_once = False
                return httpx.Response(409, json={"message": "Already exists"})
            return httpx.Response(200, json=value)
        if path.startswith("/v1/spaces/"):
            value = self.spaces.get(path.rsplit("/", 1)[-1])
            return (
                httpx.Response(200, json=value)
                if value
                else httpx.Response(404, json={"message": "No space"})
            )
        if path == "/v1/embedders":
            return httpx.Response(
                200,
                json={"embedders": [] if self.empty_embedders else [{"embedderId": EMBEDDER_ID}]},
            )
        if path == "/v1/memories:retrieve":
            if self.content_error:
                raise self.content_error
            body = "\n".join(json.dumps(event) for event in self.events) + "\n"
            if self.bad_stream:
                body += '{"retrievedItem":INVALID}\n'
            return httpx.Response(200, text=body, headers={"content-type": "application/x-ndjson"})
        if path == "/v1/memories":
            if self.fail_write and self.fail_write(data):
                return httpx.Response(503, json={"message": "Upload failed"})
            value = memory(
                memoryId=str(uuid.uuid4()),
                spaceId=data["spaceId"],
                metadata=data.get("metadata", {}),
                contentType=data["contentType"],
            )
            self.writes.append((data, value))
            return httpx.Response(200, json=value)
        raise AssertionError(f"Unexpected SDK request: {request.method} {path}")


@pytest_asyncio.fixture
async def wire():
    server = WireServer()
    async with (
        httpx.AsyncClient(
            base_url="http://server-a.test",
            headers={"x-api-key": "credential-a"},
            transport=httpx.MockTransport(server.handle),
        ) as transport,
        AsyncGoodmem(http_client=transport) as client,
    ):
        server.client, server.transport = client, transport
        yield server


async def test_two_components_in_same_session_honor_different_scopes(wire):
    plugin = GoodmemPlugin(client=wire.client, space_name="conversation", embedder_id=EMBEDDER_ID)
    tool = GoodmemSaveTool(client=wire.client, space_name="notes", embedder_id=EMBEDDER_ID)
    runner, _ = runner_for(plugin=plugin, tools=[tool])
    try:
        _, events = await turn(runner, "same-user", "SAVE: unique note")
        result = responses(events, "goodmem_save")[0]
        assert result["success"]
        destinations = {value["spaceId"]: value["name"] for value in wire.spaces.values()}
        assert [
            (destinations[request["spaceId"]], request["metadata"]["source"])
            for request, _ in wire.writes
        ] == [
            ("conversation", "adk_plugin"),
            ("notes", "adk_tool"),
            ("conversation", "adk_plugin"),
        ]
    finally:
        await runner.close()


async def test_injected_sdk_keeps_server_credentials_and_ownership(wire, monkeypatch):
    monkeypatch.setenv("GOODMEM_BASE_URL", "http://server-b.invalid")
    monkeypatch.setenv("GOODMEM_API_KEY", "credential-b")
    wire.events = [chunk()]
    runner, _ = runner_for(tools=[GoodmemFetchTool(client=wire.client, space_id=SPACE_ID)])
    try:
        _, events = await turn(runner, "user", "FETCH: memory")
        assert responses(events, "goodmem_fetch")[0]["success"]
    finally:
        await runner.close()
    assert not wire.transport.is_closed
    assert wire.transport.headers["x-api-key"] == "credential-a"
    assert (await wire.client.spaces.get(id=SPACE_ID)).space_id == SPACE_ID


async def test_sync_sdk_and_ambiguous_connection_configuration_are_rejected(wire):
    with Goodmem(base_url="http://unused.test", api_key="test") as client:
        with pytest.raises(TypeError, match="AsyncGoodmem"):
            GoodmemFetchTool(client=client)
    with pytest.raises(ValueError, match="either client"):
        GoodmemPlugin(client=wire.client, base_url="http://wrong.test")


@pytest.mark.parametrize("field,value", [("space_id", SPACE_ID), ("space_name", "chosen")])
async def test_explicit_scope_overrides_environment_scope(wire, monkeypatch, field, value):
    monkeypatch.setenv("GOODMEM_SPACE_ID", "bad-env-id")
    monkeypatch.setenv("GOODMEM_SPACE_NAME", "bad-env-name")
    runner, _ = runner_for(
        tools=[
            GoodmemSaveTool(
                client=wire.client,
                embedder_id=EMBEDDER_ID,
                **{field: value},
            )
        ]
    )
    try:
        _, events = await turn(runner, "user", "SAVE: a note")
        assert responses(events, "goodmem_save")[0]["success"]
        target = wire.spaces[wire.writes[-1][0]["spaceId"]]
        assert target["spaceId"] == value if field == "space_id" else target["name"] == value
    finally:
        await runner.close()


@pytest.mark.parametrize("field,value", [("space_id", SPACE_ID), ("space_name", "env-name")])
async def test_environment_scope_used_when_no_explicit_scope(wire, monkeypatch, field, value):
    monkeypatch.setenv("GOODMEM_" + field.upper(), value)
    runner, _ = runner_for(tools=[GoodmemSaveTool(client=wire.client, embedder_id=EMBEDDER_ID)])
    try:
        _, events = await turn(runner, "user", "SAVE: note")
        assert responses(events, "goodmem_save")[0]["success"]
        target = wire.spaces[wire.writes[-1][0]["spaceId"]]
        assert target["spaceId"] == value if field == "space_id" else target["name"] == value
    finally:
        await runner.close()


@pytest.mark.parametrize(
    "options",
    [
        {"space_id": SPACE_ID, "space_name": "wrong-name"},
        {"space_id": "missing-space"},
        {"space_id": SPACE_ID, "embedder_id": "wrong-embedder"},
    ],
)
async def test_invalid_scope_fails_without_a_write(wire, options):
    runner, _ = runner_for(tools=[GoodmemSaveTool(client=wire.client, **options)])
    try:
        _, events = await turn(runner, "user", "SAVE: must not write")
        assert not responses(events, "goodmem_save")[0]["success"]
        assert not wire.writes
    finally:
        await runner.close()


async def test_concurrent_space_creation_conflict_reuses_the_exact_name(wire):
    wire.conflict_once = True
    runner, _ = runner_for(
        tools=[
            GoodmemSaveTool(
                client=wire.client,
                space_name="raced-name",
                embedder_id=EMBEDDER_ID,
            )
        ]
    )
    try:
        _, events = await turn(runner, "user", "SAVE: note")
        assert responses(events, "goodmem_save")[0]["success"]
        lookups = [item for item in wire.requests if item[:2] == ("GET", "/v1/spaces")]
        assert len(lookups) == 2
        assert all(item[3]["name_filter"] == "raced-name" for item in lookups)
    finally:
        await runner.close()


async def test_absent_embedders_does_not_create_google_resources(wire, monkeypatch):
    wire.empty_embedders = True
    monkeypatch.setenv("GOOGLE_API_KEY", "unrelated-google-key")
    runner, _ = runner_for(tools=[GoodmemSaveTool(client=wire.client)])
    try:
        _, events = await turn(runner, "user", "SAVE: note")
        result = responses(events, "goodmem_save")[0]
        assert not result["success"] and "No embedder" in all_text(result["errors"])
        assert not any(request[:2] == ("POST", "/v1/embedders") for request in wire.requests)
    finally:
        await runner.close()


async def test_definitions_after_chunks_preserve_metadata_and_all_unique_chunks(wire):
    wire.events = [
        chunk("Alpha fact", metadata={"filename": "chunk-label"}),
        chunk("Beta fact", chunkId="second-chunk"),
        chunk("Alpha fact"),
        {"memoryDefinition": memory(metadata={"filename": "source.pdf", "role": "model"})},
    ]
    runner, _ = runner_for(tools=[GoodmemFetchTool(client=wire.client, space_id=SPACE_ID)])
    try:
        _, events = await turn(runner, "user", "FETCH: both facts")
        result = responses(events, "goodmem_fetch")[0]
        assert result["success"] and result["count"] == 2
        assert [item["content"] for item in result["memories"]] == ["Alpha fact", "Beta fact"]
        first = result["memories"][0]
        assert first["metadata"] == {"filename": "source.pdf", "role": "model"}
        assert first["chunk_metadata"] == {"filename": "chunk-label"}
        assert first["chunk_id"] == CHUNK_ID and first["memory_id"] == MEMORY_ID
        assert not any("batchGet" in request[1] for request in wire.requests)
        retrieve = next(request[2] for request in wire.requests if request[1].endswith(":retrieve"))
        assert retrieve["fetchMemory"] is True and retrieve["fetchMemoryContent"] is False
    finally:
        await runner.close()


@pytest.mark.parametrize(
    "code,partial",
    [
        ("FEATURE_DISABLED", False),
        ("LLM_CAPABILITY_INFERRED", False),
        ("EMBEDDER_FAILED", True),
        ("MEMORY_CONTENT_UNAVAILABLE", True),
        ("SERVER_FUTURE_STATUS_2099", True),
    ],
)
@pytest.mark.parametrize("has_chunks", [False, True])
async def test_statuses_are_visible_without_throwing_away_usable_chunks(
    wire, code, partial, has_chunks
):
    wire.events = ([chunk()] if has_chunks else []) + [
        {"status": {"code": code, "message": "server notice"}}
    ]
    runner, _ = runner_for(tools=[GoodmemFetchTool(client=wire.client, space_id=SPACE_ID)])
    try:
        _, events = await turn(runner, "user", "FETCH: fact")
        result = responses(events, "goodmem_fetch")[0]
        assert result["partial"] == partial and result["success"] == (has_chunks or not partial)
        assert result["statuses"][0]["code"] == (
            "UNKNOWN" if code.startswith("SERVER_FUTURE") else code
        )
        assert result["count"] == int(has_chunks)
    finally:
        await runner.close()


@pytest.mark.parametrize("failure", ["corrupt-tail", "transport"])
async def test_stream_and_transport_failures_are_reported(wire, failure):
    if failure == "corrupt-tail":
        wire.events, wire.bad_stream = [chunk()], True
    else:
        wire.content_error = httpx.ReadTimeout("read timed out")
    runner, _ = runner_for(tools=[GoodmemFetchTool(client=wire.client, space_id=SPACE_ID)])
    try:
        _, events = await turn(runner, "user", "FETCH: fact")
        result = responses(events, "goodmem_fetch")[0]
        assert result["partial"] and result["statuses"][0]["code"] == "RETRIEVAL_ERROR"
        assert result["count"] == int(failure == "corrupt-tail")
    finally:
        await runner.close()


async def test_tool_schema_exposes_only_the_model_arguments(wire):
    for tool, expected in [
        (GoodmemSaveTool(client=wire.client), {"content"}),
        (GoodmemFetchTool(client=wire.client), {"query", "top_k"}),
    ]:
        declaration = tool._get_declaration()
        assert set(declaration.parameters_json_schema["properties"]) == expected
    assert not wire.requests


async def test_prompt_injection_context_is_not_written_back_to_session_or_goodmem(wire):
    wire.events = [
        chunk("old saved reference"),
        {"memoryDefinition": memory(metadata={"filename": "source.txt", "role": "model"})},
    ]
    runner, model = runner_for(plugin=GoodmemPlugin(client=wire.client, space_id=SPACE_ID))
    try:
        sid, _ = await turn(runner, "user", "First question")
        await turn(runner, "user", "Second question", session_id=sid)
        request = all_text(model.requests[-1].contents)
        assert "source.txt" in request and "old saved reference" in request and "model" in request
        session = await runner.session_service.get_session(
            app_name=runner.app_name, user_id="user", session_id=sid
        )
        assert "BEGIN MEMORY" not in all_text(session.events)
        assert all(
            "BEGIN MEMORY" not in request.get("originalContent", "") for request, _ in wire.writes
        )
        queries = [
            request[2]["message"] for request in wire.requests if request[1].endswith(":retrieve")
        ]
        assert queries == ["First question", "Second question"]
    finally:
        await runner.close()


class ThinkingModel(RecordingModel):
    async def generate_content_async(self, llm_request, stream=False):
        yield LlmResponse(
            partial=True, content=types.Content(role="model", parts=[types.Part(text="delta")])
        )
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="private thought", thought=True),
                    types.Part(text="final answer"),
                ],
            )
        )


async def test_plugin_does_not_persist_thoughts_or_streaming_deltas(wire):
    runner, _ = runner_for(
        plugin=GoodmemPlugin(client=wire.client, space_id=SPACE_ID), model=ThinkingModel()
    )
    try:
        await turn(runner, "user", "question")
        texts = [request["originalContent"] for request, _ in wire.writes]
        assert texts == ["question", "final answer"]
    finally:
        await runner.close()


async def test_plugin_attachment_failure_surfaces_accepted_ids(wire):
    wire.fail_write = lambda data: data["contentType"] == "application/pdf"
    runner, _ = runner_for(plugin=GoodmemPlugin(client=wire.client, space_id=SPACE_ID))
    message = types.Content(
        role="user",
        parts=[
            types.Part(text="Store this PDF"),
            types.Part(
                inline_data=types.Blob(
                    data=b"PDF", mime_type="application/pdf", display_name="receipt.pdf"
                ),
            ),
        ],
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            await turn(runner, "user", message)
        assert isinstance(caught.value.__cause__, GoodmemSaveError)
        result = caught.value.__cause__.response
        assert result.partial and result.memory_id == wire.writes[0][1]["memoryId"]
        assert result.accepted[0].processing_status == "PENDING"
        assert result.errors[0].filename == "receipt.pdf"
    finally:
        await runner.close()


@pytest.mark.parametrize("split", [False, True])
async def test_internal_service_retries_failed_writes_without_repeating_accepted_ones(wire, split):
    service = GoodmemMemoryService(
        client=wire.client, space_id=SPACE_ID, config=GoodmemMemoryServiceConfig(split_turn=split)
    )
    session = Session(
        app_name="app",
        user_id="user",
        id="session",
        events=[
            Event(author="user", content=text_content("first question")),
            Event(
                author="agent",
                content=types.Content(role="model", parts=[types.Part(text="first answer")]),
            ),
            Event(author="user", content=text_content("second question")),
            Event(
                author="agent",
                content=types.Content(role="model", parts=[types.Part(text="second answer")]),
            ),
        ],
    )
    wire.fail_write = lambda data: "second answer" in data.get("originalContent", "")
    from goodmem.errors import APIError

    with pytest.raises(APIError):
        await service.add_session_to_memory(session)
    accepted = [value["memoryId"] for _, value in wire.writes]
    wire.fail_write = None
    await asyncio.gather(
        service.add_session_to_memory(session), service.add_session_to_memory(session)
    )
    assert [value["memoryId"] for _, value in wire.writes][: len(accepted)] == accepted
    assert len(wire.writes) == (4 if split else 2)


async def test_matching_space_can_be_on_a_later_sdk_page(wire):
    original_handle = wire.handle

    async def paged(request):
        if request.method == "GET" and request.url.path == "/v1/spaces":
            wire.requests.append((request.method, request.url.path, {}, dict(request.url.params)))
            if not request.url.params.get("next_token"):
                return httpx.Response(200, json={"spaces": [], "nextToken": "page-two"})
            return httpx.Response(200, json={"spaces": [space(name="wanted")]})
        return await original_handle(request)

    async with (
        httpx.AsyncClient(
            base_url="http://paged.test", transport=httpx.MockTransport(paged)
        ) as http,
        AsyncGoodmem(http_client=http) as client,
    ):
        runner, _ = runner_for(tools=[GoodmemSaveTool(client=client, space_name="wanted")])
        try:
            _, events = await turn(runner, "user", "SAVE: note")
            assert responses(events, "goodmem_save")[0]["success"]
            assert wire.writes[0][0]["spaceId"] == SPACE_ID
            assert not any(request[:2] == ("POST", "/v1/spaces") for request in wire.requests)
        finally:
            await runner.close()


async def test_ambiguous_accessible_space_names_require_an_id(wire):
    wire.spaces["another-id"] = space(spaceId="another-id")
    runner, _ = runner_for(tools=[GoodmemSaveTool(client=wire.client, space_name="test-space")])
    try:
        _, events = await turn(runner, "user", "SAVE: note")
        result = responses(events, "goodmem_save")[0]
        assert not result["success"] and "Multiple accessible spaces" in all_text(result["errors"])
        assert not wire.writes
    finally:
        await runner.close()


async def test_one_failed_attachment_does_not_hide_other_accepted_attachments(wire):
    wire.fail_write = lambda data: data["metadata"].get("filename") == "broken.pdf"
    message = types.Content(
        role="user",
        parts=[
            types.Part(text="SAVE: store both"),
            *[
                types.Part(
                    inline_data=types.Blob(
                        data=b"pdf", mime_type="application/pdf", display_name=name
                    )
                )
                for name in ("broken.pdf", "accepted.pdf")
            ],
        ],
    )
    runner, _ = runner_for(tools=[GoodmemSaveTool(client=wire.client, space_id=SPACE_ID)])
    try:
        _, events = await turn(runner, "user", message)
        result = responses(events, "goodmem_save")[0]
        assert not result["success"] and result["partial"] and result["attachments_saved"] == 1
        assert {entry["memory_id"] for entry in result["accepted"]} == {
            response["memoryId"] for _, response in wire.writes
        }
        assert result["accepted"][1]["filename"] == "accepted.pdf"
        assert result["errors"][0]["filename"] == "broken.pdf"
    finally:
        await runner.close()


@pytest.mark.parametrize("top_k", [0, 101, -1])
async def test_invalid_retrieval_size_does_not_send_a_search(wire, top_k):
    with pytest.raises(ValueError, match="top_k"):
        GoodmemFetchTool(client=wire.client, top_k=top_k)
    assert not wire.requests


async def test_cancelled_save_propagates_cancellation(wire):
    async def cancelled(request):
        raise asyncio.CancelledError()

    async with (
        httpx.AsyncClient(
            base_url="http://cancelled.test", transport=httpx.MockTransport(cancelled)
        ) as http,
        AsyncGoodmem(http_client=http) as client,
    ):
        from goodmem_adk._backend import Backend

        with pytest.raises(asyncio.CancelledError):
            await Backend(client=client, space_id=SPACE_ID).save(
                default_name="test",
                text="note",
                content=None,
                metadata={},
            )


async def test_uri_attachment_is_reported_without_downloading_it(wire):
    runner, _ = runner_for(tools=[GoodmemSaveTool(client=wire.client, space_id=SPACE_ID)])
    try:
        _, events = await turn(
            runner,
            "user",
            types.Content(
                role="user",
                parts=[
                    types.Part(text="SAVE: a note"),
                    types.Part(
                        file_data=types.FileData(
                            file_uri="file:///private/example", mime_type="text/plain"
                        )
                    ),
                ],
            ),
        )
        result = responses(events, "goodmem_save")[0]
        assert result["memory_id"] and result["partial"] and not result["success"]
        assert "URI attachments" in result["errors"][0]["message"]
    finally:
        await runner.close()
