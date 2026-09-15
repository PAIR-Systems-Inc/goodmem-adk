"""Internal-service concurrency and attachment journeys through ADK and the real SDK."""

import asyncio
import base64
import json

import httpx
import pytest
from goodmem import AsyncGoodmem
from goodmem.errors import APIError
from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import types

from goodmem_adk import GoodmemSaveTool
from goodmem_adk.memory import GoodmemMemoryService
from tests.support import responses, runner_for, text_content, turn
from tests.test_sdk_adk import wire as wire  # noqa: F401
from tests.wire import SPACE_ID

pytestmark = pytest.mark.asyncio


def session_for(user, content=None):
    return Session(
        app_name="app",
        user_id=user,
        id="session",
        events=[
            Event(author="user", content=content or text_content("question")),
            Event(author="agent", content=text_content("answer")),
        ],
    )


@pytest.mark.parametrize("cache_size", [1, 1024])
async def test_stalled_session_does_not_block_another_or_duplicate_its_own_writes(
    wire, monkeypatch, cache_size
):
    monkeypatch.setattr("goodmem_adk.memory._SESSION_CACHE_SIZE", cache_size)
    started, release, retry_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    attempts = []

    async def stalled(request):
        if request.method == "POST" and request.url.path == "/v1/memories":
            user = json.loads(request.content)["metadata"]["user_id"]
            attempts.append(user)
            if user == "alice":
                started.set()
                await release.wait()
        return await wire.handle(request)

    async with (
        httpx.AsyncClient(
            base_url="http://stalled.test", transport=httpx.MockTransport(stalled)
        ) as http,
        AsyncGoodmem(http_client=http) as client,
    ):
        service = GoodmemMemoryService(client=client, space_id=SPACE_ID)
        alice = session_for("alice")
        tasks = [asyncio.create_task(service.add_session_to_memory(alice))]

        async def retry_alice():
            retry_entered.set()
            await service.add_session_to_memory(alice)

        try:
            await asyncio.wait_for(started.wait(), 2)
            # Bob completes while Alice's HTTP request is still waiting. With a
            # one-session cache, this also pressures eviction of Alice's state.
            await asyncio.wait_for(service.add_session_to_memory(session_for("bob")), 2)
            assert not tasks[0].done()
            tasks.append(asyncio.create_task(retry_alice()))
            await asyncio.wait_for(retry_entered.wait(), 2)
            assert attempts == ["alice", "bob"]
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 2)
            assert sorted(data["metadata"]["user_id"] for data, _ in wire.writes) == [
                "alice",
                "bob",
            ]
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize(
    "part, message",
    [
        (
            types.Part(file_data=types.FileData(file_uri="file:///private/secret")),
            "URI attachments are not uploaded",
        ),
        (
            types.Part(inline_data=types.Blob(data=b"", display_name="empty.txt")),
            "Empty attachment",
        ),
    ],
)
async def test_unsaved_attachments_are_visible_through_both_adapters(wire, part, message):
    content = types.Content(role="user", parts=[types.Part(text="SAVE: a note"), part])
    runner, _ = runner_for(tools=[GoodmemSaveTool(client=wire.client, space_id=SPACE_ID)])
    try:
        _, events = await turn(runner, "user", content)
        result = responses(events, "goodmem_save")[0]
        assert not result["success"] and result["partial"]
        assert message in result["errors"][0]["message"]
        service = GoodmemMemoryService(client=wire.client, space_id=SPACE_ID)
        with pytest.raises(ValueError, match=message):
            await service.add_session_to_memory(session_for("user", content))
        # Only the tool's accepted text was written. No URI is fetched, and the
        # service does not silently continue past the unsaved attachment.
        assert len(wire.writes) == 1
    finally:
        await runner.close()


async def test_tools_and_service_upload_the_same_bytes_names_and_content_types(wire):
    data = b"\x00\xff\x80binary"
    content = types.Content(
        role="user",
        parts=[
            types.Part(text="SAVE: attachments"),
            types.Part(inline_data=types.Blob(data=data)),
            types.Part(
                inline_data=types.Blob(
                    data=data, display_name="report.pdf", mime_type="application/pdf"
                )
            ),
        ],
    )
    runner, _ = runner_for(tools=[GoodmemSaveTool(client=wire.client, space_id=SPACE_ID)])
    try:
        _, events = await turn(runner, "user", content)
        assert responses(events, "goodmem_save")[0]["attachments_saved"] == 2
        service = GoodmemMemoryService(client=wire.client, space_id=SPACE_ID)
        await service.add_session_to_memory(session_for("user", content))
        uploads = [request for request, _ in wire.writes if "originalContentB64" in request]
        assert len(uploads) == 4
        for first, second in zip(uploads[:2], uploads[2:], strict=True):
            assert base64.b64decode(first["originalContentB64"]) == data
            for key in ("originalContentB64", "spaceId", "contentType"):
                assert first[key] == second[key]
            assert first["metadata"]["filename"] == second["metadata"]["filename"]
        assert [item["metadata"]["filename"] for item in uploads[:2]] == [
            "attachment_1",
            "report.pdf",
        ]
        assert uploads[0]["contentType"] == "application/octet-stream"
    finally:
        await runner.close()


@pytest.mark.parametrize("failure", ["http", "uri", "empty"])
async def test_attachment_retry_keeps_previously_accepted_writes(wire, failure):
    parts = [
        types.Part(inline_data=types.Blob(data=b"first", display_name="first.txt")),
        types.Part(inline_data=types.Blob(data=b"second", display_name="second.txt")),
    ]
    if failure == "http":
        wire.fail_write = lambda data: data["metadata"].get("filename") == "second.txt"
    elif failure == "uri":
        parts[1] = types.Part(file_data=types.FileData(file_uri="https://example.test/file"))
    else:
        parts[1].inline_data.data = b""
    session = session_for("user", types.Content(role="user", parts=parts))
    service = GoodmemMemoryService(client=wire.client, space_id=SPACE_ID)
    with pytest.raises(APIError if failure == "http" else ValueError):
        await service.add_session_to_memory(session)
    assert [data["metadata"]["filename"] for data, _ in wire.writes] == ["first.txt"]
    wire.fail_write = None
    session.events[0].content.parts[1] = types.Part(
        inline_data=types.Blob(data=b"second", display_name="second.txt")
    )
    await service.add_session_to_memory(session)
    await service.add_session_to_memory(session)
    assert [data["metadata"].get("filename") for data, _ in wire.writes] == [
        "first.txt",
        "second.txt",
        None,  # The completed conversation is saved once too.
    ]
