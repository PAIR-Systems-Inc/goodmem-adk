"""ADK recovery and concurrency journeys with real SDK requests and server responses."""

import asyncio
import json

import httpx
import pytest
from fpdf import FPDF
from goodmem import AsyncGoodmem
from google.adk.tools.load_memory_tool import load_memory_tool
from google.genai import types

from goodmem_adk import GoodmemFetchTool, GoodmemSaveTool
from goodmem_adk.memory import GoodmemMemoryService
from tests.support import all_text, responses, runner_for, turn, unique

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_live_concurrent_ingestion_preserves_isolation_and_does_not_duplicate(live):
    """Delay one outgoing write; all writes, indexing and recall use the live server."""
    runner, _ = runner_for(tools=[load_memory_tool])
    users = [unique("alice"), unique("bob")]
    facts = [unique("launch"), unique("launch")]
    sessions = []
    paused, release = asyncio.Event(), asyncio.Event()

    async def hold_alice_write(request):
        if request.method == "POST" and request.url.path == "/v1/memories":
            metadata = json.loads(request.content).get("metadata", {})
            if metadata.get("user_id") == users[0]:
                paused.set()
                await release.wait()

    try:
        for user, fact in zip(users, facts, strict=True):
            session_id, _ = await turn(runner, user, f"My launch access code is {fact}.")
            session = await runner.session_service.get_session(
                app_name=runner.app_name, user_id=user, session_id=session_id
            )
            assert session is not None
            sessions.append(session)
        async with (
            httpx.AsyncClient(
                base_url=live.base_url,
                headers={"x-api-key": live.api_key},
                timeout=30,
                event_hooks={"request": [hold_alice_write]},
            ) as http,
            AsyncGoodmem(http_client=http) as client,
        ):
            service = GoodmemMemoryService(client=client, embedder_id=live.embedder_id)
            runner.memory_service = service
            tasks = [asyncio.create_task(service.add_session_to_memory(sessions[0]))]
            try:
                await asyncio.wait_for(paused.wait(), 15)
                tasks.append(asyncio.create_task(service.add_session_to_memory(sessions[0])))
                await asyncio.wait_for(service.add_session_to_memory(sessions[1]), 15)
                assert not tasks[0].done(), "Alice must still be waiting while Bob completes"
                assert [item["metadata"]["user_id"] for item in live.writes] == [users[1]]
                release.set()
                await asyncio.wait_for(asyncio.gather(*tasks), 30)
            finally:
                release.set()
                await asyncio.gather(*tasks, return_exceptions=True)

            await live.wait_for_writes()
            assert len(live.spaces) == 2
            assert len(live.writes) == 2
            for index, (user, session) in enumerate(zip(users, sessions, strict=True)):
                name = f"adk_memory_{runner.app_name}_{user}"
                sid = next(sid for sid, actual in live.spaces.items() if actual == name)
                stored = [item async for item in await client.memories.list(space_id=sid)]
                assert len(stored) == 1, "The overlapping retry created a duplicate memory"
                assert stored[0].metadata["user_id"] == user
                assert stored[0].metadata["session_id"] == session.id
                assert stored[0].processing_status == "COMPLETED"
                fresh_session, events = await turn(runner, user, "LOAD: my launch access code")
                assert fresh_session != session.id
                result = responses(events, "load_memory")
                assert result and facts[index] in all_text(result)
                assert facts[1 - index] not in all_text(result), "Another user's fact leaked"
    finally:
        await runner.close()


async def test_live_partial_attachment_save_can_recover_without_reuploading_accepted_files(live):
    """Reject one upload with a real server 401, then retry only that PDF in a new turn."""
    sid, user = live.new_space(), unique("receipt_user")
    facts = {"accepted.pdf": unique("north"), "retry.pdf": unique("south")}
    files = {}
    for filename, fact in facts.items():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, f"Shipping receipt. The shipping access code is {fact}.")
        files[filename] = bytes(pdf.output())

    reject_upload = True

    async def reject_second_upload(request):
        if reject_upload and request.method == "POST" and request.url.path == "/v1/memories":
            if json.loads(request.content).get("metadata", {}).get("filename") == "retry.pdf":
                # Only the credential on this request changes. The real server
                # generates the error; no SDK or HTTP response is mocked.
                request.headers["x-api-key"] = "invalid-adk-test-key"

    def attachment(filename):
        return types.Part(
            inline_data=types.Blob(
                data=files[filename], mime_type="application/pdf", display_name=filename
            )
        )

    async with (
        httpx.AsyncClient(
            base_url=live.base_url,
            headers={"x-api-key": live.api_key},
            timeout=30,
            event_hooks={"request": [reject_second_upload]},
        ) as http,
        AsyncGoodmem(http_client=http) as client,
    ):
        writer, model = runner_for(tools=[GoodmemSaveTool(client=client, space_id=sid)])
        reader, _ = runner_for(tools=[GoodmemFetchTool(**live.config, space_id=sid, top_k=10)])
        try:
            first_session, events = await turn(
                writer,
                user,
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text="SAVE: Keep both shipping receipts."),
                        attachment("accepted.pdf"),
                        attachment("retry.pdf"),
                    ],
                ),
            )
            saved = responses(events, "goodmem_save")[0]
            assert not saved["success"] and saved["partial"]
            assert saved["attachments_saved"] == 1 and len(saved["accepted"]) == 2
            assert len(saved["errors"]) == 1
            assert saved["errors"][0]["filename"] == "retry.pdf"
            assert "401" in saved["errors"][0]["message"]
            assert ("POST", "/v1/memories", 401) in live.requests
            accepted_ids = {item["memory_id"] for item in saved["accepted"]}
            assert accepted_ids == {item["memoryId"] for item in live.writes}
            # ADK must deliver the partial result and accepted IDs to the model.
            model_input = all_text(model.requests[-1].contents)
            assert all(memory_id in model_input for memory_id in accepted_ids)
            assert "retry.pdf" in model_input and "401" in model_input

            reject_upload = False
            second_session, events = await turn(
                writer,
                user,
                types.Content(
                    role="user",
                    parts=[types.Part(text="SAVE:"), attachment("retry.pdf")],
                ),
            )
            assert second_session != first_session
            retried = responses(events, "goodmem_save")[0]
            assert retried["success"] and not retried["partial"] and not retried["errors"]
            assert retried["attachments_saved"] == 1 and len(retried["accepted"]) == 1
            retry_id = retried["accepted"][0]["memory_id"]
            assert retry_id not in accepted_ids
            expected_ids = accepted_ids | {retry_id}

            await live.wait_for_writes()
            stored = {
                item.memory_id: item
                async for item in await client.memories.list(space_id=sid, include_content=True)
            }
            assert set(stored) == expected_ids, "Recovery duplicated an earlier accepted write"
            assert all(item.processing_status == "COMPLETED" for item in stored.values())
            uploaded = {
                item.metadata["filename"]: item
                for item in stored.values()
                if item.metadata.get("filename")
            }
            assert set(uploaded) == set(files)
            for filename, data in files.items():
                assert uploaded[filename].original_content == data
                assert uploaded[filename].content_type == "application/pdf"

            _, events = await turn(reader, user, "FETCH: shipping access codes from both receipts")
            result = responses(events, "goodmem_fetch")[0]
            assert result["success"] and not result["partial"]
            for filename, fact in facts.items():
                matching = [
                    item
                    for item in result["memories"]
                    if item["metadata"].get("filename") == filename
                ]
                assert matching and fact in all_text(matching)
                assert {item["memory_id"] for item in matching} == {uploaded[filename].memory_id}
        finally:
            await writer.close()
            await reader.close()
