"""Fault injection over a real HTTP socket; GoodMem/ADK code is unmodified."""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest
from google.genai import types

from goodmem_adk import GoodmemFetchTool, GoodmemPlugin, GoodmemSaveTool
from tests.support import all_text, responses, runner_for, turn, unique
from tests.wire import chunk, memory, space

pytestmark = pytest.mark.enable_socket

SPACE_ID = "00000000-0000-4000-8000-000000000001"
MEMORY_ID = "00000000-0000-4000-8000-000000000002"


class MockServer:
    def __init__(self):
        self.events = []
        self.raw = None
        self.delay = 0
        self.upload_status = 200
        self.auth_status = 200
        self.requests = []

    @property
    def config(self):
        return {"base_url": self.url, "api_key": "audit-fake-key", "space_id": SPACE_ID}


@pytest.fixture
def server():
    state = MockServer()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            path = urlparse(self.path).path
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            state.requests.append((self.command, path, body))
            status, data, content_type = 200, {}, "application/json"
            if state.auth_status != 200:
                status, data = state.auth_status, {"message": "Invalid API key"}
            elif path == "/v1/memories:retrieve":
                time.sleep(state.delay)
                data = (
                    state.raw
                    if state.raw is not None
                    else "\n".join(json.dumps(e) for e in state.events)
                )
                content_type = "application/x-ndjson"
            elif (
                path == "/v1/memories" and json.loads(body).get("contentType") == "application/pdf"
            ):
                status = state.upload_status
                data = {"message": "Upload failed"} if status != 200 else memory()
            elif path == "/v1/memories":
                data = memory()
            elif path == "/v1/memories:batchGet":
                data = {"results": [{"memory": {"memoryId": MEMORY_ID, "metadata": {}}}]}
            elif path.startswith("/v1/memories/"):
                data = {"memoryId": MEMORY_ID, "metadata": {}}
            elif path.startswith("/v1/spaces/"):
                data = space()
            else:
                status, data = 404, {"message": "Unexpected route"}
            encoded = data.encode() if isinstance(data, str) else json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{httpd.server_port}"
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2)


@pytest.mark.asyncio
async def test_unknown_status_does_not_abort_real_adk_tool_call(server):
    server.events = [
        {"status": {"code": "FUTURE_NOTICE_2099", "message": "An unfamiliar notice"}},
        chunk(),
    ]
    runner, _ = runner_for(tools=[GoodmemFetchTool(**server.config)])
    try:
        _, events = await turn(runner, unique("user"), "FETCH: test memory")
        result = responses(events, "goodmem_fetch")[0]
        assert result["success"] and "A useful test memory" in all_text(result)
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_empty_search_is_one_request_and_returns_empty(server):
    runner, _ = runner_for(tools=[GoodmemFetchTool(**server.config)])
    try:
        _, events = await turn(runner, unique("user"), "FETCH: absent fact")
        result = responses(events, "goodmem_fetch")[0]
        assert result["success"] and result["count"] == 0
        assert len([r for r in server.requests if r[1] == "/v1/memories:retrieve"]) == 1
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_server_retrieval_failure_is_visible_to_the_agent(server):
    server.events = [{"status": {"code": "EMBEDDER_FAILED", "message": "Query embedding failed"}}]
    runner, _ = runner_for(tools=[GoodmemFetchTool(**server.config)])
    try:
        _, events = await turn(runner, unique("user"), "FETCH: a stored fact")
        result = responses(events, "goodmem_fetch")[0]
        assert (
            not result["success"] or result.get("partial") or "EMBEDDER_FAILED" in all_text(result)
        ), result
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_partial_results_retain_failure_diagnostic(server):
    server.events = [
        chunk(),
        {"status": {"code": "MEMORY_CONTENT_UNAVAILABLE", "message": "One memory unavailable"}},
    ]
    runner, _ = runner_for(tools=[GoodmemFetchTool(**server.config)])
    try:
        _, events = await turn(runner, unique("user"), "FETCH: stored facts")
        result = responses(events, "goodmem_fetch")[0]
        assert "A useful test memory" in all_text(result)
        assert result.get("partial") or "MEMORY_CONTENT_UNAVAILABLE" in all_text(result), result
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_malformed_stream_is_not_reported_as_clean_empty_result(server):
    server.raw = '{"retrievedItem": INVALID}\n'
    runner, _ = runner_for(tools=[GoodmemFetchTool(**server.config)])
    try:
        _, events = await turn(runner, unique("user"), "FETCH: saved note")
        result = responses(events, "goodmem_fetch")[0]
        assert not result["success"] and result["partial"]
        assert result["statuses"][0]["code"] == "RETRIEVAL_ERROR"
    finally:
        await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["plugin", "tools"])
async def test_async_retrieval_does_not_block_other_sessions(server, path):
    server.delay = 0.25
    plugin = GoodmemPlugin(**server.config) if path == "plugin" else None
    runner, _ = runner_for(
        plugin=plugin, tools=[GoodmemFetchTool(**server.config)] if path == "tools" else []
    )
    gaps = []
    running = True

    async def heartbeat():
        previous = time.monotonic()
        while running:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - previous)
            previous = now

    pulse = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.02)
    try:
        await turn(
            runner, unique("user"), "FETCH: saved note" if path == "tools" else "Recall saved note"
        )
        await asyncio.sleep(0.02)
        assert max(gaps) < 0.15, f"Event loop was blocked for {max(gaps):.3f}s"
    finally:
        running = False
        await pulse
        await runner.close()


@pytest.mark.asyncio
async def test_failed_attachment_save_is_visible_to_agent(server):
    server.upload_status = 503
    runner, _ = runner_for(tools=[GoodmemSaveTool(**server.config)])
    message = types.Content(
        role="user",
        parts=[
            types.Part(text="SAVE: Save this PDF receipt."),
            types.Part(
                inline_data=types.Blob(mime_type="application/pdf", data=b"%PDF-1.4 audit fixture")
            ),
        ],
    )
    try:
        _, events = await turn(runner, unique("user"), message)
        result = responses(events, "goodmem_save")[0]
        assert result.get("memory_id"), "Accepted text write ID must remain visible"
        assert (
            not result["success"] or result.get("partial") or "failed" in result["message"].lower()
        ), result
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_invalid_credentials_tool_returns_visible_error(server):
    server.auth_status = 401
    runner, _ = runner_for(tools=[GoodmemFetchTool(**server.config)])
    try:
        _, events = await turn(runner, unique("user"), "FETCH: saved note")
        result = responses(events, "goodmem_fetch")[0]
        assert not result["success"] and "401" in all_text(result["statuses"])
    finally:
        await runner.close()
