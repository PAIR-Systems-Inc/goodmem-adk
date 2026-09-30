"""Configured GoodMem IDs must be UUIDs before they can reach a request URL.

The GoodMem SDK interpolates path IDs as sent (``/v1/spaces/{id}``) and httpx resolves
dot segments, so a configured ``space_id`` such as ``../embedders/<id>`` would address a
different resource. Every component resolves its space through ``GET /v1/spaces/{id}``.
These tests drive ADK, the published SDK and httpx against a local socket server that
records every request target exactly as received.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
from google.adk.events import Event
from google.adk.sessions import Session

from goodmem_adk import GoodmemFetchTool, GoodmemPlugin, GoodmemSaveTool
from goodmem_adk._backend import Backend
from goodmem_adk.memory import GoodmemMemoryService
from tests.support import all_text, responses, runner_for, text_content, turn
from tests.wire import EMBEDDER_ID, embedder, memory, space

pytestmark = [pytest.mark.enable_socket, pytest.mark.asyncio]

U = "3f1c2a4e-9b7d-4e2f-8a6b-1c0d9e8f7a65"
NOT_UUIDS = [
    f"../spaces/{U}",
    f"a/../../spaces/{U}",
    f"%2e%2e/spaces/{U}",
    f"..%2Fspaces%2F{U}",
    f"{U}/../../spaces/{U}",
    "",
    f" {U}",
    f"{U}?x=1",
    f"{U}#frag",
    f"../embedders/{EMBEDDER_ID}",
    f"{U}\n",
]


class Recorder:
    def __init__(self):
        self.requests = []

    def paths(self):
        return [(method, urlsplit(target).path) for method, target, _ in self.requests]


@pytest.fixture
def server():
    recorder = Recorder()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def do_PUT(self):
            self.respond()

        def do_PATCH(self):
            self.respond()

        def do_DELETE(self):
            self.respond()

        def respond(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.loads(raw) if raw else {}
            recorder.requests.append((self.command, self.path, body))
            path = urlsplit(self.path).path
            status, data, content_type = 200, {}, "application/json"
            if path == "/v1/spaces" and self.command == "GET":
                data = {"spaces": []}
            elif path == "/v1/spaces":
                data = space(spaceId=U, name=body["name"])
            elif path.startswith("/v1/spaces/"):
                # Answer like a permissive server, so an unrefused ID is used for writes.
                data = space(spaceId=path.rsplit("/", 1)[-1])
            elif path == "/v1/embedders":
                data = {"embedders": [embedder()]}
            elif path == "/v1/memories":
                data = memory(spaceId=body["spaceId"], metadata=body.get("metadata", {}))
            elif path == "/v1/memories:retrieve":
                data, content_type = "", "application/x-ndjson"
            else:
                status, data = 404, {"message": "Unexpected route"}
            encoded = data.encode() if isinstance(data, str) else json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    recorder.url = f"http://127.0.0.1:{httpd.server_port}"
    worker = threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True)
    worker.start()
    try:
        yield recorder
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2)


def session_for(app_name="app", user_id="user"):
    return Session(
        app_name=app_name,
        user_id=user_id,
        id="session",
        events=[
            Event(author="user", content=text_content("question")),
            Event(author="agent", content=text_content("answer")),
        ],
    )


async def _through_runner(message, name, *, tools=(), plugin=None, user_id="user"):
    runner, _ = runner_for(tools=list(tools), plugin=plugin)
    try:
        _, events = await turn(runner, user_id, message)
        return responses(events, name) if name else events
    finally:
        await runner.close()


async def _save_tool(tool):
    return await _through_runner("SAVE: a note", "goodmem_save", tools=[tool])


async def _fetch_tool(tool):
    return await _through_runner("FETCH: a note", "goodmem_fetch", tools=[tool])


async def _plugin(plugin):
    return await _through_runner("Remember this note", None, plugin=plugin)


async def _ingest(service):
    await service.add_session_to_memory(session_for())
    return "ingested"


async def _search(service):
    return await service.search_memory(app_name="app", user_id="user", query="note")


# Every public component, and the internal service's two operations, resolve their
# space through the same backend. Each entry is (constructor, operation).
ENTRY_POINTS = {
    "GoodmemSaveTool": (GoodmemSaveTool, _save_tool),
    "GoodmemFetchTool": (GoodmemFetchTool, _fetch_tool),
    "GoodmemPlugin": (GoodmemPlugin, _plugin),
    "GoodmemMemoryService.add_session_to_memory": (GoodmemMemoryService, _ingest),
    "GoodmemMemoryService.search_memory": (GoodmemMemoryService, _search),
}


async def attempt(entry, server, **settings):
    """Configure and use a component; return the refusal or outcome as text."""
    build, use = ENTRY_POINTS[entry]
    try:
        component = build(base_url=server.url, api_key="test-key", timeout=5, **settings)
    except ValueError as error:
        return str(error)
    try:
        return all_text(await use(component))
    except Exception as error:  # Plugin and service failures propagate by design.
        return str(error)


@pytest.mark.parametrize("value", NOT_UUIDS)
@pytest.mark.parametrize("entry", ENTRY_POINTS)
async def test_space_id_argument_must_be_a_uuid_before_any_request(server, entry, value):
    outcome = await attempt(entry, server, space_id=value)
    assert server.requests == [], f"{entry}(space_id={value!r}) sent {server.requests}"
    assert "space_id must be a UUID" in outcome


@pytest.mark.parametrize("value", NOT_UUIDS)
@pytest.mark.parametrize("entry", ENTRY_POINTS)
async def test_space_id_environment_setting_must_be_a_uuid_before_any_request(
    server, monkeypatch, entry, value
):
    monkeypatch.setenv("GOODMEM_SPACE_ID", value)
    outcome = await attempt(entry, server)
    assert server.requests == [], f"{entry} with GOODMEM_SPACE_ID={value!r} sent {server.requests}"
    assert "GOODMEM_SPACE_ID must be a UUID" in outcome


@pytest.mark.parametrize("value", NOT_UUIDS)
@pytest.mark.parametrize("entry", ENTRY_POINTS)
async def test_embedder_id_settings_must_be_uuids_before_any_request(
    server, monkeypatch, entry, value
):
    # Embedder IDs travel in request bodies, not paths; they share the same check.
    outcome = await attempt(entry, server, embedder_id=value)
    monkeypatch.setenv("GOODMEM_EMBEDDER_ID", value)
    from_environment = await attempt(entry, server)
    assert server.requests == [], f"{entry} embedder {value!r} sent {server.requests}"
    assert "embedder_id must be a UUID" in outcome
    assert "GOODMEM_EMBEDDER_ID must be a UUID" in from_environment


@pytest.mark.parametrize("value", NOT_UUIDS)
async def test_request_boundary_refuses_a_space_id_changed_after_configuration(server, value):
    backend = Backend(base_url=server.url, api_key="test-key", space_id=U)
    backend.space_id = value
    saved = await backend.save(default_name="scope", text="note", content=None, metadata={})
    fetched = await backend.fetch(default_name="scope", query="note", top_k=1)
    service = GoodmemMemoryService(base_url=server.url, api_key="test-key", space_id=U)
    service._backend.space_id = value
    with pytest.raises(ValueError, match="space_id must be a UUID"):
        await service.add_session_to_memory(session_for())
    assert server.requests == [], f"space_id={value!r} sent {server.requests}"
    assert not saved.success and not saved.accepted
    assert "space_id must be a UUID" in all_text(saved.errors)
    assert fetched.partial and fetched.statuses[0].code == "RETRIEVAL_ERROR"
    assert "space_id must be a UUID" in fetched.statuses[0].message


@pytest.mark.parametrize("configured", [U, U.upper()], ids=["lowercase", "uppercase"])
@pytest.mark.parametrize("source", ["argument", "environment"])
@pytest.mark.parametrize("entry", ENTRY_POINTS)
async def test_valid_uuid_reaches_exactly_the_configured_space(
    server, monkeypatch, entry, source, configured
):
    # Uppercase input is accepted and sent in GoodMem's lowercase form.
    if source == "environment":
        monkeypatch.setenv("GOODMEM_SPACE_ID", configured)
        monkeypatch.setenv("GOODMEM_EMBEDDER_ID", EMBEDDER_ID)
        outcome = await attempt(entry, server)
    else:
        outcome = await attempt(entry, server, space_id=configured, embedder_id=EMBEDDER_ID)
    assert "must be a UUID" not in outcome
    (method, target, _), *rest = server.requests
    assert (method, target) == ("GET", f"/v1/spaces/{U}")
    assert rest, f"{entry} stopped after resolving its space: {outcome}"
    for method, target, body in rest:
        if target == "/v1/memories":
            assert (method, body["spaceId"]) == ("POST", U)
        else:
            assert (method, target) == ("POST", "/v1/memories:retrieve")
            assert body["spaceKeys"] == [{"spaceId": U}]


async def test_adk_user_and_app_names_stay_out_of_request_paths(server):
    """ADK-supplied names select default scopes through a query and a body only."""
    hostile = f"../spaces/{U}"
    tool = GoodmemSaveTool(base_url=server.url, api_key="test-key", embedder_id=EMBEDDER_ID)
    runner, _ = runner_for(tools=[tool])
    try:
        _, events = await turn(runner, hostile, "SAVE: a note")
        assert responses(events, "goodmem_save")[0]["success"]
    finally:
        await runner.close()
    service = GoodmemMemoryService(base_url=server.url, api_key="test-key", embedder_id=EMBEDDER_ID)
    await service.add_session_to_memory(session_for(app_name=hostile, user_id=hostile))
    await service.search_memory(app_name=hostile, user_id=hostile, query="note")
    assert {path for _, path in server.paths()} == {
        "/v1/spaces",
        "/v1/memories",
        "/v1/memories:retrieve",
    }
    lookups = [
        parse_qs(urlsplit(target).query)["name_filter"][0]
        for method, target, _ in server.requests
        if method == "GET" and urlsplit(target).path == "/v1/spaces"
    ]
    assert f"adk_tool_{hostile}" in lookups
    assert f"adk_memory_{hostile}_{hostile}" in lookups
