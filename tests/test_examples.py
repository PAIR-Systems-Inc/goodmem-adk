"""Execute each README/example agent through ADK with only model/HTTP boundaries replaced."""

import re
import runpy
from pathlib import Path

import httpx
import pytest
from google.adk.runners import InMemoryRunner

from tests.support import RecordingModel, all_text, responses, runner_for, turn
from tests.test_sdk_adk import wire as wire  # noqa: F401
from tests.wire import CHUNK_ID, EMBEDDER_ID, MEMORY_ID, SPACE_ID, chunk, memory

_README = Path("README.md").read_text()
_CHANGELOG = Path("CHANGELOG.md").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["readme-tools", "readme-plugin", "tools-demo", "plugin-demo"])
async def test_documented_app_runs(source, wire, monkeypatch):  # noqa: F811
    monkeypatch.setenv("GOODMEM_BASE_URL", "http://example.test")
    monkeypatch.setenv("GOODMEM_API_KEY", "test-key")
    monkeypatch.setenv("GOODMEM_SPACE_ID", SPACE_ID)
    monkeypatch.setenv("GOODMEM_EMBEDDER_ID", EMBEDDER_ID)
    monkeypatch.setenv("ADK_MODEL", "cohere_chat/command-a-03-2025")
    original_init = httpx.AsyncClient.__init__

    def with_mock_transport(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(wire.handle)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", with_mock_transport)
    if source.startswith("readme"):
        code = re.findall(r"```python\n(.*?)```", _README, re.S)
        namespace = {}
        exec(code[0 if source.endswith("tools") else 1], namespace)
    else:
        mode = "tools" if source.startswith("tools") else "plugin"
        namespace = runpy.run_path(f"examples/goodmem_{mode}_demo/agent.py")
    app = namespace["app"]
    model = RecordingModel()
    app.root_agent.model = model
    runner = InMemoryRunner(app=app)
    wire.events = [
        chunk("The saved label is evergreen-537"),
        {
            "memoryDefinition": memory(metadata={"filename": "note.txt"}),
        },
    ]
    try:
        if "tools" in source:
            _, saved = await turn(runner, "example-user", "SAVE: a label")
            assert responses(saved, "goodmem_save")[0]["success"]
            _, events = await turn(runner, "example-user", "FETCH: saved label")
            result = responses(events, "goodmem_fetch")[0]
            assert result["success"] and "evergreen-537" in all_text(result)
            assert result["memories"][0]["chunk_id"] == CHUNK_ID
            assert result["memories"][0]["memory_id"] == MEMORY_ID
        else:
            await turn(runner, "example-user", "Recall my label")
            assert "evergreen-537" in all_text(model.requests[-1].contents)
            assert len(wire.writes) == 2
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_migration_example_connects_with_explicit_environment_settings(wire, monkeypatch):  # noqa: F811
    monkeypatch.setenv("GOODMEM_BASE_URL", "http://migration.test")
    monkeypatch.setenv("GOODMEM_API_KEY", "migration-key")
    monkeypatch.setenv("GOODMEM_SPACE_ID", SPACE_ID)
    original_init = httpx.AsyncClient.__init__
    requests = []

    async def record(request):
        requests.append(request)
        return await wire.handle(request)

    def with_mock_transport(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(record)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", with_mock_transport)
    namespace = {}
    exec(re.findall(r"```python\n(.*?)```", _CHANGELOG, re.S)[0], namespace)
    wire.events = [chunk("The label is migration-812")]

    async def run_agent(tools):
        runner, _ = runner_for(tools=tools)
        try:
            _, saved = await turn(runner, "user", "SAVE: label migration-812")
            assert responses(saved, "goodmem_save")[0]["success"]
            _, found = await turn(runner, "user", "FETCH: label")
            result = responses(found, "goodmem_fetch")[0]
            assert result["success"] and "migration-812" in all_text(result)
        finally:
            await runner.close()

    await namespace["run_with_memory"](run_agent)
    assert requests
    assert all(request.url.host == "migration.test" for request in requests)
    assert all(request.headers["x-api-key"] == "migration-key" for request in requests)


@pytest.mark.asyncio
async def test_readme_names_every_model_facing_name_and_setting(wire):  # noqa: F811
    """The README documents what the model sees and what the code reads, taken from the code."""
    import goodmem_adk
    from goodmem_adk import GoodmemFetchTool, GoodmemPlugin, GoodmemSaveTool

    tools = [
        GoodmemSaveTool(client=wire.client, embedder_id=EMBEDDER_ID),
        GoodmemFetchTool(client=wire.client, embedder_id=EMBEDDER_ID),
    ]
    names = {tool.name for tool in tools}
    arguments = {
        name
        for tool in tools
        for name in (tool._get_declaration().parameters_json_schema or {}).get("properties", {})
    }
    runner, _ = runner_for(
        plugin=GoodmemPlugin(client=wire.client, space_id=SPACE_ID), tools=tools[:1]
    )
    try:
        await turn(runner, "readme-user", "SAVE: a note")
    finally:
        await runner.close()
    metadata = {key for request, _ in wire.writes for key in request["metadata"]}
    values = {request["metadata"][key] for request, _ in wire.writes for key in ("role", "source")}
    source = "".join(
        Path(goodmem_adk.__file__).with_name(module).read_text()
        for module in ("_backend.py", "plugin.py", "tools.py")
    )
    settings = set(re.findall(r"getenv\(\"(\w+)\"\)", source))
    assert names == {"goodmem_save", "goodmem_fetch"}
    assert values == {"user", "model", "adk_plugin", "adk_tool"} and len(settings) == 5
    documented = set(re.findall(r"`([^`\n]+)`", _README))
    missing = sorted((names | arguments | metadata | values | settings) - documented)
    assert not missing, f"README does not name: {missing}"
