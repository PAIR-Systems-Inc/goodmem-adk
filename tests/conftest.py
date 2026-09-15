"""SDK-backed fixtures; live resource cleanup runs even after failed assertions."""

import asyncio
import json
import os
from contextlib import ExitStack

import httpx
import pytest
from goodmem import AsyncGoodmem, Goodmem
from goodmem.errors import NotFoundError

from tests.support import unique


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    for name in (
        "GOODMEM_SPACE_ID",
        "GOODMEM_SPACE_NAME",
        "GOODMEM_EMBEDDER_ID",
        "GOODMEM_BASE_URL",
        "GOODMEM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class LiveServer:
    def __init__(self, monkeypatch):
        self.base_url = os.environ["GOODMEM_TEST_BASE_URL"].rstrip("/")
        self.api_key = os.environ["GOODMEM_TEST_API_KEY"]
        self.embedder_id = os.environ["GOODMEM_TEST_EMBEDDER_ID"]
        self.spaces, self.writes, self.requests = {}, [], []
        self._resources = ExitStack()
        original_send, original_asend = httpx.Client.send, httpx.AsyncClient.send

        def record(request, response):
            if str(request.url).startswith(self.base_url + "/"):
                self.requests.append((request.method, request.url.path, response.status_code))
                if request.method == "POST" and response.is_success:
                    if request.url.path == "/v1/spaces":
                        result = response.json()
                        self.register_space(result["spaceId"], result["name"])
                    elif request.url.path == "/v1/memories":
                        self.writes.append(response.json())

        def observe(client, request, *args, **kwargs):
            response = original_send(client, request, *args, **kwargs)
            record(request, response)
            return response

        async def observe_async(client, request, *args, **kwargs):
            response = await original_asend(client, request, *args, **kwargs)
            record(request, response)
            return response

        monkeypatch.setattr(httpx.Client, "send", observe)
        monkeypatch.setattr(httpx.AsyncClient, "send", observe_async)
        self.sdk = self._resources.enter_context(
            Goodmem(base_url=self.base_url, api_key=self.api_key)
        )
        self.http = self._resources.enter_context(
            httpx.Client(
                base_url=self.base_url,
                headers={"x-api-key": self.api_key},
                timeout=30,
            )
        )

    @property
    def config(self):
        return {"base_url": self.base_url, "api_key": self.api_key, "embedder_id": self.embedder_id}

    def new_space(self, name=None, chunk_size=None):
        chunking = (
            {"none": {}}
            if chunk_size is None
            else {
                "recursive": {
                    "chunkSize": chunk_size,
                    "chunkOverlap": 0,
                    "keepStrategy": "KEEP_END",
                    "lengthMeasurement": "CHARACTER_COUNT",
                }
            }
        )
        created = self.sdk.spaces.create(
            name=name or unique(),
            space_embedders=[{"embedderId": self.embedder_id}],
            default_chunking_config=chunking,
        )
        self.register_space(created.space_id, created.name)
        return created.space_id

    def register_space(self, space_id, name):
        if space_id in self.spaces:
            return
        self.spaces[space_id] = name
        journal = os.environ.get("GOODMEM_TEST_RESOURCE_JOURNAL")
        if journal:
            with open(journal, "a") as file:
                file.write(json.dumps({"space_id": space_id, "name": name}) + "\n")

    async def wait_for_writes(self):
        async with AsyncGoodmem(base_url=self.base_url, api_key=self.api_key) as client:
            # Each accepted ID gets a bounded wait. Never poll an empty search.
            for written in list(self.writes):
                deadline = asyncio.get_running_loop().time() + 60
                while True:
                    memory = await client.memories.get(id=written["memoryId"])
                    if memory.processing_status == "COMPLETED":
                        break
                    assert memory.processing_status != "FAILED", (
                        f"Indexing failed: {memory.memory_id}"
                    )
                    assert asyncio.get_running_loop().time() < deadline, (
                        f"Indexing timeout: {memory.memory_id}"
                    )
                    await asyncio.sleep(0.2)

    def close(self):
        cleanup, errors = [], []
        try:
            for sid, name in self.spaces.items():
                try:
                    self.sdk.spaces.delete(id=sid)
                    try:
                        self.sdk.spaces.get(id=sid)
                    except NotFoundError:
                        cleanup.append({"space_id": sid, "name": name, "deleted": True})
                    else:
                        raise AssertionError(f"Space still exists after DELETE: {sid}")
                except Exception as error:
                    # Keep cleaning up all remaining resources, then fail teardown.
                    errors.append(f"{sid}: {error}")
            evidence = os.environ.get("GOODMEM_TEST_EVIDENCE")
            if evidence:
                with open(evidence, "a") as file:
                    file.write(
                        json.dumps(
                            {
                                "cleanup": cleanup,
                                "errors": errors,
                                "requests": self.requests,
                                "accepted_memory_ids": [item["memoryId"] for item in self.writes],
                            }
                        )
                        + "\n"
                    )
        finally:
            self._resources.close()
        assert not errors, "Cleanup failed: " + "; ".join(errors)


@pytest.fixture
def live(monkeypatch):
    if os.environ.get("GOODMEM_TEST_LIVE") != "1":
        pytest.skip("Set GOODMEM_TEST_LIVE=1 and GOODMEM_TEST_{BASE_URL,API_KEY,EMBEDDER_ID}")
    server = LiveServer(monkeypatch)
    try:
        yield server
    finally:
        server.close()
