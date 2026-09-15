"""Regression checks for the test harness's real-SDK resource lifecycle."""

import httpx
import pytest

from tests.conftest import LiveServer
from tests.wire import EMBEDDER_ID, space


def setup_server(monkeypatch, fail_first_delete=False):
    monkeypatch.setenv("GOODMEM_TEST_BASE_URL", "http://cleanup.test")
    monkeypatch.setenv("GOODMEM_TEST_API_KEY", "test-key")
    monkeypatch.setenv("GOODMEM_TEST_EMBEDDER_ID", EMBEDDER_ID)
    active, deletes = {}, []

    def handle(request):
        sid = request.url.path.rsplit("/", 1)[-1]
        if request.method == "POST":
            sid = f"test-space-{len(active) + 1}"
            active[sid] = space(spaceId=sid)
            return httpx.Response(200, json=active[sid])
        if request.method == "DELETE":
            deletes.append(sid)
            if fail_first_delete and len(deletes) == 1:
                return httpx.Response(400, json={"message": "Deletion failed"})
            del active[sid]
            return httpx.Response(204)
        return (
            httpx.Response(200, json=active[sid])
            if sid in active
            else httpx.Response(404, json={"message": "Deleted"})
        )

    original_init = httpx.Client.__init__

    def mock_transport(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handle)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", mock_transport)
    return LiveServer(monkeypatch), active, deletes


def test_precreated_sdk_spaces_are_deleted_after_a_failed_assertion(monkeypatch, tmp_path):
    journal = tmp_path / "resources.jsonl"
    monkeypatch.setenv("GOODMEM_TEST_RESOURCE_JOURNAL", str(journal))
    server, active, deletes = setup_server(monkeypatch)
    with pytest.raises(AssertionError, match="deliberate test failure"):
        try:
            sid = server.new_space()
            assert sid in journal.read_text(), "Resource must be journaled before test work"
            raise AssertionError("deliberate test failure")
        finally:
            server.close()
    assert deletes == [sid] and not active


def test_cleanup_attempts_every_space_and_reports_failure(monkeypatch):
    server, active, deletes = setup_server(monkeypatch, fail_first_delete=True)
    first, second = server.new_space(), server.new_space()
    with pytest.raises(AssertionError, match="Cleanup failed"):
        server.close()
    assert deletes == [first, second]
    assert first in active and second not in active
