# Changelog

## 0.2.0 — 2026-09-15

This release uses the official `goodmem` SDK and native asynchronous requests.
It is a deliberate API break from 0.1.1. Requires Python 3.10+, Google ADK 2.9+,
and GoodMem's Python SDK 0.1.34+.

### Fixed

- Each configured plugin/tool owns its scope cache. A session's plugin can no
  longer redirect a tool to a different configured space. Explicit scope arguments
  override all environment scope settings. Conflicting ID/name/embedder settings fail.
- Retrieval preserves all distinct matching chunks from a document. Typed memory
  definitions supply source metadata without the broken batch lookup or N+1 reads.
- Server statuses pass through; unrecognized codes become `UNKNOWN`. Noninformational
  statuses mark results `partial`. Stream/transport errors preserve chunks already
  received and remain visible to the model. Empty searches perform one request.
- All callbacks/tools use native async HTTP. Caller-injected SDKs retain their
  connection, credentials, and ownership; sync clients are rejected explicitly.
- Saves retain accepted memory IDs, processing states, attachment IDs and individual
  errors. A failed attachment cannot silently become a successful save.
- Prompt augmentation no longer modifies persisted session events. Model thoughts
  and streaming deltas are excluded from automatic persistence.
- Internal session ingestion propagates SDK write errors and records individual
  accepted writes, so a retry doesn't skip failures or repeat confirmed writes.
- Live tests register created spaces immediately, delete them using the SDK, verify
  deletion, and fail visibly on cleanup errors. Tests exercise actual ADK runners,
  HTTP fault injection, independent event-loop activity, and exact random facts
  recalled by real models in fresh sessions with different-user controls.

### Migration

| 0.1 API / behavior | 0.2 replacement |
|---|---|
| `GoodmemClient` / `plugin.goodmem_client` | Import `Goodmem` or `AsyncGoodmem` from `goodmem` for administrative calls. |
| Bare `goodmem_save` / `goodmem_fetch` functions | Pass configured `GoodmemSaveTool` / `GoodmemFetchTool` instances to your agent. Model-facing tool names are unchanged. |
| `debug=True`, printed content tables | Use application/SDK logging. Debug arguments and content-printing helpers are removed. |
| Implicit Google embedder creation | Configure an embedder on GoodMem first; select it with `embedder_id` or `GOODMEM_EMBEDDER_ID`. No Google credentials are required when using another provider. |
| One result per memory | One result per matching chunk, including `chunk_id`, `space_id`, original `metadata` and separate `chunk_metadata`. |
| Save `success` alone | Check `accepted`, `errors`, and `partial`; acceptance does not imply indexing completion. |
| Silent automatic-save failures | ADK raises an error caused by `GoodmemSaveError`; its `response` and error text retain accepted IDs and individual errors. |
| Fetch `success` alone | Check `partial` and `statuses` as well. Usable chunks can accompany a partial result. |

A caller-owned async SDK can be shared by tools and plugins:

```python
import os

from goodmem import AsyncGoodmem
from goodmem_adk import GoodmemFetchTool, GoodmemSaveTool

async def run_with_memory(run_agent):
    async with AsyncGoodmem(
        base_url=os.environ["GOODMEM_BASE_URL"],
        api_key=os.environ["GOODMEM_API_KEY"],
    ) as client:
        tools = [GoodmemSaveTool(client=client), GoodmemFetchTool(client=client)]
        await run_agent(tools)
```

For a private certificate authority, construct `AsyncGoodmem` with a custom
`httpx.AsyncClient(verify=ssl.create_default_context(cafile=...))` and pass that
SDK into the integration. Keep both clients open while the ADK app runs.

The internal `goodmem_adk.memory` service remains outside the supported public
exports. Its retry bookkeeping is process-local and bounded; it does not provide
restart-durable ingestion deduplication. No new public MemoryService interface is
being introduced in this release.
