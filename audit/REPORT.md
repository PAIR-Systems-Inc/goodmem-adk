# GoodMem ADK audit — 2026-09-15 UTC

The basic integration works with current Google ADK. Both public paths—automatic
plugin callbacks and explicit save/fetch tools—successfully stored and recalled
facts through real ADK runners and a live GoodMem server. Real Cohere agents
recalled exact random identifiers in fresh sessions and answered questions from
PDFs extracted by GoodMem. Different users' default spaces remained isolated in
these tests.

There are nevertheless reproducible correctness and reliability problems. These
need targeted repairs; passing the existing happy paths does not cover them.
Production code was not changed for this audit.

The published wheel is `goodmem-adk==0.1.1`; all five Python source modules match
repository commit `c9745e336a9d675a4c8da24ace38ea0c6556165e` byte for byte. Tests
imported the installed wheel, not the checkout. Dependencies resolved to Google
ADK 2.9.0, google-genai 2.23.0, httpx 0.28.1 and Python 3.13.2. Real model tests
used ADK's LiteLLM adapter with Cohere `command-a-03-2025`. The local server was
GoodMem `server-v1.0.311`, image digest
`sha256:0897f60a0700d2bd023db69f0907923ad0e25b8ae836e8152d93095417cf5a60`,
using Cohere `embed-v4.0`. This is the existing local test server, not a claim
that the latest server release was tested.

| Test group | Passed | Failed | Skipped | What actually ran |
|---|---:|---:|---:|---|
| Existing unit suite | 112 | 1 | 0 | Published integration; network disabled; the missing delete method fails. |
| Original live suite with Cohere | 13 | 0 | 2 | Original text/configuration assertions, actual ADK and GoodMem. Model factory and credential gate were adapted for Cohere. |
| Additional real-model journeys | 4 | 0 | 0 | Exact random text recall plus PDF-derived answers, through both public paths. |
| Additional live GoodMem journeys | 5 | 3 | 0 | Real HTTP, embeddings, indexing, ADK callbacks/tools; deterministic model boundary for precise assertions. |
| HTTP failure/concurrency checks | 3 | 6 | 0 | Real local mock server sockets; production client and ADK unchanged. |

The two skipped original cases ask Gemini to read an inline PDF directly. They
were not run with Gemini. Separate tests verified inline PDF upload through both
ADK integration paths using the controlled model, and real Cohere answers from
PDF content extracted and retrieved by GoodMem. This covers the memory path
without claiming Gemini-specific model coverage.

The live audit waits on accepted memory IDs reaching `COMPLETED`; it does not
retry empty searches. All recorded temporary spaces were deleted successfully,
including spaces created by the original suite. An additional standalone batch
metadata probe also cleaned up its space.

| Priority | Finding | Reproduction and cause | Suggested repair |
|---|---|---|---|
| P1 | A tool can write to the wrong configured space. | **Live.** Configure a plugin with one `space_name` and a save tool with another in the same app. The tool reports success but uses the plugin's space; its own named space is never created. Both components share `_goodmem_space_id` in session state, and a cache hit bypasses the tool's configured name. | Resolve/cache by connection and configured scope. An explicit developer configuration must not be displaced by another component's cached value. |
| P2 | The fetch tool discards relevant chunks from the same document. | **Live.** A document contains two required identifiers in different chunks. The raw retrieval returns both, but `goodmem_fetch` returns only one. Deduplication uses `memoryId`, so every further matching chunk from that memory is skipped. | Keep matching chunks, or combine their text without discarding content. |
| P2 | Plugin retrieval loses memory metadata. | **Live.** The batch endpoint returns `results[].memory` with the filename and role. `get_memories_batch` reads a nonexistent top-level `memories` field and returns `[]`. The model receives content without the attachment filename; missing role metadata also falls back to `user`. | Parse the real batch response, preferably through the official SDK. |
| P2 | Retrieval failures and corrupt streams look like successful searches. | **Mock HTTP.** An `EMBEDDER_FAILED` status becomes `success=true`, count zero, “No memories found matching the query.” Good chunks plus `MEMORY_CONTENT_UNAVAILABLE` lose the diagnostic. Malformed NDJSON is silently ignored too. The parser keeps only `retrievedItem` events and catches JSON errors without reporting them. | Preserve statuses and usable chunks, expose incomplete results, and report malformed transport data. Unknown status codes must remain nonfatal. |
| P2 | Async callbacks and tools block the event loop. | **Mock HTTP + actual ADK.** A 250 ms retrieval response stalls an independent 10 ms heartbeat for about 254–263 ms, through both public paths. Their async functions call synchronous httpx methods directly. | Use the SDK's async client or offload blocking I/O. Test that other sessions continue while one request is slow. |
| P2 | A failed PDF attachment upload can still be reported as success. | **Mock HTTP + actual ADK.** Text insertion succeeds, multipart upload returns 503, and the tool returns `success=true`, the text memory ID and zero attachments. The attachment exception is visible only with debug enabled. | Preserve the accepted text ID and expose attachment failures/partial completion so the agent can recover accurately. |
| P2, tests | Existing live cleanup silently fails. | **Existing unit failure + source inspection.** `GoodmemClient` has no `delete_space`. Both live fixtures call it and suppress every exception; the unit test for it fails. | Repair cleanup and surface teardown errors. Register resources immediately so failures before the last assertion do not leak them. |

Source locations at the audited commit:

- Routing: [tools.py:372](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/c9745e336a9d675a4c8da24ace38ea0c6556165e/goodmem_adk/tools.py#L372), [plugin.py:254](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/c9745e336a9d675a4c8da24ace38ea0c6556165e/goodmem_adk/plugin.py#L254).
- Chunk loss: [tools.py:985](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/c9745e336a9d675a4c8da24ace38ea0c6556165e/goodmem_adk/tools.py#L985).
- Batch metadata: [client.py:422](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/c9745e336a9d675a4c8da24ace38ea0c6556165e/goodmem_adk/client.py#L422).
- Retrieval parsing: [client.py:266](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/c9745e336a9d675a4c8da24ace38ea0c6556165e/goodmem_adk/client.py#L266).
- Blocking calls: [plugin.py:670](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/c9745e336a9d675a4c8da24ace38ea0c6556165e/goodmem_adk/plugin.py#L670), [tools.py:943](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/c9745e336a9d675a4c8da24ace38ea0c6556165e/goodmem_adk/tools.py#L943).
- Attachment errors: [tools.py:652](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/c9745e336a9d675a4c8da24ace38ea0c6556165e/goodmem_adk/tools.py#L652).
- Cleanup: [test_integration.py:123](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/c9745e336a9d675a4c8da24ace38ea0c6556165e/tests/test_integration.py#L123), [test_optional_env_vars.py:136](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/c9745e336a9d675a4c8da24ace38ea0c6556165e/tests/test_optional_env_vars.py#L136).

Several things are already right. Unknown status codes do not abort retrieval.
Empty searches return after one request. Space listing includes pagination.
Authentication failures are visible through the tools. The existing configuration
tests cover useful cases, including plugin-write/tool-read interoperability by
name and ID. This integration had meaningful end-to-end coverage.

Some assertions are too permissive: the goldfish recall check accepts any answer
containing “water,” including “I don't know whether you live in water.” The tools
test checks that save/fetch were called but does not require successful tool
responses containing the fact. The new tests require exact random facts and
inspect tool results or the request delivered to the model, with different-user
negative controls. The current monthly integration harness also has no ADK entry.

`GoodmemMemoryService` is present in the wheel but explicitly excluded from the
public exports and examples. Its basic native `load_memory` journey, repeated
session ingestion and default app/user isolation passed a separate live check.
That does not establish that the internal service is ready to become public.
The optional newer ADK memory methods were not audited.

No Gemini authentication setup is needed for the completed Cohere tests. Gemini
itself, automatic Gemini embedder creation on an otherwise empty server, large
scale/load behavior and restart durability of the internal service's deduplication
were not verified. No quality/latency benchmark against another memory backend
was attempted.

Reproduction sources and the original tests are preserved unchanged in the
[versioned baseline archive](baseline-0.1.1.tar.gz); the [runner instructions](README.md)
describe its contents and pinned environment. [results.json](results.json) records
individual outcomes; [model-evidence.jsonl](model-evidence.jsonl) contains the
actual synthetic-fact answers, [batch-metadata-evidence.json](batch-metadata-evidence.json)
shows the batch parsing defect, and [cleanup-evidence.json](cleanup-evidence.json)
records the deletions. The failing audit assertions intentionally remain failing
against the unmodified 0.1.1 release.
