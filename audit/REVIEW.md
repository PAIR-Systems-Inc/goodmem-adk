# GoodMem ADK 0.2 validation record

All seven audited defects have repairs and regression coverage. This record
captures the validation reviewed and approved for the 0.2.0 release on September
15, 2026. The release deliberately breaks the 0.1.1 API.

## What changed

| Audited defect | Repair and validation |
|---|---|
| Plugin/tool cache collision routes writes to the wrong space | Component-local caches; explicit scopes override environment scopes. Live tests exercise separate scopes and deliberate sharing by ID/name. |
| Fetch drops other chunks from the same document | Deduplicate by chunk ID. Live retrieval retains both facts from different chunks of one document. |
| Broken batch parsing loses source metadata | Use SDK memory-definition events in the retrieval stream. Source filename and role reach ADK; no extra metadata lookup is needed. |
| Retrieval errors and malformed NDJSON look successful | Preserve statuses and received chunks; mark incomplete results partial. Unknown codes become UNKNOWN and do not abort retrieval. |
| Async callbacks/tools block the event loop | Native AsyncGoodmem throughout. A separate heartbeat keeps running during delayed socket responses. |
| Failed attachment uploads look successful | Return accepted IDs, processing states, individual errors and partial completion. The automatic plugin raises an error with accepted IDs. |
| Live cleanup silently fails | Use SDK deletion, register every created space immediately, verify deletion, and report teardown failures after attempting all resources. |

The SDK is a net simplification: production source fell from **3,548 to 962 lines
(73%)**, including removal of the duplicate HTTP client and verbose debug helpers.
The README is **399 words**. The remaining adapter code handles ADK context,
scoping, attachments and compact results; it does not reimplement HTTP or NDJSON.

## Maintainability review follow-up

All four findings in the September 15 review were valid and are addressed:

| Finding | Change and regression evidence |
|---|---|
| Internal service blocks unrelated sessions | Each session has its own lock and accepted-write record. Active and queued sessions cannot be evicted. Tests stall one user's HTTP write, finish another user's ingestion, and concurrently retry the first without duplicate writes, including under cache pressure. |
| Attachment logic is duplicated and inconsistent | Tools/plugin and the internal service share validation, byte encoding, filenames, metadata construction and SDK submission. URI and empty attachments are explicit failures. HTTP/validation failure retries preserve earlier accepted attachments. |
| Migration example calls `AsyncGoodmem()` without its required URL | The snippet passes URL and API key explicitly from the environment. A new documentation test executes the actual snippet through an ADK save/fetch conversation and checks the outgoing SDK host and credentials. CI copies the changelog when testing the installed wheel. |
| Historical audit runs today's tests against 0.1.1 | Original tests and reproduction helpers are frozen in a versioned archive with source/commit hashes and pinned dependencies. The runner checks versions before collection and extracts only that snapshot. The loose duplicate helper/test files are removed. |

These changes add **nine** offline regressions. The corrected baseline runner
reproduces **112 passed / 1 known failure** in the original unit suite and
**3 passed / 6 known failures** in its HTTP suite. There are no collection errors.
It also rejects an installed 0.2.0 package before collecting the 0.1.1 baseline.
The initial baseline unit replay stalled on asyncio thread wakeups inside the
socket-restricted sandbox; permitting local sockets reproduced the original result.
The snapshot's original sources match the recorded Git commit byte for byte.

Two additional end-to-end tests now cover those behaviors against live GoodMem.
Concurrent ADK session ingestions overlap while one request is held, and the test
checks server-side memory counts and user-isolated recall through `load_memory`.
A separate save/fetch journey gives one PDF upload an invalid credential, verifies
the real HTTP 401 and accepted IDs delivered to the model, then retries only that
file in a fresh turn. It checks original PDF bytes, absence of duplicate memories,
indexing completion, source filenames and exact facts retrieved by a fresh reader.
Both tests run in the live workflow and are included in the recurring harness.
No production changes were needed for either test.

## Validation of the built wheel

| Environment / checks | Result |
|---|---|
| Python 3.10, installed wheel outside checkout | 63 passed, zero failures/errors/skips |
| Python 3.13, installed wheel outside checkout | 63 passed, zero failures/errors/skips |
| Live GoodMem with real ADK, latest full run | 16 passed, including the two new concurrency/recovery journeys |
| Live Cohere conversations, latest full run | 4 passed: text and PDF recall through plugin and tools |
| Ruff lint/format, strict mypy, diff checks | Passed |
| Wheel and source archive; rebuild wheel from source archive | Passed; identical file contents |
| Website MDX compilation | Passed; full website build was not run |
| Recurring harness manifest and feature aliases | Passed; all six advertised features map to passing live tests |

Dependencies tested: GoodMem Python SDK **0.1.34**, Google ADK **2.9.0**, and
Cohere **command-a-03-2025**. The local GoodMem server remains the existing
`server-v1.0.311` with Cohere `embed-v4.0`. This is not a latest-server, scale, or
latency benchmark. Gemini-specific model behavior was not exercised.

[Candidate results](candidate/results.json) and JUnit files are in `candidate/`.
All **27** spaces from the latest complete live run were independently confirmed
deleted. The previous complete run's **24** spaces and the maintainability
follow-up's **six** spaces were verified deleted as well. A fresh server inventory
found **zero** remaining ADK audit spaces. The latest run passed all **20** live
tests, including the real-model conversations, in approximately one minute.

An earlier candidate's test observer missed precreated spaces because the sync SDK
captured its HTTP method before interception. The final audit caught this. Registration
now happens at creation, with two regression tests and an optional persistent resource
journal. The 12 leftovers from that fixture defect and the interrupted run were
matched to this audit, deleted, and verified separately. This was a defect in my new
fixture, not another bug attributed to the original integration.

One early Cohere run answered without calling fetch. The tool description and agent
instructions now state that saved facts must be looked up even in a fresh session.
The subsequent complete real-model runs passed; model tool choice is still model
behavior, not something the integration can guarantee.

## Review decisions

- `GoodmemClient`, the bare helper functions and `debug` arguments are removed.
  SDK administration and configured FunctionTools replace them; model-facing names
  remain `goodmem_save` and `goodmem_fetch`.
- Embedder creation is explicit. The integration uses an existing server embedder
  instead of implicitly provisioning a Google embedder with unrelated credentials.
- `GoodmemMemoryService` remains internal. Its native ADK journey and retry handling
  are tested, but its deduplication record is process-local, not durable across restarts.
- Inject an SDK for connection pooling/custom TLS. Otherwise each operation owns and
  closes its async client, avoiding global client caches and missing tool shutdown hooks.
- The SDK attaches API groups dynamically without type annotations. A small internal
  typing Protocol describes those groups; this does not add a runtime client layer.

The [changelog](../CHANGELOG.md) describes the public migration and behavior changes.
Baseline assertions and evidence are preserved in the frozen snapshot; the
runner and instructions now select it explicitly. The audit remains excluded
from both distributions.

## Working directories and release order

- Integration: `/home/amin/clients/pairsys/goodmem-adk`, branch `audit/published-0.1.1`.
- Website: `/home/amin/clients/pairsys/goodmem-wt-adk-0.2`, branch `docs/adk-0.2`.
- Monthly harness: `/home/amin/clients/pairsys/goodmem-integrations-e2e-adk-0.2`,
  branch `update/adk-0.2`.

The main GoodMem checkout and retired documentation repository were not edited.
Release order: merge/publish the integration before its website and harness changes.
The results above were collected locally before publication. GitHub Actions results
are recorded separately on the release pull request. No remote test credentials
were added as part of this validation.
