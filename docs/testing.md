# Validation

Install development dependencies, then run:

```bash
pip install -e '.[dev,live]'
pytest -m 'not integration'
ruff check goodmem_adk tests examples
ruff format --check goodmem_adk tests examples
mypy goodmem_adk
python -m build
```

The offline tests run actual Google ADK runners, callbacks and FunctionTools
against the published GoodMem SDK. They mock HTTP responses, not the integration's
own methods. Socket tests independently delay a real local HTTP server and verify
that another coroutine continues running. Failure cases include invalid credentials,
corrupt NDJSON, unknown server statuses and partially failed attachment uploads.
The README agents, example apps, and changelog's SDK migration snippet are executed
through actual ADK runners too.

For deterministic live ADK/GoodMem journeys, configure:

- `GOODMEM_TEST_LIVE=1`
- `GOODMEM_TEST_BASE_URL` and `GOODMEM_TEST_API_KEY`
- `GOODMEM_TEST_EMBEDDER_ID` (an existing embedder)

```bash
pytest -m 'integration and not model'
```

For real model conversations, also set `GOODMEM_TEST_MODEL`, for example
`cohere_chat/command-a-03-2025`, and the provider's credential (`COHERE_API_KEY` for
Cohere). A Gemini model identifier can be used with Google credentials instead.

```bash
pytest -m model
```

These tests create temporary spaces and delete them in teardown, including after
failed assertions. Cleanup failures fail the suite. Indexing waits inspect accepted
memory IDs; they never retry an empty search. Optional `GOODMEM_TEST_EVIDENCE` and
`GOODMEM_TEST_MODEL_EVIDENCE` paths record cleanup and synthetic-fact answers.
`GOODMEM_TEST_RESOURCE_JOURNAL` records each created space immediately for recovery
if a process is forcibly terminated before teardown.

Real-model cases require successful save/fetch results and exact random identifiers
in fresh sessions, with different-user negative controls. PDF cases exercise
GoodMem extraction; they do not claim model-provider-specific inline PDF support.
The deterministic scope cases also cover explicit arguments, environment defaults,
plugin/tool sharing and separate scopes, ID/name mismatches, and user isolation.

`tests/test_live_recovery.py` adds two server-backed recovery journeys. One holds
an outgoing ingestion request while another user completes, then verifies one
stored memory per session and isolated recall through ADK's native `load_memory`.
The other sends an invalid credential on one PDF upload, checks that the model
receives the real server rejection and accepted IDs, and retries only the failed
file. It verifies original bytes, memory counts, and source metadata on retrieval.
Both run in the live workflow and recurring integration harness.

CI builds the distribution, installs its wheel outside the checkout, runs tests
against that installed package, and rebuilds a wheel from the source archive.
The old 0.1 tests of removed HTTP/debug helpers are replaced by boundary and user
journey tests. The original audit evidence and executable 0.1.1 snapshot remain in
`audit/`, excluded from both published distributions. Its runner uses frozen tests
and pinned dependencies, never the current test tree. The internal MemoryService
retains native ADK, concurrent-session and attachment-retry checks; it is not
advertised as a new supported integration.
