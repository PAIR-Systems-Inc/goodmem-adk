# Running the published-package audit

These are historical review artifacts for `goodmem-adk` **0.1.1**, not a second
test suite for the current integration. See [REPORT.md](REPORT.md) for findings
and which checks used a live server, a real model or fault injection.

[baseline-0.1.1.tar.gz](baseline-0.1.1.tar.gz) freezes the original tests from
commit `c9745e336a9d675a4c8da24ace38ea0c6556165e` under `original/` and the audit's
additional reproductions and helpers under `reproductions/`. Assertions and helper
sources are unchanged. The [manifest](baseline-0.1.1.json) records their SHA-256
hashes and the archive hash. This snapshot is not synchronized with today's tests.
To inspect its sources: `tar -xzf audit/baseline-0.1.1.tar.gz -C /path/to/scratch`.

The runner checks the package/framework versions and archive hash, extracts into
a temporary directory, and runs only those frozen tests. Neither the current
`tests/` directory nor its fixtures participate. The entire audit is excluded from
the wheel and source distribution.

Create an isolated environment outside this checkout and install the published
wheel with the recorded Python 3.13 dependency versions:

```bash
uv venv --python 3.13 /tmp/adk-audit-venv
uv pip install --python /tmp/adk-audit-venv/bin/python \
  -r audit/baseline-requirements.txt
```

Run each suite using that environment's Python:

```bash
/tmp/adk-audit-venv/bin/python audit/run.py unit --output /tmp/adk-unit
/tmp/adk-audit-venv/bin/python audit/run.py http --output /tmp/adk-http
/tmp/adk-audit-venv/bin/python audit/run.py live --output /tmp/adk-live
```

The HTTP suite starts local mock servers. The live suite uses the existing local
RAG test instance on port 8088. `ADK_AUDIT_RUNTIME` can select the directory
containing its `credentials.json` (`base_url`, `api_key`) and `state.json`
(`embedder_id`). The default is the sibling `Agentic_RAG_GoodMem/.runtime`
checkout used in this session. All spaces created by the audit are tracked and
deleted. Existing spaces and credentials are left alone.

For real-model checks, install `litellm==1.101.0` into the test environment, then supply a
local `.env` file containing `COHERE_API_KEY`:

```bash
uv pip install --python /tmp/adk-audit-venv/bin/python \
  -c audit/baseline-requirements.txt litellm==1.101.0
/tmp/adk-audit-venv/bin/python audit/run.py real \
  --env-file /path/to/provider.env --model cohere_chat/command-a-03-2025 \
  --output /tmp/adk-real
/tmp/adk-audit-venv/bin/python audit/run.py original-live \
  --env-file /path/to/provider.env --model cohere_chat/command-a-03-2025 \
  --output /tmp/adk-original
```

The latter substitutes the original suite's Gemini model factory and credential
gate for Cohere. Its two Gemini-native PDF cases are skipped; assertions in the
other thirteen tests are unchanged. The outer audit adds reliable cleanup because
the original fixtures call a nonexistent method and suppress the exception.

`--match` selects test names for focused runs. Results are written as JUnit XML;
live runs also record cleanup and synthetic-fact evidence. Expect failed checks:
these reproduce defects in the published package, not fixes to it.
