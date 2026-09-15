# GoodMem for Google ADK

Give your ADK agent memory across conversations. GoodMem stores and indexes
messages and documents; the agent retrieves relevant passages when it needs them.

Choose **tools** when the agent should decide what to remember, or the **plugin**
when you want automatic conversation capture and context retrieval.

## Install

```bash
pip install goodmem-adk
```

You need a [GoodMem server](https://docs.goodmem.ai), an API key, and an existing
embedder. Set `GOODMEM_BASE_URL` and `GOODMEM_API_KEY`; optionally select an
embedder with `GOODMEM_EMBEDDER_ID`. Your agent can use any ADK-supported model.

## Give your agent memory tools

This example uses Cohere through ADK's LiteLLM adapter. Install `litellm>=1.84`
and set `COHERE_API_KEY` separately from your GoodMem credentials.

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm
from goodmem_adk import GoodmemFetchTool, GoodmemSaveTool

root_agent = LlmAgent(
    name="assistant",
    model=LiteLlm(model="cohere_chat/command-a-03-2025"),
    instruction=(
        "Save facts when asked to remember them. Before answering questions "
        "about saved facts, call goodmem_fetch, even in a fresh conversation. "
        "Check tool results and report errors honestly."
    ),
    tools=[GoodmemSaveTool(), GoodmemFetchTool()],
)
app = App(name="memory_agent", root_agent=root_agent)
```

Save this as `memory_agent/agent.py`, then run `adk run memory_agent`.
Ask it to remember a fact, start a fresh session with the same user ID, and ask
for that fact. [Complete examples](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/main/examples/README.md) include both integration paths.

## Automatic memory

Instead of adding memory tools, attach the plugin to your app:

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm
from goodmem_adk import GoodmemPlugin

root_agent = LlmAgent(
    name="assistant",
    model=LiteLlm(model="cohere_chat/command-a-03-2025"),
    instruction="Answer using relevant memory context.",
)
app = App(
    name="memory_agent",
    root_agent=root_agent,
    plugins=[GoodmemPlugin()],
)
```

The plugin saves visible user and model messages, uploads inline attachments,
and supplies relevant context before model calls.

## Scopes and results

Defaults are `adk_tool_{user_id}` for tools and `adk_chat_{user_id}` for the
plugin. Set the same `space_id` or `space_name` on both to share memory. Explicit
scopes are shared by everyone using that configuration; defaults separate users,
not applications. Explicit arguments override environment scope settings.

Writes return accepted IDs and processing states. Indexing happens asynchronously;
empty searches are never retried automatically. Failed attachments are reported
alongside accepted writes. Automatic persistence failures raise an error containing
those IDs. Fetch results preserve distinct chunks, source metadata, statuses,
and a `partial` flag when retrieval may be incomplete.

For connection pooling or custom TLS, pass a caller-owned `AsyncGoodmem` as
`client=`. Otherwise each operation creates and closes its own asynchronous SDK.

See the [0.2 migration notes](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/main/CHANGELOG.md) and [validation guide](https://github.com/PAIR-Systems-Inc/goodmem-adk/blob/main/docs/testing.md).
