# Goodmem ADK Plugin — Examples

Three examples demonstrating the different ways to integrate Goodmem memory with Google ADK agents.

## Prerequisites

1. Install the plugin:

   ```bash
   pip install goodmem-adk
   ```

2. Set environment variables:

   ```bash
   export GOODMEM_BASE_URL="https://api.goodmem.ai"
   export GOODMEM_API_KEY="your-api-key"
   export GOOGLE_API_KEY="your-google-api-key"          # for Gemini
   # Optional:
   export GOODMEM_EMBEDDER_ID="your-embedder-id"        # pin a specific embedder
   export EMBEDDER_ID="your-embedder-id"                 # used by plugin demo
   ```

---

## 1. Plugin Demo — Automatic Memory

**Directory:** `goodmem_plugin_demo/`

The plugin intercepts every user message and LLM response automatically — no agent instructions needed. Before each model call, it retrieves relevant past conversations and injects them into the prompt.

```bash
cd examples
adk web .
# Open http://localhost:8000 and select goodmem_plugin_demo
```

**How it works:**

```python
from goodmem_adk import GoodmemChatPlugin

plugin = GoodmemChatPlugin(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
    top_k=5,
)

app = App(
    name="goodmem_plugin_demo",
    root_agent=root_agent,
    plugins=[plugin],
)
```

The plugin hooks into three ADK callbacks:
- `on_user_message_callback` — logs user messages and file attachments to Goodmem
- `before_model_callback` — retrieves top-k relevant memories and augments the prompt
- `after_model_callback` — logs the LLM response to Goodmem

---

## 2. Tools Demo — Explicit Memory

**Directory:** `goodmem_tools_demo/`

The agent decides when to save and fetch memories using tools. This gives the agent full control over what gets stored and when to search.

```bash
cd examples
adk web .
# Open http://localhost:8000 and select goodmem_tools_demo
```

**How it works:**

```python
from goodmem_adk import GoodmemSaveTool, GoodmemFetchTool

save_tool = GoodmemSaveTool(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
)
fetch_tool = GoodmemFetchTool(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
    top_k=5,
)

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="goodmem_tools_agent",
    instruction="Answer user questions to the best of your knowledge",
    tools=[save_tool, fetch_tool],
)
```

- `goodmem_save` — saves text content (and auto-saves supported file attachments)
- `goodmem_fetch` — semantic search over stored memories

---

## 3. Memory Service Demo — Session-Based Memory

**Directory:** `goodmem_memory_service_demo/`

Uses ADK's built-in `BaseMemoryService` interface. Conversation turns are stored automatically via an `after_agent_callback`, and the agent retrieves memories using ADK's `load_memory` / `preload_memory` tools.

```bash
cd examples
adk web --memory_service_uri="goodmem://env" .
# Open http://localhost:8000 and select goodmem_memory_service_demo
```

The `--memory_service_uri="goodmem://env"` flag tells ADK to use the factory registered in `services.py`, which creates a `GoodmemMemoryService` from environment variables.

**How it works:**

`services.py` (loaded automatically by `adk web` from the agents root):

```python
from goodmem_adk import GoodmemMemoryService
from google.adk.cli.service_registry import get_service_registry

def _goodmem_factory(uri, **kwargs):
    return GoodmemMemoryService(
        base_url=os.getenv("GOODMEM_BASE_URL"),
        api_key=os.getenv("GOODMEM_API_KEY"),
        top_k=5,
        split_turn=True,
    )

get_service_registry().register_memory_service("goodmem", _goodmem_factory)
```

`agent.py`:

```python
from google.adk import Agent
from google.adk.tools import load_memory, preload_memory

async def save_to_memory(callback_context):
    await callback_context.add_session_to_memory()

root_agent = Agent(
    model="gemini-2.5-flash",
    name="goodmem_memory_agent",
    instruction="You are a helpful assistant with persistent memory.",
    after_agent_callback=save_to_memory,
    tools=[preload_memory, load_memory],
)
```

**Key concepts:**
- `after_agent_callback=save_to_memory` — stores conversation turns after each agent response
- `preload_memory` — automatically searches memory before each model call
- `load_memory` — lets the agent explicitly search memory during a conversation
- `GoodmemMemoryServiceConfig` — configure `top_k`, `timeout`, `split_turn`

---

## Choosing an Integration

| Approach | Control | Setup | Best For |
|----------|---------|-------|----------|
| **Plugin** | Automatic | Lowest | Chat apps where all conversations should be remembered |
| **Tools** | Agent-driven | Medium | Agents that selectively save/recall information |
| **Memory Service** | ADK-managed | Highest | Full ADK integration with session management |

You can also combine approaches — e.g., use the plugin for automatic logging alongside tools for explicit recall.
