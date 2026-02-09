# Goodmem ADK Plugin

Persistent memory plugin for [Google ADK](https://google.github.io/adk-docs/) agents, powered by [Goodmem.ai](https://goodmem.ai).

## Installation

```bash
pip install goodmem-adk
```

## Three Integration Points

| Approach | Class | Description |
|----------|-------|-------------|
| **Plugin** | `GoodmemChatPlugin` | Automatic memory — intercepts user/LLM messages via ADK callbacks |
| **Tools** | `GoodmemSaveTool`, `GoodmemFetchTool` | Explicit memory — agent decides when to save/fetch |
| **Memory Service** | `GoodmemMemoryService` | Session-based — full ADK `BaseMemoryService` implementation |

See [examples/README.md](examples/README.md) for detailed usage of each approach with runnable demo agents.

## Quick Start

```python
import os
from google.adk.agents import LlmAgent
from google.adk.apps import App
from goodmem_adk import GoodmemChatPlugin, GoodmemSaveTool, GoodmemFetchTool

plugin = GoodmemChatPlugin(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
)

save_tool = GoodmemSaveTool(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
)
fetch_tool = GoodmemFetchTool(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
)

agent = LlmAgent(
    name="memory_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant with persistent memory.",
    tools=[save_tool, fetch_tool],
)

app = App(name="my_app", root_agent=agent, plugins=[plugin])
```

## License

Apache 2.0
