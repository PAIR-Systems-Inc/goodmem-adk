# Goodmem ADK Plugin

Persistent memory plugin for [Google ADK](https://google.github.io/adk-docs/) agents, powered by [Goodmem.ai](https://goodmem.ai).

There are two main integration points for Goodmem ADK:

| Approach | Class | Description | Coverage |
|----------|-------|-------------|----------|
| **Plugin** | `GoodmemPlugin` | Implicit but deterministic memory reads and writes at every agent–user turn, triggered by callbacks predefined in ADK. | Saves all conversation turns, including file attachments. Retrieves memory at every turn. |
| **Tools** | `GoodmemSaveTool`, `GoodmemFetchTool` | Explicit but non-deterministic memory reads and writes, decided by the agent. | Saves information that the agent decides is important to remember. Retrieves memory when the agent decides it needs to recall. |


See [examples/README.md](examples/README.md) for detailed usage of each integration point with runnable demo agents.

## Quick Start

### Installation

For stable release, install from PyPI.
```bash
pip install goodmem-adk
```

For local development, install in editable mode. Run the command below in the root of the repository.
```bash
pip install -e .
```

### Set Environment Variables

```bash
export GOODMEM_BASE_URL="http://localhost:8080"
export GOODMEM_API_KEY="goodmem-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

Note that `GOODMEM_BASE_URL` shall not have the `/v1` suffix. 

**Optional Environment Variables:**

```bash
export GOODMEM_EMBEDDER_ID="your-embedder-id"
export GOODMEM_SPACE_ID="your-space-id"
export GOODMEM_SPACE_NAME="your-space-name"
```

**Which space is used?**

| Condition | Behavior |
|---|---|
| `GOODMEM_SPACE_ID` set | Must exist — `ValueError` if not found |
| `GOODMEM_SPACE_NAME` set | Looked up by name; auto-created if missing |
| Both set | `GOODMEM_SPACE_ID` must exist and its name must match `GOODMEM_SPACE_NAME` |
| Neither set | Auto-created as `adk_chat_{user_id}` (plugin) or `adk_tool_{user_id}` (tools) |

**Which embedder is used when creating a space?**

| Condition | Behavior |
|---|---|
| `GOODMEM_EMBEDDER_ID` set | Must exist — `ValueError` if not found |
| Not set, embedders exist in Goodmem instance | First available embedder is used |
| Not set, no embedders in Goodmem instance | `gemini-embedding-001` auto-created via `GOOGLE_API_KEY` |

### Examples

We provide the following examples: 
* See [examples/README.md](examples/README.md) for demos that can be immediately run with the command `adk web .`.
* See [tests/test_integration.py](tests/test_integration.py) for integration tests that invoke agents and process agent responses which use the plugin and tools.

### Using the plugin

```python
import os
from google.adk.agents import LlmAgent
from google.adk.apps import App
from goodmem_adk import GoodmemPlugin

plugin = GoodmemPlugin(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
)

agent = LlmAgent(
    name="memory_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant with persistent memory.",
)

app = App(name="GoodmemPluginDemo", root_agent=agent, plugins=[plugin])
```

### Using the tools

```python
import os
from google.adk.agents import LlmAgent
from google.adk.apps import App
from goodmem_adk import GoodmemSaveTool, GoodmemFetchTool


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

app = App(name="GoodmemToolsDemo", root_agent=agent)
```

## Testing

Unit tests (no server required):

```bash
pytest tests/test_client.py tests/test_plugin.py tests/test_tools.py -v
```

Integration tests (require a live Goodmem server and Gemini API key):

```bash
GOODMEM_BASE_URL=http://localhost:8080 \
GOODMEM_API_KEY=<key> \
GOOGLE_API_KEY=<key> \
pytest -m integration -v -s
```

We perform two integration tests:
* [tests/test_integration.py](tests/test_integration.py) tests the plugin and tools by invoking agents and processing agent responses, covering both text and PDF content.
* [tests/test_optional_env_vars.py](tests/test_optional_env_vars.py) tests the optional environment variables by invoking agents and processing agent responses.


## Local Development

To use a local checkout instead of the PyPI release, install in editable mode. Run the command below in the root of the repository.

```bash
pip install -e .
```


## License

Apache 2.0
