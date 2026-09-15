"""Run with adk run examples/goodmem_tools_demo; see examples/README.md."""

import os

from google.adk.agents import LlmAgent
from google.adk.apps import App

from goodmem_adk import GoodmemFetchTool, GoodmemSaveTool

model_name = os.environ["ADK_MODEL"]
if "/" in model_name:
    from google.adk.models.lite_llm import LiteLlm

    model = LiteLlm(model=model_name)
else:
    model = model_name

root_agent = LlmAgent(
    name="assistant",
    model=model,
    instruction=(
        "Save facts when asked to remember them. Before answering questions about saved facts, "
        "call goodmem_fetch, even in a fresh conversation. Check tool results and report errors "
        "honestly. Do not re-upload accepted writes when another attachment fails."
    ),
    tools=[GoodmemSaveTool(), GoodmemFetchTool()],
)
app = App(name="goodmem_tools_demo", root_agent=root_agent)
