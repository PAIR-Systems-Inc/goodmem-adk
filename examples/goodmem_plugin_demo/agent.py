"""Run with adk run examples/goodmem_plugin_demo; see examples/README.md."""

import os

from google.adk.agents import LlmAgent
from google.adk.apps import App

from goodmem_adk import GoodmemPlugin

model_name = os.environ["ADK_MODEL"]
if "/" in model_name:
    from google.adk.models.lite_llm import LiteLlm

    model = LiteLlm(model=model_name)
else:
    model = model_name

root_agent = LlmAgent(
    name="assistant",
    model=model,
    instruction="Answer using relevant memory context. Report retrieval limitations honestly.",
)
app = App(name="goodmem_plugin_demo", root_agent=root_agent, plugins=[GoodmemPlugin()])
