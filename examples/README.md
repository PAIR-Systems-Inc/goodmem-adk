# Goodmem ADK Plugin — Examples

Two demo agents showing how to use Goodmem with Google ADK.

For installation and environment setup, see [../README.md](../README.md).

We provide two demos: 
1. **Plugin Demo**, under `goodmem_plugin_demo/` directory — implicit memory management via callbacks. 
   This demo shows how to use the Goodmem plugin to automatically save all conversation turns (including file attachments like PDFs) and retrieve relevant memory at every turn. Useful for agents that benefit from automatic, implicit memory without explicit tool calls.
2. **Tools Demo**, under `goodmem_tools_demo/` directory — explicit memory management via agent tools.
   This demo shows how to use the Goodmem tools to give the agent explicit control over what to save and when to retrieve from memory. The agent decides which information is important to remember. Useful for agents that need fine-grained control over memory usage.

To start the demos, run the following command in the root of the repository:
```bash
cd examples
adk web .
```
Then open http://localhost:8000 and select the demo you want to run from the left sidebar.
