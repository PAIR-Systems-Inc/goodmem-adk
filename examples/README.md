# Run the examples

Install the package and the optional model adapter:

```bash
pip install 'goodmem-adk[live]'
```

Configure `GOODMEM_BASE_URL`, `GOODMEM_API_KEY`, and optionally
`GOODMEM_EMBEDDER_ID`. An embedder must already exist on your server.
For Cohere, set `COHERE_API_KEY` and:

```bash
export ADK_MODEL=cohere_chat/command-a-03-2025
adk run examples/goodmem_tools_demo
```

Use `examples/goodmem_plugin_demo` for automatic memory instead. The tools demo
lets the model decide when to save or search; the plugin records visible messages
and retrieves context automatically. Both upload inline file attachments.

You can also use a Gemini model name in `ADK_MODEL` with its Google credentials,
or another LiteLLM provider/model identifier with that provider's credentials.

Ask the agent to remember a unique fact. Start a new session using the same ADK
user ID, then ask for the fact. A different user gets a different default space.
The two examples use different default space prefixes; set `GOODMEM_SPACE_ID`
or `GOODMEM_SPACE_NAME` to deliberately share one space between them.

Writes are accepted before indexing finishes. To wait in application code, check
the returned memory ID's `processing_status` using the SDK's `memories.get()`.
An empty search is not an indexing signal.
