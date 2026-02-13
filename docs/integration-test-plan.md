# Integration Test Plan — Core Memory Functionality

End-to-end tests that verify `GoodmemPlugin` and `GoodmemSaveTool`/`GoodmemFetchTool` can store and recall information across independent ADK sessions using a live Goodmem backend and real Gemini LLM.

Implemented in `tests/test_integration.py`.

## Test Infrastructure

- **Runner**: `InMemoryRunner` from ADK. Sessions are purely in-memory — no `.adk/` directory, no SQLite. This guarantees session 2 starts with a blank slate; the only way the LLM can answer questions about session 1's content is through Goodmem memory retrieval.
- **Cleanup**: `cleanup_spaces` fixture collects space IDs during the test and deletes them via `GoodmemClient.delete_space()` in teardown.
- **PDF generation**: `mock_receipt_pdf` fixture (in `tests/conftest.py`) uses `fpdf2` to generate a mock receipt PDF in memory (no file on disk).
- **Indexing wait**: Fixed-time waits after session 1 (`_INDEX_WAIT = 5s` for text, `_INDEX_WAIT_PDF = 8s` for PDF). If indexing is not complete by the time session 2 runs, the assertions will fail — acting as a timeout error.
- **Space isolation**: Each test uses a unique space name (`{prefix}_{uuid8}`) to avoid cross-test interference.

## Prerequisites

```bash
GOODMEM_BASE_URL=http://localhost:8080 \
GOODMEM_API_KEY=<key> \
GOOGLE_API_KEY=<key> \
pytest -m integration -v
```

Tests are automatically skipped when these env vars are not set.

---

## Test 1: Goldfish Memory — Plugin Path

**Class**: `TestPluginIntegration.test_goldfish_memory_across_sessions`

| Session | User message | Expected behavior |
|---------|-------------|-------------------|
| 1 | "I am a goldfish" | Plugin auto-stores the text via `on_user_message_callback`. LLM acknowledges. |
| 2 | "Do I live in water?" | Plugin retrieves memory via `before_model_callback`. LLM answers "yes" with goldfish-related reasoning. |

**Assertions**:
- Session 2 response contains at least one of: "water", "goldfish", "fish", "aquatic", "aquarium".
- Direct `retrieve_memories("Do I live in water?", ...)` returns chunks mentioning "goldfish".

**Why it's effective**: Session 2 is a brand-new in-memory session. The LLM has no conversational context about goldfish. The only source of this information is Goodmem retrieval.

---

## Test 2: Goldfish Memory — Tools Path

**Class**: `TestToolsIntegration.test_goldfish_memory_via_tools`

| Session | User message | Expected behavior |
|---------|-------------|-------------------|
| 1 | "Remember this: I am a goldfish" | LLM calls `goodmem_save` tool. |
| 2 | "Do I live in water?" | LLM calls `goodmem_fetch` tool, retrieves the goldfish fact, answers accordingly. |

**Assertions**:
- Session 1: `goodmem_save` function call detected in events.
- Session 2: `goodmem_fetch` function call detected in events.
- Session 2 response contains goldfish-related keywords.

**Why it's effective**: Same session isolation as Test 1. Additionally verifies the LLM correctly decides to use the save/fetch tools.

---

## Test 3: PDF Receipt Memory — Plugin Path

**Class**: `TestPluginIntegration.test_pdf_receipt_memory_across_sessions`

The mock receipt contains:
- **From**: Acme Corp, 123 Innovation Drive, San Francisco, CA 94105
- **To**: GoodMind Inc., 456 Memory Lane, Palo Alto, CA 94301
- **Line items**: Cloud Computing ($2,450.00), Data Processing ($1,275.50), Technical Support ($500.00)
- **Total**: $4,225.50

| Session | User message | Expected behavior |
|---------|-------------|-------------------|
| 1 | "In the attached receipt, how much did GoodMind pay Acme?" + PDF attachment | Plugin auto-stores the PDF via `insert_memory_binary`. LLM reads PDF natively and answers with the total. |
| 2 | "What's the address of Acme?" | Plugin retrieves memory. LLM answers with Acme's address from the PDF. |

**Assertions**:
- Session 1 response mentions the total ($4,225.50).
- Session 2 response contains Acme address keywords: "Innovation Drive", "San Francisco", "94105", or "123".
- Direct `retrieve_memories("Acme address", ...)` returns chunks mentioning "Acme" or "Innovation".

**Why it's effective**: Session 1 only asks about the **total amount**. Acme's **address** is never mentioned in conversation text — it only exists in the PDF. Session 2 can only answer the address question if Goodmem successfully extracted text from the PDF, embedded it, and retrieved it.

---

## Test 4: PDF Receipt Memory — Tools Path

**Class**: `TestToolsIntegration.test_pdf_receipt_memory_via_tools`

| Session | User message | Expected behavior |
|---------|-------------|-------------------|
| 1 | "Save this receipt to memory and tell me the total amount." + PDF attachment | LLM calls `goodmem_save`, which auto-saves the binary attachment from `tool_context.user_content`. |
| 2 | "What's the address of Acme? Check your memory." | LLM calls `goodmem_fetch`, retrieves PDF content, answers with Acme's address. |

**Assertions**:
- Session 1: `goodmem_save` function call detected.
- Session 2: `goodmem_fetch` function call detected.
- Session 2 response contains Acme address keywords.

**Why it's effective**: Same reasoning as Test 3, plus verifies the tools correctly handle binary attachment saving and retrieval.
