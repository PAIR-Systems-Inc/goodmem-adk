# Integration Test Plan — Optional Env Var Combinations

Tests that verify all meaningful combinations of optional configuration parameters (`space_id`, `space_name`, `embedder_id`) work correctly, including cross-session scenarios where different sessions use different configurations pointing to the same underlying space.

To be implemented in `tests/test_integration.py` alongside the existing core memory tests.

## How to Control Config Per Test

- **Constructor parameters** (`space_id=`, `space_name=`, `embedder_id=`) and env vars (`GOODMEM_SPACE_ID`, etc.) feed into the same internal fields. Most tests pass constructor parameters directly for clarity.
- **Env var fallback** is tested separately using `monkeypatch.setenv` (Group E).
- **Pre-existing resources** are created via `GoodmemClient` in test setup and registered with `cleanup_spaces` for teardown.
- All happy-path tests reuse the **goldfish scenario** (fast indexing, already proven) — no need for PDF.

## Resolution Logic Reference

**Space:**

| Condition | Behavior |
|---|---|
| `space_id` set | Must exist — `ValueError` if not found |
| `space_name` set | Looked up by name; auto-created if missing |
| Both set | `space_id` must exist and its name must match `space_name` |
| Neither set | Auto-created as `adk_chat_{user_id}` (plugin) or `adk_tool_{user_id}` (tools) |

**Embedder:**

| Condition | Behavior |
|---|---|
| `embedder_id` set | Must exist — `ValueError` if not found |
| Not set, embedders exist | First available embedder is used |
| Not set, no embedders exist | `gemini-embedding-001` auto-created via `GOOGLE_API_KEY` |

---

## Group A: Space Resolution — Happy Paths

### A1. `space_name` only (auto-create)

Already covered by the existing goldfish and PDF tests. **Skip**.

### A2. `space_id` only (pre-created)

**Setup**: Use `GoodmemClient` to create a space with a known name. Record its `space_id`.

| Session | Config | User message |
|---------|--------|-------------|
| 1 | `space_id=<pre-created>` | "I am a goldfish" |
| 2 | `space_id=<pre-created>` | "Do I live in water?" |

**Assertions**: Session 2 recalls the goldfish fact. Verifies pinning a pre-existing space ID works end-to-end.

### A3. `space_id` + `space_name` (both match)

**Setup**: Pre-create a space with name "integ_both_xyz". Record its `space_id`.

| Session | Config | User message |
|---------|--------|-------------|
| 1 | `space_id=<id>`, `space_name="integ_both_xyz"` | "I am a goldfish" |
| 2 | `space_id=<id>`, `space_name="integ_both_xyz"` | "Do I live in water?" |

**Assertions**: No `ValueError` raised. Session 2 recalls the goldfish fact. Verifies the consistency check passes.

---

## Group B: Cross-Session Config Changes

The most interesting tests — sessions use **different configurations** that resolve to the same underlying space.

### B1. Session 1 via `space_name`, Session 2 via `space_id`

| Session | Config | User message |
|---------|--------|-------------|
| 1 | `space_name="cross_name_xyz"` | "I am a goldfish" |
| *(between)* | Look up space by name to discover `space_id` | |
| 2 | `space_id=<discovered>` (no name) | "Do I live in water?" |

**Assertions**: Session 2 recalls the goldfish fact even though the config changed. Verifies that `space_name` auto-creation and `space_id` pinning refer to the same space.

### B2. Plugin writes, Tools read (same `space_name`)

| Session | Component | Config | User message |
|---------|-----------|--------|-------------|
| 1 | `GoodmemPlugin` | `space_name="shared_xyz"` | "I am a goldfish" |
| 2 | `GoodmemFetchTool` | `space_name="shared_xyz"` | "Do I live in water? Check your memory." |

**Assertions**: `goodmem_fetch` is called. Session 2 recalls the goldfish fact. Verifies cross-component interoperability — data written by the plugin can be read by the tools.

### B3. Plugin writes, Tools read (via `space_id`)

| Session | Component | Config | User message |
|---------|-----------|--------|-------------|
| 1 | `GoodmemPlugin` | `space_name="shared_xyz"` | "I am a goldfish" |
| *(between)* | Look up space by name to discover `space_id` | |
| 2 | `GoodmemFetchTool` | `space_id=<discovered>` | "Do I live in water? Check your memory." |

**Assertions**: Same as B2 but verifies cross-component interop when session 2 uses a `space_id` instead of a name.

---

## Group C: Space Resolution — Error Paths

These verify invalid configurations fail fast. Only need one session.

### C1. `space_id` non-existent

**Config**: `space_id="00000000-0000-0000-0000-000000000000"` (valid UUID format, does not exist).

**Expected**: `ValueError` raised with message containing "not found" and "must already exist".

**Plugin**: Wrap the session run in `pytest.raises(ValueError)`.
**Tools**: Check that the tool returns an error string (tools return error messages instead of raising).

### C2. `space_id` + `space_name` mismatch

**Setup**: Pre-create a space with name "alpha". Record its `space_id`.

**Config**: `space_id=<alpha's id>`, `space_name="beta"`.

**Expected**: `ValueError` raised with message about mismatch.

---

## Group D: Embedder Resolution

### D1. `embedder_id` set (valid)

**Setup**: Use `GoodmemClient` to list existing embedders and pick the first one's ID.

| Session | Config | User message |
|---------|--------|-------------|
| 1 | `embedder_id=<valid>`, `space_name=<unique>` | "I am a goldfish" |
| 2 | `embedder_id=<valid>`, `space_name=<unique>` | "Do I live in water?" |

**Assertions**: No errors. Session 2 recalls the goldfish fact. Verifies that pinning a specific embedder works.

### D2. `embedder_id` non-existent

**Config**: `embedder_id="00000000-0000-0000-0000-000000000000"`.

**Expected**: `ValueError` raised with message about embedder not found.

---

## Group E: Env Var Fallback — Smoke Tests

These specifically test that env vars are picked up when constructor parameters are not provided.

### E1. `GOODMEM_SPACE_NAME` env var

**Setup**: Use `monkeypatch.setenv("GOODMEM_SPACE_NAME", "envvar_name_xyz")`.

**Config**: Construct plugin/tool with **no** `space_id` or `space_name` params.

| Session | User message |
|---------|-------------|
| 1 | "I am a goldfish" |
| 2 | "Do I live in water?" |

**Assertions**: Session 2 recalls the goldfish fact. The space name used matches the env var value (verify via `_find_space_id`).

### E2. `GOODMEM_SPACE_ID` env var

**Setup**: Pre-create a space. Use `monkeypatch.setenv("GOODMEM_SPACE_ID", "<pre-created id>")`.

**Config**: Construct plugin/tool with **no** `space_id` or `space_name` params.

| Session | User message |
|---------|-------------|
| 1 | "I am a goldfish" |
| 2 | "Do I live in water?" |

**Assertions**: Session 2 recalls the goldfish fact. Verifies the env var is picked up correctly.

---

## Component Coverage

Most tests use `GoodmemPlugin` since the space/embedder resolution logic in the plugin and tools ultimately delegates to the same `GoodmemClient` methods. Testing one path is sufficient to verify the resolution logic.

The **cross-component tests** (B2, B3) specifically use `GoodmemPlugin` for session 1 (write) and `GoodmemFetchTool` for session 2 (read), verifying that plugin and tools can interoperate on the same space.

Error tests (C1, C2, D2) use only `GoodmemPlugin`. Note that ADK's `PluginManager` wraps plugin exceptions in `RuntimeError`, so these tests catch `RuntimeError` rather than the underlying `ValueError`.

## Summary

| Group | Test | Component | space_id | space_name | embedder_id | Sessions | Expected |
|-------|------|-----------|----------|------------|-------------|----------|----------|
| A | A2 | Plugin | pre-made | - | - | 2 | recall OK |
| A | A3 | Plugin | pre-made | matching | - | 2 | recall OK |
| B | B1 | Plugin | (sess 2) | (sess 1) | - | 2 | recall OK |
| B | B2 | Plugin + FetchTool | - | same | - | 2 | plugin->tools OK |
| B | B3 | Plugin + FetchTool | (sess 2) | (sess 1) | - | 2 | plugin->tools OK |
| C | C1 | Plugin | bogus | - | - | 1 | RuntimeError |
| C | C2 | Plugin | pre-made | wrong | - | 1 | RuntimeError |
| D | D1 | Plugin | - | - | valid | 2 | recall OK |
| D | D2 | Plugin | - | - | bogus | 1 | RuntimeError |
| E | E1 | Plugin | - | env var | - | 2 | recall OK |
| E | E2 | Plugin | env var | - | - | 2 | recall OK |

**Total**: 11 tests (7 happy paths with 2 sessions, 4 error paths with 1 session).
