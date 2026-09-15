"""Audit helpers: real ADK runners, synthetic facts, and recorded HTTP traffic."""

import json
import uuid

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import Field


def unique(prefix="adk_audit"):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def append_evidence(path, evidence):
    with open(path, "a") as file:
        file.write(all_text(evidence) + "\n")


def text_content(text):
    return types.Content(role="user", parts=[types.Part(text=text)])


def all_text(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        default=lambda item: (
            item.model_dump(mode="json", exclude_none=True)
            if hasattr(item, "model_dump")
            else str(item)
        ),
    )


class RecordingModel(BaseLlm):
    """Deterministic model boundary; ADK callbacks and tools execute normally.

    A SAVE/FETCH directive causes one real ADK tool call. Otherwise the model
    acknowledges the message. Requests are copied so tests inspect exactly what
    reached the model, rather than accepting a plausible generated answer.
    """

    model: str = "audit-recording-model"
    requests: list = Field(default_factory=list)

    async def generate_content_async(self, llm_request, stream=False):
        self.requests.append(llm_request.model_copy(deep=True))
        last = llm_request.contents[-1]
        if any(part.function_response for part in (last.parts or [])):
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="Tool response received.")]
                )
            )
            return
        text = " ".join(part.text for part in (last.parts or []) if part.text)
        for directive, name, argument in [
            ("SAVE:", "goodmem_save", "content"),
            ("FETCH:", "goodmem_fetch", "query"),
            ("LOAD:", "load_memory", "query"),
        ]:
            if text.startswith(directive):
                # Plugin-added memory is context, not an additional directive.
                value = text[len(directive) :].split("\n\nBEGIN MEMORY")[0].strip()
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(name=name, args={argument: value})
                            )
                        ],
                    )
                )
                return
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="Acknowledged.")])
        )


def runner_for(*, plugin=None, tools=None, model=None, app_name=None):
    model = model or RecordingModel()
    app = App(
        name=app_name or unique("auditapp"),
        root_agent=LlmAgent(name="audit_agent", model=model, tools=tools or []),
        plugins=[plugin] if plugin else [],
    )
    return InMemoryRunner(app=app), model


async def turn(runner, user_id, message, session_id=None):
    if session_id is None:
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id=user_id
        )
        session_id = session.id
    events = [
        event
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=text_content(message) if isinstance(message, str) else message,
        )
    ]
    return session_id, events


def responses(events, name):
    found = [
        part.function_response.response
        for event in events
        if event.content
        for part in (event.content.parts or [])
        if part.function_response and part.function_response.name == name
    ]
    # ADK wraps non-dict FunctionTool results in {"result": ...}.
    normalized = []
    for result in found:
        if set(result) == {"result"}:
            result = result["result"]
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")
        normalized.append(result)
    return normalized
