# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
# SPDX-License-Identifier: Apache-2.0

"""Shared attachment validation and SDK writes for ADK content."""

import base64
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from goodmem.api.memories import AsyncMemoriesAPI
from goodmem.models.memory import Memory
from google.genai import types

from ._results import SaveFailure


@dataclass(frozen=True)
class Attachment:
    """Validated inline bytes, retaining their original position for retry tracking."""

    part_index: int
    filename: str
    data: bytes
    content_type: str


def attachments_from_content(content: types.Content | None) -> Iterator[Attachment | SaveFailure]:
    """Yield inline attachments or explicit failures; never download a supplied URI."""
    number = 0
    for index, part in enumerate((content.parts or []) if content else []):
        if part.inline_data is not None:
            number += 1
            blob = part.inline_data
            filename = blob.display_name or f"attachment_{number}"
            if blob.data:
                yield Attachment(
                    index, filename, blob.data, blob.mime_type or "application/octet-stream"
                )
            else:
                yield SaveFailure(filename=filename, message="Empty attachment")
        if part.file_data is not None:
            yield SaveFailure(
                filename=part.file_data.display_name,
                message="URI attachments are not uploaded. Supply inline bytes from the application.",
            )


async def write_attachment(
    memories: AsyncMemoriesAPI,
    attachment: Attachment,
    *,
    space_id: str,
    metadata: dict[str, Any],
) -> Memory:
    """Encode validated bytes and submit the same SDK request from every ADK adapter."""
    return await memories.create(
        space_id=space_id,
        original_content_b64=base64.b64encode(attachment.data).decode("ascii"),
        content_type=attachment.content_type,
        metadata={**metadata, "filename": attachment.filename},
    )
