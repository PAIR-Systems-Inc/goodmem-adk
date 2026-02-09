# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Goodmem ADK Plugin — persistent memory for Google ADK agents.

Provides three integration points:

- **Plugin**: ``GoodmemChatPlugin`` — automatic memory interception via ADK callbacks
- **Tools**: ``GoodmemSaveTool`` / ``GoodmemFetchTool`` — explicit agent memory management
- **Memory Service**: ``GoodmemMemoryService`` — full session-based persistent memory
"""

__version__ = "0.1.0"

from .client import GoodmemClient
from .plugin import GoodmemChatPlugin
from .tools import (
    GoodmemFetchResponse,
    GoodmemFetchTool,
    GoodmemSaveResponse,
    GoodmemSaveTool,
    MemoryItem,
    goodmem_fetch,
    goodmem_save,
)
from .memory import (
    GoodmemMemoryService,
    GoodmemMemoryServiceConfig,
    format_memory_block_for_prompt,
)

__all__ = [
    "GoodmemClient",
    "GoodmemChatPlugin",
    "GoodmemSaveTool",
    "GoodmemFetchTool",
    "GoodmemSaveResponse",
    "GoodmemFetchResponse",
    "MemoryItem",
    "goodmem_save",
    "goodmem_fetch",
    "GoodmemMemoryService",
    "GoodmemMemoryServiceConfig",
    "format_memory_block_for_prompt",
]
