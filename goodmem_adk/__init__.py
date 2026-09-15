# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
# SPDX-License-Identifier: Apache-2.0

"""Persistent memory for Google ADK, using the official GoodMem SDK."""

from ._results import (
    AcceptedMemory,
    GoodmemFetchResponse,
    GoodmemSaveError,
    GoodmemSaveResponse,
    MemoryItem,
    RetrievalStatus,
    SaveFailure,
)
from .plugin import GoodmemPlugin
from .tools import GoodmemFetchTool, GoodmemSaveTool

__version__ = "0.2.0"

__all__ = [
    "AcceptedMemory",
    "GoodmemFetchResponse",
    "GoodmemFetchTool",
    "GoodmemPlugin",
    "GoodmemSaveError",
    "GoodmemSaveResponse",
    "GoodmemSaveTool",
    "MemoryItem",
    "RetrievalStatus",
    "SaveFailure",
]
