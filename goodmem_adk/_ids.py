# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
# SPDX-License-Identifier: Apache-2.0

"""The single check applied to GoodMem resource IDs before any request is made."""

import re

# GoodMem IDs are UUIDs. The SDK places path IDs into URLs as given and httpx resolves
# dot segments, so a value such as "../embedders/<id>" would address another resource.
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def require_uuid(value: object, field: str) -> str:
    """Return the canonical lowercase form of a UUID, refusing anything else.

    Args:
        value: The ID as supplied by a caller or the environment.
        field: The setting name reported when the value is refused.

    Raises:
        ValueError: The value is not exactly a UUID; no request may use it.
    """
    if not isinstance(value, str) or _UUID.fullmatch(value) is None:
        raise ValueError(f"{field} must be a UUID (GoodMem IDs are UUIDs); got {value!r}")
    return value.lower()
