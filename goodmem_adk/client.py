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

"""Goodmem API client for interacting with Goodmem.ai.

Lives under plugins/goodmem and is shared: used by GoodmemPlugin and
re-exported for use by tools (goodmem_save, goodmem_fetch). Uses httpx for
HTTP calls.
"""

import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx


class GoodmemClient:
  """Client for interacting with the Goodmem API.

  Attributes:
    _base_url: The base URL for the Goodmem API.
    _api_key: The API key for authentication.
    _headers: HTTP headers for API requests.
  """

  def __init__(self, base_url: str, api_key: str, debug: bool = False) -> None:
    """Initializes the Goodmem client.

    Args:
      base_url: The base URL for the Goodmem API, without the /v1 suffix
        (e.g., "https://api.goodmem.ai").
      api_key: The Goodmem API key for authentication.
      debug: Whether to enable debug mode.
    """
    self._base_url = base_url.rstrip("/")
    self._api_key = api_key.strip()
    self._headers = {"x-api-key": self._api_key}
    self._debug = debug
    self._client = httpx.Client(
        base_url=self._base_url,
        headers=self._headers,
        timeout=30.0,
    )

  def close(self) -> None:
    """Closes the underlying HTTP client."""
    self._client.close()

  def __enter__(self) -> "GoodmemClient":
    return self

  def __exit__(self, *args: Any) -> None:
    self.close()

  def _safe_json_dumps(self, value: Any) -> str:
    try:
      return json.dumps(value, indent=2)
    except (TypeError, ValueError):
      return f"<non-serializable: {type(value).__name__}>"

  def get_space(self, space_id: str) -> Optional[Dict[str, Any]]:
    """Gets a space by its ID.

    Args:
      space_id: The ID of the space to retrieve.

    Returns:
      The space object, or ``None`` if the space does not exist (404).

    Raises:
      httpx.HTTPStatusError: If the API request fails with a non-404 error.
      httpx.RequestError: If the request fails (e.g. connection, timeout).
    """
    encoded_space_id = quote(space_id, safe="")
    url = f"/v1/spaces/{encoded_space_id}"
    response = self._client.get(url, timeout=30.0)
    if response.status_code == 404:
      return None
    response.raise_for_status()
    return response.json()

  def create_space(
      self,
      space_name: str,
      embedder_id: str,
      space_id: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Creates a new Goodmem space.

    Args:
      space_name: The name of the space to create.
      embedder_id: The embedder ID to use for the space.
      space_id: Optional UUID to assign to the new space. If not provided,
        the server generates one.

    Returns:
      The response JSON containing spaceId.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails (e.g. connection, timeout).
    """
    url = "/v1/spaces"
    payload: Dict[str, Any] = {
        "name": space_name,
        "spaceEmbedders": [
            {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}
        ],
        "defaultChunkingConfig": {
            "recursive": {
                "chunkSize": 512,
                "chunkOverlap": 64,
                "keepStrategy": "KEEP_END",
                "lengthMeasurement": "CHARACTER_COUNT",
            }
        },
    }
    if space_id is not None:
      payload["spaceId"] = space_id
    response = self._client.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()

  def insert_memory(
      self,
      space_id: str,
      content: str,
      content_type: str = "text/plain",
      metadata: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Inserts a text memory into a Goodmem space.

    Args:
      space_id: The ID of the space to insert into.
      content: The content of the memory.
      content_type: The content type (default: text/plain).
      metadata: Optional metadata dict (e.g., session_id, user_id).

    Returns:
      The response JSON containing memoryId and processingStatus.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    url = "/v1/memories"
    payload: Dict[str, Any] = {
        "spaceId": space_id,
        "originalContent": content,
        "contentType": content_type,
    }
    if metadata:
      payload["metadata"] = metadata
    response = self._client.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()

  def insert_memory_binary(
      self,
      space_id: str,
      content_bytes: bytes,
      content_type: str,
      metadata: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Inserts a binary memory into a Goodmem space using multipart upload.

    Args:
      space_id: The ID of the space to insert into.
      content_bytes: The raw binary content as bytes.
      content_type: The MIME type (e.g., application/pdf, image/png).
      metadata: Optional metadata dict (e.g., session_id, user_id, filename).

    Returns:
      The response JSON containing memoryId and processingStatus.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    url = "/v1/memories"

    if self._debug:
      print("[DEBUG] insert_memory_binary called:")
      print(f"  - space_id: {space_id}")
      print(f"  - content_type: {content_type}")
      print(f"  - content_bytes length: {len(content_bytes)} bytes")
      if metadata:
        print(f"  - metadata:\n{self._safe_json_dumps(metadata)}")

    request_data: Dict[str, Any] = {
        "spaceId": space_id,
        "contentType": content_type,
    }
    if metadata:
      request_data["metadata"] = metadata

    if self._debug:
      print(f"[DEBUG] request_data:\n{self._safe_json_dumps(request_data)}")

    data = {"request": json.dumps(request_data)}
    files = {"file": ("upload", content_bytes, content_type)}

    if self._debug:
      print(f"[DEBUG] Making POST request to {url}")
    response = self._client.post(
        url,
        data=data,
        files=files,
        timeout=120.0,
    )
    if self._debug:
      print(f"[DEBUG] Response status: {response.status_code}")

    response.raise_for_status()
    result = response.json()
    if self._debug:
      print(f"[DEBUG] Response:\n{self._safe_json_dumps(result)}")
    return result

  def retrieve_memories(
      self,
      query: str,
      space_ids: List[str],
      request_size: int = 5,
  ) -> List[Dict[str, Any]]:
    """Searches for chunks matching a query in given spaces.

    Args:
      query: The search query message.
      space_ids: List of space IDs to search in.
      request_size: The number of chunks to retrieve.

    Returns:
      List of matching chunks (parsed from NDJSON response).

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    url = "/v1/memories:retrieve"
    headers = {**self._headers, "Accept": "application/x-ndjson"}
    payload = {
        "message": query,
        "spaceKeys": [{"spaceId": sid} for sid in space_ids],
        "requestedSize": request_size,
    }

    response = self._client.post(
        url, json=payload, headers=headers, timeout=30.0
    )
    response.raise_for_status()

    chunks: List[Dict[str, Any]] = []
    for line in response.text.strip().split("\n"):
      if line.strip():
        try:
          tmp_dict = json.loads(line)
          if "retrievedItem" in tmp_dict:
            chunks.append(tmp_dict)
        except json.JSONDecodeError:
          continue
    return chunks

  def list_spaces(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists spaces, optionally filtering by name.

    Returns:
      List of spaces (optionally filtered by name).

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    url = "/v1/spaces"
    all_spaces: List[Dict[str, Any]] = []
    next_token: Optional[str] = None
    max_results = 1000

    while True:
      params: Dict[str, Any] = {"maxResults": max_results}
      if next_token:
        params["nextToken"] = next_token
      if name:
        params["nameFilter"] = name

      response = self._client.get(url, params=params, timeout=30.0)
      response.raise_for_status()

      data = response.json()
      spaces = data.get("spaces", [])
      all_spaces.extend(spaces)

      next_token = data.get("nextToken")
      if not next_token:
        break

    return all_spaces

  def list_embedders(self) -> List[Dict[str, Any]]:
    """Lists all embedders.

    Returns:
      List of embedders.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    url = "/v1/embedders"
    response = self._client.get(url, timeout=30.0)
    response.raise_for_status()
    return response.json().get("embedders", [])

  def create_embedder(
      self,
      display_name: str,
      provider_type: str,
      endpoint_url: str,
      model_identifier: str,
      dimensionality: int,
      api_key: str,
      distribution_type: str = "DENSE",
      embedder_id: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Creates a new embedder.

    Args:
      display_name: Human-readable name for the embedder.
      provider_type: Provider type (e.g., "OPENAI").
      endpoint_url: The endpoint URL for the embedding API.
      model_identifier: The model identifier (e.g., "gemini-embedding-001").
      dimensionality: The embedding vector dimensionality.
      api_key: The API key for the embedding provider.
      distribution_type: The distribution type (default: "DENSE").
      embedder_id: Optional UUID to assign to the new embedder. If not
        provided, the server generates one.

    Returns:
      The response JSON containing embedderId.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    url = "/v1/embedders"
    payload: Dict[str, Any] = {
        "displayName": display_name,
        "providerType": provider_type,
        "endpointUrl": endpoint_url,
        "modelIdentifier": model_identifier,
        "dimensionality": dimensionality,
        "distributionType": distribution_type,
        "credentials": {
            "kind": "CREDENTIAL_KIND_API_KEY",
            "apiKey": {
                "inlineSecret": api_key,
            },
        },
    }
    if embedder_id is not None:
      payload["embedderId"] = embedder_id
    response = self._client.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()

  def get_memory_by_id(self, memory_id: str) -> Dict[str, Any]:
    """Gets a memory by its ID.

    Args:
      memory_id: The ID of the memory to retrieve.

    Returns:
      The memory object including metadata, contentType, etc.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    encoded_memory_id = quote(memory_id, safe="")
    url = f"/v1/memories/{encoded_memory_id}"
    response = self._client.get(url, timeout=30.0)
    response.raise_for_status()
    return response.json()

  def get_memories_batch(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
    """Gets multiple memories by ID in a single request (batch get).

    Uses POST /v1/memories:batchGet to avoid N+1 queries when enriching
    many chunks with full memory metadata.

    Args:
      memory_ids: List of memory IDs to fetch.

    Returns:
      List of memory objects (same shape as get_memory_by_id). Order and
      presence may not match request; missing or failed IDs are omitted.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    if not memory_ids:
      return []
    url = "/v1/memories:batchGet"
    payload = {"memoryIds": list(memory_ids)}
    response = self._client.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    return data.get("memories", [])

  # -- embedder helpers ------------------------------------------------------

  # Default Google embedder configuration
  _GOOGLE_EMBEDDER_DISPLAY_NAME = "gemini-embedding-001"
  _GOOGLE_EMBEDDER_PROVIDER_TYPE = "OPENAI"
  _GOOGLE_EMBEDDER_ENDPOINT_URL = (
      "https://generativelanguage.googleapis.com/v1beta/openai"
  )
  _GOOGLE_EMBEDDER_MODEL_ID = "gemini-embedding-001"
  _GOOGLE_EMBEDDER_DIMENSIONALITY = 1536
  _GOOGLE_EMBEDDER_DISTRIBUTION_TYPE = "DENSE"

  def ensure_embedder(
      self,
      embedder_id: Optional[str] = None,
      debug: bool = False,
  ) -> str:
    """Return a valid embedder ID, creating a Google embedder if needed.

    Resolution order:
    1. If ``embedder_id`` is provided, it must exist — ``ValueError``
       if not found.
    2. If ``embedder_id`` is not provided and embedders already exist,
       return the first available one.
    3. If ``embedder_id`` is not provided and no embedders exist,
       auto-create a ``gemini-embedding-001`` embedder using
       ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``).

    Args:
      embedder_id: Optional embedder ID that must already exist.
      debug: Whether to print debug messages.

    Returns:
      A valid embedder ID.

    Raises:
      ValueError: If ``embedder_id`` is set but not found, or if no
        embedders can be resolved and neither ``GOOGLE_API_KEY`` nor
        ``GEMINI_API_KEY`` is set.
    """
    embedders = self.list_embedders()

    if embedder_id is not None:
      valid_ids = [e.get("embedderId") for e in embedders]
      if embedder_id in valid_ids:
        return embedder_id
      raise ValueError(
          f"GOODMEM_EMBEDDER_ID '{embedder_id}' not found. "
          f"Available embedders: {valid_ids}"
      )

    if embedders:
      eid = embedders[0].get("embedderId")
      if eid:
        if debug:
          print(f"[DEBUG] Using existing embedder: {eid}")
        return eid

    # No embedders at all — auto-create with server-generated ID
    if debug:
      print(
          "[DEBUG] No embedders found. Auto-creating Google Gemini embedder "
          f"({self._GOOGLE_EMBEDDER_MODEL_ID}) using GOOGLE_API_KEY"
      )
    return self._auto_create_google_embedder(debug=debug)

  def _auto_create_google_embedder(
      self,
      debug: bool = False,
  ) -> str:
    """Create a ``gemini-embedding-001`` embedder using ``GOOGLE_API_KEY``.

    Args:
      debug: Whether to print debug messages.

    Returns:
      The newly created embedder ID.

    Raises:
      ValueError: If neither ``GOOGLE_API_KEY`` nor ``GEMINI_API_KEY`` is
        set.
    """
    google_api_key = (
        os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    )
    if not google_api_key:
      raise ValueError(
          "No embedders available in Goodmem and neither GOOGLE_API_KEY "
          "nor GEMINI_API_KEY is set. Please create an embedder manually "
          "or set one of these environment variables to auto-create a "
          "Google Gemini embedder."
      )

    response = self.create_embedder(
        display_name=self._GOOGLE_EMBEDDER_DISPLAY_NAME,
        provider_type=self._GOOGLE_EMBEDDER_PROVIDER_TYPE,
        endpoint_url=self._GOOGLE_EMBEDDER_ENDPOINT_URL,
        model_identifier=self._GOOGLE_EMBEDDER_MODEL_ID,
        dimensionality=self._GOOGLE_EMBEDDER_DIMENSIONALITY,
        api_key=google_api_key,
        distribution_type=self._GOOGLE_EMBEDDER_DISTRIBUTION_TYPE,
    )

    new_id = response.get("embedderId")
    if not new_id:
      raise ValueError(
          "Failed to auto-create Google embedder: no embedderId in response"
      )

    if debug:
      print(f"[DEBUG] Auto-created Google embedder: {new_id}")

    return new_id
