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

"""Service registration for Goodmem ADK integration.

NOTE: GoodmemMemoryService is currently not exposed in the public API.
This file is just here for reference.
"""

# Commented out: GoodmemMemoryService is not yet exposed in the public API.
# 
# import os
# from google.adk.cli.service_registry import get_service_registry
# from goodmem_adk import GoodmemMemoryService
#
# def _goodmem_factory(uri: str, **kwargs):
#     return GoodmemMemoryService(
#         # mandatory parameters
#         base_url=os.getenv("GOODMEM_BASE_URL"),
#         api_key=os.getenv("GOODMEM_API_KEY"),
#         # optional parameters
#         embedder_id=os.getenv("GOODMEM_EMBEDDER_ID"),
#         space_id=os.getenv("GOODMEM_SPACE_ID"),
#         space_name=os.getenv("GOODMEM_SPACE_NAME"),
#         top_k=5,
#         timeout=30.0,
#         split_turn=True,
#         debug=False,
#     )
#
# get_service_registry().register_memory_service("goodmem", _goodmem_factory)
