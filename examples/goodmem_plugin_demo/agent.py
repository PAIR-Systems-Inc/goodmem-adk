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

"""Example agent using the Goodmem Chat Plugin.

For usage instructions, see ../README.md.
"""

import os

from google.adk.agents import LlmAgent
from google.adk.apps import App
from goodmem_adk import GoodmemPlugin

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)

goodmem_chat_plugin = GoodmemPlugin(
    # mandatory parameters
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
    # optional parameters
    embedder_id=os.getenv("GOODMEM_EMBEDDER_ID"), # pin an embedder; must exist
    space_id=os.getenv("GOODMEM_SPACE_ID"), # pin a space by ID; must exist
    space_name=os.getenv("GOODMEM_SPACE_NAME"), # override the default space name; auto-creates if not found
    top_k=5,
    debug=False
)

app = App(
    name="goodmem_plugin_demo",
    root_agent=root_agent,
    plugins=[goodmem_chat_plugin]
)
