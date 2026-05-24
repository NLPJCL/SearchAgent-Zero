# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import asyncio
import json

from verl.experimental.agent_loop.tool_agent_loop_credit_assignment import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.workers.rollout.replica import TokenOutput


class DummyServerManager:
    async def generate(self, **kwargs):
        return TokenOutput(token_ids=[42, 43], log_probs=[-0.1, -0.2], extra_fields={})


class DummyToolParser:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls

    async def extract_tool_calls(self, response_ids, tools):
        return None, self.tool_calls


def _make_tool_agent_loop(tool_calls, max_queries_per_tool_call: int | None = None) -> ToolAgentLoop:
    loop = object.__new__(ToolAgentLoop)
    loop.max_model_len = None
    loop.response_length = 16
    loop.max_assistant_turns = None
    loop.max_user_turns = None
    loop.turn_limit_schedule = []
    loop.server_manager = DummyServerManager()
    loop.tools = {}
    loop.tool_parser = DummyToolParser(tool_calls)
    loop.max_queries_per_tool_call = max_queries_per_tool_call
    return loop


def _make_agent_data() -> AgentData:
    agent_data = AgentData(
        messages=[],
        image_data=None,
        video_data=None,
        metrics={},
        request_id="test-request",
        tools_kwargs={},
    )
    agent_data.prompt_ids = [1, 2, 3]
    agent_data.response_mask = [1, 0, 1]
    return agent_data


def _search_call(arguments: dict | str) -> FunctionCall:
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return FunctionCall(name="search", arguments=arguments)


def test_tool_parser_error_masks_previous_turns_only():
    loop = _make_tool_agent_loop([_search_call({"query_list": "not-a-list"})])
    agent_data = _make_agent_data()

    state = asyncio.run(loop._handle_generating_state(agent_data, sampling_params={}))

    assert state == AgentState.TERMINATED
    assert agent_data.abnormal_trajectory_dic["tool_parser_error_count"] == 1
    assert agent_data.response_mask == [0, 0, 0, 1, 1]


def test_too_many_tool_calls_masks_previous_turns_only():
    loop = _make_tool_agent_loop(
        [_search_call({"query_list": ["first query", "second query"]})],
        max_queries_per_tool_call=1,
    )
    agent_data = _make_agent_data()

    state = asyncio.run(loop._handle_generating_state(agent_data, sampling_params={}))

    assert state == AgentState.TERMINATED
    assert agent_data.abnormal_trajectory_dic["too_many_tool_call_count"] == 1
    assert agent_data.response_mask == [0, 0, 0, 1, 1]


def test_repeated_search_query_masks_previous_turns_only():
    loop = _make_tool_agent_loop([_search_call({"query_list": ["Repeated Query"]})])
    agent_data = _make_agent_data()
    agent_data.searched_query.add("repeated query")

    state = asyncio.run(loop._handle_generating_state(agent_data, sampling_params={}))

    assert state == AgentState.TERMINATED
    assert agent_data.abnormal_trajectory_dic["searched_query_count"] == 1
    assert agent_data.response_mask == [0, 0, 0, 1, 1]
