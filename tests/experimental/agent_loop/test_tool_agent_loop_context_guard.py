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

import pytest

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.workers.rollout.replica import TokenOutput


class DummyServerManager:
    def __init__(self, extra_fields=None):
        self.called = False
        self.extra_fields = extra_fields or {}

    async def generate(self, **kwargs):
        self.called = True
        return TokenOutput(token_ids=[42], log_probs=[-0.1], extra_fields=self.extra_fields)


class DummyToolParser:
    async def extract_tool_calls(self, response_ids, tools):
        return None, []


def _make_tool_agent_loop(max_model_len: int | None, extra_fields=None) -> ToolAgentLoop:
    loop = object.__new__(ToolAgentLoop)
    loop.max_model_len = max_model_len
    loop.response_length = 16
    loop.max_assistant_turns = None
    loop.max_user_turns = None
    loop.turn_limit_schedule = []
    loop.server_manager = DummyServerManager(extra_fields=extra_fields)
    loop.tools = {}
    loop.tool_parser = DummyToolParser()
    return loop


def _make_agent_data(prompt_len: int, response_mask_len: int = 3) -> AgentData:
    agent_data = AgentData(
        messages=[],
        image_data=None,
        video_data=None,
        metrics={},
        request_id="test-request",
        tools_kwargs={},
    )
    agent_data.prompt_ids = list(range(prompt_len))
    agent_data.response_mask = [1] * response_mask_len
    return agent_data


def test_tool_agent_loop_terminates_before_generation_when_prompt_near_max_model_len():
    loop = _make_tool_agent_loop(max_model_len=10)
    agent_data = _make_agent_data(prompt_len=9)

    state = asyncio.run(loop._handle_generating_state(agent_data, sampling_params={}))

    assert state == AgentState.TERMINATED
    assert loop.server_manager.called is False
    assert agent_data.response_mask == [0, 0, 0]
    assert agent_data.abnormal_trajectory_dic["too_long_seq_truncated_count"] == 1


def test_tool_agent_loop_still_generates_when_prompt_has_context_budget():
    loop = _make_tool_agent_loop(max_model_len=10)
    agent_data = _make_agent_data(prompt_len=8)

    state = asyncio.run(loop._handle_generating_state(agent_data, sampling_params={}))

    assert state == AgentState.TERMINATED
    assert loop.server_manager.called is True
    assert agent_data.prompt_ids[-1] == 42
    assert agent_data.response_mask[-1] == 1
    assert agent_data.abnormal_trajectory_dic["too_long_seq_truncated_count"] == 0


def test_parse_turn_limit_schedule_sorts_and_validates_milestones():
    assert ToolAgentLoop._parse_turn_limit_schedule("10:64,0:16,5:32") == [(0, 16), (5, 32), (10, 64)]

    with pytest.raises(ValueError, match="expected '<step>:<turn_limit>'"):
        ToolAgentLoop._parse_turn_limit_schedule("0:16,bad")
    with pytest.raises(ValueError, match="non-negative"):
        ToolAgentLoop._parse_turn_limit_schedule("-1:16")
    with pytest.raises(ValueError, match="positive"):
        ToolAgentLoop._parse_turn_limit_schedule("0:0")
    with pytest.raises(ValueError, match="duplicate step"):
        ToolAgentLoop._parse_turn_limit_schedule("0:16,0:32")


def test_effective_turn_limits_use_schedule_and_static_caps():
    loop = _make_tool_agent_loop(max_model_len=None)
    loop.turn_limit_schedule = ToolAgentLoop._parse_turn_limit_schedule("0:16,5:32,10:64,20:100")
    loop.max_assistant_turns = 80
    loop.max_user_turns = 100
    agent_data = _make_agent_data(prompt_len=1)

    assert loop._effective_turn_limits(agent_data, {"global_steps": 0}) == (16, 16)
    assert loop._effective_turn_limits(agent_data, {"global_steps": 5}) == (32, 32)
    assert loop._effective_turn_limits(agent_data, {"global_steps": 10}) == (64, 64)
    assert loop._effective_turn_limits(agent_data, {"global_steps": 20}) == (80, 100)


def test_effective_turn_limits_fall_back_to_static_limits_without_step_or_schedule():
    loop = _make_tool_agent_loop(max_model_len=None)
    loop.max_assistant_turns = 12
    loop.max_user_turns = 14
    agent_data = _make_agent_data(prompt_len=1)

    assert loop._effective_turn_limits(agent_data, {}) == (12, 14)


def test_all_search_results_are_duplicate_requires_every_current_query_to_match_history():
    loop = _make_tool_agent_loop(max_model_len=None)
    historical_signatures = [{"doc a", "doc b"}, {"doc c", "doc d"}]

    current_signatures = [{"doc a", "doc b"}, {"doc e", "doc f"}]

    assert (
        loop._all_search_results_are_duplicate(
            current_signatures,
            historical_signatures,
            overlap_threshold=1.0,
        )
        is False
    )


def test_all_search_results_are_duplicate_returns_true_when_all_current_queries_match_history():
    loop = _make_tool_agent_loop(max_model_len=None)
    historical_signatures = [{"doc a", "doc b"}, {"doc c", "doc d"}]

    current_signatures = [{"doc a", "doc b"}, {"doc c", "doc d"}]

    assert (
        loop._all_search_results_are_duplicate(
            current_signatures,
            historical_signatures,
            overlap_threshold=1.0,
        )
        is True
    )


def test_all_search_results_are_duplicate_returns_false_without_current_or_history():
    loop = _make_tool_agent_loop(max_model_len=None)

    assert loop._all_search_results_are_duplicate([], [{"doc a"}], overlap_threshold=1.0) is False
    assert loop._all_search_results_are_duplicate([{"doc a"}], [], overlap_threshold=1.0) is False


def test_all_search_results_are_duplicate_uses_overlap_threshold_inclusively():
    loop = _make_tool_agent_loop(max_model_len=None)
    historical_signatures = [{"doc a", "doc b", "doc c"}]

    current_signatures = [{"doc a", "doc b", "doc d"}]

    assert (
        loop._all_search_results_are_duplicate(
            current_signatures,
            historical_signatures,
            overlap_threshold=2 / 3,
        )
        is True
    )


def test_tool_agent_loop_terminates_at_scheduled_assistant_turn_limit():
    loop = _make_tool_agent_loop(max_model_len=None, extra_fields={"global_steps": 5})
    loop.response_length = 64
    loop.max_assistant_turns = 100
    loop.max_user_turns = 100
    loop.turn_limit_schedule = ToolAgentLoop._parse_turn_limit_schedule("0:16,5:32,10:64,20:100")
    agent_data = _make_agent_data(prompt_len=1, response_mask_len=0)
    agent_data.assistant_turns = 31

    state = asyncio.run(loop._handle_generating_state(agent_data, sampling_params={}))

    assert state == AgentState.TERMINATED
    assert loop.server_manager.called is True
    assert agent_data.assistant_turns == 32
    assert agent_data.response_mask == [0]
    assert agent_data.abnormal_trajectory_dic["too_many_turn_count"] == 1
