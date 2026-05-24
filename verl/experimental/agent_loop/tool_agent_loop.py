# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from enum import Enum
from typing import Any
from uuid import uuid4
import re
import random
import string

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


PROMPT_TEMPLATE = """
**TASK:**
Analyze the **[Retrieved Documents]** to generate a summary that answers the **[Current Query]**. You must separate your output into a Reasoning section and a Summary section.

**INSTRUCTIONS:**
1.  **Analyze Relevance:** First, determine if the **[Retrieved Documents]** actually contain information relevant to the **[Current Query]**.
2.  **Handle Insufficient Information:**
    - If the documents are irrelevant, empty, or do not provide enough context to answer the query, the **Summary** must explicitly state: "Information Insufficient".
    - Do not attempt to fabricate an answer or use outside knowledge not present in the documents.
3.  **Synthesize (If Relevant):**
    - If relevant information is found, extract key facts and synthesize them into a coherent summary.
    - Focus on factual accuracy based strictly on the provided documents.
4.  **Format:** Your output must use the exact headers below.

**INPUT DATA:**
- **[Current Query]:** {query}
- **[Retrieved Documents]:** {documents}

**OUTPUT FORMAT:**

## Reasoning
[Write your analysis here. Step 1: Evaluate if documents match the query. Step 2: Select key information or identify gaps.]

## Summary
[Write the final synthesized summary here. If documents are irrelevant, output "Information Insufficient".]
"""


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        question: str = "",
        ground_truth_list: list[str] = [],
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.ground_truth_list = ground_truth_list

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.tool_turns = 0
        self.all_call_tool_counts = 0
        self.all_call_tool_success_counts = 0
        #异常轨迹监督
        #重复query数量
        self.abnormal_trajectory_dic = {}
        self.abnormal_trajectory_dic['searched_query_count'] = 0
        self.abnormal_trajectory_dic['tool_parser_error_count'] = 0
        self.abnormal_trajectory_dic['too_many_tool_call_count'] = 0
        self.abnormal_trajectory_dic['too_many_turn_count'] = 0
        self.abnormal_trajectory_dic['response_truncated_count'] = 0
        self.abnormal_trajectory_dic['too_long_seq_truncated_count'] = 0
        self.abnormal_trajectory_dic['duplicate_search_result_count'] = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        self.tool_call_contexts: list[dict[str, Any]] = []
        self.question = question

        self.routed_experts = None

        #维护搜索过的query
        self.searched_query = set()
        #维护搜索结果签名，用于过滤换词但返回内容高度重合的搜索
        self.searched_result_signatures: list[set[str]] = []
        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.turn_limit_schedule = self._parse_turn_limit_schedule(
            self.rollout_config.multi_turn.turn_limit_schedule
        )
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.enable_tool_response_summary = self.rollout_config.multi_turn.enable_tool_response_summary
        self.summary_temperature = self.rollout_config.multi_turn.summary_temperature
        self.summary_top_p = self.rollout_config.multi_turn.summary_top_p
        self.summary_top_k = self.rollout_config.multi_turn.summary_top_k
        self.summary_max_tokens = self.rollout_config.multi_turn.summary_max_tokens
        self.summary_use_external_model = self.rollout_config.multi_turn.summary_use_external_model
        self.summary_external_base_urls = self.rollout_config.multi_turn.summary_external_base_urls
        self.summary_external_model = self.rollout_config.multi_turn.summary_external_model
        self.summary_external_api_key = self.rollout_config.multi_turn.summary_external_api_key
        self.summary_external_timeout = self.rollout_config.multi_turn.summary_external_timeout
        self.summary_external_enable_thinking = self.rollout_config.multi_turn.summary_external_enable_thinking
        self.summary_external_clients = []
        if self.summary_use_external_model:
            self._init_summary_external_clients()
        self.max_queries_per_tool_call = self.rollout_config.multi_turn.max_queries_per_tool_call
        self.duplicate_search_result_overlap_threshold = (
            self.rollout_config.multi_turn.duplicate_search_result_overlap_threshold
        )
        self.summary_result_separator = self.rollout_config.multi_turn.summary_result_separator.replace("\\n", "\n")
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_model_len = self.rollout_config.max_model_len

    @staticmethod
    def _parse_turn_limit_schedule(schedule: str | None) -> list[tuple[int, int]]:
        if not schedule:
            return []

        milestones: list[tuple[int, int]] = []
        for raw_item in schedule.split(","):
            item = raw_item.strip()
            if not item:
                continue
            try:
                step_text, limit_text = item.split(":", maxsplit=1)
                step = int(step_text.strip())
                limit = int(limit_text.strip())
            except ValueError as exc:
                raise ValueError(
                    "Invalid multi_turn.turn_limit_schedule item "
                    f"{item!r}; expected '<step>:<turn_limit>'."
                ) from exc

            if step < 0:
                raise ValueError("multi_turn.turn_limit_schedule steps must be non-negative.")
            if limit <= 0:
                raise ValueError("multi_turn.turn_limit_schedule turn limits must be positive.")
            milestones.append((step, limit))

        if not milestones:
            return []

        milestones.sort(key=lambda x: x[0])
        for idx in range(1, len(milestones)):
            if milestones[idx][0] == milestones[idx - 1][0]:
                raise ValueError(
                    "multi_turn.turn_limit_schedule contains duplicate step "
                    f"{milestones[idx][0]}."
                )
        return milestones

    def _resolve_turn_limit_step(
        self, agent_data: AgentData, output_extra_fields: dict[str, Any] | None = None
    ) -> int | None:
        output_extra_fields = output_extra_fields or {}
        for key in ("max_global_steps", "global_steps", "min_global_steps"):
            value = output_extra_fields.get(key)
            if value is not None:
                return int(value)
        for key in ("max_global_steps", "global_steps", "min_global_steps"):
            value = agent_data.extra_fields.get(key)
            if value is not None:
                return int(value)
        return None

    def _scheduled_turn_limit(self, step: int | None) -> int | None:
        if not self.turn_limit_schedule or step is None:
            return None

        active_limit = self.turn_limit_schedule[0][1]
        for milestone_step, limit in self.turn_limit_schedule:
            if step < milestone_step:
                break
            active_limit = limit
        return active_limit

    @staticmethod
    def _cap_turn_limit(dynamic_limit: int | None, static_limit: int | None) -> int | None:
        if dynamic_limit is None:
            return static_limit
        if static_limit is None:
            return dynamic_limit
        return min(dynamic_limit, static_limit)

    def _effective_turn_limits(
        self, agent_data: AgentData, output_extra_fields: dict[str, Any] | None = None
    ) -> tuple[int | None, int | None]:
        step = self._resolve_turn_limit_step(agent_data, output_extra_fields)
        dynamic_limit = self._scheduled_turn_limit(step)
        assistant_limit = self._cap_turn_limit(dynamic_limit, self.max_assistant_turns)
        user_limit = self._cap_turn_limit(dynamic_limit, self.max_user_turns)
        return assistant_limit, user_limit

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}
        question = extra_info.get("question", "")
        reward_model = kwargs.get("reward_model", {}) or {}
        ground_truth_list = list((reward_model.get("ground_truth", {}) or {}).get("target", []))
        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            question=question,
            ground_truth_list=ground_truth_list,
        )

        # Per-sample tool selection: filter global tools by extra_info.tool_selection
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            tool_turns=agent_data.tool_turns,
            all_call_tool_success_counts=agent_data.all_call_tool_success_counts,
            all_call_tool_counts=agent_data.all_call_tool_counts,
            abnormal_trajectory_dic=agent_data.abnormal_trajectory_dic,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        if self.max_model_len is not None and len(agent_data.prompt_ids) >= self.max_model_len - 1:
            logger.warning(
                "Terminating trajectory before generation because prompt length %s is near max_model_len %s. "
                "request_id=%s",
                len(agent_data.prompt_ids),
                self.max_model_len,
                agent_data.request_id,
            )
            agent_data.response_mask = [0] * len(agent_data.response_mask)
            agent_data.abnormal_trajectory_dic["too_long_seq_truncated_count"] += 1
            return AgentState.TERMINATED

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            for key in ("global_steps", "min_global_steps", "max_global_steps"):
                value = output.extra_fields.get(key, None)
                if value is not None:
                    agent_data.extra_fields[key] = value

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            agent_data.response_mask = [0]*len(agent_data.response_mask)
            agent_data.abnormal_trajectory_dic['too_long_seq_truncated_count'] +=1
            return AgentState.TERMINATED
        max_assistant_turns, max_user_turns = self._effective_turn_limits(agent_data, output.extra_fields)
        if max_assistant_turns and agent_data.assistant_turns >= max_assistant_turns:
            agent_data.response_mask = [0]*len(agent_data.response_mask)
            agent_data.abnormal_trajectory_dic['too_many_turn_count'] += 1
            return AgentState.TERMINATED
        if max_user_turns and agent_data.user_turns >= max_user_turns:
            agent_data.response_mask = [0]*len(agent_data.response_mask)
            agent_data.abnormal_trajectory_dic['too_many_turn_count'] += 1
            return AgentState.TERMINATED

        # Extract tool calls (use per-sample tools if routed)
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        if agent_data.tool_calls:
            try:
                agent_data.tool_call_contexts = []
                for tool_call in agent_data.tool_calls:
                    parsed_args = json.loads(tool_call.arguments)
                    agent_data.tool_call_contexts.append({"tool_call": tool_call, "parsed_args": parsed_args})
                    if tool_call.name != "search":
                        continue
                    query_list = parsed_args.get("query_list")
                    if not query_list or not isinstance(query_list, list):
                        agent_data.abnormal_trajectory_dic['tool_parser_error_count'] += 1
                        return AgentState.TERMINATED
                    else:
                        if self.max_queries_per_tool_call and len(query_list) > self.max_queries_per_tool_call:
                            agent_data.abnormal_trajectory_dic['too_many_tool_call_count'] += 1
                            return  AgentState.TERMINATED
                        for query in query_list:
                            query = normalize_answer(query)
                            if query in agent_data.searched_query:
                                agent_data.abnormal_trajectory_dic['searched_query_count'] += 1
                                return AgentState.TERMINATED
                            else:
                                agent_data.searched_query.add(query)
            except:
                agent_data.abnormal_trajectory_dic['tool_parser_error_count'] += 1
                return AgentState.TERMINATED
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED

    def _extract_search_result_signatures(self, tool_response_text: str) -> list[set[str]]:
        """Build per-query document signatures from the raw search tool response."""
        try:
            result_text = json.loads(tool_response_text).get("result", "")
        except Exception:
            return []

        if not isinstance(result_text, str) or not result_text:
            return []

        signatures: list[set[str]] = []
        result_blocks = result_text.split(self.summary_result_separator)
        for result_block in result_blocks:
            result_block = result_block.strip()
            if not result_block:
                continue

            doc_blocks = [
                doc.strip()
                for doc in re.split(r"(?=Doc\s+\d+\s+\(Title:)", result_block)
                if doc.strip()
            ]
            if not doc_blocks:
                doc_blocks = [result_block]

            signature = {normalize_answer(doc) for doc in doc_blocks if normalize_answer(doc)}
            if signature:
                signatures.append(signature)
        return signatures

    @staticmethod
    def _search_result_overlap(first: set[str], second: set[str]) -> float:
        if not first or not second:
            return 0.0
        return len(first & second) / min(len(first), len(second))

    def _all_search_results_are_duplicate(
        self,
        current_signatures: list[set[str]],
        historical_signatures: list[set[str]],
        overlap_threshold: float = 2 / 3,
    ) -> bool:
        if not current_signatures or not historical_signatures:
            return False
        for current_signature in current_signatures:
            if not any(
                self._search_result_overlap(current_signature, historical_signature) >= overlap_threshold
                for historical_signature in historical_signatures
            ):
                return False
        return True

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        tool_call_contexts = agent_data.tool_call_contexts[: self.max_parallel_calls]
        for context in tool_call_contexts:
            tool_call = context["tool_call"]
            tasks.append(self._call_tool(tool_call, context["parsed_args"], agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for context, (tool_response, tool_reward, metadata) in zip(tool_call_contexts, responses):
            tool_call = context["tool_call"]
            parsed_args = context["parsed_args"]
            query_list = parsed_args.get("query_list", [])
            tool_response_text = tool_response.text or ""
            message_text = tool_response_text
            query_list_len = 1
            try :
                if "query_list_len" in metadata :
                    query_list_len = metadata['query_list_len']
            except:
                query_list_len = 1

            if 'error_code' in metadata  and metadata['error_code'] == 1:
                agent_data.all_call_tool_success_counts+= query_list_len
            else:
                print(metadata)
            agent_data.all_call_tool_counts += query_list_len
            if tool_call.name == "search" and tool_response_text:
                current_signatures = self._extract_search_result_signatures(tool_response_text)
                if self._all_search_results_are_duplicate(
                    current_signatures,
                    agent_data.searched_result_signatures,
                    self.duplicate_search_result_overlap_threshold,
                ):
                    agent_data.abnormal_trajectory_dic['duplicate_search_result_count'] += 1
                    # return AgentState.TERMINATED
                agent_data.searched_result_signatures.extend(current_signatures)

            if self.enable_tool_response_summary and tool_call.name == "search" and tool_response_text and query_list:
                try:
                    tool_response_text_lst = json.loads(tool_response_text)['result'].split(self.summary_result_separator)
                except Exception as e:
                    print(f"[tool_response_text parser Error] {e}")
                    tool_response_text_lst = []
                # 修复逻辑：处理长度不一致的情况
                # 如果 split 后的数量和 query 数量对不上，记录日志并尽量匹配
                if len(tool_response_text_lst) != len(query_list):
                    print(
                        f"Mismatch in summarization: queries={len(query_list)}, "
                        f"responses={len(tool_response_text_lst)}. "
                        "Zip will truncate to the shorter list."
                    )

                summary_tasks = []
                for q_idx, (query, tool_response_text_item) in enumerate(zip(query_list, tool_response_text_lst)):
                    tool_response_text_item, truncated = await self._truncate_text_by_tokens(
                        tool_response_text_item,
                        self.max_tool_response_length,
                        self.tool_response_truncate_side,
                    )
                    if truncated:
                        agent_data.abnormal_trajectory_dic['response_truncated_count'] += 1

                    task = self._generate_single_summary(query, tool_response_text_item, q_idx + 1)
                    summary_tasks.append(task)

                try:
                    summary_results = await asyncio.gather(*summary_tasks) if summary_tasks else []
                except Exception as e:
                    print(f"[Summary Error] {e}")
                    summary_results = [""]
                all_summary_text = self.summary_result_separator.join(summary_results)
                if all_summary_text:
                    message_text = all_summary_text
                else:
                    message_text, truncated = await self._truncate_text_by_tokens(
                        tool_response_text,
                        self.max_tool_response_length,
                        self.tool_response_truncate_side,
                    )
                    if truncated:
                        agent_data.abnormal_trajectory_dic['response_truncated_count'] += 1
            elif tool_response_text:
                message_text, truncated = await self._truncate_text_by_tokens(
                    tool_response_text,
                    self.max_tool_response_length,
                    self.tool_response_truncate_side,
                )
                if truncated:
                    agent_data.abnormal_trajectory_dic['response_truncated_count'] += 1

            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if message_text:
                    content.append({"type": "text", "text": message_text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": message_text}

            add_messages.append(message)
            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            agent_data.response_mask = len(agent_data.response_mask) * [0]
            agent_data.abnormal_trajectory_dic['too_long_seq_truncated_count'] +=1
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        agent_data.tool_turns += 1
        return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tool_args: dict[str, Any], tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool = active_tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text or "" # 确保非 None
        tool_response_kwargs = {"text": tool_response_text}
        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    async def _truncate_text_by_tokens(
        self, text: str, max_tokens: int | None, truncate_side: str
    ) -> tuple[str, bool]:
        if not text or not max_tokens:
            return text, False
        token_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(text, add_special_tokens=False)
        )
        if len(token_ids) <= max_tokens:
            return text, False

        if truncate_side == "left":
            truncated_ids = token_ids[-max_tokens:]
            marker_prefix = "(truncated)..."
            marker_suffix = ""
        elif truncate_side == "right":
            truncated_ids = token_ids[:max_tokens]
            marker_prefix = ""
            marker_suffix = "...(truncated)"
        else:
            left_n = max_tokens // 2
            right_n = max_tokens - left_n
            left_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(token_ids[:left_n], skip_special_tokens=True)
            )
            right_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(token_ids[-right_n:], skip_special_tokens=True)
            )
            return f"{left_text}...(truncated)...{right_text}", True

        truncated_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
        )
        truncated_text = f"{marker_prefix}{truncated_text}{marker_suffix}"
        return truncated_text, True

    def _init_summary_external_clients(self) -> None:
        base_urls = [
            base_url.strip()
            for base_url in (self.summary_external_base_urls or "").split(",")
            if base_url.strip()
        ]
        if not base_urls:
            raise ValueError(
                "summary_use_external_model=True requires "
                "actor_rollout_ref.rollout.multi_turn.summary_external_base_urls"
            )
        if not self.summary_external_model:
            raise ValueError(
                "summary_use_external_model=True requires "
                "actor_rollout_ref.rollout.multi_turn.summary_external_model"
            )

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("summary_use_external_model=True requires the openai package") from exc

        self.summary_external_clients = [
            AsyncOpenAI(
                api_key=self.summary_external_api_key,
                base_url=base_url,
                timeout=self.summary_external_timeout,
            )
            for base_url in base_urls
        ]

    async def _generate_single_summary(self, query, document, idx):
        if self.summary_use_external_model:
            return await self.generate_single_summary(query, document, idx)
        return await self.generate_single_summary_self(query, document, idx)

    async def generate_single_summary(self,query, document, idx):
        summary_prompt = PROMPT_TEMPLATE.format(query=query, documents=document)
        new_messages = [{"role": "user", "content": summary_prompt or ""}]

        client_order = list(self.summary_external_clients)
        random.shuffle(client_order)
        last_error = None
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": self.summary_external_enable_thinking},
        }
        if self.summary_top_k is not None:
            extra_body["top_k"] = self.summary_top_k

        try:
            for client in client_order:
                try:
                    request_kwargs = dict(
                        model=self.summary_external_model,
                        messages=new_messages,
                        temperature=self.summary_temperature,
                        top_p=self.summary_top_p,
                        extra_body=extra_body,
                    )
                    if self.summary_max_tokens:
                        request_kwargs["max_tokens"] = self.summary_max_tokens
                    summary_response = await client.chat.completions.create(**request_kwargs)
                    summary_text = summary_response.choices[0].message.content
                    return f"the summary of the query {idx} search result is : {summary_text}"
                except Exception as e:
                    last_error = e
                    logger.warning(f"External summary client failed: {e}")
            raise RuntimeError(f"All external summary clients failed: {last_error}")
        except Exception as e:
            print(f"[Summary Error] {e}")
            return f"Error: summary generation failed: {e}"

    async def generate_single_summary_self(self,query, document, idx):
        summary_prompt = PROMPT_TEMPLATE.format(query=query, documents=document)
        new_messages = [{"role": "user", "content": summary_prompt or ""}]

        # 使用本地变量避免并发冲突
        tool_response_prompt_ids = await self.apply_chat_template(
            new_messages,
            images=None,
            videos=None,
            remove_system_prompt=False,
        )
        sampling_params = dict(
            temperature=self.summary_temperature,
            top_p=self.summary_top_p,
            top_k=self.summary_top_k,
            repetition_penalty=1.0,
            logprobs=False,
        )
        if self.summary_max_tokens:
            sampling_params["max_new_tokens"] = self.summary_max_tokens
        summary_req_id = uuid4().hex
        output = await self.server_manager.generate(
            request_id=summary_req_id,
            prompt_ids=tool_response_prompt_ids,
            sampling_params=sampling_params,
            image_data=None,
            video_data=None,
        )

        summary_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        )
        return f"the summary of the query {idx} search result is : {summary_text}"
