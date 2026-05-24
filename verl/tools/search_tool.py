# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
import time
import uuid
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
import requests
from requests.exceptions import RequestException, Timeout

from verl.utils.rollout_trace import rollout_trace_op
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

# ==========================================
# Configuration & Constants
# ==========================================

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
RETRY_DELAY = 1

# ==========================================
# Utilities & Helpers
# ==========================================

def _format_passages(retrieval_result: List[Dict]) -> str:
    """Format the raw retrieval JSON list into a readable string."""
    formatted_texts = []
    for idx, doc_item in enumerate(retrieval_result):
        try:
            content_block = doc_item.get("document", {}).get("contents", "")
            lines = content_block.split("\n")
            title = lines[0] if lines else "No Title"
            text = "\n".join(lines[1:]) if len(lines) > 1 else content_block
            formatted_texts.append(f"Doc {idx + 1} (Title: {title})\n{text}")
        except Exception:
            continue

    return "\n\n".join(formatted_texts)

# ==========================================
# Ray Actors & Remote Tasks
# ==========================================

@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class GlobalRateLimiter:
    """
    A simple Token Bucket Rate Limiter implemented as a Ray Actor.
    Ensures the search API is not overwhelmed across distributed workers.
    """
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()


@ray.remote
def perform_search_remote(
    query_list: List[str],
    url: str,
    topk: int,
    timeout: int,
    rate_limiter: Optional[ray.actor.ActorHandle] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Executes the search request remotely via Ray.
    Handles rate limiting, retries, and result formatting.
    """

    # 1. Rate Limiting (Acquire)
    if rate_limiter:
        ray.get(rate_limiter.acquire.remote())

    result_text = ""
    # 初始化 Metadata，默认 error_code 为 0 (假设失败/无结果)
    metadata = {
        "query_count": len(query_list),
        "status": "initializing",
        "total_results": 0,
        "error_code": 0,  # 0: Error or No Results, 1: Success with Results
        "error_details": None
    }

    payload = {"queries": query_list, "topk": topk, "return_scores": True}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    try:
        # 2. HTTP Request with Retries
        response_data = None
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=timeout)

                # Handle Server Errors (Retryable)
                if response.status_code >= 500:
                    last_error = f"Server Error {response.status_code}"
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue

                response.raise_for_status()
                response_data = response.json()
                break # Success

            except Timeout:
                last_error = "Request Timeout"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
            except (RequestException, json.JSONDecodeError) as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

        # 3. Process Response
        if response_data:
            raw_results = response_data.get("result", [])
            formatted_chunks = []
            total_count = 0

            # 解析结果
            try:
                for single_query_result in raw_results:
                    formatted_str = _format_passages(single_query_result)
                    formatted_chunks.append(formatted_str)
                    total_count += len(single_query_result) if isinstance(single_query_result, list) else 1
            except Exception as parse_e:
                # 解析过程出错
                error_msg = f"Result parsing failed: {str(parse_e)}"
                result_text = json.dumps({"result": error_msg}, ensure_ascii=False)
                metadata["status"] = "parsing_error"
                metadata["error_code"] = 0
                metadata["error_details"] = str(parse_e)
                return result_text, metadata

            if total_count > 0:
                # 成功且有结果
                result_text = json.dumps({"result": "\n-*-*-\n".join(formatted_chunks)}, ensure_ascii=False)
                metadata["status"] = "success"
                metadata["total_results"] = total_count
                metadata["error_code"] = 1 # 成功标记
            else:
                # 成功但无结果
                result_text = json.dumps({"result": "No search results found."}, ensure_ascii=False)
                metadata["status"] = "no_results"
                metadata["error_code"] = 0 # 无结果视为 0
        else:
            # 重试后依然失败
            error_msg = f"Search failed after {MAX_RETRIES} retries. Last error: {last_error}"
            logger.warning(f"[SearchTool] {error_msg}")
            result_text = json.dumps({"result": error_msg}, ensure_ascii=False)

            metadata["error_code"] = 0
            metadata["error_details"] = last_error
            if "Timeout" in str(last_error):
                metadata["status"] = "api_timeout"
            elif "Server Error" in str(last_error):
                metadata["status"] = "server_error"
            else:
                metadata["status"] = "request_failed"

    except Exception as e:
        # 意外的执行错误
        error_msg = f"Unexpected execution error: {str(e)}"
        logger.error(f"[SearchTool] {error_msg}")
        result_text = json.dumps({"result": error_msg}, ensure_ascii=False)
        metadata["status"] = "unknown_error"
        metadata["error_code"] = 0
        metadata["error_details"] = str(e)

    finally:
        # 4. Rate Limiting (Release)
        if rate_limiter:
            rate_limiter.release.remote()

    return result_text, metadata

# ==========================================
# Main Tool Class
# ==========================================

class SearchTool(BaseTool):
    """
    Search tool for retrieving information using external retrieval services.
    Supports parallel execution and rate limiting via Ray.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

        # Configuration
        self.retrieval_service_url = config.get("retrieval_service_url")
        if not self.retrieval_service_url:
            raise ValueError("Config must include 'retrieval_service_url'")

        self.topk = config.get("topk", 3)
        self.timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self.num_workers = config.get("num_workers", 10) # Used as max_concurrency hint if needed

        # Initialize Rate Limiter
        self.enable_rate_limit = config.get("enable_global_rate_limit", True)
        self.rate_limit_actor = None

        if self.enable_rate_limit:
            limit = config.get("rate_limit", 120)
            try:
                self.rate_limit_actor = GlobalRateLimiter.options(
                    name="search-global-rate-limiter",
                    get_if_exists=True,
                    lifetime="detached"
                ).remote(limit)
            except Exception as e:
                logger.warning(f"Failed to init global rate limiter: {e}. Running without limit.")

        self._instance_dict = {}
        logger.info(f"Initialized SearchTool with URL: {self.retrieval_service_url}")

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
        """Create a new session/trajectory for the tool."""
        if instance_id is None:
            instance_id = str(uuid.uuid4())

        self._instance_dict[instance_id] = {
            "reward": [],
            "history": []
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, dict]:
        """
        Execute the search asynchronously.

        Args:
            instance_id: Unique session ID.
            parameters: Dict containing 'query_list'.

        Returns:
            (ToolResponse, reward, metrics)
        """
        query_list = parameters.get("query_list")
        query_list_len = len(query_list)
        # Basic Validation: 参数缺失也是一种错误
        if not query_list or not isinstance(query_list, list):
            msg = "Error: 'query_list' parameter is missing or not a list."
            error_metadata = {
                "status": "invalid_parameters",
                "error_code": 0,
                "error_details": msg,
                "query_list_len": query_list_len
            }
            return ToolResponse(text=json.dumps({"result": msg})), 0.0, error_metadata

        try:
            # Delegate to Ray Remote Task
            future = perform_search_remote.remote(
                query_list=query_list,
                url=self.retrieval_service_url,
                topk=self.topk,
                timeout=self.timeout,
                rate_limiter=self.rate_limit_actor
            )

            result_text, metadata = await future

            metadata['query_list_len'] = query_list_len

            # Update State
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["reward"].append(result_text)

            return ToolResponse(text=result_text), 0.0, metadata

        except Exception as e:
            # 捕获本地执行或 Ray 调用过程中的异常
            err_msg = f"Search execution exception: {str(e)}"
            logger.error(err_msg)
            error_metadata = {
                "status": "execution_exception",
                "error_code": 0,
                "error_details": str(e),
                "query_list_len": query_list_len
            }
            return ToolResponse(text=json.dumps({"result": err_msg})), 0.0, error_metadata

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def calc_reward(self, instance_id: str, **kwargs) -> Any:
        return self._instance_dict.get(instance_id, {}).get("reward", [])

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]