# SearchAgent-Zero: 基于 verl 的可扩展多轮 Search Agent RL 训练框架

SearchAgent-Zero 是一个基于 verl 的 Search Agent 强化学习训练框架，覆盖 Search-R1 式短程问答搜索，也支持 ASearch 式长程多轮搜索。

项目提供一套可复现、可扩展的训练 recipe，整合了多轮工具调用、检索服务、异常轨迹处理、summary 压缩，以及同步/异步 RL 训练流程，方便研究者和工程师复现并扩展 Search Agent RL。

## News

- **[2026/05/24] SearchAgent-Zero 发布 Search-R1 与 ASearch recipes。** Search-R1 recipe 在 Qwen2.5-3B-Instruct 上将完整 Search-R1 评测平均分从 `0.325` 提升到 `0.407`；ASearch recipe 支持长程多轮搜索训练，SearchAgent-Zero (Qwen3-8B, 300 steps) 在 BrowseComp-Plus 达到 `37.95%` Accuracy 和 `50.87%` Recall，在 14B 以下模型中达到 SOTA。实现细节见 `examples/search_agent_rl/`、`verl/experimental/agent_loop/` 和 `verl/tools/search_tool.py`。

## 主要功能
- **稳定可扩展的 RL infra**：基于最新的 verl RL infra 与 GRPO 训练流程，SearchAgent-Zero 已验证可稳定训练 Search-R1 等 search agent 任务，并支持扩展到更多轮数的多轮搜索训练，避免原框架在长程 rollout 中容易崩溃的问题。
- **异常轨迹监控**：训练中记录工具调用成功率、平均搜索轮数、重复 query、并发 query 过多、工具解析失败、轨迹截断等 Search Agent 专属指标。
- **异常轨迹过滤与信用分配**：对重复搜索、单轮并发过多、工具格式错误等低质量轨迹进行过滤或惩罚；对于只发生在某一轮工具调用中的异常，只惩罚出错轮次相关 token，减少对前序有效搜索行为的误伤。
- **搜索结果 summary 压缩**：支持 self-summary 与 external-summary，在有限上下文内保留更多轮搜索中的关键信息。
- **同步与 fully async 训练**：提供标准 GRPO 训练脚本，也提供 fully async policy 训练入口，用于探索更高吞吐的长程 Search Agent RL。

## 实验结果

### Search-R1

在 Search-R1 设置下，SearchAgent-Zero 使用 Qwen2.5-3B-Instruct 与相同训练数据进行复现。相比原 Search-R1 结果，基于 verl AgentLoop 的训练在多个开放域问答数据集上取得稳定提升。

![Search-R1 evaluation comparison](docs/_static/search_r1_comparison.png)

| 数据集 | Search-R1 (Qwen2.5-3B-Instruct) | verl AgentLoop 复现 (Qwen2.5-3B-Instruct) | Abs. Gain | Rel. Gain |
| --- | ---: | ---: | ---: | ---: |
| NQ | 0.341 | 0.4640 | +0.1230 | +36.1% |
| TriviaQA* | 0.545 | 0.6164 | +0.0714 | +13.1% |
| PopQA* | 0.378 | 0.4239 | +0.0459 | +12.1% |
| HotpotQA† | 0.324 | 0.4225 | +0.0985 | +30.4% |
| 2Wiki* | 0.319 | 0.3979 | +0.0789 | +24.7% |
| Musique* | 0.103 | 0.1808 | +0.0778 | +75.5% |
| Bamboogle* | 0.264 | 0.3440 | +0.0800 | +30.3% |
| Avg | 0.325 | 0.40707 | +0.08207 | +25.3% |

在完整 Search-R1 评测集合上，SearchAgent-Zero 将平均分从 `0.325` 提升到约 `0.407`，且所有数据集均取得正向提升，说明稳定的 RL infra、AgentLoop rollout 与异常轨迹处理会显著影响 Search Agent 训练效果。

### BrowseComp-Plus

在更长程的 BrowseComp-Plus 搜索场景中，Qwen3-8B 通过纯 RL 学会了更主动的多轮搜索策略。加入 self-summary 与带信用分配的异常轨迹过滤后，模型能够在固定上下文预算内扩展到更多搜索轮次。下表中前两个训练模型为 100 step 结果，`SearchAgent-Zero (300 step)` 为 300 step 训练结果。

![BrowseComp-Plus performance comparison](docs/_static/browsecomp_plus_comparison.jpg)

| 设置 | Accuracy | Recall | 平均搜索次数 |
| --- | ---: | ---: | ---: |
| Qwen3-32B base | 10.72% | 7.28% | 0.94 |
| Qwen3-8B + 异常轨迹过滤 + self-summary（100 step） | 24.21% | 33.14% | 10.11 |
| Qwen3-8B + 信用分配异常轨迹过滤 + self-summary（100 step） | 28.19% | 40.10% | 14.22 |
| SearchAgent-Zero (300 step) | 37.95% | 50.87% | 38.47 |

结果表明，SearchAgent-Zero 不只是提升 benchmark 分数，更重要的是验证了 Search Agent RL 可以稳定 scale up 到多轮搜索场景。

## Quick Start

### 1. 安装训练环境

建议参考 verl 官方安装流程，使用一个全新的 conda 环境。vLLM、SGLang 等推理框架通常会严格约束 PyTorch 版本，因此先安装训练/推理后端依赖，再以 `--no-deps` 方式安装当前仓库本体。

```bash
# 在仓库根目录执行
conda create -n verl python==3.12 -y
conda activate verl

# SearchAgent-Zero 默认使用 FSDP/FSDP2；如需 Megatron，请见下方可选命令。
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
pre-commit install
```

如需 Megatron 后端，使用完整依赖安装：

```bash
bash scripts/install_vllm_sglang_mcore.sh
```

如果你的 CUDA、PyTorch、vLLM 或 SGLang 版本与脚本默认配置不同，请按当前机器环境调整该脚本后再执行。

### 2. 安装 retriever 环境

本地 dense retriever 推荐使用独立 conda 环境。GPU 版本检索速度和精度更适合 RL rollout；CPU 版本可用于简单连通性测试，但可能影响训练效果。

```bash
conda create -n retriever python=3.10 -y
conda activate retriever

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers datasets pyserini huggingface_hub uvicorn fastapi
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y
```

### 3. 启动本地搜索服务

SearchAgent-Zero 默认使用 Wikipedia 语料和 E5 dense index。索引和语料较大，请预留足够磁盘空间。

```bash
# 在仓库根目录执行
conda activate retriever

save_path=examples/search_agent_rl/local_dense_retriever/search_data
python examples/search_agent_rl/local_dense_retriever/download.py --save_path "$save_path"
cat "$save_path"/part_* > "$save_path"/e5_Flat.index
gzip -dk "$save_path"/wiki-18.jsonl.gz
```

启动检索服务：

```bash
conda activate retriever

CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/search_agent_rl/local_dense_retriever/start_retrieval.sh
```

默认服务地址为：

```text
http://127.0.0.1:8000/retrieve
```

如需替换为自定义检索服务，请修改：

```text
examples/search_agent_rl/config/tool_config/search_tool_config.yaml
```

检索服务输入输出格式与 Search-R1 风格保持一致：

```python
# request
{
    "queries": ["What is Python?", "Tell me about neural networks."],
    "topk": 3,
    "return_scores": True
}

# response
{
    "result": [
        [
            {"document": "...", "score": 0.9}
        ]
    ]
}
```

## 数据集

### Search-R1 数据集

Search-R1 recipe 使用 `PeterJinGo/nq_hotpotqa_train`，覆盖 NQ、HotpotQA 等开放域问答任务，适合复现短程 reasoning + search 训练。

预处理数据：

```bash
conda activate verl

bash examples/search_agent_rl/preprocess_search_r1_dataset_new.sh
```

默认输出：

```text
examples/search_agent_rl/search_r1_processed/train_search_r1.parquet
examples/search_agent_rl/search_r1_processed/test_search_r1.parquet
```

启动训练：

```bash
conda activate verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=YOUR_WANDB_API_KEY

bash run_qwen2.5_3b_instruct_search_multiturn_SearchR1.sh
```

如需只做验证，可使用：

```bash
bash run_qwen2.5_3b_instruct_search_multiturn_SearchR1_eval.sh
```

### ASearcher 数据集

ASearch recipe 使用 `aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample`，更适合训练长程多轮搜索 Agent。该数据会被切分为训练集与测试集，并转换为 verl 多轮工具调用格式。

预处理数据：

```bash
conda activate verl

bash examples/search_agent_rl/preprocess_ASearcher_dataset.sh
```

默认输出：

```text
examples/search_agent_rl/ASearcher/ASearcher_train.parquet
examples/search_agent_rl/ASearcher/ASearcher_test.parquet
```

启动同步 GRPO 训练：

```bash
conda activate verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=YOUR_WANDB_API_KEY

bash run_qwen3_8b_instruct_search_multiturn_ASearch.sh
```

启动 fully async 训练：

```bash
conda activate verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=YOUR_WANDB_API_KEY

bash run_qwen3_8b_instruct_search_multiturn_ASearch_fully_async.sh
```


## 配置说明

核心配置位于：

```text
examples/search_agent_rl/config/search_multiturn_grpo.yaml
examples/search_agent_rl/config/tool_config/search_tool_config.yaml
examples/search_agent_rl/config/agent_loop/tool_agent_credit_assignment.yaml
```

关键开关包括：

- `actor_rollout_ref.rollout.multi_turn.enable=True`：启用多轮工具调用。
- `actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent`：使用工具 AgentLoop。
- `actor_rollout_ref.rollout.multi_turn.tool_config_path`：指定搜索工具配置。
- `actor_rollout_ref.rollout.multi_turn.enable_tool_response_summary=True`：启用搜索结果 summary。
- `actor_rollout_ref.rollout.multi_turn.summary_use_external_model=False`：默认使用 self-summary。
- `actor_rollout_ref.rollout.multi_turn.max_queries_per_tool_call`：控制单轮最多并发 query 数。
- `actor_rollout_ref.rollout.multi_turn.turn_limit_schedule`：控制训练过程中最大搜索轮次 schedule。

## 监控指标

SearchAgent-Zero 为 Search Agent RL 增加了更细粒度的训练指标，便于判断模型是否真的学会稳定搜索。

| 指标 | 含义 |
| --- | --- |
| `turn/tool_call_success_rate/mean` | 工具调用成功率 |
| `turn/tool_call_turn/mean` | 平均工具调用轮数 |
| `turn/tool_call_success_counts/mean` | 成功搜索 query 数 |
| `turn/all_call_tool_counts/mean` | 总搜索 query 数 |
| `abnormal_trajectory/tool_parser_error_count_percentage` | 工具调用格式错误比例 |
| `abnormal_trajectory/searched_query_count_percentage` | 重复搜索 query 比例 |
| `abnormal_trajectory/too_many_tool_call_count_percentage` | 单轮并发 query 过多比例 |
| `abnormal_trajectory/duplicate_search_result_count_percentage` | 搜索结果无增量信息比例 |
| `abnormal_trajectory/too_many_turn_count_percentage` | 超过最大轮数比例 |
| `abnormal_trajectory/too_long_seq_truncated_count_percentage` | 轨迹过长被截断比例 |
| `abnormal_trajectory/response_truncated_count_percentage` | 单轮回复被截断比例 |

这些指标可以帮助区分三类问题：环境不稳定、工具调用格式错误、模型策略退化。对于长程搜索训练，建议同时观察 reward、平均搜索轮数、搜索 query 数、异常轨迹比例和回复长度。

## 项目结构

```text
examples/search_agent_rl/
  config/
    search_multiturn_grpo.yaml
    tool_config/search_tool_config.yaml
    agent_loop/tool_agent_credit_assignment.yaml
  local_dense_retriever/
    download.py
    retrieval_server.py
    start_retrieval.sh
  preprocess_search_r1_dataset_new.sh
  preprocess_ASearcher_dataset.sh

run_qwen2.5_3b_instruct_search_multiturn_SearchR1.sh
run_qwen2.5_3b_instruct_search_multiturn_SearchR1_eval.sh
run_qwen3_8b_instruct_search_multiturn_ASearch.sh
run_qwen3_8b_instruct_search_multiturn_ASearch_fully_async.sh

verl/experimental/agent_loop/
verl/tools/search_tool.py
```

## Citation

如果你觉得 SearchAgent-Zero 对你的研究或工程有帮助，欢迎引用：

```bibtex
@software{li2026searchagentzero,
  title  = {SearchAgent-Zero: Scalable Reinforcement Learning for Multi-Turn Search Agents},
  author = {Li, Jiacheng},
  url    = {https://github.com/verl-project/verl},
  year   = {2026}
}
```

## Acknowledgement

在开发过程中，我们借鉴或基于了以下项目的实现。我们由衷感谢这些团队为开源研究与开发所做出的贡献。

- [verl](https://github.com/verl-project/verl)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- [Cut the Bill, Keep the Turns: Affordable Multi-Turn Search RL](https://agate-slipper-ef0.notion.site/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c)
