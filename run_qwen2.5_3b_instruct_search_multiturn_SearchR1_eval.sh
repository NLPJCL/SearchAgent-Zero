# run on 8xH20
# make sure your current working directory is the root of the project

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$SCRIPT_DIR"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/examples/search_agent_rl/config}"
TOOL_CONFIG="${TOOL_CONFIG:-${CONFIG_PATH}/tool_config/search_tool_config.yaml}"
AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-${CONFIG_PATH}/agent_loop/tool_agent_credit_assignment.yaml}"

CACHE_BASE="${CACHE_BASE:-/tmp/temp_cache}"

mkdir -p "$CACHE_BASE/huggingface"
mkdir -p "$CACHE_BASE/hf_datasets"
mkdir -p "$CACHE_BASE/tmp"
mkdir -p "$CACHE_BASE/ray_tmp"

export HF_HOME="$CACHE_BASE/huggingface"
export HF_DATASETS_CACHE="$CACHE_BASE/hf_datasets"
export TMPDIR="$CACHE_BASE/tmp"
export RAY_TMPDIR="$CACHE_BASE/ray_tmp"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export no_proxy="*"
export NO_PROXY="*"
export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""
export ALL_PROXY=""
export VERL_DISABLE_RAY_RUNTIME_ENV=1
export VERL_FORCE_RAY_LOCALHOST=1
export TOKENIZERS_PARALLELISM=true
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_DISABLE_COMPILE_CACHE=1
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HSA_NO_SCRATCH_RECLAIM=1

set -x

ulimit -n 65535

#ps -ef | grep "VLLM" | awk '{print $2}' | xargs kill -9

# Importance Sampling (IS) weights configuration
rollout_is="token"                       # Token-level IS for metrics/analysis
rollout_is_threshold=2.0                 # Upper threshold for IS weights
rollout_is_batch_normalize="false"       # Keep raw truncated weights

# Rejection Sampling (RS) configuration (multi-criteria)
# - token_k1 keeps per-token ratios inside [lower, upper]
# - seq_max_k2 rejects sequences with extreme chi-square spikes
rollout_rs="token_k1,seq_max_k2"
rollout_rs_threshold="0.6_1.6,2.5"

# Bypass PPO mode (reuse rollout_log_prob)
bypass_mode="False"
loss_type="ppo_clip"

if [ ! -d "./output" ]; then
    mkdir -p ./output
    echo "mkdir output"
fi

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
    echo "mkdir logs"
fi

if [ ! -d "./rollout_data" ]; then
    mkdir -p ./rollout_data
    echo "mkdir rollout_data"
fi

SEARCH_R1_DATA_DIR="${SEARCH_R1_DATA_DIR:-${REPO_ROOT}/examples/search_agent_rl/search_r1_processed}"
TRAIN_DATA="${TRAIN_DATA:-${SEARCH_R1_DATA_DIR}/train_search_r1.parquet}"
VAL_DATA="${VAL_DATA:-${SEARCH_R1_DATA_DIR}/test_search_r1.parquet}"



python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="search_multiturn_grpo" \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=3000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=./output/qwen2.5-3b-instruct_searchr1/global_step_500/merged_hf_model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=5000 \
    actor_rollout_ref.rollout.multi_turn.enable_tool_response_summary=False \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    actor_rollout_ref.rollout.multi_turn.max_queries_per_tool_call=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=disable \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='search_r1_like_async_rl' \
    trainer.experiment_name='qwen2.5-3b-instruct_searchr1_eval' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=250 \
    trainer.test_freq=50 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.validation_data_dir="./rollout_data/qwen2.5-3b-instruct_searchr1_eval"\
    trainer.total_epochs=2 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    trainer.default_local_dir='./output/qwen2.5-3b-instruct_searchr1'  2>&1 | tee ./logs/qwen2.5-3b-instruct_searchr1_eval.log
