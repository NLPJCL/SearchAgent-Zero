# run on 8xH20
# make sure your current working directory is the root of the project
#ps -ef | grep "VLLM" | awk '{print $2}' | xargs kill -9
#ps -ef | grep "ray" | awk '{print $2}' | xargs kill -9

set -x

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
mkdir -p ./outputs
mkdir -p ./logs

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
export VLLM_USE_V1=1
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HSA_NO_SCRATCH_RECLAIM=1

ulimit -n 65535

ASEARCH_DATA_DIR="${ASEARCH_DATA_DIR:-${REPO_ROOT}/examples/search_agent_rl/ASearcher}"
TRAIN_DATA="${TRAIN_DATA:-${ASEARCH_DATA_DIR}/ASearcher_train.parquet}"
VAL_DATA="${VAL_DATA:-${ASEARCH_DATA_DIR}/ASearcher_test.parquet}"
MODEL_PATH="${MODEL_PATH:-/mnt/bn/ttsa-relevance-2/jcli/models/Qwen3-8B}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
N_GPUS_TRAINING="${N_GPUS_TRAINING:-$((NGPUS_PER_NODE - N_GPUS_ROLLOUT))}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-36864}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20000}"
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-999999999}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"
TRIGGER_PARAMETER_SYNC_STEP="${TRIGGER_PARAMETER_SYNC_STEP:-2}"
STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-0.5}"
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-True}"
TURN_LIMIT_SCHEDULE="${TURN_LIMIT_SCHEDULE:-0:100,50:100,100:100,200:100,300:100}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qinstruct_ASearch_abnormal_trajectory_filter_ca_fully_async}"
PROJECT_NAME="${PROJECT_NAME:-search_r1_like_async_rl}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-./output/$EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-./rollout_data/$EXPERIMENT_NAME}"
LOG_FILE="${LOG_FILE:-./logs/$EXPERIMENT_NAME.log}"

# Importance Sampling (IS) weights configuration
rollout_is="token"                       # Token-level IS for metrics/analysis
rollout_is_threshold=2.0                 # Upper threshold for IS weights
rollout_is_batch_normalize="false"       # Keep raw truncated weights

# Rejection Sampling (RS) configuration (multi-criteria)
# - token_k1 keeps per-token ratios inside [lower, upper]
# - seq_max_k2 rejects sequences with extreme chi-square spikes
# rollout_rs="token_k1,seq_max_k2"
# rollout_rs_threshold="0.6_1.6,2.5"

rollout_rs="token_k1"
rollout_rs_threshold="0.6_1.6"
# Bypass PPO mode (reuse rollout_log_prob)
bypass_mode="False"
loss_type="ppo_clip"

python -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=\'${rollout_rs}\' \
    algorithm.rollout_correction.rollout_rs_threshold=\'${rollout_rs_threshold}\' \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    algorithm.rollout_correction.loss_type=${loss_type} \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.val_batch_size=128 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1) / 2)) \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.34 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1536 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=100 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=100 \
    actor_rollout_ref.rollout.multi_turn.turn_limit_schedule="'$TURN_LIMIT_SCHEDULE'" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=20000 \
    actor_rollout_ref.rollout.multi_turn.enable_tool_response_summary=True \
    actor_rollout_ref.rollout.multi_turn.summary_result_separator='\\n-*-*-\\n' \
    actor_rollout_ref.rollout.multi_turn.summary_temperature=0.6 \
    actor_rollout_ref.rollout.multi_turn.summary_top_p=0.95 \
    actor_rollout_ref.rollout.multi_turn.summary_top_k=20 \
    actor_rollout_ref.rollout.multi_turn.summary_max_tokens=1024 \
    actor_rollout_ref.rollout.multi_turn.summary_use_external_model=False \
    actor_rollout_ref.rollout.multi_turn.summary_external_base_urls="" \
    actor_rollout_ref.rollout.multi_turn.summary_external_model="" \
    actor_rollout_ref.rollout.multi_turn.max_queries_per_tool_call=4 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    rollout.total_rollout_steps="$TOTAL_ROLLOUT_STEPS" \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.nnodes="$NNODES" \
    trainer.n_gpus_per_node="$N_GPUS_TRAINING" \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    trainer.total_epochs=3 \
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
    rollout.nnodes="$NNODES" \
    rollout.n_gpus_per_node="$N_GPUS_ROLLOUT" \
    async_training.staleness_threshold="$STALENESS_THRESHOLD" \
    async_training.trigger_parameter_sync_step="$TRIGGER_PARAMETER_SYNC_STEP" \
    async_training.require_batches="$REQUIRE_BATCHES" \
    async_training.partial_rollout="$PARTIAL_ROLLOUT" \
    async_training.use_trainer_do_validate=False \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    "$@" 2>&1 | tee "$LOG_FILE"
