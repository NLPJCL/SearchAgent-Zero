#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

DATASET_REPO=${DATASET_REPO:-aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample}
RAW_DIR=${RAW_DIR:-"${SCRIPT_DIR}/ASearcher_en_no-math_Qwen3-8B-reject-sample"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/ASearcher"}
PYTHON=${PYTHON:-python3}

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${REPO_ROOT}/$1" ;;
  esac
}

RAW_DIR=$(resolve_path "${RAW_DIR}")
OUTPUT_DIR=$(resolve_path "${OUTPUT_DIR}")

"${PYTHON}" "${REPO_ROOT}/examples/search_agent_rl/preprocess_ASearcher_dataset.py" \
  --hf_repo_id "${DATASET_REPO}" \
  --local_dir "${RAW_DIR}" \
  --output_path "${OUTPUT_DIR}/ASearcher_train.parquet" \
  --test_output_path "${OUTPUT_DIR}/ASearcher_test.parquet" \
  --train_ratio 0.95 \
  --seed 42
