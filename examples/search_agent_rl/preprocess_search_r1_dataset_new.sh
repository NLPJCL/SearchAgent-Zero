#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON=${PYTHON:-python3}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/search_r1_processed"}

"${PYTHON}" "${REPO_ROOT}/examples/search_agent_rl/preprocess_search_r1_dataset_new.py" \
    --local_dir "${OUTPUT_DIR}"
