#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

DATA_DIR=${DATA_DIR:-"${SCRIPT_DIR}/search_data"}
INDEX_FILE=${INDEX_FILE:-"${DATA_DIR}/e5_Flat.index"}
CORPUS_FILE=${CORPUS_FILE:-"${DATA_DIR}/wiki-18.jsonl"}
RETRIEVER_NAME=${RETRIEVER_NAME:-e5}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-intfloat/e5-base-v2}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
PYTHON=${PYTHON:-python3}

if [[ ! -f "${INDEX_FILE}" || ! -f "${CORPUS_FILE}" ]]; then
    echo "Missing retrieval files under ${DATA_DIR}." >&2
    echo "Run: ${PYTHON} ${SCRIPT_DIR}/download.py --save_path ${DATA_DIR}" >&2
    echo "Then run: cat ${DATA_DIR}/part_* > ${INDEX_FILE} && gzip -dk ${DATA_DIR}/wiki-18.jsonl.gz" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" "${REPO_ROOT}/examples/search_agent_rl/local_dense_retriever/retrieval_server.py" \
        --index_path "${INDEX_FILE}" \
        --corpus_path "${CORPUS_FILE}" \
        --topk 3 \
        --retriever_name "${RETRIEVER_NAME}" \
        --retriever_model "${RETRIEVER_MODEL}" \
        --faiss_gpu
