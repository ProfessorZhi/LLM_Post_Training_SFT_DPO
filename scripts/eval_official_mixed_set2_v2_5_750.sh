#!/usr/bin/env bash
set -euo pipefail

ROOT="${WORK_ROOT:-/workspace/llm4se}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/eval_benchmarks/code_fim_test}"
RUNTIME_REPO="${RUNTIME_REPO:-/workspace/MindSpeed-LLM}"
RUN_ID_ROOT=20260519_142908_fim_sft_v2_5_fnonly_filtered_1000step
STEP=750
STEP_PADDED="$(printf "%07d" "${STEP}")"

SOURCE_CKPT_ROOT="${ROOT}/runs/fim_sft/v2_5_fnonly_filtered/${RUN_ID_ROOT}/checkpoints/ckpt_save_qwen25_coder_fim_sft"
ITER_DIR="${SOURCE_CKPT_ROOT}/iter_${STEP_PADDED}"
VIEW_ROOT="${ROOT}/tmp/ckpt_views/official_mixed_set2_v2_5_fnonly_filtered_750step"
MODEL_MCORE="${VIEW_ROOT}"
TOKENIZER_PATH="${RUNTIME_REPO}/model_from_hf/qwen25-coder-1point5b-hf"
MODEL_RUN_NAME="v2_5_fnonly_filtered_750step_official_set2"

SET2_FILE="${EVAL_ROOT}/HumanEvalFIM/data/mixed_test_set_2_20pct_disjoint_canonical_le32.jsonl"
OUT_DIR="${ROOT}/eval/official_mixed_set2_20260524/$(date +%Y%m%d_%H%M%S)_${MODEL_RUN_NAME}"
LOG_FILE="${OUT_DIR}/official_set2_run.log"

echo "[official-set2] started $(date -Is)"
echo "[official-set2] eval_root=${EVAL_ROOT}"
echo "[official-set2] set2_file=${SET2_FILE}"
echo "[official-set2] iter_dir=${ITER_DIR}"
echo "[official-set2] output=${OUT_DIR}"

[ -d "${EVAL_ROOT}" ] || { echo "missing official eval repo: ${EVAL_ROOT}"; exit 1; }
[ -f "${SET2_FILE}" ] || { echo "missing set2 file: ${SET2_FILE}"; exit 1; }
[ -d "${RUNTIME_REPO}" ] || { echo "missing runtime repo: ${RUNTIME_REPO}"; exit 1; }
[ -d "${ITER_DIR}" ] || { echo "missing checkpoint iter dir: ${ITER_DIR}"; exit 1; }
[ -d "${TOKENIZER_PATH}" ] || { echo "missing tokenizer path: ${TOKENIZER_PATH}"; exit 1; }
[ -f "${TOKENIZER_PATH}/tokenizer.json" ] || { echo "missing tokenizer.json under ${TOKENIZER_PATH}"; exit 1; }

rm -rf "${VIEW_ROOT}"
mkdir -p "${VIEW_ROOT}" "${OUT_DIR}"
ln -s "${ITER_DIR}" "${VIEW_ROOT}/iter_${STEP_PADDED}"
printf "%s" "${STEP}" > "${VIEW_ROOT}/latest_checkpointed_iteration.txt"

cat > "${OUT_DIR}/manifest.txt" <<EOF
label=${MODEL_RUN_NAME}
eval_root=${EVAL_ROOT}
set2_file=${SET2_FILE}
runtime_repo=${RUNTIME_REPO}
model_mcore=${MODEL_MCORE}
tokenizer_path=${TOKENIZER_PATH}
checkpoint_iteration=${STEP}
humanevalfim_gen_length=64
humanevalfim_max_seq_length=2048
ms_eval_batch_size=8
started=$(date -Is)
EOF

cd "${EVAL_ROOT}"

export EVAL_ROOT
export PROJECT_ROOT="${EVAL_ROOT}"
export MSLLM_ROOT="${RUNTIME_REPO}"
export MODEL_MCORE
export MODEL_HF="${TOKENIZER_PATH}"
export TOKENIZER_PATH
export MODEL_RUN_NAME
export RESULT_ROOT="${ROOT}/eval/official_mixed_set2_20260524/results"
export PROMPT_CACHE_ROOT="${OUT_DIR}/_prompt_cache"
export MIXED_HUMANEVALFIM_INPUT_FILE="${SET2_FILE}"
export MIXED_HUMANEVALFIM_OUT="${OUT_DIR}"
export HUMANEVALFIM_GEN_LENGTH=64
export HUMANEVALFIM_MAX_SEQ_LENGTH=2048
export MS_EVAL_BATCH_SIZE="${MS_EVAL_BATCH_SIZE:-8}"
export MS_MICRO_BATCH_SIZE="${MS_MICRO_BATCH_SIZE:-${MS_EVAL_BATCH_SIZE}}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-16864}"
export RUN_REPETITION_ANALYSIS=1

bash "${EVAL_ROOT}/scripts/run_mixed_humanevalfim_mindspeed.sh" 2>&1 | tee "${LOG_FILE}"

{
  echo "status=completed"
  echo "finished=$(date -Is)"
  echo "output=${OUT_DIR}"
  [ -f "${OUT_DIR}/results.json" ] && echo "results_json=${OUT_DIR}/results.json"
  [ -f "${OUT_DIR}/repetition_report.json" ] && echo "repetition_report=${OUT_DIR}/repetition_report.json"
} >> "${OUT_DIR}/manifest.txt"

echo "[official-set2] finished $(date -Is)"
