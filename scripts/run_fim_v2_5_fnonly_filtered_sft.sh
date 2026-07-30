#!/bin/bash
set -euo pipefail

RUN_ID="$(date +%Y%m%d_%H%M%S)_fim_sft_v2_5_fnonly_filtered_${TRAIN_ITERS:-1000}step"
BRANCH_ID="v2_5_fnonly_filtered"
DATASET_ID="v2_5_fnonly_filtered_20260519"
WORK_ROOT="${WORK_ROOT:-/workspace/llm4se}"
REPO_DIR="${RUNTIME_REPO:-/workspace/MindSpeed-LLM}"
FIM_TOOL_DIR="${FIM_TOOL_DIR:-/workspace/fim_tools/fim}"
SOURCE_DATA="${WORK_ROOT}/datasets/fim_sft/${DATASET_ID}/source/fim_v2_5_fnonly_filtered_train.source.parquet"
MANIFEST="${WORK_ROOT}/datasets/fim_sft/${DATASET_ID}/manifest/fim_v2_5_fnonly_filtered_manifest.json"

TRAIN_ITERS="${TRAIN_ITERS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
MASTER_PORT="${MASTER_PORT:-6021}"
RUN_DIR="${WORK_ROOT}/runs/fim_sft/${BRANCH_ID}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
SCRIPT_DIR="${RUN_DIR}/scripts"
CKPT_DIR="${RUN_DIR}/checkpoints/ckpt_save_qwen25_coder_fim_sft"
DATASET_BACKUP_DIR="${RUN_DIR}/converted/alpaca_fim_dataset"
DATA_CONVERT_LOG="${LOG_DIR}/data_convert_json.log"
SFT_LOG="${LOG_DIR}/sft_${TRAIN_ITERS}step.log"
OUTER_LOG="${LOG_DIR}/sft_${TRAIN_ITERS}step.outer.log"
RUN_MANIFEST="${RUN_DIR}/manifest.json"

echo "[fim-v2.5-filtered] run_id=${RUN_ID}"
echo "[fim-v2.5-filtered] repo=${REPO_DIR}"
echo "[fim-v2.5-filtered] source_data=${SOURCE_DATA}"
echo "[fim-v2.5-filtered] train_iters=${TRAIN_ITERS}"
echo "[fim-v2.5-filtered] save_interval=${SAVE_INTERVAL}"
echo "[fim-v2.5-filtered] master_port=${MASTER_PORT}"
echo "[fim-v2.5-filtered] run_dir=${RUN_DIR}"

mkdir -p "${LOG_DIR}" "${SCRIPT_DIR}" "$(dirname "${CKPT_DIR}")" "$(dirname "${DATASET_BACKUP_DIR}")"
cd "${REPO_DIR}"

echo "[0/5] boundary and file checks"
[ -d "${REPO_DIR}" ] || { echo "missing repo dir: ${REPO_DIR}"; exit 1; }
[ -d "${FIM_TOOL_DIR}" ] || { echo "missing fim tool dir: ${FIM_TOOL_DIR}"; exit 1; }
[ -f "${FIM_TOOL_DIR}/data_convert_qwen25_fim_sft.sh" ] || { echo "missing latest data convert script"; exit 1; }
[ -f "${FIM_TOOL_DIR}/tune_qwen25_coder_1point5b_fim_sft_ptd.sh" ] || { echo "missing latest tune script"; exit 1; }
[ -f "${SOURCE_DATA}" ] || { echo "missing source parquet: ${SOURCE_DATA}"; exit 1; }
[ -f "${MANIFEST}" ] || { echo "missing manifest: ${MANIFEST}"; exit 1; }
[ -d "./model_from_hf/qwen25-coder-1point5b-hf" ] || { echo "missing HF-format local model dir"; exit 1; }
[ -d "./model_weights/qwen25-coder-1point5b-mcore" ] || { echo "missing mcore model dir"; exit 1; }
[ -n "$(ls -A ./model_weights/qwen25-coder-1point5b-mcore 2>/dev/null)" ] || { echo "empty mcore model dir"; exit 1; }

echo "[1/5] record run manifest"
cat > "${RUN_MANIFEST}" <<JSON
{
  "run_id": "${RUN_ID}",
  "method": "fim_sft",
  "branch_id": "${BRANCH_ID}",
  "branch_reason": "data_filter",
  "parent_checkpoint": "./model_weights/qwen25-coder-1point5b-mcore",
  "dataset_id": "${DATASET_ID}",
  "source_parquet": "${SOURCE_DATA}",
  "source_manifest": "${MANIFEST}",
  "train_iters": ${TRAIN_ITERS},
  "save_interval": ${SAVE_INTERVAL},
  "master_port": ${MASTER_PORT},
  "repo_dir": "${REPO_DIR}",
  "fim_tool_dir": "${FIM_TOOL_DIR}",
  "run_dir": "${RUN_DIR}",
  "logs": {
    "data_convert": "${DATA_CONVERT_LOG}",
    "sft": "${SFT_LOG}",
    "outer": "${OUTER_LOG}"
  },
  "checkpoint_dir": "${CKPT_DIR}",
  "converted_dataset_backup": "${DATASET_BACKUP_DIR}"
}
JSON
echo "[ok] run manifest: ${RUN_MANIFEST}"

echo "[2/5] normalize latest FIM tool line endings"
find "${FIM_TOOL_DIR}" -type f -name "*.sh" -exec sed -i 's/\r$//' {} +

echo "[3/5] FIM SFT data conversion from V2.5 filtered train parquet"
mkdir -p logs dataset
rm -rf ./alpaca_fim_dataset
cp "${SOURCE_DATA}" ./dataset/fim_sft_source.parquet
bash "${FIM_TOOL_DIR}/data_convert_qwen25_fim_sft.sh" > "${DATA_CONVERT_LOG}" 2>&1
[ -n "$(ls -A ./alpaca_fim_dataset 2>/dev/null)" ] || { echo "empty converted fim dataset"; exit 1; }
rm -rf "${DATASET_BACKUP_DIR}"
cp -a ./alpaca_fim_dataset "${DATASET_BACKUP_DIR}"
echo "[ok] data convert log: ${DATA_CONVERT_LOG}"
echo "[ok] converted dataset backup: ${DATASET_BACKUP_DIR}"

echo "[4/5] ${TRAIN_ITERS}-step FIM SFT on V2.5 fn-only filtered train"
TUNE_COPY="${SCRIPT_DIR}/tune_qwen25_${RUN_ID}.sh"
cp "${FIM_TOOL_DIR}/tune_qwen25_coder_1point5b_fim_sft_ptd.sh" "${TUNE_COPY}"
sed -i "s#CKPT_SAVE_DIR=\"./ckpt_save_qwen25_coder_fim_sft\"#CKPT_SAVE_DIR=\"${CKPT_DIR}\"#" "${TUNE_COPY}"
sed -i -E "s#^MASTER_PORT=.*#MASTER_PORT=${MASTER_PORT}#" "${TUNE_COPY}"
sed -i "s#TRAIN_ITERS=\"10\"#TRAIN_ITERS=\"${TRAIN_ITERS}\"#" "${TUNE_COPY}"
sed -i 's#SAVE_INTERVAL="${TRAIN_ITERS}"#SAVE_INTERVAL="'"${SAVE_INTERVAL}"'"#' "${TUNE_COPY}"
sed -i 's#EVAL_INTERVAL="${TRAIN_ITERS}"#EVAL_INTERVAL="'"${SAVE_INTERVAL}"'"#' "${TUNE_COPY}"
(cd "${REPO_DIR}" && bash "${TUNE_COPY}") > "${OUTER_LOG}" 2>&1
echo "[ok] sft outer log: ${OUTER_LOG}"

echo "[5/5] checkpoint check"
[ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ] || { echo "empty V2.5 checkpoint"; exit 1; }

echo "[done] fim v2.5 fn-only filtered sft complete"
echo "run_manifest=${RUN_MANIFEST}"
echo "data_convert_log=${DATA_CONVERT_LOG}"
echo "sft_log=${SFT_LOG}"
echo "outer_log=${OUTER_LOG}"
echo "sft_ckpt=${CKPT_DIR}"
echo "dataset_backup=${DATASET_BACKUP_DIR}"
