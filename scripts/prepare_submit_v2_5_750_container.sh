#!/usr/bin/env bash
set -euo pipefail

ROOT="${WORK_ROOT:-/workspace/llm4se}"
RUNTIME_REPO="${RUNTIME_REPO:-/workspace/MindSpeed-LLM}"
RUN_ID_ROOT=20260519_142908_fim_sft_v2_5_fnonly_filtered_1000step
STEP=750
STEP_PADDED="$(printf "%07d" "${STEP}")"
SOURCE_CKPT_ROOT="${ROOT}/runs/fim_sft/v2_5_fnonly_filtered/${RUN_ID_ROOT}/checkpoints/ckpt_save_qwen25_coder_fim_sft"
ITER_DIR="${SOURCE_CKPT_ROOT}/iter_${STEP_PADDED}"
VIEW_ROOT="${ROOT}/tmp/ckpt_views/submission_v2_5_fnonly_filtered_750step"
CHECKPOINT_VIEW="${VIEW_ROOT}/iter_${STEP_PADDED}"
BASE_HF_DIR="${RUNTIME_REPO}/model_from_hf/qwen25-coder-1point5b-hf"
CANDIDATE_ROOT="${ROOT}/archives/submission_candidates/v2_5_fnonly_filtered_750step_hf_20260522"
BASE_HF_COPY="${CANDIDATE_ROOT}/base_hf_root"
EXPORT_DIR="${BASE_HF_COPY}/mg2hf"
MANIFEST="${CANDIDATE_ROOT}/submission_prepare_manifest.txt"

echo "[submit-prepare] started $(date -Is)"
[ -d "${ITER_DIR}" ] || { echo "missing iter dir: ${ITER_DIR}"; exit 1; }
[ -d "${BASE_HF_DIR}" ] || { echo "missing base hf dir: ${BASE_HF_DIR}"; exit 1; }
[ -f "${BASE_HF_DIR}/config.json" ] || { echo "missing base config.json"; exit 1; }
[ -f "${BASE_HF_DIR}/tokenizer.json" ] || { echo "missing base tokenizer.json"; exit 1; }

rm -rf "${VIEW_ROOT}"
mkdir -p "${CHECKPOINT_VIEW}"
ln -s "${ITER_DIR}" "${CHECKPOINT_VIEW}/iter_${STEP_PADDED}"
printf "%s" "${STEP}" > "${CHECKPOINT_VIEW}/latest_checkpointed_iteration.txt"

rm -rf "${CANDIDATE_ROOT}"
mkdir -p "${CANDIDATE_ROOT}"
cp -a "${BASE_HF_DIR}" "${BASE_HF_COPY}"

cd "${RUNTIME_REPO}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt.py \
  --use-mcore-models \
  --model-type GPT \
  --model-type-hf llama2 \
  --load-model-type mg \
  --save-model-type hf \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 1 \
  --add-qkv-bias \
  --load-dir "${CHECKPOINT_VIEW}" \
  --save-dir "${BASE_HF_COPY}"

test -d "${EXPORT_DIR}" || { echo "missing export dir: ${EXPORT_DIR}"; exit 1; }
for f in tokenizer.json tokenizer_config.json special_tokens_map.json merges.txt vocab.json LICENSE README.md generation_config.json; do
  if [ -f "${BASE_HF_COPY}/${f}" ] && [ ! -f "${EXPORT_DIR}/${f}" ]; then
    cp -f "${BASE_HF_COPY}/${f}" "${EXPORT_DIR}/${f}"
  fi
done

test -f "${EXPORT_DIR}/config.json" || { echo "missing config.json after convert"; exit 1; }
compgen -G "${EXPORT_DIR}/model.safetensors" >/dev/null || \
compgen -G "${EXPORT_DIR}/model-*.safetensors" >/dev/null || \
compgen -G "${EXPORT_DIR}/pytorch_model.bin" >/dev/null || \
compgen -G "${EXPORT_DIR}/pytorch_model-*.bin" >/dev/null || {
  echo "missing model weights after convert"
  exit 1
}

{
  echo "step=${STEP}"
  echo "run_id_root=${RUN_ID_ROOT}"
  echo "iter_dir=${ITER_DIR}"
  echo "checkpoint_view=${CHECKPOINT_VIEW}"
  echo "candidate_root=${CANDIDATE_ROOT}"
  echo "base_hf_copy=${BASE_HF_COPY}"
  echo "export_dir=${EXPORT_DIR}"
  date -Is
  ls -lh "${EXPORT_DIR}"
} > "${MANIFEST}"

echo "[submit-prepare] export_dir=${EXPORT_DIR}"
echo "[submit-prepare] finished $(date -Is)"
