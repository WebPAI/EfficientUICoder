#!/bin/bash

MODEL_PATH=${MODEL_PATH:-"Your_model_path"}
EVAL_DIR=${EVAL_DIR:-"Your_eval_dir"}
RESULT_DIR=${RESULT_DIR:-"Your_result_dir"}
KEEP_INDICES_DIR=${KEEP_INDICES_DIR:-"Your_keep_indices_dir"}

if [[ -z "$MODEL_PATH" || "$MODEL_PATH" == "Your_model_path" ]]; then
    echo "Error: MODEL_PATH is not set correctly"
    exit 1
fi
if [[ -z "$EVAL_DIR" || "$EVAL_DIR" == "Your_eval_dir" ]]; then
    echo "Error: EVAL_DIR is not set correctly"
    exit 1
fi
if [[ -z "$RESULT_DIR" || "$RESULT_DIR" == "Your_result_dir" ]]; then
    echo "Error: RESULT_DIR is not set correctly"
    exit 1
fi
if [[ -z "$KEEP_INDICES_DIR" || "$KEEP_INDICES_DIR" == "Your_keep_indices_dir" ]]; then
    echo "Error: KEEP_INDICES_DIR is not set correctly"
    exit 1
fi

CONV_MODE=${CONV_MODE:-"llava_v1"} # "'llava_v1' is used for LLaVA-v1.6-7B/13B, 'chatml_direct' is used for LLaVA-v1.6-34B"
EXP_MODE=${EXP_MODE:-1} # help="0=end-to-end, 1=EfficientUICoder"
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
DECAY_FACTOR=${DECAY_FACTOR:-0.5}
PENALTY_STEP=${PENALTY_STEP:-3}

SCRIPT_DIR=$(dirname $(realpath "$0"))
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")

cd "$PROJECT_ROOT"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

python -m efficient_uicoder.LLaVA_v1_6.eval \
    --model-path "$MODEL_PATH" \
    --eval-dir "$EVAL_DIR" \
    --result-dir "$RESULT_DIR" \
    --keep-indices-dir "$KEEP_INDICES_DIR" \
    --conv-mode "$CONV_MODE" \
    --exp-mode "$EXP_MODE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --decay-factor "$DECAY_FACTOR" \
    --penalty-step "$PENALTY_STEP"
