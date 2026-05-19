PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
DTYPE="${DTYPE:-auto}"
NUM_BITS="${NUM_BITS:-4}"
QUANT_MODE="${QUANT_MODE:-k-channel}"
GROUP_SIZE="${GROUP_SIZE:-128}"
KV_ROTATION="${KV_ROTATION:-none}"
KV_NORM="${KV_NORM:-0}"
ATTN_BACKEND="${ATTN_BACKEND:-oscar}"

"${PYTHON_BIN}" example.py \
    --model_path "${MODEL_PATH}" \
    --max_length "${MAX_LENGTH}" \
    --dtype "${DTYPE}" \
    --num_bits "${NUM_BITS}" \
    --quant_mode "${QUANT_MODE}" \
    --group_size "${GROUP_SIZE}" \
    --kv_rotation "${KV_ROTATION}" \
    --kv_norm "${KV_NORM}" \
    --attn_backend "${ATTN_BACKEND}" # flash_attention_2, flash_decoding, oscar
