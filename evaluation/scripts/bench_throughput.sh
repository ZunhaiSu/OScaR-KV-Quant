PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DTYPE="${DTYPE:-auto}"
DECODE_LEN="${DECODE_LEN:-100}"
ITERATION="${ITERATION:-1}"
NUM_BITS="${NUM_BITS:-4}"
QUANT_MODE="${QUANT_MODE:-k-channel}"
GROUP_SIZE="${GROUP_SIZE:-128}"
KV_ROTATION="${KV_ROTATION:-none}"
KV_NORM="${KV_NORM:-0}"
ATTN_BACKEND="${ATTN_BACKEND:-flash_attention_2}"

# BUDGET_POOL=('1024' '2048' '4096' '8192' '16384' '32768')
# BATCH_SIZE=('1' '2' '4' '8' '16' '32')

BUDGET_POOL=("${BUDGET_POOL:-16384}")
BATCH_SIZE=("${BATCH_SIZE:-1}")

for batch_size in ${BATCH_SIZE[@]}; do
    for budget in ${BUDGET_POOL[@]}; do
        "${PYTHON_BIN}" bench_throughput.py \
            --model_path "${MODEL_PATH}" \
            --batch_size $batch_size \
            --context_len $budget \
            --decode_len "${DECODE_LEN}" \
            --iteration "${ITERATION}" \
            --dtype "${DTYPE}" \
            --num_bits "${NUM_BITS}" \
            --quant_mode "${QUANT_MODE}" \
            --group_size "${GROUP_SIZE}" \
            --kv_rotation "${KV_ROTATION}" \
            --kv_norm "${KV_NORM}" \
            --attn_backend "${ATTN_BACKEND}"
    done
done


# flash_attention_2, flash_decoding, oscar
