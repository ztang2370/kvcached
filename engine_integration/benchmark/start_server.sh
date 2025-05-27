#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)


MODEL=meta-llama/Llama-3.1-8B
VLLM_PORT=12346
SGL_PORT=30000

op=$1

if [ "$op" == "vllm" ]; then
    export PYTHONPATH="$KVCACHED_DIR:$PYTHONPATH"
    export VLLM_USE_V1=1
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    vllm serve "$MODEL" --disable-log-requests --no-enable-prefix-caching --port="$VLLM_PORT"
elif [ "$op" == "sgl" -o "$op" == "sglang" ]; then
    export PYTHONPATH="$KVCACHED_DIR:$PYTHONPATH"
    python -m sglang.launch_server --model "$MODEL" --disable-radix-cache --trust-remote-code --port "$SGL_PORT"
else
    echo "Invalid option: $op"
    exit 1
fi
