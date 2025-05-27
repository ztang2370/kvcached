#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

setup_vllm() {
    pushd "$ENGINE_DIR"

    git clone -b v0.8.4 https://github.com/vllm-project/vllm.git vllm-v0.8.4
    cd vllm-v0.8.4
    pip install --upgrade pip
    VLLM_USE_PRECOMPILED=1 pip install --editable .

    git apply "$SCRIPT_DIR/kvcached-vllm-v0.8.4.patch"

    popd
}

setup_sgl() {
    pushd "$ENGINE_DIR"

    git clone -b v0.4.6.post2 https://github.com/sgl-project/sglang.git sglang-v0.4.6.post2
    cd sglang-v0.4.6.post2
    pip install --upgrade pip
    pip install -e "python[all]"

    git apply "$SCRIPT_DIR/kvcached-sglang-v0.4.6.post2.patch"

    popd
}

setup_vllm
setup_sgl
