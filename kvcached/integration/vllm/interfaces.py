# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, Optional, Tuple

import torch

from kvcached.kv_cache_manager import KVCacheManager
from kvcached.tp_ipc_util import start_worker_listener_thread
from kvcached.utils import CONTIGUOUS_LAYOUT, PAGE_SIZE, get_kvcached_logger
from kvcached.vmm_ops import (
    create_kv_tensors,
    init_kvcached as _init_kvcached_impl,
    shutdown_kvcached as _shutdown_kvcached_impl,
)

logger = get_kvcached_logger()

_kvcached_initialized: bool = False
_kvcached_device = None
_async_sched = False
_world_size: int = 1
_pp_rank: int = 0
_contiguous_layout: bool = CONTIGUOUS_LAYOUT
_is_worker: bool = False


def should_use_worker_ipc() -> bool:
    return _kvcached_initialized and not _is_worker

def init_kvcached(
    tp_rank: int = 0,
    world_size: int = 1,
    pp_rank: int = 0,
    is_worker: bool = False,
    device: Optional[str] = None,
    async_sched: bool = False,
) -> None:
    global _kvcached_initialized, _kvcached_device, _world_size, _async_sched, _pp_rank, _is_worker
    if _kvcached_initialized:
        # EngineCore call init_kvcached(is_worker=False) first. When TP=1 GPUModelRunner
        # then calls init_kvcached(is_worker=True) in the same process; without this branch
        # the early return would leave _is_worker False, so KVCacheManager would try Unix IPC
        # (broadcast_kv_tensors_created) and fail with ENOENT on the socket path.
        if is_worker and not _is_worker:
            _is_worker = True
            start_worker_listener_thread(tp_rank, pp_rank)
        if async_sched and not _async_sched:
            _async_sched = True
            logger.info("kvcached async scheduler enabled")
        _pp_rank = pp_rank
        _world_size = world_size
        return

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"

    _init_kvcached_impl(device, PAGE_SIZE, _contiguous_layout)
    _kvcached_initialized = True
    _kvcached_device = device
    _world_size = world_size
    _pp_rank = pp_rank
    _async_sched = async_sched
    _is_worker = is_worker

    if _async_sched:
        logger.info("kvcached async scheduler enabled")

    if is_worker:
        # start the listener thread for kv cache management regardless of TP size
        # because the vLLM EngineCore might need to reach this worker if PP > 1
        start_worker_listener_thread(tp_rank, pp_rank)


def shutdown_kvcached() -> None:
    global _kvcached_initialized, _kvcached_device, _async_sched
    if not _kvcached_initialized:
        return

    _shutdown_kvcached_impl()
    _kvcached_initialized = False
    _kvcached_device = None
    _async_sched = False


def alloc_kv_cache(
    kvcache_shape: Tuple[int, ...],
    block_size: int,
    dtype: torch.dtype,
    device: str,
    num_layers: int,
    attention_type: str = "MHA",  # MHA, GQA, or MLA.
    kv_layout: str = "NHD",  # NHD: (num_tokens, head_num, head_dim)
    group_id: int = 0,
    return_raw_buffers: bool = False,
    num_kv_buffers_override: Optional[int] = None,
) -> List[torch.Tensor]:
    """Allocate KV cache tensors for all supported attention types.

    For MHA/GQA, kvcache_shape is expected to be:
      - FlashAttn:  (2, num_blocks, block_size, head_num, head_dim)
      - FlashInfer: (num_blocks, 2, block_size, head_num, head_dim)
    For MLA, kvcache_shape is expected to be:
      - (num_blocks, block_size, head_size)

    For hybrid models (attention + mamba), the caller should set
    ``num_layers`` to the group_size, ``return_raw_buffers=True``,
    and ``num_kv_buffers_override=1``.  Using a single FTensor per
    pool ensures VM page mappings use page_size_bytes granularity
    (K+V combined), matching the ``as_strided_`` access pattern that
    vLLM's ``_update_hybrid_attention_mamba_layout`` applies.

    Returns:
        List[torch.Tensor] when return_raw_buffers is False.
        Otherwise (kv_tensors, raw_info) where raw_info is a dict with:
          buffers            - flat int8 tensors (one per pool when
                               non-contiguous, single base when contiguous)
          num_blocks         - number of blocks per pool
          page_size_bytes    - uniform page size (bytes) shared by all groups
          block_stride_bytes - byte stride between consecutive blocks of the
                               same pool
          num_pools          - number of pools (== num_layers)
          is_contiguous      - whether contiguous layout is used
    """
    if not _kvcached_initialized:
        raise RuntimeError("kvcached is not initialized. Please call init_kvcached() first.")

    if attention_type not in ["MHA", "GQA", "MLA"]:
        raise ValueError(f"Attention type {attention_type} is not supported.")

    if kv_layout != "NHD":
        raise ValueError(f"KV layout {kv_layout} is not supported.")

    is_mla = attention_type == "MLA"
    num_k_or_v = 1 if is_mla else 2

    # --- Validate shape and determine layout indices ---
    if is_mla:
        # MLA shape: (num_blocks, block_size, head_size)
        if len(kvcache_shape) <= 2:
            raise ValueError(f"Unsupported MLA kv cache shape: {kvcache_shape}")
        if kvcache_shape[1] != block_size:
            raise ValueError(
                f"block_size mismatch: kvcache_shape[1]={kvcache_shape[1]} != block_size={block_size}"
            )
        blocks_dim_idx = 0
        permute_order = list(range(len(kvcache_shape)))
        block_mem_bytes = math.prod(kvcache_shape[1:]) * dtype.itemsize
    else:
        # MHA/GQA shape with K/V dimension
        if (len(kvcache_shape) <= 3
                or (kvcache_shape[0] != num_k_or_v and kvcache_shape[1] != num_k_or_v)
                or kvcache_shape[2] != block_size):
            raise ValueError(f"Unsupported kv cache shape: {kvcache_shape}")

        # FlashAttn (num_k_or_v, num_blocks, block_size, head_num, head_dim)
        if kvcache_shape[0] == num_k_or_v:
            blocks_dim_idx = 1
            permute_order = [1, 0] + list(range(2, len(kvcache_shape)))
        # FlashInfer (num_blocks, num_k_or_v, block_size, head_num, head_dim)
        elif kvcache_shape[1] == num_k_or_v:
            blocks_dim_idx = 0
            permute_order = [0, 1] + list(range(2, len(kvcache_shape)))
        else:
            raise ValueError(f"Unsupported kv cache shape: {kvcache_shape}")

        block_mem_bytes = math.prod(kvcache_shape[2:]) * dtype.itemsize

    requested_num_blocks = kvcache_shape[blocks_dim_idx]

    assert torch.cuda.is_available(), "CUDA is not available."

    # --- Compute per-layer memory budget and number of blocks ---
    gpu_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    gpu_mem_bytes_per_layer_k_or_v = gpu_mem_bytes // num_layers // num_k_or_v
    # round down to page size
    gpu_mem_bytes_per_layer_k_or_v = (gpu_mem_bytes_per_layer_k_or_v // PAGE_SIZE) * PAGE_SIZE

    num_blocks_per_layer = gpu_mem_bytes_per_layer_k_or_v // block_mem_bytes
    if requested_num_blocks > num_blocks_per_layer:
        logger.warning(
            f"Requested {requested_num_blocks} blocks, but only {num_blocks_per_layer} blocks are available."
        )

    actual_num_kv_buffers = num_kv_buffers_override if num_kv_buffers_override is not None else num_k_or_v
    raw_kv_tensors = create_kv_tensors(
        gpu_mem_bytes_per_layer_k_or_v * num_k_or_v, dtype.itemsize, device, num_layers,
        num_kv_buffers=actual_num_kv_buffers, group_id=group_id,
    )

    actual_kvcache_shape: List[int] = list(kvcache_shape)
    actual_kvcache_shape[blocks_dim_idx] = num_blocks_per_layer

    page_size_bytes = math.prod(
        actual_kvcache_shape[:blocks_dim_idx] + actual_kvcache_shape[blocks_dim_idx + 1:]
    ) * dtype.itemsize

    # --- Reshape raw tensors into per-layer KV cache views ---
    if not _contiguous_layout:
        kv_tensors: List[torch.Tensor] = []
        if is_mla:
            num_eles = math.prod(actual_kvcache_shape)
            kv_tensors = [
                t.view(dtype=dtype)[:num_eles].view(actual_kvcache_shape)
                for t in raw_kv_tensors
            ]
        else:
            # In the per-layer FTensors, K occupies [0, v_offset) and V
            # occupies [v_offset, 2*v_offset) where v_offset equals
            # gpu_mem_bytes_per_layer_k_or_v.  A plain contiguous view
            # would place V at num_blocks_per_layer * block_mem_bytes,
            # which differs from v_offset when block_mem_bytes does not
            # evenly divide gpu_mem_bytes_per_layer_k_or_v.  Use
            # as_strided so the K/V dimension stride matches the real
            # V offset in the underlying virtual memory.
            v_offset_eles = gpu_mem_bytes_per_layer_k_or_v // dtype.itemsize
            shape = list(actual_kvcache_shape)
            strides = [0] * len(shape)
            strides[-1] = 1
            for i in range(len(shape) - 2, 1, -1):
                strides[i] = strides[i + 1] * shape[i + 1]
            if blocks_dim_idx == 1:              # FlashAttn (2, N, ...)
                strides[1] = strides[2] * shape[2]
                strides[0] = v_offset_eles
            else:                                 # FlashInfer (N, 2, ...)
                strides[0] = strides[2] * shape[2]
                strides[1] = v_offset_eles
            for t in raw_kv_tensors:
                kv_tensors.append(
                    torch.as_strided(t.view(dtype=dtype), shape, strides))
    else:
        layer_elem_shape = actual_kvcache_shape[:blocks_dim_idx] + actual_kvcache_shape[blocks_dim_idx + 1:]
        contiguous_shape = [num_blocks_per_layer, num_layers] + layer_elem_shape
        num_eles = math.prod(contiguous_shape)
        contiguous_tensor = raw_kv_tensors[0].view(dtype=dtype)[:num_eles].view(contiguous_shape)
        kv_tensors = [
            contiguous_tensor[:, i].permute(*permute_order) for i in range(num_layers)
        ]

    if not return_raw_buffers:
        return kv_tensors

    # --- Build raw int8 buffers for hybrid model (mamba) support ---
    if not _contiguous_layout:
        pool_bytes = num_blocks_per_layer * page_size_bytes
        raw_int8 = [t.view(torch.int8)[:pool_bytes] for t in raw_kv_tensors]
        block_stride_bytes = page_size_bytes
    else:
        raw_int8 = [raw_kv_tensors[0].view(torch.int8)]
        block_stride_bytes = num_layers * page_size_bytes

    raw_info = {
        "buffers": raw_int8,
        "num_blocks": num_blocks_per_layer,
        "page_size_bytes": page_size_bytes,
        "block_stride_bytes": block_stride_bytes,
        "num_pools": num_layers,
        "is_contiguous": _contiguous_layout,
    }
    return kv_tensors, raw_info  # type: ignore[return-value]


def get_kv_cache_manager(
    num_blocks: int,
    block_size: int,
    cell_size: int,
    num_layers: int,
    num_kv_buffers: int = 2,
    group_id: int = 0,
) -> KVCacheManager:
    if not _kvcached_initialized:
        raise RuntimeError("kvcached is not initialized. Please call init_kvcached() first.")

    return KVCacheManager(
        num_blocks,
        block_size,
        cell_size,
        num_layers,
        _world_size,
        pp_rank=_pp_rank,
        async_sched=_async_sched,
        num_kv_buffers=num_kv_buffers,
        group_id=group_id,
    )
