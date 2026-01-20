# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Hyper-Connections implementation for Megatron-LM."""

from .hyper_connections import (
    HyperConnections,
    ManifoldConstrainedHyperConnections,
    mHC,
    Residual,
    StreamEmbed,
    AttentionPoolReduceStream,
    get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions,
    sinkhorn_knopps,
    log_domain_sinkhorn_knopps,
)

__all__ = [
    "HyperConnections",
    "ManifoldConstrainedHyperConnections",
    "mHC",
    "Residual",
    "StreamEmbed",
    "AttentionPoolReduceStream",
    "get_expand_reduce_stream_functions",
    "get_init_and_expand_reduce_stream_functions",
    "sinkhorn_knopps",
    "log_domain_sinkhorn_knopps",
]

