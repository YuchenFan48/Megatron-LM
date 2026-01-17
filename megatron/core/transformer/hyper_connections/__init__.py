# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Hyper-Connections implementation for Megatron-LM."""

from .hyper_connections import (
    HyperConnections,
    Residual,
    StreamEmbed,
    AttentionPoolReduceStream,
    get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions,
)

__all__ = [
    "HyperConnections",
    "Residual",
    "StreamEmbed",
    "AttentionPoolReduceStream",
    "get_expand_reduce_stream_functions",
    "get_init_and_expand_reduce_stream_functions",
]

