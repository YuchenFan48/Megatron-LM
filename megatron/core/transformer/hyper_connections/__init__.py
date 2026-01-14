# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Hyper-Connections implementation for Megatron-LM."""

from .hyper_connections import (
    HyperConnections,
    get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions,
)
from .residuals import Residual

__all__ = [
    "HyperConnections",
    "Residual",
    "get_expand_reduce_stream_functions",
    "get_init_and_expand_reduce_stream_functions",
]

