# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_context_parallel import (
    _all_to_all_cp2hp,
    _all_to_all_hp2cp,
    _redo_attention_load_balancing,
    _undo_attention_load_balancing,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push


class KDACPMode(str, Enum):
    """Context Parallel mode for KDA (Kimi Delta Attention).
    
    HEAD_PARALLEL (default): All-to-all based, each rank processes full sequence with partial heads.
        - Memory usage: O(seq_len * hidden/cp_size)
        - Suitable for: shorter sequences with many heads
        
    SEQUENCE_PARALLEL: Ring-style, each rank processes partial sequence with full heads.
        - Memory usage: O(seq_len/cp_size * hidden)  
        - Suitable for: very long sequences (128K+)
        - Note: Requires sequential state passing between CP ranks
    """
    HEAD_PARALLEL = "head_parallel"
    SEQUENCE_PARALLEL = "sequence_parallel"


import sys
import os
 
if not os.path.exists('/apdcephfs/mnt/cephfs/users/yuchenfan/flash-linear-attention'):
    raise ImportError("Hard code the path to flash-linear-attention in the code -> kda.py")
sys.path.append('/apdcephfs/mnt/cephfs/users/yuchenfan/flash-linear-attention')
from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
from fla.modules import FusedRMSNormGated
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate
from einops import rearrange

HAVE_FLA = True

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


def _create_seq_idx_from_cu_seqlens(
    cu_seqlens: torch.Tensor, total_length: int, device: torch.device
) -> torch.Tensor:
    """
    Create seq_idx tensor from cumulative sequence lengths.
    
    Args:
        cu_seqlens: Cumulative sequence lengths tensor of shape (num_seqs + 1,)
        total_length: Total number of tokens
        device: Device to create tensor on
        
    Returns:
        seq_idx: Tensor of shape (1, total_length) mapping each token to its sequence index
    """
    num_seqs = cu_seqlens.shape[0] - 1
    seq_idx = torch.zeros(total_length, dtype=torch.int32, device=device)
    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_idx[start:end] = i
    return seq_idx.unsqueeze(0)  # Shape: (1, total_length)


def _check_chunk_kda_varlen_support() -> bool:
    """Check if chunk_kda supports cu_seqlens parameter."""
    if not HAVE_FLA:
        return False
    import inspect

    sig = inspect.signature(chunk_kda)
    return 'cu_seqlens' in sig.parameters


logger = logging.getLogger(__name__)


@dataclass
class KDASubmodules:
    """
    Contains the module specs for the input linear, output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class KDA(MegatronModule):
    """KDA (Kimi Delta Attention) layer class

    KDA layer takes input with size [s, b, h]
    and returns output of the same size.
    
    Supports two Context Parallel modes:
    - HEAD_PARALLEL (default): All-to-all based, each rank has full sequence, partial heads
    - SEQUENCE_PARALLEL: Ring-style, each rank has partial sequence, full heads (for long sequences)
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: KDASubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
        cp_mode: Union[str, KDACPMode, None] = None,
        **kwargs,
    ):
        """
        Args:
            config: The config of the model.
            submodules: Contains the module specs for the input and output linear layers.
            layer_number: The layer number of this KDA layer.
            bias: Whether to use bias in the linear layers.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: The initialization range for the causal convolution weights.
            use_qk_l2norm: Whether to use L2 normalization in the kernel of the kda.
            A_init_range: The initialization range for the attention weights.
            pg_collection: The required process groups to use for tensor model parallel and context
                parallel.
            cp_mode: Context parallel mode. If None, reads from config.kda_cp_mode. Options:
                - "head_parallel" or KDACPMode.HEAD_PARALLEL: 
                    Each CP rank processes full sequence with partial heads.
                    Memory: O(seq_len * hidden/cp_size). Good for many heads.
                - "sequence_parallel" or KDACPMode.SEQUENCE_PARALLEL (default):
                    Each CP rank processes partial sequence with full heads.
                    Memory: O(seq_len/cp_size * hidden). Good for very long sequences (128K+).
        """

        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed. Please install it with `pip install flash-linear-attention`."
            )

        super().__init__(config)

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        assert pg_collection is not None, "pg_collection must be provided for KDA"
        self.pg_collection = pg_collection
        self.cp_size = self.pg_collection.cp.size()
        self.cp_rank = self.pg_collection.cp.rank()
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1
        
        # Parse and validate CP mode (read from config if not provided)
        if cp_mode is None:
            cp_mode = getattr(config, 'kda_cp_mode', 'sequence_parallel')
        if isinstance(cp_mode, str):
            cp_mode = KDACPMode(cp_mode)
        self.cp_mode = cp_mode
        
        if self.cp_mode == KDACPMode.SEQUENCE_PARALLEL and self.cp_size > 1:
            logger.info(
                f"KDA layer {layer_number}: Using SEQUENCE_PARALLEL mode for CP. "
                f"Each rank processes seq_len/{self.cp_size} tokens with all heads. "
                f"This is optimal for very long sequences."
            )

        # Attributes from config
        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size

        # Input projection (hidden_states -> q, k, v, f_gate_a, g_gate_a, beta)
        # f_gate_a: for forget gate (first stage projection to head_dim)
        # g_gate_a: for output gate (first stage projection to head_dim)
        # beta: for the beta parameter
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim + self.value_head_dim * 2 + self.num_value_heads
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the KDA layer "
                "input projection output tensor must be a multiple of 16."
            )
        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )

        # Conv1d for QKV (no gate)
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size

        # weight shape: [conv_dim, 1, d_conv]
        # bias shape: [conv_dim]
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.conv1d.weight, "tensor_model_parallel", True)
        if conv_bias:
            setattr(self.conv1d.bias, "tensor_model_parallel", True)

        # KDA-specific parameters
        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size
        
        # A_log parameter for forget gate
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)
        
        # dt_bias for forget gate
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.v_dim_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        
        # Second stage projection for forget gate: head_dim -> v_dim
        self.f_b_proj = nn.Linear(
            self.value_head_dim,
            self.v_dim_local_tp,
            bias=False,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.f_b_proj.weight, "tensor_model_parallel", True)
        
        # Second stage projection for output gate: head_dim -> v_dim
        self.g_b_proj = nn.Linear(
            self.value_head_dim,
            self.v_dim_local_tp,
            bias=False,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.g_b_proj.weight, "tensor_model_parallel", True)

        # Output layernorm with gating (FusedRMSNormGated from FLA)
        self.out_norm = FusedRMSNormGated(
            self.value_head_dim,
            eps=self.config.layernorm_epsilon,
            activation='sigmoid',
        )

        self.out_proj = build_module(
            submodules.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
                # conv1d.weight
                if self.conv_init is not None:
                    nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
                # dt_bias
                nn.init.uniform_(
                    self.dt_bias,
                    0.001,
                    0.01,
                )
                # A_log
                A = torch.empty(
                    self.num_v_heads_local_tp,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(torch.log(A))

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        """
        Perform a forward pass through the KDA module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            packed_seq_params (Optional[PackedSeqParams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) KDA output and bias.

        """
        # TODO: Deal with attention_mask

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "KDA does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("KDA does not support inference for now.")

        # Route to appropriate CP mode implementation
        if self.cp_size > 1 and self.cp_mode == KDACPMode.SEQUENCE_PARALLEL:
            return self._forward_sequence_parallel(
                hidden_states, attention_mask, packed_seq_params
            )
        else:
            return self._forward_head_parallel(
                hidden_states, attention_mask, packed_seq_params
            )

    def _forward_sequence_parallel(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass using SEQUENCE_PARALLEL mode for CP.
        
        Each CP rank processes a chunk of the sequence with ALL heads.
        Recurrent state is passed between CP ranks using P2P communication.
        
        Memory usage: O(seq_len/cp_size * hidden)
        This is optimal for very long sequences.
        
        Note: This mode is NOT compatible with Megatron's sequence_parallel (SP).
        When using SEQUENCE_PARALLEL CP mode, set --sequence-parallel to False.
        """
        # Check if using packed sequences (THD format) - not supported in sequence parallel mode
        is_packed = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if is_packed:
            raise NotImplementedError(
                "SEQUENCE_PARALLEL CP mode does not support packed sequences (THD format). "
                "Please use HEAD_PARALLEL mode for packed sequences."
            )
        
        # Check for incompatible Megatron sequence_parallel
        if self.config.sequence_parallel:
            raise RuntimeError(
                "SEQUENCE_PARALLEL CP mode is NOT compatible with Megatron's sequence_parallel. "
                "Please set --sequence-parallel to False, or use HEAD_PARALLEL CP mode. "
                "SEQUENCE_PARALLEL CP mode already provides sequence memory reduction."
            )
        
        # hidden_states shape: [local_seq_len, batch, hidden]
        # In SEQUENCE_PARALLEL CP mode, local_seq_len = global_seq_len / cp_size
        local_seq_len, batch, _ = hidden_states.shape
        
        # Input projection (full hidden dimension, no head sharding)
        nvtx_range_push(suffix="in_proj")
        proj_out, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")
        
        # In sequence parallel mode, we don't do all-to-all
        # Each rank processes its local sequence chunk with full heads
        
        # Transpose: s b x --> b s x
        proj_out = proj_out.transpose(0, 1)
        
        # Split into q, k, v, f_gate_a, g_gate_a, beta
        qkv, f_gate_a, g_gate_a, beta = torch.split(
            proj_out,
            [
                self.qk_dim_local_tp * 2 + self.v_dim_local_tp,
                self.value_head_dim,  # f_gate_a is first stage projection
                self.value_head_dim,  # g_gate_a is first stage projection
                self.num_value_heads // self.tp_size,
            ],
            dim=-1,
        )
        
        beta = beta.reshape(batch, local_seq_len, -1)
        
        # Convolution on qkv (using full weights, no CP slicing)
        nvtx_range_push(suffix="conv1d")
        
        # For sequence parallel, we need to handle convolution at sequence boundaries
        # This requires receiving the last (kernel_size-1) tokens from previous CP rank
        qkv = self._sequence_parallel_conv1d(qkv, batch, local_seq_len)
        
        nvtx_range_pop(suffix="conv1d")
        
        # Split qkv into query, key, value (full heads)
        query, key, value = torch.split(
            qkv,
            [
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
            ],
            dim=-1,
        )
        query = query.reshape(batch, local_seq_len, -1, self.key_head_dim)
        key = key.reshape(batch, local_seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, local_seq_len, -1, self.value_head_dim)
        
        # Note: L2 norm is applied inside chunk_kda when use_qk_l2norm_in_kernel=True
        # Handle GQA (grouped query attention) if num_value_heads > num_key_heads
        if self.num_value_heads // self.num_key_heads > 1:
            query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
            key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
        
        # Compute forget gate: g = fused_kda_gate(f_b_proj(f_gate_a), A_log, head_dim, dt_bias)
        nvtx_range_push(suffix="forget_gate")
        f_gate = self.f_b_proj(f_gate_a)  # (batch, local_seq_len, v_dim)
        g = fused_kda_gate(f_gate, self.A_log, self.value_head_dim, g_bias=self.dt_bias)
        nvtx_range_pop(suffix="forget_gate")
        
        # Compute beta
        beta = beta.sigmoid()

        # Make contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        
        # Get initial state from previous CP rank (if not first rank)
        nvtx_range_push(suffix="kda")
        initial_state = self._receive_state_from_prev_rank(batch)
        
        # Compute KDA with state passing
        # Note: use_qk_l2norm_in_kernel=True to apply L2 norm inside the kernel (matches reference)
        if self.config.deterministic_mode:
            core_attn_out, final_state = torch_chunk_kda(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=(self.cp_rank < self.cp_size - 1),  # Output state if not last rank
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            )
        else:
            core_attn_out, final_state = chunk_kda(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=(self.cp_rank < self.cp_size - 1),
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            )
        
        # Send final state to next CP rank (if not last rank)
        self._send_state_to_next_rank(final_state)
        nvtx_range_pop(suffix="kda")
        
        # Compute output gate and apply gated norm
        nvtx_range_push(suffix="gated_norm")
        output_gate = self.g_b_proj(g_gate_a)  # (batch, local_seq_len, v_dim)
        output_gate = rearrange(output_gate, 'b s (h d) -> b s h d', d=self.value_head_dim)
        norm_out = self.out_norm(core_attn_out, output_gate)
        nvtx_range_pop(suffix="gated_norm")
        
        # Transpose: b s x --> s b x
        norm_out = norm_out.reshape(batch, local_seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()
        
        # Output projection
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")
        
        return out, out_bias
    
    def _sequence_parallel_conv1d(
        self,
        qkv: Tensor,
        batch: int,
        local_seq_len: int,
    ) -> Tensor:
        """
        Perform causal conv1d for sequence parallel mode.
        
        Handles sequence boundary by receiving context from previous CP rank.
        
        Note: We always use F.conv1d here (not causal_conv1d_fn) because we need
        explicit control over the padding/context at sequence boundaries.
        """
        # qkv: (batch, local_seq_len, conv_dim)
        kernel_size = self.conv_kernel_dim
        
        if self.cp_rank > 0:
            # Receive context (last kernel_size-1 tokens) from previous rank
            context_size = kernel_size - 1
            context = torch.empty(
                batch, context_size, self.conv_dim_local_tp,
                dtype=qkv.dtype, device=qkv.device
            )
            torch.distributed.recv(context, src=self.cp_rank - 1, group=self.pg_collection.cp)
            # Prepend context to qkv
            qkv_with_context = torch.cat([context, qkv], dim=1)
        else:
            # First rank: pad with zeros (causal: no future context)
            qkv_with_context = F.pad(qkv, (0, 0, kernel_size - 1, 0))
        
        # Send last (kernel_size-1) tokens to next rank
        if self.cp_rank < self.cp_size - 1:
            send_buffer = qkv[:, -(kernel_size - 1):, :].contiguous()
            torch.distributed.send(send_buffer, dst=self.cp_rank + 1, group=self.pg_collection.cp)
        
        # Perform conv1d with explicit padding control
        # Input shape: (b, local_seq_len + kernel_size - 1, d)
        qkv_with_context = qkv_with_context.transpose(1, 2).contiguous()  # (b, d, s+pad)
        
        # Always use F.conv1d for explicit control over sequence boundaries
        conv_out = F.conv1d(
            input=qkv_with_context,
            weight=self.conv1d.weight,
            bias=self.conv1d.bias if self.conv_bias else None,
            stride=self.conv1d.stride,
            padding=0,  # We already handled padding manually
            dilation=self.conv1d.dilation,
            groups=self.conv_dim_local_tp,
        )
        # conv_out shape: (b, d, local_seq_len) - F.conv1d with padding=0 reduces length
        qkv_out = self.act_fn(conv_out)
        
        qkv_out = qkv_out.transpose(1, 2)  # (b, local_seq_len, d)
        return qkv_out
    
    def _receive_state_from_prev_rank(self, batch: int) -> Optional[Tensor]:
        """Receive recurrent state from previous CP rank."""
        if self.cp_rank == 0:
            return None
        
        # State shape: (batch, num_heads, key_dim, value_dim)
        num_heads = self.num_value_heads // self.tp_size
        state = torch.empty(
            batch, num_heads, self.key_head_dim, self.value_head_dim,
            dtype=torch.float32,  # State is in fp32
            device=torch.cuda.current_device(),
        )
        torch.distributed.recv(state, src=self.cp_rank - 1, group=self.pg_collection.cp)
        return state
    
    def _send_state_to_next_rank(self, state: Optional[Tensor]):
        """Send recurrent state to next CP rank."""
        if self.cp_rank >= self.cp_size - 1 or state is None:
            return
        
        # State should already be in fp32
        torch.distributed.send(state.contiguous(), dst=self.cp_rank + 1, group=self.pg_collection.cp)

    def _forward_head_parallel(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass using HEAD_PARALLEL mode for CP (original implementation).
        
        All-to-all based: each CP rank processes full sequence with partial heads.
        
        Memory usage: O(seq_len * hidden/cp_size)
        Note: Sequence length is NOT reduced, only head dimension is sharded.
        """
        # Check if using packed sequences (THD format)
        is_packed = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'

        if is_packed:
            # Packed sequence: hidden_states shape is (total_tokens_local, batch=1, hidden_size)
            # where total_tokens_local = total_tokens_global / cp_size
            total_tokens_local, batch, _ = hidden_states.shape
            assert batch == 1, "Packed sequences should have batch size 1"
            # cu_seqlens represents GLOBAL sequence boundaries (same on all CP ranks)
            cu_seqlens_global = packed_seq_params.cu_seqlens_q
            # After all-to-all, we'll have the full sequence gathered
            total_tokens_global = total_tokens_local * self.cp_size
            seq_len = total_tokens_global * self.sp_size
            # We'll compute cu_seqlens and seq_idx after all-to-all
            cu_seqlens = None
            seq_idx = None
        else:
            seq_len, batch, _ = hidden_states.shape
            seq_len = seq_len * self.sp_size * self.cp_size
            cu_seqlens = None
            cu_seqlens_global = None
            seq_idx = None

        # Input projection
        nvtx_range_push(suffix="in_proj")
        proj_out, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        # CP All to All: CP to HP (gather sequence, shard heads)
        # This applies to both packed and non-packed sequences when CP > 1
        proj_out = tensor_a2a_cp2hp(
            proj_out,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
            split_sections=[
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.value_head_dim,  # f_gate_a - NOT split by CP (low-rank)
                self.value_head_dim,  # g_gate_a - NOT split by CP (low-rank)
                self.num_value_heads // self.tp_size,
            ],
        )
        
        # For packed sequences, use global cu_seqlens after all-to-all gathering
        if is_packed:
            # After all-to-all, we have the full sequence gathered
            # cu_seqlens_global already represents global sequence boundaries
            cu_seqlens = cu_seqlens_global
            # Create seq_idx for the gathered sequence
            seq_idx = _create_seq_idx_from_cu_seqlens(
                cu_seqlens, total_tokens_global, hidden_states.device
            )

        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        proj_out = proj_out.transpose(0, 1)

        # Split, reorder, and reshape the tensor into q, k, v, f_gate_a, g_gate_a, beta
        # After all-to-all, the head dimension is sharded by cp_size for QKV and beta
        qkv, f_gate_a, g_gate_a, beta = torch.split(
            proj_out,
            [
                (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,
                self.value_head_dim,  # f_gate_a is NOT sharded by CP
                self.value_head_dim,  # g_gate_a is NOT sharded by CP
                self.num_value_heads // self.tp_size // self.cp_size,
            ],
            dim=-1,
        )
        
        if is_packed:
            # For packed: batch=1, seq_len=total_tokens_global (gathered across CP)
            actual_seq_len = total_tokens_global
        else:
            actual_seq_len = seq_len
            
        beta = beta.reshape(batch, actual_seq_len, -1)

        # Convolution on qkv
        nvtx_range_push(suffix="conv1d")
        
        # Get CP-local weights for convolution
        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=self.pg_collection.cp,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            )
            if self.conv_bias
            else None
        )
        
        if is_packed:
            # For packed sequences, use FLA's causal_conv1d with native cu_seqlens support
            # Input qkv is in (batch=1, seq_len, d) format
            
            if HAVE_FLA and not self.config.deterministic_mode:
                # Use FLA's causal_conv1d directly with cu_seqlens support (Triton kernel)
                # Weight shape: (d, 1, w) -> (d, w)
                conv1d_weight_reshaped = rearrange(conv1d_weight, "d 1 w -> d w")
                
                qkv, _ = fla_causal_conv1d(
                    x=qkv,
                    weight=conv1d_weight_reshaped,
                    bias=conv1d_bias,
                    activation=self.activation,
                    cu_seqlens=cu_seqlens,
                )
            elif causal_conv1d_fn is not None and not self.config.deterministic_mode:
                # Fallback to causal_conv1d_fn with seq_idx
                qkv = qkv.transpose(1, 2)  # (1, seq_len, d) -> (1, d, seq_len)
                
                assert self.activation in ["silu", "swish"]
                qkv = causal_conv1d_fn(
                    x=qkv,
                    weight=conv1d_weight.squeeze(1),  # d, 1, w -> d, w
                    bias=conv1d_bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                )
                qkv = qkv.transpose(1, 2)  # (1, d, seq_len) -> (1, seq_len, d)
            else:
                # Fall back to vectorized conv1d
                qkv = qkv.transpose(1, 2)  # (1, seq_len, d) -> (1, d, seq_len)
                qkv = self._packed_conv1d_fallback(
                    qkv, conv1d_weight, conv1d_bias, cu_seqlens
                )
                qkv = qkv.transpose(1, 2)  # (1, d, seq_len) -> (1, seq_len, d)
        else:
            # Non-packed path: weights already sliced above
            qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
            if (causal_conv1d_fn is None) or self.config.deterministic_mode:
                conv_out = F.conv1d(
                    input=qkv,
                    weight=conv1d_weight,
                    bias=conv1d_bias,
                    stride=self.conv1d.stride,
                    padding=self.conv1d.padding,
                    dilation=self.conv1d.dilation,
                    groups=self.conv_dim_local_tp // self.cp_size,
                )
                qkv = self.act_fn(conv_out[..., :seq_len])
            else:
                assert self.activation in ["silu", "swish"]
                qkv = causal_conv1d_fn(
                    x=qkv,
                    weight=conv1d_weight.squeeze(1),  # d, 1, w -> d, w
                    bias=conv1d_bias,
                    activation=self.activation,
                )
        nvtx_range_pop(suffix="conv1d")
        
        # Split qkv into query, key, and value
        # For non-packed: qkv is in (b, d, s), need to transpose to (b, s, d)
        # For packed: qkv is already in (b, s, d) format
        if not is_packed:
            qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
        
        # After all-to-all, the head dimension is sharded by cp_size for both packed and non-packed
        query, key, value = torch.split(
            qkv,
            [
                self.qk_dim_local_tp // self.cp_size,
                self.qk_dim_local_tp // self.cp_size,
                self.v_dim_local_tp // self.cp_size,
            ],
            dim=-1,
        )
        query = query.reshape(batch, actual_seq_len, -1, self.key_head_dim)
        key = key.reshape(batch, actual_seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, actual_seq_len, -1, self.value_head_dim)
            
        # Note: L2 norm is applied inside chunk_kda when use_qk_l2norm_in_kernel=True
        # Handle GQA (grouped query attention) if num_value_heads > num_key_heads
        if self.num_value_heads // self.num_key_heads > 1:
            query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
            key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)

        # Make contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        beta = beta.contiguous()

        # Calculate forget gate g using fused_kda_gate
        # f_gate_a is already extracted, now apply f_b_proj and fused_kda_gate
        nvtx_range_push(suffix="forget_gate")
        # Get CP-local parameters for f_b_proj
        f_b_proj_weight = get_parameter_local_cp(
            self.f_b_proj.weight, dim=0, cp_group=self.pg_collection.cp
        )
        f_gate = F.linear(f_gate_a, f_b_proj_weight)  # (batch, seq_len, v_dim_local_cp)
        
        # Get CP-local A_log and dt_bias
        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
        dt_bias_local_cp = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=self.pg_collection.cp)
        
        g = fused_kda_gate(f_gate, A_log_local_cp, self.value_head_dim, g_bias=dt_bias_local_cp)
        nvtx_range_pop(suffix="forget_gate")
        
        # Compute beta
        beta = beta.sigmoid()

        g = g.contiguous()

        nvtx_range_push(suffix="kda")
        # Note: use_qk_l2norm_in_kernel=True to apply L2 norm inside the kernel (matches reference)
        if is_packed:
            # For packed sequences, process each sequence separately
            core_attn_out = self._packed_kda(
                query, key, value, g, beta, cu_seqlens
            )
            last_recurrent_state = None
        elif self.config.deterministic_mode:
            core_attn_out, last_recurrent_state = torch_chunk_kda(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_kda(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            )
        nvtx_range_pop(suffix="kda")

        # Compute output gate and apply gated norm
        nvtx_range_push(suffix="gated_norm")
        # Get CP-local parameters for g_b_proj
        g_b_proj_weight = get_parameter_local_cp(
            self.g_b_proj.weight, dim=0, cp_group=self.pg_collection.cp
        )
        output_gate = F.linear(g_gate_a, g_b_proj_weight)  # (batch, seq_len, v_dim_local_cp)
        output_gate = rearrange(output_gate, 'b s (h d) -> b s h d', d=self.value_head_dim)
        norm_out = self.out_norm(core_attn_out, output_gate)
        nvtx_range_pop(suffix="gated_norm")

        # Transpose: b s x --> s b x
        # From bshd back to sbhd format
        norm_out = norm_out.reshape(batch, actual_seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()

        # CP all to all: HP to CP (scatter sequence back to CP ranks)
        # This applies to both packed and non-packed sequences when using CP
        norm_out = tensor_a2a_hp2cp(
            norm_out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp
        )

        # Output projection
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        return out, out_bias

    def _packed_conv1d_fallback(
        self,
        qkv: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        cu_seqlens: Tensor,
    ) -> Tensor:
        """
        Fallback conv1d for packed sequences when causal_conv1d_fn is not available.
        
        Uses vectorized approach with sequence boundary masking for better performance.
        
        Args:
            qkv: Input tensor of shape (batch=1, d, total_tokens)
            weight: Conv1d weight of shape (d, 1, kernel_size)
            bias: Conv1d bias of shape (d,) or None
            cu_seqlens: Cumulative sequence lengths
            
        Returns:
            Output tensor of shape (batch=1, d, total_tokens)
        """
        batch, d, total_tokens = qkv.shape
        assert batch == 1, "Packed sequences should have batch size 1"
        
        kernel_size = weight.shape[-1]
        weight_2d = weight.squeeze(1)  # (d, kernel_size)
        
        # Create position indices for masking
        # We need to mask out positions where convolution would cross sequence boundaries
        positions = torch.arange(total_tokens, device=qkv.device)
        
        # Find which sequence each position belongs to using searchsorted
        # seq_ids[i] = sequence index for position i
        # searchsorted returns the index where position would be inserted in cu_seqlens
        # We subtract 1 and clamp to get the sequence index
        seq_ids = torch.searchsorted(cu_seqlens[1:], positions, right=True)
        seq_ids = seq_ids.clamp(max=cu_seqlens.shape[0] - 2)
        
        # Get sequence starts for each position
        seq_starts = cu_seqlens[:-1].long()
        
        # For each position, compute the distance to the start of its sequence
        # This tells us how many valid positions we can look back
        pos_in_seq = positions - seq_starts[seq_ids]  # (total_tokens,)
        
        # Pad input for causal convolution
        padded = F.pad(qkv, (kernel_size - 1, 0))  # (1, d, total_tokens + kernel_size - 1)
        
        # Unfold to get sliding windows
        # Shape: (1, d, total_tokens, kernel_size)
        unfolded = padded.unfold(dimension=2, size=kernel_size, step=1)
        
        # Create causal mask that respects sequence boundaries
        # For position i, we can only look back min(kernel_size, pos_in_seq[i] + 1) positions
        # Create mask: (total_tokens, kernel_size)
        kernel_positions = torch.arange(kernel_size, device=qkv.device)  # [0, 1, ..., k-1]
        # Distance from current position: kernel_size - 1 - kernel_positions gives [k-1, k-2, ..., 0]
        # i.e., how far back each kernel position looks
        lookback = kernel_size - 1 - kernel_positions  # (kernel_size,)
        
        # Valid if lookback <= pos_in_seq (we have enough history within the sequence)
        # mask[i, j] = 1 if position i can use kernel position j
        mask = (lookback.unsqueeze(0) <= pos_in_seq.unsqueeze(1)).float()  # (total_tokens, kernel_size)
        
        # Apply mask to unfolded tensor
        # unfolded: (1, d, total_tokens, kernel_size)
        # mask: (total_tokens, kernel_size) -> (1, 1, total_tokens, kernel_size)
        masked_unfolded = unfolded * mask.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution weights
        # weight_2d: (d, kernel_size) -> (1, d, 1, kernel_size)
        conv_out = (masked_unfolded * weight_2d.unsqueeze(0).unsqueeze(2)).sum(dim=-1)  # (1, d, total_tokens)
        
        # Add bias
        if bias is not None:
            conv_out = conv_out + bias.view(1, -1, 1)
        
        # Apply activation
        output = self.act_fn(conv_out)
        
        return output

    def _packed_kda(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        g: Tensor,
        beta: Tensor,
        cu_seqlens: Tensor,
    ) -> Tensor:
        """
        Process KDA for packed sequences.
        
        First tries to use FLA's native cu_seqlens support if available.
        Falls back to per-sequence processing if not supported.
        
        Args:
            query: Query tensor of shape (batch=1, total_tokens, num_heads, head_dim)
            key: Key tensor of shape (batch=1, total_tokens, num_heads, head_dim)
            value: Value tensor of shape (batch=1, total_tokens, num_heads, head_dim)
            g: Gating tensor of shape (batch=1, total_tokens, num_heads, head_dim)
            beta: Beta tensor of shape (batch=1, total_tokens, num_heads)
            cu_seqlens: Cumulative sequence lengths
            
        Returns:
            Output tensor of shape (batch=1, total_tokens, num_heads, head_dim)
        """
        batch, total_tokens, num_heads, head_dim = query.shape
        assert batch == 1, "Packed sequences should have batch size 1"
        
        # Check if FLA supports cu_seqlens natively
        if _check_chunk_kda_varlen_support() and not self.config.deterministic_mode:
            # Use FLA's native varlen support
            core_attn_out, _ = chunk_kda(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                cu_seqlens=cu_seqlens,
            )
            return core_attn_out
        
        # Fallback: process each sequence separately
        output = torch.zeros_like(value)
        num_seqs = cu_seqlens.shape[0] - 1
        
        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            
            # Extract this sequence's tensors
            seq_query = query[:, start:end, :, :]  # (1, seq_len, num_heads, head_dim)
            seq_key = key[:, start:end, :, :]
            seq_value = value[:, start:end, :, :]
            seq_g = g[:, start:end, :, :]
            seq_beta = beta[:, start:end, :]
            
            # Apply KDA for this sequence
            if self.config.deterministic_mode:
                seq_out, _ = torch_chunk_kda(
                    seq_query,
                    seq_key,
                    seq_value,
                    g=seq_g,
                    beta=seq_beta,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                )
            else:
                seq_out, _ = chunk_kda(
                    seq_query,
                    seq_key,
                    seq_value,
                    g=seq_g,
                    beta=seq_beta,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                )
            
            output[:, start:end, :, :] = seq_out
            
        return output

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                "A_log": 0,
                "dt_bias": 0,
            },  # parameters sharded across TP
            sharded_offsets=sharded_offsets,
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata['dp_cp_group'],
        )
        # Submodules
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        for name, module in self.named_children():
            if name == "conv1d":
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {f"weight": 0}
                if self.conv_bias:
                    tp_sharding_map[f"bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    tp_sharding_map,
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            elif name in ["f_b_proj", "g_b_proj"]:
                # Add TP sharding for projection layers
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {"weight": 0}
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    tp_sharding_map,
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            elif name == "out_norm":
                # FusedRMSNormGated has its own state dict
                module_sd = module.state_dict(prefix="", keep_vars=True)
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    {},  # No TP sharding for norm
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )

            sharded_state_dict.update(module_sharded_sd)

        # At this point the TP sharding is correctly defined for each tensor, but some of the
        # tensors must be additionally split into separate parts
        in_proj_dim_local_tp = self.in_proj_dim // self.tp_size
        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim_local_tp, (
            in_proj_dim_local_tp,
            sharded_state_dict[f"{prefix}in_proj.weight"],
        )

        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.value_head_dim,  # f_gate_a
                self.value_head_dim,  # g_gate_a
                self.num_value_heads // self.tp_size,  # beta
            ],
            ["query", "key", "value", "f_gate_a", "g_gate_a", "beta"],
            0,
        )

        conv_layer_name_list = ["conv1d.weight"]
        assert (
            sharded_state_dict[f"{prefix}conv1d.weight"].data.size(0) == self.conv_dim_local_tp
        ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.weight"])
        if self.conv_bias:
            conv_layer_name_list.append("conv1d.bias")
            assert (
                sharded_state_dict[f"{prefix}conv1d.bias"].data.size(0) == self.conv_dim_local_tp
            ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.bias"])
        for conv_layer_name in conv_layer_name_list:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp],
                ["query", "key", "value"],
                0,
            )

        return sharded_state_dict


####################
# Sharded state dict utilities
####################
def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: List[int], split_names: List[str], split_dim: int
) -> ShardedTensorFactory:
    """Builds a factory that splits a given ShardedTensor into several independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()  # remove `data` reference

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimensions size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )

    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )

        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size

        assert split_start == orig_sh_ten_no_data.local_shape[split_dim], (
            split_start,
            orig_sh_ten_no_data.local_shape[split_dim],
        )
        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel(), (
            chunk_sh_tens,
            t.shape,
        )
        return chunk_sh_tens

    @torch.no_grad()
    def sh_ten_merge_fn(sub_state_dict):
        return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        orig_sh_ten.key, orig_sh_ten.data, sh_ten_build_fn, sh_ten_merge_fn, orig_sh_ten.replica_id
    )


####################
# Context parallel utilities
####################
def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """Get the local parameter for the current context parallel rank.

    Args:
        param (torch.Tensor): The entire parameter to get the local parameter for.
        dim (int): The dimension to split the parameter along. Usually the dimension of head.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None,
            first split the parameter along the dimension dim into sections,
            then get the local hidden parallel weights separately,
            finally concatenate the local hidden parallel weights along the dimension dim.

    Returns:
        torch.Tensor: The local parameter for the current context parallel rank.
    """

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    # No need to split if CP size is 1.
    if cp_size == 1:
        return param

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(param, split_sections, dim=dim)
        outputs = []
        for p in inputs:
            p = get_parameter_local_cp(p, dim, cp_group)
            outputs.append(p)
        return torch.cat(outputs, dim=dim)

    # Slice the parameter.
    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    param = param[slices]
    return param


def tensor_a2a_cp2hp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    undo_attention_load_balancing: bool = True,
):
    """All-to-all context parallel to hidden parallel.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None, split the tensor along the dimension
            head_dim into sections first, then do all-to-all for each section separately,
            finally concatenate the separated tensors along the dimension head_dim.
        undo_attention_load_balancing (bool): Whether to undo the attention load balancing of CP.

    Returns:
        torch.Tensor: The all-to-all tensor.
    """

    cp_size = cp_group.size()

    # No need to all-to-all if CP size is 1.
    if cp_size == 1:
        return tensor

    # Limitations of mamba_context_parallel._all_to_all_cp2hp.
    assert seq_dim == 0, f"tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got {seq_dim=}"
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_cp2hp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    assert (
        tensor.dim() == 3
    ), f"tensor_a2a_cp2hp only supports 3-d input tensor for now, but got {tensor.dim()=}"

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_cp2hp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                undo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_cp2hp(tensor, cp_group)

    # Undo attention load balancing last if needed.
    if undo_attention_load_balancing:
        tensor = _undo_attention_load_balancing(tensor, cp_size)
    return tensor


def tensor_a2a_hp2cp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    redo_attention_load_balancing: bool = True,
):
    """All-to-all hidden parallel to context parallel.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None, first split the tensor along the
            dimension head_dim into sections, then do all-to-all for each section separately,
            finally concatenate the separated tensors along the dimension head_dim.
        redo_attention_load_balancing (bool): Whether to redo the attention load balancing of HP.

    Returns:
        torch.Tensor: The all-to-all tensor.
    """

    cp_size = cp_group.size()

    # No need to all-to-all if CP size is 1.
    if cp_size == 1:
        return tensor

    # Limitations of mamba_context_parallel._all_to_all_hp2cp.
    assert seq_dim == 0, f"tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got {seq_dim=}"
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_cp2hp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    assert (
        tensor.dim() == 3
    ), f"tensor_a2a_cp2hp only supports 3-d input tensor for now, but got {tensor.dim()=}"

    # Redo attention load balancing first if needed.
    if redo_attention_load_balancing:
        tensor = _redo_attention_load_balancing(tensor, cp_size)

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_hp2cp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                redo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_hp2cp(tensor, cp_group)

    return tensor


####################
# Torch native KDA
####################
def torch_chunk_kda(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    '''
    Torch-native implementation of chunked KDA for deterministic mode.
    Need this because FLA is not deterministic.

    This is an approximation based on the gated delta rule structure.
    For exact KDA computation, use the FLA implementation.
    '''
    from fla.modules.l2norm import l2norm

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    
    query, key, value, beta = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta)
    ]
    # g has shape (batch, seq, heads, head_dim), needs different handling
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size, g.shape[-1])
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # chunk decay - for KDA, g has per-element decay (not per-head like GDN)
    g_cumsum = g.cumsum(dim=-2)  # cumsum over sequence dimension within chunk
    
    # For the decay mask, we need to handle the per-element g
    # decay_mask[i,j] = exp(sum_{t=j+1}^{i} g[t]) for i > j
    # This is a simplification - proper KDA has more complex decay
    g_sum = g_cumsum.sum(dim=-1)  # (batch, heads, num_chunks, chunk_size)
    decay_mask = ((g_sum.unsqueeze(-1) - g_sum.unsqueeze(-2)).tril().exp().float()).tril()
    
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_sum.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        g_i = g_sum[:, :, i]  # (batch, heads, chunk_size)
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g_i[..., None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_i[:, :, -1, None, None].exp()
            + (k_i * (g_i[:, :, -1, None] - g_i).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
