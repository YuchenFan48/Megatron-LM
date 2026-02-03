# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Some of this code was adopted from https://github.com/zhuzilin/ring-flash-attention/
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


@torch.no_grad
def eager_attn_fwd(q, k, v, attn_bias, sinks, scale, dropout):
    """Forward pass for eager attention"""

    # Rearrange query, key, value to (b, h, s, d)
    b, sq, h, d = q.shape
    sk = k.shape[1]
    _q = einops.rearrange(q, 'b s h d -> b h s d')
    _k = einops.rearrange(k, 'b s h d -> b h d s')
    _v = einops.rearrange(v, 'b s h d -> b h s d')

    # Compute attention weights
    attn_w = torch.matmul(_q, _k) * scale
    attn_w = attn_w + attn_bias

    # Add sinks to attention weights
    if sinks is None:
        logits = attn_w
    else:
        _sinks = sinks.reshape(1, h, 1, 1).expand(b, -1, sq, 1)
        logits = torch.cat([attn_w, _sinks], dim=-1)

    # Compute attention scores
    probs = F.softmax(logits, dim=-1, dtype=logits.dtype)
    if sinks is None:
        attn_w = probs
    else:
        attn_w = probs[..., :-1]  # Drop the sink

    # Compute attention output
    attn_output = torch.matmul(attn_w, _v)
    attn_output = einops.rearrange(attn_output, 'b h s d -> b s h d')
    attn_output = attn_output.contiguous()

    return attn_output, probs


@torch.no_grad
def eager_attn_bwd(q, k, v, attn_bias, sinks, scale, dropout, attn_output, probs, grad_output):
    """Backward pass for eager attention"""

    # Rearrange query, key, value to (b, h, s, d)
    b, sq, h, d = q.shape
    sk = k.shape[1]
    _q_T = einops.rearrange(q, 'b s h d -> b h d s')
    _k_T = einops.rearrange(k, 'b s h d -> b h s d')
    _v_T = einops.rearrange(v, ' b s h d -> b h d s')

    # Backward pass for score @ value
    if sinks is None:
        attn_w = probs
    else:
        attn_w = probs[..., :-1]  # Drop the sink
    grad_output = einops.rearrange(grad_output, 'b s h d -> b h s d')
    attn_w_T = einops.rearrange(attn_w, ' b h sq sk -> b h sk sq')
    grad__v = torch.matmul(attn_w_T, grad_output)
    grad_attn_w = torch.matmul(grad_output, _v_T)

    # Backward pass for softmax
    if sinks is None:
        grad_probs = grad_attn_w
    else:
        dummy = torch.zeros((b, h, sq, 1), device=q.device, dtype=q.dtype)
        grad_probs = torch.cat([grad_attn_w, dummy], dim=3)
    del grad_attn_w
    grad_logits = torch._softmax_backward_data(
        grad_probs, probs, -1, probs.dtype
    )  # [b, h, sq, sk+1]

    # Backward pass for adding sinks
    if sinks is None:
        grad_sinks = None
        grad_attn_w = grad_logits
    else:
        grad__sinks = grad_logits[:, :, :, -1]  # [b, h, sq]
        grad_sinks = einops.rearrange(grad__sinks, 'b h s -> h (b s)').sum(-1)
        grad_attn_w = grad_logits[:, :, :, :-1].contiguous()  # [b, h, sq, sk]

    # Backward pass for q @ K^T
    grad_attn_w *= scale
    grad__q = torch.matmul(grad_attn_w, _k_T)
    grad__k = torch.matmul(_q_T, grad_attn_w)

    # Rearrange grads to (b, s, h, d)
    grad_v = einops.rearrange(grad__v, 'b h s d -> b s h d')
    grad_k = einops.rearrange(grad__k, 'b h d s -> b s h d')
    grad_q = einops.rearrange(grad__q, 'b h s d -> b s h d')
    return grad_q, grad_k, grad_v, grad_sinks


class AllGatherComm:
    """All gather communication with async operations"""

    def __init__(self, group=None) -> None:
        self.group = group
        self.handles = []

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        '''All gather the input tensor to the output tensor'''

        if self.group is None:
            output_tensor.copy_(input_tensor)
        else:
            handle = torch.distributed.all_gather_into_tensor(
                output_tensor, input_tensor, group=self.group, async_op=True
            )
            self.handles.append(handle)

    def wait(self):
        '''Wait for all gather operations to complete'''

        if self.group is not None:
            for handle in self.handles:
                handle.wait()
            self.handles = []


def to_zz_mask_attn_bias(attention_mask, cp_size, nheads, nheads_k, heads_k_stride, device, dtype):
    '''Convert the attention mask to the attention bias'''

    if cp_size == 1:
        zz_mask = attention_mask
    else:
        chunked = attention_mask.chunk(dim=3, chunks=cp_size * 2)
        zz_mask = [_x for _p in zip(chunked[:cp_size], reversed(chunked[cp_size:])) for _x in _p]
        zz_mask = torch.cat(zz_mask, dim=3)
    attn_bias = torch.zeros(zz_mask.shape, device=device, dtype=dtype)
    attn_bias.masked_fill_(zz_mask, float('-inf'))
    attn_bias = attn_bias.expand(-1, heads_k_stride * (nheads // nheads_k), -1, -1)
    return attn_bias


def get_swa_mask_for_cp(sq_local, sk_total, window_size, cp_size, cp_rank, device):
    """
    Generate sliding window attention mask for context parallel.
    
    In zigzag (ZZ) partitioning with cp_size ranks:
    - The sequence is split into 2*cp_size chunks
    - rank i holds chunk[i] and chunk[2*cp_size-1-i]
    
    Args:
        sq_local: local query sequence length per rank
        sk_total: total key sequence length (after all-gather)
        window_size: tuple of (left_window, right_window), where -1 means infinite
        cp_size: context parallel size
        cp_rank: current rank in context parallel group
        device: device to create the mask on
        
    Returns:
        mask: boolean mask of shape [sq_local, sk_total], True means masked (not attended)
    """
    left_window, right_window = window_size
    
    # Calculate chunk size
    chunk_size = sq_local // 2  # Each rank holds 2 chunks
    
    # Create local query indices and map to global positions
    # In ZZ: rank i holds positions from chunk[i] (first half) and chunk[2*cp_size-1-i] (second half)
    local_q_indices = torch.arange(sq_local, device=device)
    
    # First half of local queries come from chunk[cp_rank]
    # Second half come from chunk[2*cp_size-1-cp_rank]
    first_chunk_global_start = cp_rank * chunk_size
    second_chunk_global_start = (2 * cp_size - 1 - cp_rank) * chunk_size
    
    global_q_pos = torch.where(
        local_q_indices < chunk_size,
        local_q_indices + first_chunk_global_start,
        (local_q_indices - chunk_size) + second_chunk_global_start
    )
    
    # Create global key indices - after all-gather in ZZ order
    # The all-gathered KV has layout: [chunk0, chunk7, chunk1, chunk6, chunk2, chunk5, chunk3, chunk4] for cp_size=4
    local_k_indices = torch.arange(sk_total, device=device)
    
    # Map ZZ-ordered local k indices to global positions
    k_chunk_idx = local_k_indices // chunk_size
    k_pos_in_chunk = local_k_indices % chunk_size
    
    # In ZZ order: even positions (0,2,4,...) are from first half of chunks (0,1,2,...)
    #              odd positions (1,3,5,...) are from second half reversed (2*cp_size-1, 2*cp_size-2, ...)
    global_k_chunk = torch.where(
        k_chunk_idx % 2 == 0,
        k_chunk_idx // 2,  # Even: chunk[i//2]
        2 * cp_size - 1 - k_chunk_idx // 2  # Odd: chunk[2*cp_size-1-i//2]
    )
    global_k_pos = global_k_chunk * chunk_size + k_pos_in_chunk
    
    # Calculate distance: positive means key is after query (future), negative means before (past)
    # For causal attention with SWA: attend to keys within [q_pos - left_window, q_pos + right_window]
    # distance[i, j] = global_k_pos[j] - global_q_pos[i]
    distance = global_k_pos.unsqueeze(0) - global_q_pos.unsqueeze(1)  # [sq_local, sk_total]
    
    # Create mask: mask out positions outside the window
    # Attend if: -left_window <= distance <= right_window
    # Note: For causal SWA, right_window is typically 0 (can't attend to future)
    if left_window == -1:
        left_mask = torch.zeros_like(distance, dtype=torch.bool)
    else:
        left_mask = distance < -left_window  # Too far in the past
    
    if right_window == -1:
        right_mask = torch.zeros_like(distance, dtype=torch.bool)
    else:
        right_mask = distance > right_window  # Too far in the future (or non-causal)
    
    mask = left_mask | right_mask
    
    return mask


def to_zz_swa_attn_bias(sq_local, sk_total, window_size, cp_size, cp_rank, nheads, nheads_k, heads_k_stride, device, dtype):
    """
    Create attention bias for sliding window attention with context parallel.
    
    Args:
        sq_local: local query sequence length
        sk_total: total key sequence length after all-gather  
        window_size: tuple of (left_window, right_window)
        cp_size: context parallel size
        cp_rank: current rank
        nheads: number of query heads
        nheads_k: number of key/value heads  
        heads_k_stride: stride for iterating over kv heads
        device: device
        dtype: data type
        
    Returns:
        attn_bias: attention bias of shape [1, heads_k_stride * (nheads // nheads_k), sq_local, sk_total]
    """
    # Get SWA mask for CP
    swa_mask = get_swa_mask_for_cp(sq_local, sk_total, window_size, cp_size, cp_rank, device)
    
    # Convert to attention bias: [sq_local, sk_total] -> [1, 1, sq_local, sk_total]
    swa_mask = swa_mask.unsqueeze(0).unsqueeze(0)
    
    attn_bias = torch.zeros(swa_mask.shape, device=device, dtype=dtype)
    attn_bias.masked_fill_(swa_mask, float('-inf'))
    attn_bias = attn_bias.expand(-1, heads_k_stride * (nheads // nheads_k), -1, -1)
    
    return attn_bias


class AttentionFuncionWithContextParallel(torch.autograd.Function):
    """Native attention function with context parallelism."""

    @staticmethod
    def forward(ctx, q, k, v, attention_mask, attention_dropout, softmax_scale, pg, window_size=None):
        '''Forward pass for the native attention function with context parallelism
        
        Args:
            q: query tensor of shape [s, b, h, d]
            k: key tensor of shape [s, b, h_kv, d]
            v: value tensor of shape [s, b, h_kv, d]
            attention_mask: attention mask tensor
            attention_dropout: dropout rate
            softmax_scale: scale factor for softmax
            pg: context parallel process group
            window_size: optional tuple (left_window, right_window) for sliding window attention
        '''

        # Assert einops exists
        if not HAVE_EINOPS:
            raise ImportError("einops is required by the attention CP but cannot be imported.")

        # Initialize communication group and constants
        cp_size = 1
        cp_rank = 0
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
            cp_rank = torch.distributed.get_rank(pg)
        comm = AllGatherComm(group=pg)
        nheads = q.shape[2]
        nheads_k = k.shape[2]
        heads_k_stride = 1
        assert nheads % nheads_k == 0 and nheads_k % heads_k_stride == 0
        outs = []
        probs = []

        # Initialize KV buffers
        kv_buffer = torch.empty(
            (2, k.shape[0] * cp_size, k.shape[1], heads_k_stride, k.shape[3]),
            dtype=k.dtype,
            device=k.device,
        )
        kv_buffer_copy = torch.empty_like(kv_buffer)

        # All-gather first chunk of KV buffers
        k_0 = k[:, :, :heads_k_stride].contiguous()
        v_0 = v[:, :, :heads_k_stride].contiguous()
        comm.all_gather(kv_buffer_copy[0], k_0)
        comm.all_gather(kv_buffer_copy[1], v_0)

        # Prepare attention bias - use SWA mask if window_size is provided
        sq_local = q.shape[0]  # local query sequence length
        sk_total = k.shape[0] * cp_size  # total key sequence length after all-gather
        if window_size is not None:
            attn_bias = to_zz_swa_attn_bias(
                sq_local, sk_total, window_size, cp_size, cp_rank,
                nheads, nheads_k, heads_k_stride, q.device, q.dtype
            )
        else:
            attn_bias = to_zz_mask_attn_bias(
                attention_mask, cp_size, nheads, nheads_k, heads_k_stride, q.device, q.dtype
            )

        # Iterate over heads
        for i in range(0, nheads_k, heads_k_stride):
            # Wait for previous all-gather to complete
            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
            # All-gather the next portion of KV buffers if not the last iteration
            if i < nheads_k - heads_k_stride:
                kvsl = i + heads_k_stride
                kvsr = kvsl + heads_k_stride
                send_k = k[:, :, kvsl:kvsr].contiguous()
                send_v = v[:, :, kvsl:kvsr].contiguous()
                comm.all_gather(kv_buffer_copy[0], send_k)
                comm.all_gather(kv_buffer_copy[1], send_v)

            # Prepare query, key, value for attention
            q_i = q[:, :, i * nheads // nheads_k : (i + heads_k_stride) * nheads // nheads_k]
            k_i = kv_buffer[0]
            v_i = kv_buffer[1]

            # Rearrange query, key, value to (b, s, h, d)
            q_i = einops.rearrange(q_i, 's b h d -> b s h d')
            k_i = einops.rearrange(k_i, 's b h d -> b s h d')
            v_i = einops.rearrange(v_i, 's b h d -> b s h d')

            # Forward pass
            out_i, probs_i = eager_attn_fwd(
                q_i, k_i, v_i, attn_bias, None, softmax_scale, attention_dropout
            )
            outs.append(out_i)
            probs.append(probs_i)

        # Concatenate outputs and rearrange to (s, b, h, d)
        out = torch.cat(outs, dim=2)
        out = einops.rearrange(out, 'b s h d -> s b h d')

        # Save contexts for backward pass
        ctx.save_for_backward(q, k, v, attention_mask, *outs, *probs)
        ctx.dropout = attention_dropout
        ctx.scale = softmax_scale
        ctx.heads_k_stride = heads_k_stride  # TODO make it configurable
        ctx.pg = pg
        ctx.window_size = window_size

        return out

    @staticmethod
    def backward(ctx, dout):
        '''Backward pass for the native attention function with context parallelism'''

        # Initialize or resume constants and communication group
        q, k, v, attention_mask, *rest = ctx.saved_tensors
        nheads = q.shape[2]
        nheads_k = k.shape[2]
        heads_k_stride = ctx.heads_k_stride
        window_size = ctx.window_size
        assert nheads_k % heads_k_stride == 0
        outs = rest[: nheads_k // heads_k_stride]
        probs = rest[nheads_k // heads_k_stride :]
        pg = ctx.pg
        cp_size = 1
        cp_rank = 0
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
            cp_rank = torch.distributed.get_rank(pg)
        comm = AllGatherComm(group=pg)

        # Initialize KV buffers
        kv_buffer = torch.empty(
            (2, k.shape[0] * cp_size, k.shape[1], heads_k_stride, k.shape[3]),
            dtype=k.dtype,
            device=k.device,
        )
        kv_buffer_copy = torch.empty_like(kv_buffer)

        # All-gather first chunk of KV buffers
        dq = []
        dk = []
        dv = []
        k_0 = k[:, :, :heads_k_stride].contiguous()
        v_0 = v[:, :, :heads_k_stride].contiguous()
        comm.all_gather(kv_buffer_copy[0], k_0)
        comm.all_gather(kv_buffer_copy[1], v_0)

        # Prepare attention bias - use SWA mask if window_size is provided
        sq_local = q.shape[0]
        sk_total = k.shape[0] * cp_size
        if window_size is not None:
            attn_bias = to_zz_swa_attn_bias(
                sq_local, sk_total, window_size, cp_size, cp_rank,
                nheads, nheads_k, heads_k_stride, q.device, q.dtype
            )
        else:
            attn_bias = to_zz_mask_attn_bias(
                attention_mask, cp_size, nheads, nheads_k, heads_k_stride, q.device, q.dtype
            )

        # Iterate over heads
        for i in range(0, nheads_k, heads_k_stride):
            # Slice query and output for this iteration
            q_slice = slice(i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k)
            q_i = q[:, :, q_slice]
            dout_i = dout[:, :, q_slice]

            # Wait for previous all-gather to complete
            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

            # All-gather the next portion of KV buffers if not the last iteration
            if i < nheads_k - heads_k_stride:
                kvsl = i + heads_k_stride
                kvsr = kvsl + heads_k_stride
                send_k = k[:, :, kvsl:kvsr].contiguous()
                send_v = v[:, :, kvsl:kvsr].contiguous()
                comm.all_gather(kv_buffer_copy[0], send_k)
                comm.all_gather(kv_buffer_copy[1], send_v)

            # Prepare key, value for attention
            k_i = kv_buffer[0]
            v_i = kv_buffer[1]

            # Rearrange query, key, value to (b, s, h, d)
            q_i = einops.rearrange(q_i, 's b h d -> b s h d')
            k_i = einops.rearrange(k_i, 's b h d -> b s h d')
            v_i = einops.rearrange(v_i, 's b h d -> b s h d')
            dout_i = einops.rearrange(dout_i, 's b h d -> b s h d')

            # Backward pass
            dq_i, _dk_i, _dv_i, _ = eager_attn_bwd(
                q_i, k_i, v_i, attn_bias, None, ctx.scale, ctx.dropout, outs[i], probs[i], dout_i
            )

            # Rearrange gradients to (s, b, h, d)
            dq_i = einops.rearrange(dq_i, 'b s h d -> s b h d')
            _dk_i = einops.rearrange(_dk_i, 'b s h d -> s b h d')
            _dv_i = einops.rearrange(_dv_i, 'b s h d -> s b h d')
            if pg is None:
                dk_i = _dk_i
                dv_i = _dv_i
            else:
                # Reduce-scatter gradients if CP > 1
                dk_i = torch.zeros(
                    (k_i.shape[1] // cp_size, k_i.shape[0], k_i.shape[2], k_i.shape[3]),
                    device=k_i.device,
                    dtype=k_i.dtype,
                )
                dv_i = torch.zeros(
                    (v_i.shape[1] // cp_size, v_i.shape[0], v_i.shape[2], v_i.shape[3]),
                    device=v_i.device,
                    dtype=v_i.dtype,
                )
                torch.distributed.reduce_scatter_tensor(dk_i, _dk_i, group=pg)
                torch.distributed.reduce_scatter_tensor(dv_i, _dv_i, group=pg)

            # Collect gradients
            dq.append(dq_i)
            dk.append(dk_i)
            dv.append(dv_i)

        # Concatenate gradients and return
        dq = torch.cat(dq, dim=2)
        dk = torch.cat(dk, dim=2)
        dv = torch.cat(dv, dim=2)
        return dq, dk, dv, None, None, None, None, None
