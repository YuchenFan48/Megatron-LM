# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for Sliding Window Attention (SWA) with Context Parallelism (CP).
"""

import pytest
import torch
import einops

from megatron.core.transformer.dot_product_attention_context_parallel import (
    AttentionFuncionWithContextParallel,
    get_swa_mask_for_cp,
    to_zz_swa_attn_bias,
)


class TestSWAMaskForCP:
    """Test the sliding window attention mask generation for context parallel."""

    def test_swa_mask_single_rank(self):
        """Test SWA mask with cp_size=1 (no context parallel)."""
        sq_local = 8
        sk_total = 8
        window_size = (2, 0)  # Look back 2 positions, causal (no future)
        cp_size = 1
        cp_rank = 0
        device = 'cuda'

        mask = get_swa_mask_for_cp(sq_local, sk_total, window_size, cp_size, cp_rank, device)

        # Verify mask shape
        assert mask.shape == (sq_local, sk_total)

        # For causal SWA with window (2, 0):
        # Position i can attend to positions [max(0, i-2), i]
        # Check specific positions
        expected_attend = [
            [True, False, False, False, False, False, False, False],  # pos 0: attend to [0]
            [True, True, False, False, False, False, False, False],  # pos 1: attend to [0,1]
            [True, True, True, False, False, False, False, False],  # pos 2: attend to [0,1,2]
            [False, True, True, True, False, False, False, False],  # pos 3: attend to [1,2,3]
            [False, False, True, True, True, False, False, False],  # pos 4: attend to [2,3,4]
            [False, False, False, True, True, True, False, False],  # pos 5: attend to [3,4,5]
            [False, False, False, False, True, True, True, False],  # pos 6: attend to [4,5,6]
            [False, False, False, False, False, True, True, True],  # pos 7: attend to [5,6,7]
        ]
        expected_mask = torch.tensor(expected_attend, device=device, dtype=torch.bool)
        expected_mask = ~expected_mask  # Invert: True means masked out

        assert torch.equal(mask, expected_mask), f"Mask mismatch:\n{mask}\n vs expected:\n{expected_mask}"

    def test_swa_mask_infinite_left_window(self):
        """Test SWA mask with infinite left window (full causal attention)."""
        sq_local = 4
        sk_total = 4
        window_size = (-1, 0)  # Infinite left window, causal
        cp_size = 1
        cp_rank = 0
        device = 'cuda'

        mask = get_swa_mask_for_cp(sq_local, sk_total, window_size, cp_size, cp_rank, device)

        # With infinite left window, this is standard causal attention
        # Position i can attend to all positions [0, i]
        expected_attend = [
            [True, False, False, False],  # pos 0
            [True, True, False, False],  # pos 1
            [True, True, True, False],  # pos 2
            [True, True, True, True],  # pos 3
        ]
        expected_mask = torch.tensor(expected_attend, device=device, dtype=torch.bool)
        expected_mask = ~expected_mask

        assert torch.equal(mask, expected_mask)

    def test_swa_attn_bias_shape(self):
        """Test the attention bias tensor shape and values."""
        sq_local = 8
        sk_total = 16
        window_size = (4, 0)
        cp_size = 2
        cp_rank = 0
        nheads = 4
        nheads_k = 4
        heads_k_stride = 1
        device = 'cuda'
        dtype = torch.float32

        attn_bias = to_zz_swa_attn_bias(
            sq_local, sk_total, window_size, cp_size, cp_rank,
            nheads, nheads_k, heads_k_stride, device, dtype
        )

        # Check shape
        expected_shape = (1, heads_k_stride * (nheads // nheads_k), sq_local, sk_total)
        assert attn_bias.shape == expected_shape

        # Check that masked positions have -inf and unmasked have 0
        assert torch.all((attn_bias == 0) | (attn_bias == float('-inf')))


class TestSWAWithContextParallel:
    """Test SWA integration with context parallel attention."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        torch.manual_seed(42)

    def test_swa_cp_forward_basic(self):
        """Test basic forward pass with SWA and CP."""
        # Configuration
        batch_size = 2
        num_heads = 4
        head_dim = 32
        seq_len = 16
        window_size = (4, 0)  # Look back 4 positions, causal

        # Initialize inputs
        q = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
            requires_grad=True,
        )
        k = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
            requires_grad=True,
        )
        v = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
            requires_grad=True,
        )
        attention_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.bool, device='cuda')

        # Run forward pass with SWA
        output = AttentionFuncionWithContextParallel.apply(
            q, k, v, attention_mask, 0.0, 1.0, None, window_size
        )

        # Check output shape
        assert output.shape == (seq_len, batch_size, num_heads, head_dim)

        # Check that output is not all zeros or NaN
        assert not torch.all(output == 0)
        assert not torch.any(torch.isnan(output))

    def test_swa_cp_backward(self):
        """Test backward pass with SWA and CP."""
        # Configuration
        batch_size = 2
        num_heads = 4
        head_dim = 32
        seq_len = 16
        window_size = (4, 0)

        # Initialize inputs
        q = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
            requires_grad=True,
        )
        k = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
            requires_grad=True,
        )
        v = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
            requires_grad=True,
        )
        attention_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.bool, device='cuda')

        # Run forward pass
        output = AttentionFuncionWithContextParallel.apply(
            q, k, v, attention_mask, 0.0, 1.0, None, window_size
        )

        # Run backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are valid
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.any(torch.isnan(q.grad))
        assert not torch.any(torch.isnan(k.grad))
        assert not torch.any(torch.isnan(v.grad))

    def test_swa_vs_full_attention_equivalence(self):
        """Test that SWA with infinite window equals full causal attention."""
        # Configuration
        batch_size = 2
        num_heads = 4
        head_dim = 32
        seq_len = 16
        
        # Initialize inputs
        torch.manual_seed(42)
        q = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
            requires_grad=True,
        )
        k = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
            requires_grad=True,
        )
        v = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
            requires_grad=True,
        )
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device='cuda', dtype=torch.bool),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # Run with no window size (full attention)
        output_full = AttentionFuncionWithContextParallel.apply(
            q.clone(), k.clone(), v.clone(), causal_mask, 0.0, 1.0, None, None
        )

        # Run with infinite window size
        window_size = (-1, 0)  # Infinite left window
        output_swa = AttentionFuncionWithContextParallel.apply(
            q.clone(), k.clone(), v.clone(), causal_mask, 0.0, 1.0, None, window_size
        )

        # They should be approximately equal
        torch.testing.assert_close(
            output_full, output_swa,
            atol=1e-5, rtol=1e-5,
            msg="SWA with infinite window should equal full causal attention"
        )

    def test_swa_window_effect(self):
        """Test that window size actually limits attention range."""
        # Configuration
        batch_size = 1
        num_heads = 1
        head_dim = 32
        seq_len = 8
        
        # Create specific inputs to verify window effect
        torch.manual_seed(42)
        q = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
        )
        k = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
        )
        v = torch.zeros(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
        )
        
        # Set different values for each position
        for i in range(seq_len):
            v[i] = i + 1
        
        attention_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.bool, device='cuda')
        
        # With window_size=(1, 0), position 7 should only see positions 6 and 7
        window_size = (1, 0)
        output = AttentionFuncionWithContextParallel.apply(
            q, k, v, attention_mask, 0.0, 1.0, None, window_size
        )
        
        # The last position should only attend to positions 6 and 7
        # So its output should be a weighted combination of values 7 and 8 (1-indexed)
        last_output = output[-1]
        
        # Check that the output is in the expected range
        # It should be between 7 and 8 (the values at positions 6 and 7)
        min_val = 7.0
        max_val = 8.0
        avg_last = last_output.mean().item()
        assert min_val <= avg_last <= max_val, f"Last position output {avg_last} not in expected range [{min_val}, {max_val}]"


def reference_swa_attention(q, k, v, window_size, scale=1.0):
    """Reference implementation of sliding window attention."""
    # q, k, v: [s, b, h, d]
    sq = q.shape[0]
    sk = k.shape[0]
    
    # Rearrange to [b, h, s, d]
    q = einops.rearrange(q, 's b h d -> b h s d')
    k = einops.rearrange(k, 's b h d -> b h s d')
    v = einops.rearrange(v, 's b h d -> b h s d')
    
    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Create SWA mask
    left_window, right_window = window_size
    q_idx = torch.arange(sq, device=q.device).unsqueeze(1)
    k_idx = torch.arange(sk, device=q.device).unsqueeze(0)
    distance = k_idx - q_idx
    
    if left_window == -1:
        left_mask = torch.zeros_like(distance, dtype=torch.bool)
    else:
        left_mask = distance < -left_window
    
    if right_window == -1:
        right_mask = torch.zeros_like(distance, dtype=torch.bool)
    else:
        right_mask = distance > right_window
    
    mask = left_mask | right_mask
    attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Compute attention weights and output
    attn_weights = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    
    # Rearrange back to [s, b, h, d]
    output = einops.rearrange(output, 'b h s d -> s b h d')
    return output


class TestSWAReferenceComparison:
    """Test SWA CP implementation against reference implementation."""

    def test_swa_matches_reference(self):
        """Test that SWA CP matches reference implementation for cp_size=1."""
        # Configuration
        batch_size = 2
        num_heads = 4
        head_dim = 32
        seq_len = 16
        window_size = (4, 0)
        scale = 1.0 / (head_dim ** 0.5)

        # Initialize inputs
        torch.manual_seed(42)
        q = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
        )
        k = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
        )
        v = torch.rand(
            (seq_len, batch_size, num_heads, head_dim),
            device='cuda',
            dtype=torch.float32,
        )
        attention_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.bool, device='cuda')

        # Run CP implementation
        output_cp = AttentionFuncionWithContextParallel.apply(
            q.clone(), k.clone(), v.clone(), attention_mask, 0.0, scale, None, window_size
        )

        # Run reference implementation
        output_ref = reference_swa_attention(q.clone(), k.clone(), v.clone(), window_size, scale)

        # Compare results
        torch.testing.assert_close(
            output_cp, output_ref,
            atol=1e-4, rtol=1e-4,
            msg="SWA CP output should match reference implementation"
        )
