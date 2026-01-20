# Hyper-Connections Integration for Megatron-LM

This directory contains the Hyper-Connections implementation integrated into Megatron-LM.

## Overview

Hyper-Connections is a technique that uses multiple residual streams to improve information flow in transformer layers. See the [Hyper-Connections paper](https://arxiv.org/abs/2409.19606) for details.

This implementation includes two variants:
1. **Hyper-Connections (HC)**: The original implementation
2. **Manifold-Constrained Hyper-Connections (mHC)**: A more stable variant that uses Sinkhorn projection

## Dependencies

To use Hyper-Connections, you need to install:
```bash
pip install einops>=0.8.0
```

## Usage

### Standard Hyper-Connections (HC)

To enable Hyper-Connections in your model, set the following configuration options in `TransformerConfig`:

```python
config = TransformerConfig(
    # ... other config options ...
    use_hyper_connections=True,       # Enable Hyper-Connections
    num_residual_streams=4,           # Number of residual streams (default: 1, disabled)
    hyper_connections_dropout=0.0,    # Dropout for Hyper-Connections (default: 0.0)
    num_fracs=1,                      # Number of fractions for Frac-Connections (default: 1)
)
```

### Manifold-Constrained Hyper-Connections (mHC)

For more stable training, use mHC which uses Sinkhorn projection to constrain mixing matrices:

```python
config = TransformerConfig(
    # ... other config options ...
    use_hyper_connections=True,               # Enable Hyper-Connections
    use_manifold_hyper_connections=True,      # Use mHC instead of HC
    num_residual_streams=4,                   # Number of residual streams
    hyper_connections_dropout=0.0,            # Dropout
    num_fracs=1,                              # Fractions for Frac-Connections
    
    # mHC-specific options
    mhc_sinkhorn_iters=20,                    # Sinkhorn iterations (default: 20)
    mhc_log_domain_sinkhorn=False,            # Use log-domain Sinkhorn for numerical stability
    mhc_num_dynamic_alpha_proposals=1,        # Number of alpha proposals for averaging
)
```

## Why mHC?

mHC addresses training instability issues in the original HC by:
- Using **Sinkhorn-Knopp algorithm** to normalize mixing matrices (doubly stochastic)
- Keeping **Amax stable at ~1.0** instead of letting it grow unbounded
- Using **sigmoid constraints** for pre-branch alpha and beta weights
- Eliminating the need for explicit clamping

Reference: [mHC Reproduction Blog](https://taylorkolasinski.com/notes/mhc-reproduction-part2/)

## Implementation Details

The integration applies Hyper-Connections to both:
- Self-attention branch (in `_forward_attention`)
- MLP branch (in `_forward_mlp`)

The implementation:
1. Expands residual streams before processing (in `TransformerBlock`)
2. Applies width connection (alpha weights) to combine streams for branch input
3. Processes through attention/MLP
4. Applies depth connection (beta weights) to distribute output back to streams
5. Reduces streams back to single residual stream (in `TransformerBlock`)

## Notes

- Hyper-Connections are disabled by default (`use_hyper_connections=False`)
- When disabled (`num_residual_streams=1`), the model behaves exactly as before
- Memory usage increases approximately by `num_residual_streams` times
- The implementation handles dimension conversions between Megatron's `[s, b, h]` format and Hyper-Connections' `[b, s, h]` format
- **MTP (Multi-Token Prediction) layers** automatically disable Hyper-Connections since they use standalone TransformerLayers

