# Hyper-Connections Integration for Megatron-LM

This directory contains the Hyper-Connections implementation integrated into Megatron-LM.

## Overview

Hyper-Connections is a technique that uses multiple residual streams to improve information flow in transformer layers. See the [Hyper-Connections paper](https://arxiv.org/abs/2409.19606) for details.

## Dependencies

To use Hyper-Connections, you need to install:
```bash
pip install einops>=0.8.0
```

## Usage

To enable Hyper-Connections in your model, set the following configuration options in `TransformerConfig`:

```python
config = TransformerConfig(
    # ... other config options ...
    use_hyper_connections=True,      # Enable Hyper-Connections
    num_residual_streams=4,           # Number of residual streams (default: 1, disabled)
    hyper_connections_dropout=0.0,    # Dropout for Hyper-Connections (default: 0.0)
    num_fracs=1,                      # Number of fractions for Frac-Connections (default: 1)
)
```

## Implementation Details

The integration applies Hyper-Connections to both:
- Self-attention branch (in `_forward_attention`)
- MLP branch (in `_forward_mlp`)

The implementation:
1. Expands residual streams before processing
2. Applies width connection (alpha weights) to combine streams for branch input
3. Processes through attention/MLP
4. Applies depth connection (beta weights) to distribute output back to streams
5. Reduces streams back to single residual stream

## Notes

- Hyper-Connections are disabled by default (`use_hyper_connections=False`)
- When disabled (`num_residual_streams=1`), the model behaves exactly as before
- Memory usage increases approximately by `num_residual_streams` times
- The implementation handles dimension conversions between Megatron's `[s, b, h]` format and Hyper-Connections' `[b, s, h]` format

