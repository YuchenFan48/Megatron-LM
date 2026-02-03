# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

# Use lazy imports to avoid circular import issues
# Import directly from submodules when needed:
#   from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules, GatedDeltaNetCPMode
#   from megatron.core.ssm.kda import KDA, KDASubmodules, KDACPMode
#   from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
#   from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
#   from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
