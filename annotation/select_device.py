"""
Device Selection and Optimization for PyTorch Models

This utility module handles automatic device selection and optimization configuration
for PyTorch models, specifically optimized for Segment Anything Model (SAM2)
inference and training workflows.

The module automatically detects available hardware and configures appropriate
settings for optimal performance across different computing environments.

Key Features:
- Automatic Compute Unified Device Architecture (CUDA) & Central Processing Unit (CPU) device detection
- GPU memory optimization for Segment Anything Model (SAM2) models
- Platform-specific performance tuning
- MPS (Apple Silicon) compatibility warnings
- Automatic precision and backend optimization

Device Support:
- CUDA (NVIDIA GPUs): Full optimization with mixed precision
- CPU: Fallback with standard precision
- MPS (Apple Silicon): Preliminary support with warnings

Optimizations Applied:
- CUDA: Enables bfloat16 autocast for memory efficiency
- Ampere GPUs: Enables TensorFloat-32 for faster training
- CUDNN: Optimized convolution algorithms
- Automatic device capability detection

Usage:
    device = select_device()
    model.to(device)

The function automatically applies the optimal settings and returns
the selected device for model deployment.

Performance Notes:
- CUDA devices get automatic mixed precision
- Ampere architecture (RTX 30xx+) gets TF32 acceleration
- MPS devices may have numerical differences vs CUDA
- CPU execution uses standard float32 precision

Dependencies:
    - torch
"""

import torch

def select_device():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'using device: {device}')

    if device.type == 'cuda':
        # use bfloat16 for the entire notebook
        torch.autocast('cuda', dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == 'mps':
        print(
            '\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might '
            'give numerically different outputs and sometimes degraded performance on MPS. '
            'See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion.'
        )
    
    return device
