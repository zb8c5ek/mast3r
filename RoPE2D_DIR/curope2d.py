# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import sys
import torch

# Add CUDA DLL directory for Windows (Python 3.8+ requirement)
if sys.platform == 'win32':
    cuda_path = os.environ.get('CUDA_PATH', os.environ.get('CUDA_HOME', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'))
    cuda_bin = os.path.join(cuda_path, 'bin')
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)

try:
    import curope as _kernels # run `python setup.py install`
except ModuleNotFoundError:
    from . import curope as _kernels # run `python setup.py build_ext --inplace`


class cuRoPE2D_func (torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        # tokens = tokens.clone() # uncomment this if inplace doesn't work
        _kernels.rope_2d( tokens, positions, base, F0 )
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        _kernels.rope_2d( grad_res, positions, base, -F0 )
        ctx.mark_dirty(grad_res)
        return grad_res, None, None, None


class cuRoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq 
        self.F0 = F0

    def forward(self, tokens, positions): 
        cuRoPE2D_func.apply( tokens.transpose(1,2), positions, self.base, self.F0 )
        return tokens
