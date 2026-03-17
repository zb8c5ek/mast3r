# RoPE2D (curope) Installation Guide

RoPE2D is a CUDA-accelerated implementation of 2D Rotary Position Embedding used by DUSt3R and MASt3R.

## Prerequisites

- Python 3.11
- PyTorch with CUDA support (tested with torch 2.8.0+cu126)
- CUDA Toolkit 12.6
- Visual Studio 2017-2025 (Windows) or GCC (Linux)

## Quick Install (Recommended)

The simplest method is to build in-place in the original location:

```powershell
# Set CUDA_HOME
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

# Build in-place
cd d:\mast3r\dust3rDir\croco\models\curope
python setup.py build_ext --inplace
```

### Verify Installation

```powershell
cd d:\mast3r\dust3rDir\croco\models
python -c "from curope import cuRoPE2D; print('Success!')"
```

## Alternative Locations

RoPE2D source code is available in two locations:
- `dust3rDir/croco/models/curope/` - Original location (recommended)
- `RoPE2D_DIR/` - Standalone copy at project root

## Troubleshooting

### Visual Studio Version Error

If you see an error about unsupported Visual Studio version, the `setup.py` has been modified to include `--allow-unsupported-compiler` flag.

If you still have issues, clean and rebuild:

```powershell
cd d:\mast3r\dust3rDir\croco\models\curope
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
python setup.py build_ext --inplace
```

### PyTorch Compatibility Error (`tokens.type()`)

If you see an error about `tokens.type()`, edit `kernels.cu` line 101:

Change:
```cpp
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type(), "rope_2d_cuda", ([&] {
```

To:
```cpp
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.scalar_type(), "rope_2d_cuda", ([&] {
```

### DLL Load Failed (Windows)

If you get `ImportError: DLL load failed while importing curope`:

1. **Use in-place build** (recommended) - Build in `dust3rDir/croco/models/curope/` and import from there
2. **Add CUDA to PATH**:
   ```powershell
   $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;" + $env:PATH
   ```
3. **Use os.add_dll_directory** in Python:
   ```python
   import os
   os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
   import curope
   ```

### Fallback to PyTorch Implementation

If you see this warning:
```
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
```

The code will still work using a slower PyTorch implementation. This is not a fatal error, just a performance notice.

## How It Works

The import chain in `dust3rDir/croco/models/pos_embed.py`:

1. First tries: `from RoPE2D_DIR import cuRoPE2D`
2. Then tries: `from models.curope import cuRoPE2D`
3. Falls back to: Pure PyTorch implementation (slower)

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CUDA_HOME` | CUDA Toolkit installation path | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6` |

## Files

| File | Description |
|------|-------------|
| `setup.py` | Build configuration for CUDA extension |
| `kernels.cu` | CUDA kernel implementation |
| `curope.cpp` | C++ wrapper and CPU fallback |
| `curope2d.py` | Python wrapper class |
| `__init__.py` | Package exports |

After successful build, a new file appears:
- Windows: `curope.cp311-win_amd64.pyd`
- Linux: `curope.cpython-311-x86_64-linux-gnu.so`
