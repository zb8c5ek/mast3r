# Installation Guide

This guide covers the installation of MASt3R and its dependencies.

## Prerequisites

- Python 3.11
- PyTorch with CUDA support
- CUDA Toolkit 12.6
- Visual Studio 2017-2025 (Windows) or GCC (Linux)

## Step 1: Create Environment

```bash
# Using micromamba
micromamba create -n sfm3r python=3.11
micromamba activate sfm3r

# Install PyTorch 2.8.0 with CUDA 12.6
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Step 2: Install Base Requirements

```bash
pip install -r requirements.txt
pip install -r dust3rDir/requirements.txt

# Optional requirements
pip install -r dust3rDir/requirements_optional.txt
```

## Step 3: Build RoPE2D CUDA Extension

The RoPE2D (curope) CUDA kernels provide faster runtime for positional embeddings.

```bash
# Set CUDA_HOME (Windows)
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

# Build in-place
cd dust3rDir/croco/models/curope
python setup.py build_ext --inplace
cd ../../../..
```

> **For detailed instructions and troubleshooting, see [INSTALL_RoPE2D.md](INSTALL_RoPE2D.md)**

## Step 4: Install dust3r Package

```bash
cd dust3rDir
pip install -e .
cd ..
```

## Step 5: Install mast3r Package

```bash
# From the mast3r root directory
pip install -e .
```

## Step 6: Install Additional Dependencies

```bash
pip install einops
```

## Step 7: Install ASMK (Optional - for retrieval)

ASMK is required for image retrieval functionality.

```bash
pip install cython

cd asmk/cython
cythonize *.pyx
cd ..
pip install .
cd ..
```

## Step 8: Verify Installation

```bash
python -c "from mast3r.model import AsymmetricMASt3R; from dust3r.inference import inference; print('Success!')"
```

## Step 9: Download Checkpoints

```bash
mkdir -p checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
```

Or use the Hugging Face integration (models download automatically).

---

## Optional: AWS ECR Deployment

If you want to push Docker images to AWS ECR:

### Install AWS CLI

**Windows:**
```powershell
winget install Amazon.AWSCLI
```

**Linux/Mac:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### Configure Credentials

```bash
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (e.g., us-east-1), Output format (json)
```

### Push to ECR

```bash
# Get account ID
aws sts get-caller-identity

# Authenticate Docker
aws ecr get-login-password --region YOUR_REGION | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com

# Create repository
aws ecr create-repository --repository-name mast3r --region YOUR_REGION

# Tag and push
docker tag mast3r:latest YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/mast3r:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/mast3r:latest
```

---

## Quick Install (All-in-One)

```bash
# From the mast3r root directory
pip install -r requirements.txt
pip install -r dust3rDir/requirements.txt
pip install einops cython

# Build RoPE2D CUDA extension (set CUDA_HOME first on Windows)
cd dust3rDir/croco/models/curope
python setup.py build_ext --inplace
cd ../../../..

# Build and install ASMK (optional, for retrieval)
cd asmk/cython && cythonize *.pyx && cd ..
pip install . && cd ..

# Install packages
cd dust3rDir && pip install -e . && cd ..
pip install -e .

# Verify
python -c "from mast3r.model import AsymmetricMASt3R; print('Done!')"
```
