# Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **Storage**: 2 GB free disk space
- **CPU**: Multi-core processor (4+ cores recommended)

### Optional (for GPU acceleration)
- **GPU**: NVIDIA GPU with CUDA support
- **CUDA**: 11.0 or higher
- **cuDNN**: Compatible version with your CUDA installation

---

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/sauwal948-ops/equitable-hrc-dqn.git
cd equitable-hrc-dqn

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

### Method 2: Conda Install

```bash
# 1. Clone the repository
git clone https://github.com/sauwal948-ops/equitable-hrc-dqn.git
cd equitable-hrc-dqn

# 2. Create conda environment
conda create -n hrc-dqn python=3.9
conda activate hrc-dqn

# 3. Install PyTorch (with CUDA if available)
# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
# For CPU only:
conda install pytorch torchvision cpuonly -c pytorch

# 4. Install remaining dependencies
pip install -r requirements.txt
```

### Method 3: Docker (Advanced)

```bash
# Coming soon - Docker container for reproducible environment
```

---

## Verification

### Test Your Installation

Run the verification script:

```bash
python code/verify_installation.py
```

Expected output:
```
✓ PyTorch installed: 1.13.0
✓ CUDA available: True (or False for CPU)
✓ NumPy installed: 1.23.0
✓ Pandas installed: 1.5.0
✓ Matplotlib installed: 3.6.0
✓ All dependencies satisfied!
```

### Run Quick Test

```bash
# Test the DQN agent initialization
python -c "from code.dqn_agent import DQNAgent; print('DQN Agent loaded successfully!')"

# Test the environment
python -c "from code.environment import CementBaggingEnv; print('Environment loaded successfully!')"
```

---

## Troubleshooting

### Common Issues

#### Issue 1: PyTorch CUDA not available

**Symptoms:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solution:**
```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Issue 2: ImportError for missing packages

**Symptoms:**
```
ModuleNotFoundError: No module named 'tqdm'
```

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

#### Issue 3: Permission denied on Linux/macOS

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Use virtual environment instead of system Python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Issue 4: Matplotlib backend issues on headless servers

**Symptoms:**
```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```

**Solution:**
Add to your script:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

---

## GPU Acceleration (Optional)

### Check CUDA Availability

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
```

### Install CUDA Toolkit (if needed)

**Ubuntu/Linux:**
```bash
# CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

**Windows:**
Download from: https://developer.nvidia.com/cuda-downloads

---

## Development Installation

For contributing to the project:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

---

## Updating the Repository

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
```

---

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv/

# Remove repository (if desired)
cd ..
rm -rf equitable-hrc-dqn/
```

---

## Platform-Specific Notes

### Windows
- Use `python` instead of `python3`
- Use backslashes `\` for paths or forward slashes `/` in Python
- PowerShell may require execution policy changes: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### macOS
- May need Xcode Command Line Tools: `xcode-select --install`
- Use `python3` explicitly
- Some packages may require Homebrew dependencies

### Linux (Ubuntu/Debian)
- May need system packages: `sudo apt-get install python3-dev build-essential`
- Use `python3` and `pip3` explicitly

---

## Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: https://github.com/sauwal948-ops/equitable-hrc-dqn/issues
2. **Open a new issue**: Provide your OS, Python version, and error message
3. **Contact**: salisuauwalm@gmail.com

---

## Next Steps

After successful installation, proceed to:
- **[USAGE.md](docs/USAGE.md)** - Learn how to run experiments
- **[README.md](README.md)** - Understand the project structure
- **Quick Start**: `python code/training.py --config baseline --episodes 10`
