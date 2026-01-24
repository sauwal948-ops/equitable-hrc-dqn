#!/bin/bash

# Repository Reorganization Script
# This script reorganizes your existing files into the recommended structure
# Run this from the root of your repository: bash reorganize_repository.sh

echo "=========================================="
echo "Repository Reorganization Script"
echo "=========================================="
echo ""

# Safety check
read -p "This will reorganize your repository. Have you backed up your files? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Please backup your repository first, then run this script again."
    exit 1
fi

echo "Starting reorganization..."
echo ""

# Create directory structure
echo "1. Creating directory structure..."
mkdir -p code
mkdir -p experiments
mkdir -p data
mkdir -p results/figures
mkdir -p results/tables
mkdir -p results/models
mkdir -p results/raw_outputs
mkdir -p economic_analysis
mkdir -p supplementary
mkdir -p docs
mkdir -p configs

echo "   ✓ Directories created"

# Move existing files to code/
echo ""
echo "2. Moving existing code files to code/ directory..."

# Rename and move environment file (fix typo)
if [ -f "enviroment.py" ]; then
    mv enviroment.py code/environment.py
    echo "   ✓ Moved and renamed: enviroment.py -> code/environment.py"
fi

# Move DQN agent
if [ -f "dqn_agent.py" ]; then
    mv dqn_agent.py code/dqn_agent.py
    echo "   ✓ Moved: dqn_agent.py -> code/"
fi

# Move main simulation
if [ -f "hrc_rl_simulation.py" ]; then
    mv hrc_rl_simulation.py code/hrc_rl_simulation.py
    echo "   ✓ Moved: hrc_rl_simulation.py -> code/"
fi

# Move supplementary codes (rename to fix typo and spaces)
if [ -f "suplimentary codes.py" ]; then
    mv "suplimentary codes.py" supplementary/supplementary_codes.py
    echo "   ✓ Moved and renamed: 'suplimentary codes.py' -> supplementary/supplementary_codes.py"
fi

# Move pseudocode to docs
if [ -f "pseudocode.py" ]; then
    mv pseudocode.py docs/pseudocode.py
    echo "   ✓ Moved: pseudocode.py -> docs/"
fi

# Remove duplicate readme (keep README.md)
echo ""
echo "3. Cleaning up duplicate files..."
if [ -f "readme.md" ]; then
    rm readme.md
    echo "   ✓ Removed duplicate: readme.md (keeping README.md)"
fi

# Create placeholder files for missing scripts
echo ""
echo "4. Creating placeholder files for missing components..."

# Create baseline comparison script
cat > experiments/baseline_comparison.py << 'EOF'
"""
Baseline Comparison: Equitable DQN vs Productivity-Only DQN
Reproduces Table 4 and Figure 2 from the manuscript
"""

import sys
sys.path.append('..')

from code.dqn_agent import DQNAgent
from code.environment import CementBaggingEnv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def run_comparison(n_seeds=20, n_episodes=100):
    """
    Compare Equitable vs Productivity-Only DQN across multiple seeds
    
    Args:
        n_seeds: Number of random seeds for statistical validation
        n_episodes: Episodes per training run
    """
    
    print("="*60)
    print("Baseline Comparison: Equitable vs Productivity-Only DQN")
    print("="*60)
    print(f"Seeds: {n_seeds} | Episodes: {n_episodes}")
    print()
    
    equitable_rewards = []
    productivity_rewards = []
    
    for seed in range(n_seeds):
        print(f"Running seed {seed+1}/{n_seeds}...")
        
        # TODO: Implement training logic using your existing code
        # This is a placeholder - adapt to your actual implementation
        
        # Equitable DQN (w3=0.1, w4=0.1)
        # equitable_reward = train_dqn(weights=[0.5, 0.3, 0.1, 0.1], seed=seed)
        # equitable_rewards.append(equitable_reward)
        
        # Productivity-Only (w3=0, w4=0)
        # productivity_reward = train_dqn(weights=[0.5, 0.3, 0.0, 0.0], seed=seed)
        # productivity_rewards.append(productivity_reward)
    
    # Statistical analysis
    # t_stat, p_value = stats.ttest_rel(equitable_rewards, productivity_rewards)
    
    # Print results
    # print("\nResults:")
    # print(f"Equitable DQN: {np.mean(equitable_rewards):.2f} ± {np.std(equitable_rewards):.2f}")
    # print(f"Productivity-Only: {np.mean(productivity_rewards):.2f} ± {np.std(productivity_rewards):.2f}")
    # print(f"p-value: {p_value:.6f}")
    
    print("\nTODO: Implement actual training logic using your existing code")
    print("See hrc_rl_simulation.py for reference")

if __name__ == "__main__":
    run_comparison(n_seeds=20, n_episodes=100)
EOF

echo "   ✓ Created: experiments/baseline_comparison.py"

# Create sensitivity analysis script
cat > experiments/sensitivity_analysis.py << 'EOF'
"""
Sensitivity Analysis: Test different reward weight configurations
Reproduces Section 3.5 scenarios A, B, and C
"""

# TODO: Implement sensitivity analysis
# Scenario A: Safety-First (w3=0.5)
# Scenario B: Production-Critical (w1=0.8)
# Scenario C: Skill-Level Generalization

print("TODO: Implement sensitivity analysis")
print("See Section 3.5 and Supplementary S4.1 for specifications")
EOF

echo "   ✓ Created: experiments/sensitivity_analysis.py"

# Create cross-industry validation
cat > experiments/cross_industry_validation.py << 'EOF'
"""
Cross-Industry Validation: Test framework across 5 manufacturing sectors
Reproduces Table 6 and Figure 4
"""

# TODO: Implement cross-industry validation
# Industries: Cement, Electronics, Textiles, Food, Automotive

print("TODO: Implement cross-industry validation")
print("See Section 4.6 and Table 6 for industry parameters")
EOF

echo "   ✓ Created: experiments/cross_industry_validation.py"

# Create Monte Carlo analysis
cat > experiments/monte_carlo_analysis.py << 'EOF'
"""
Monte Carlo Robustness Testing: 1,000-iteration validation
Reproduces Figure 8 and Section S4.6
"""

# TODO: Implement Monte Carlo analysis with sensor noise and randomization

print("TODO: Implement Monte Carlo analysis (1,000 iterations)")
print("See Section S4.6 for noise injection specifications")
EOF

echo "   ✓ Created: experiments/monte_carlo_analysis.py"

# Create ROI calculator
cat > economic_analysis/roi_calculator.py << 'EOF'
"""
ROI Calculator: 5-year NPV and payback analysis
Reproduces Figure 5 and Table 7
"""

# TODO: Implement economic analysis
# Based on Section 4.7 and Table 7

print("TODO: Implement ROI calculation")
print("See Section 4.7 and Table 7 for financial parameters")
EOF

echo "   ✓ Created: economic_analysis/roi_calculator.py"

# Create verification script
cat > code/verify_installation.py << 'EOF'
"""
Installation Verification Script
"""

def verify_installation():
    print("Verifying installation...\n")
    
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("✗ PyTorch not installed")
    
    try:
        import numpy as np
        print(f"✓ NumPy installed: {np.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
    
    try:
        import pandas as pd
        print(f"✓ Pandas installed: {pd.__version__}")
    except ImportError:
        print("✗ Pandas not installed")
    
    try:
        import matplotlib
        print(f"✓ Matplotlib installed: {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib not installed")
    
    print("\nInstallation verification complete!")

if __name__ == "__main__":
    verify_installation()
EOF

echo "   ✓ Created: code/verify_installation.py"

# Create default config
cat > configs/baseline.yaml << 'EOF'
# Baseline Configuration for Equitable DQN
# Corresponds to default weights: w = [0.5, 0.3, 0.1, 0.1]

reward_weights:
  throughput: 0.5
  error: 0.3
  fatigue: 0.1
  equity: 0.1

training:
  episodes: 100
  learning_rate: 0.001
  gamma: 0.95
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 60
  batch_size: 64
  replay_buffer_size: 10000
  target_update_freq: 100

environment:
  bag_arrival_rate: 1.2  # Poisson lambda (bags/min)
  bagging_time_mean: 45  # seconds
  bagging_time_std: 5    # seconds
  infrastructure_downtime: 0.05  # 5% probability
  
worker:
  fatigue_accumulation: 0.02
  fatigue_recovery: 0.95
  initial_fatigue: 0.0
EOF

echo "   ✓ Created: configs/baseline.yaml"

# Create data directory README
cat > data/README.md << 'EOF'
# Data Directory

This directory should contain:

1. **manufacturing_parameters.csv** - Industry operational parameters (Table S6)
2. **industry_profiles.json** - Cross-industry configurations (Section 4.6)
3. **systematic_review_data.csv** - Literature review synthesis (Tables S1-S3)

Data files should be generated from your research or added manually.
See Supplementary Materials S3.1 and S4.1 for parameter specifications.
EOF

echo "   ✓ Created: data/README.md"

# Create results README
cat > results/README.md << 'EOF'
# Results Directory

This directory stores:

- **figures/** - All manuscript figures (Figure 1-9)
- **tables/** - CSV exports of all tables
- **models/** - Trained DQN model checkpoints
- **raw_outputs/** - Training logs and episode data

Files are generated when running experiments.
EOF

echo "   ✓ Created: results/README.md"

echo ""
echo "=========================================="
echo "Reorganization Complete!"
echo "=========================================="
echo ""
echo "Summary of changes:"
echo "  ✓ Created organized directory structure"
echo "  ✓ Moved existing files to appropriate locations"
echo "  ✓ Fixed file naming (enviroment -> environment)"
echo "  ✓ Removed duplicate readme.md"
echo "  ✓ Created placeholder scripts for missing experiments"
echo "  ✓ Added configuration files"
echo ""
echo "Next steps:"
echo "  1. Review the new structure: ls -R"
echo "  2. Update your existing code to work with new paths"
echo "  3. Fill in the TODO placeholders in experiments/"
echo "  4. Add your data files to data/"
echo "  5. Generate figures and save to results/figures/"
echo ""
echo "See USAGE.md for how to run experiments"
echo ""
