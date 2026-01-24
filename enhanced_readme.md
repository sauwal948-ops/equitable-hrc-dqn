# Equitable Human-Robot Collaboration: Deep Reinforcement Learning Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

> **A Deep Reinforcement Learning Framework for Equitable Human-Robot Collaboration: Multi-Objective Task Allocation in Resource-Constrained Manufacturing**

*Salisu Auwal Musa¹, Bashir Muhammad Ahmad²*

¹ Department of Mechanical Engineering, Vivekananda Global University, Jaipur, India  
² School of Computer Science Education, Southwest University, Chongqing, China

**Corresponding Author:** salisuauwalm@gmail.com

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Main Results](#main-results)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Reproducing Results](#reproducing-results)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This repository contains the complete implementation of an equitable Deep Q-Network (DQN) framework for human-robot collaboration in resource-constrained manufacturing environments. Unlike traditional approaches that optimize solely for productivity, our framework explicitly balances:

- **Economic Efficiency** (Throughput & Quality)
- **Worker Welfare** (Fatigue Management)
- **Social Equity** (Fair Task Distribution)

**Research Type:** Simulation-based proof-of-concept demonstrating computational feasibility and generating testable hypotheses for empirical validation.

### Context

The transition to Industry 5.0 requires human-centric automation that augments rather than replaces workers. Our systematic review of 25 studies (from 1,247 records) reveals:

- 90% of HRC frameworks optimize only for productivity
- Only 36% address developing economies (despite 60% of manufacturing employment)
- 0% explicitly model equity in task allocation

This work addresses these gaps through multi-objective reinforcement learning with explicit equity weights.

---

## ✨ Key Features

### Methodological Innovation
- **First framework** treating equity (δ=0.1) and welfare (γ=0.1) as explicit parametric objectives
- **Composite reward function** enabling transparent stakeholder negotiation
- **Replicable template** for operationalizing "human-centric automation"

### Technical Robustness
- Cross-industry validation (5 manufacturing sectors)
- Monte Carlo analysis (1,000 iterations)
- Stress testing under sensor noise and infrastructure instability
- Statistical validation across 20 independent random seeds

### Economic Viability
- Complete NPV analysis ($185K, 2.4-year payback)
- 89% profitability probability
- Boundary condition analysis (30% worker resistance threshold)
- Break-even robustness testing

### Practical Applicability
- Deployment readiness checklist (Appendix A in Supplementary Materials)
- Three-phase validation pathway
- Resource-constrained context modeling

---

## 📊 Main Results

### Performance Metrics (Simulation-Based)

| Metric | Improvement | Statistical Significance |
|--------|-------------|-------------------------|
| **Error Reduction** | 14.0% pooled (15.6% baseline) | 95% CI: 11.2-16.8%, p<0.001 |
| **Fatigue Reduction** | 13.2% pooled (13.0% baseline) | 95% CI: 10.8-15.6%, p<0.001 |
| **Throughput Cost** | -4.8% | Modest trade-off |
| **Bias Reduction** | 37.3% | More equitable task distribution |
| **Skill Development** | +30.9% | Increased learning exposure |

### Robustness
- **70% error-reduction advantage** maintained under 10% sensor noise
- **Effective until 30% worker resistance** threshold
- **Converges in 60 episodes** with stable policies

### Economic Analysis
- **NPV:** $185K over 5 years (7% discount rate)
- **Payback Period:** 2.4 years (95% CI: 1.9-3.1 years)
- **Break-even:** Profitable even if error reduction is only 8% (vs. 15.6% simulated)

---

## 📁 Repository Structure

```
equitable-hrc-dqn/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── CITATION.bib                       # Citation information
├── .gitignore                         # Git ignore rules
│
├── code/                              # Core implementation
│   ├── dqn_agent.py                  # DQN agent class
│   ├── environment.py                # Cement bagging simulation
│   ├── reward_functions.py           # Composite reward implementation
│   ├── training.py                   # Training loop
│   ├── evaluation.py                 # Performance metrics
│   └── verify_installation.py        # Installation checker
│
├── experiments/                       # Reproduce paper results
│   ├── baseline_comparison.py        # Table 4, Figure 2
│   ├── sensitivity_analysis.py       # Section 3.5 scenarios
│   ├── robustness_testing.py         # Section 4.5.2
│   ├── cross_industry_validation.py  # Table 6, Figure 4
│   ├── monte_carlo_analysis.py       # Figure 8
│   ├── pareto_analysis.py            # Figure 3
│   └── resistance_testing.py         # Figure 6
│
├── data/                              # Input data
│   ├── manufacturing_parameters.csv   # Industry parameters
│   ├── industry_profiles.json         # Cross-industry configs
│   └── systematic_review_data.csv     # Literature synthesis
│
├── results/                           # Output directory
│   ├── figures/                       # Generated plots
│   ├── tables/                        # CSV exports
│   ├── models/                        # Trained checkpoints
│   └── raw_outputs/                   # Training logs
│
├── economic_analysis/                 # ROI modeling
│   ├── roi_calculator.py             # NPV & payback
│   └── sensitivity_scenarios.xlsx    # Economic Monte Carlo
│
├── supplementary/                     # Extended materials
│   ├── supplementary_codes.py        # Additional analyses
│   └── deployment_checklist.pdf      # Appendix A
│
├── configs/                           # Configuration files
│   ├── baseline.yaml                 # Default settings
│   └── custom_weights.yaml           # Example customization
│
└── docs/                              # Documentation
    ├── INSTALLATION.md               # Setup guide
    ├── USAGE.md                      # How to run experiments
    └── methodology.md                # Detailed MDP formulation
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/sauwal948-ops/equitable-hrc-dqn.git
cd equitable-hrc-dqn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python code/verify_installation.py
```

**Full installation guide:** [INSTALLATION.md](docs/INSTALLATION.md)

### 2. Run Baseline Training

```bash
python code/training.py --config baseline --episodes 100 --seed 42
```

Expected output:
```
Episode 10/100 | Reward: 142.3 | Error: 8.2% | Fatigue: 0.156
Episode 50/100 | Reward: 164.1 | Error: 5.1% | Fatigue: 0.125
Episode 100/100 | Reward: 168.5 | Error: 4.36% | Fatigue: 0.118
✓ Training completed! Model saved to results/models/baseline_dqn.pth
```

### 3. Reproduce Key Results

```bash
# Table 4: Comparative Performance
python experiments/baseline_comparison.py --seeds 20

# Figure 3: Pareto Frontier
python experiments/pareto_analysis.py

# Figure 8: Monte Carlo Validation
python experiments/monte_carlo_analysis.py --iterations 1000
```

**Full usage guide:** [USAGE.md](docs/USAGE.md)

---

## 🔬 Reproducing Results

All results from the manuscript can be reproduced using the scripts in `experiments/`:

| Paper Section | Script | Runtime | Output |
|---------------|--------|---------|--------|
| Table 4 | `baseline_comparison.py` | ~30 min | CSV + Figure |
| Figure 2 | `baseline_comparison.py --visualize` | ~30 min | PNG |
| Figure 3 | `pareto_analysis.py` | ~45 min | PNG + CSV |
| Table 6 | `cross_industry_validation.py` | ~2 hours | CSV + Figure |
| Figure 8 | `monte_carlo_analysis.py` | ~3 hours | PNG + CSV |
| Section 3.5 | `sensitivity_analysis.py` | ~1 hour | Console + CSV |

**Note:** Runtimes are for 4-core CPU. GPU acceleration provides 2-3x speedup.

See [USAGE.md](docs/USAGE.md) for detailed instructions.

---

## 📚 Documentation

- **[INSTALLATION.md](docs/INSTALLATION.md)** - Complete setup instructions for Linux/macOS/Windows
- **[USAGE.md](docs/USAGE.md)** - How to reproduce all paper results
- **[Supplementary Materials](supplementary/)** - Extended methodology and validation protocols
- **[API Documentation](docs/api.md)** - Code reference (coming soon)

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{musa2025equitable,
  title={A Deep Reinforcement Learning Framework for Equitable Human-Robot 
         Collaboration: Multi-Objective Task Allocation in Resource-Constrained 
         Manufacturing},
  author={Musa, Salisu Auwal and Ahmad, Bashir Muhammad},
  journal={[Journal Name - To Be Updated Upon Acceptance]},
  year={2025},
  url={https://github.com/sauwal948-ops/equitable-hrc-dqn},
  note={Preprint with open-source implementation}
}
```

For the software implementation:

```bibtex
@software{musa2025equitable_code,
  title={Equitable HRC-DQN: Implementation and Supplementary Code},
  author={Musa, Salisu Auwal and Ahmad, Bashir Muhammad},
  year={2025},
  url={https://github.com/sauwal948-ops/equitable-hrc-dqn},
  version={1.0.0}
}
```

See [CITATION.bib](CITATION.bib) for BibTeX entries.

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- **Empirical validation:** Pilot studies in actual manufacturing facilities
- **Multi-agent extension:** Coordination of multiple robots/workers
- **Additional industries:** Validation beyond the 5 tested sectors
- **Real-world data:** Integration of actual factory operational data
- **User interface:** Dashboard for non-technical stakeholders

Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

---

## ⚠️ Important Notes

### Simulation vs. Reality

**This is a simulation-based proof-of-concept study.** Results demonstrate:

✅ **What We Have Proven:**
- Multi-objective HRC optimization is computationally feasible
- Equitable policies can maintain competitive throughput in simulation
- Framework provides replicable methodological template

❌ **What Requires Validation:**
- Specific improvements (15.6% error reduction) will materialize in real factories
- Workers will accept and engage with the system
- Economic projections reflect real-world costs/benefits

**Interpretation:** Results establish that equitable HRC is worth testing empirically, not proven to work in practice. See Section 6.3 for validation requirements.

---

## � Data Availability

All code, data, and materials supporting this study are openly available at 
https://github.com/sauwal948-ops/equitable-hrc-dqn under the MIT License. 
The repository includes: 
1. Complete DQN implementation with composite reward functions
2. Simulation environment for cement bagging operations
3. Cross-industry validation scripts for five manufacturing sectors
4. Monte Carlo robustness testing code (n=1,000 iterations)
5. Economic ROI analysis templates
6. Deployment readiness assessment tools

Manufacturing operational parameters are derived from publicly available sources cited in 
references [7, 8, 11, 15]. Detailed parameter tables, validation protocols, 
and systematic review data extraction are provided in Supplementary Materials 
S3.1, S4.1-S4.8, and Appendix A.

---

## ⚖️ Ethics Declaration

**Ethics approval and consent to participate:** Not applicable (simulation-based study).

**Consent for publication:** Not applicable.

**Competing interests:** The authors declare no competing interests.

**Funding:** [Specify funding source or state "No funding was received for this research"]

**Authors' contributions:** S.A.M. conceptualized the framework, conducted simulations, 
and drafted the manuscript. B.M.A. contributed to methodology and manuscript revision. 
Both authors approved the final version.

---

## �📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Summary:** You are free to use, modify, and distribute this code, even for commercial purposes, with attribution.

---

## 📧 Contact

**Corresponding Author:**  
Salisu Auwal Musa  
Department of Mechanical Engineering  
Vivekananda Global University  
Jaipur, Rajasthan, India

**Email:** salisuauwalm@gmail.com

**Issues & Questions:**  
- Technical issues: [GitHub Issues](https://github.com/sauwal948-ops/equitable-hrc-dqn/issues)
- Research collaboration: salisuauwalm@gmail.com

---

## 🙏 Acknowledgments

- Manufacturing parameters sourced from publicly available literature (references [7, 8, 11, 15] in paper)
- DQN implementation inspired by standard PyTorch examples
- Systematic review follows PRISMA guidelines
- Economic analysis methodology from standard NPV frameworks

---

## 🔗 Related Links

- **Paper:** [Preprint - Link to be updated]
- **Supplementary Materials:** [Full document with extended analysis](supplementary/)
- **Data Repository:** [Zenodo DOI - To be created]
- **Project Website:** [Coming soon]

---

## 📊 Project Status

- ✅ **Simulation framework:** Complete and validated
- ✅ **Cross-industry validation:** Complete (5 sectors)
- ✅ **Economic modeling:** Complete with sensitivity analysis
- 🔄 **Empirical validation:** In planning (seeking partners)
- 📋 **Documentation:** Ongoing improvements

**Last Updated:** January 2025  
**Version:** 1.0.0  
**Status:** Active development

---

<p align="center">
  <strong>Building human-centric automation for inclusive manufacturing</strong>
</p>

<p align="center">
  <a href="#quick-start">Get Started</a> •
  <a href="docs/USAGE.md">Usage Guide</a> •
  <a href="docs/INSTALLATION.md">Installation</a> •
  <a href="https://github.com/sauwal948-ops/equitable-hrc-dqn/issues">Report Bug</a>
</p>
