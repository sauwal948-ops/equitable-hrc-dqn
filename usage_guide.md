# Usage Guide

This guide demonstrates how to reproduce all results from the paper "A Deep Reinforcement Learning Framework for Equitable Human-Robot Collaboration."

---

## Quick Start

### Run Baseline Training (5 minutes)

```bash
python code/training.py --config baseline --episodes 100 --seed 42
```

Expected output:
```
Episode 10/100 | Reward: 142.3 | Error: 8.2% | Fatigue: 0.156
Episode 20/100 | Reward: 156.7 | Error: 6.1% | Fatigue: 0.134
...
Episode 100/100 | Reward: 168.5 | Error: 4.36% | Fatigue: 0.118
Training completed! Model saved to results/models/baseline_dqn.pth
```

---

## Reproducing Paper Results

### 1. Table 4 - Comparative Performance Analysis

**Reproduce the Equitable vs. Productivity-Only comparison:**

```bash
python experiments/baseline_comparison.py --seeds 20 --episodes 100
```

**Output:**
- CSV file: `results/table4_comparison.csv`
- Statistical test results printed to console
- Expected runtime: ~30 minutes

**What it does:**
- Trains 20 independent models with different random seeds
- Compares Equitable DQN (w₃=0.1, w₄=0.1) vs. Productivity-Only (w₃=0, w₄=0)
- Performs paired t-test for statistical validation
- Generates Table 4 from the manuscript

### 2. Figure 2 - Robustness Validation

**Reproduce the 20-seed cumulative reward distribution:**

```bash
python experiments/baseline_comparison.py --seeds 20 --visualize
```

**Output:**
- Figure: `results/figures/figure2_robustness_validation.png`
- Shows violin plots comparing Equitable vs. Productivity-Only DQN
- Expected runtime: ~30 minutes

### 3. Figure 3 - Pareto Frontier

**Reproduce the productivity-equity trade-off analysis:**

```bash
python experiments/pareto_analysis.py --weight_range 0.0:0.3:0.05
```

**Output:**
- Figure: `results/figures/figure3_pareto_frontier.png`
- CSV: `results/pareto_frontier_data.csv`
- Tests equity weights from 0% to 30% in 5% increments
- Expected runtime: ~45 minutes

### 4. Table 6 & Figure 4 - Cross-Industry Validation

**Reproduce the 5-industry meta-analysis:**

```bash
python experiments/cross_industry_validation.py --industries all --episodes 100 --seeds 20
```

**Output:**
- Table: `results/table6_cross_industry.csv`
- Figure: `results/figures/figure4_industry_comparison.png`
- Meta-analysis results printed to console
- Expected runtime: ~2 hours

**Industries tested:**
- Cement Bagging (Baseline)
- Electronics Assembly
- Textile Inspection
- Food Processing
- Automotive Parts

### 5. Figure 8 - Monte Carlo Validation

**Reproduce the 1,000-iteration statistical reliability test:**

```bash
python experiments/monte_carlo_analysis.py --iterations 1000
```

**Output:**
- Figure: `results/figures/figure8_monte_carlo.png`
- Shows probability density functions for error and fatigue reduction
- CSV: `results/monte_carlo_results.csv`
- Expected runtime: ~3 hours

**Warning:** This is computationally intensive. For quick testing, use:
```bash
python experiments/monte_carlo_analysis.py --iterations 100
```

### 6. Section 3.5 - Sensitivity Analysis

**Reproduce Scenarios A, B, and C:**

```bash
# Scenario A: Safety-First Policy
python experiments/sensitivity_analysis.py --scenario safety_first

# Scenario B: Production-Critical Policy
python experiments/sensitivity_analysis.py --scenario production_critical

# Scenario C: Skill-Level Generalization
python experiments/sensitivity_analysis.py --scenario skill_generalization
```

**Output:**
- Results for each scenario in `results/sensitivity/`
- Console output shows KPI comparisons
- Expected runtime: ~20 minutes per scenario

### 7. Figure 5 - ROI Projection

**Reproduce the 5-year economic analysis:**

```bash
python economic_analysis/roi_calculator.py --scenario baseline
```

**Output:**
- Figure: `results/figures/figure5_roi_projection.png`
- Detailed cash flow breakdown
- NPV, payback period, and probability calculations
- Expected runtime: <1 minute

### 8. Figure 6 - Worker Resistance Analysis

**Reproduce the resistance threshold testing:**

```bash
python experiments/resistance_testing.py --resistance_range 0:0.5:0.05
```

**Output:**
- Figure: `results/figures/figure6_resistance_threshold.png`
- Shows performance degradation vs. compliance rate
- Expected runtime: ~40 minutes

---

## Running Individual Components

### Training a Custom Model

```bash
python code/training.py \
    --config custom \
    --episodes 100 \
    --w_throughput 0.5 \
    --w_error 0.3 \
    --w_fatigue 0.1 \
    --w_equity 0.1 \
    --learning_rate 0.001 \
    --gamma 0.95 \
    --seed 42
```

### Evaluating a Trained Model

```bash
python code/evaluation.py \
    --model_path results/models/baseline_dqn.pth \
    --episodes 20 \
    --visualize
```

### Testing the Environment

```bash
python code/test_environment.py
```

---

## Advanced Usage

### Custom Reward Weights

Create a configuration file `configs/my_config.yaml`:

```yaml
# Custom reward configuration
reward_weights:
  throughput: 0.6
  error: 0.25
  fatigue: 0.1
  equity: 0.05

training:
  episodes: 100
  learning_rate: 0.001
  gamma: 0.95
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 60

environment:
  bag_arrival_rate: 1.2
  bagging_time_mean: 45
  bagging_time_std: 5
  infrastructure_downtime: 0.05
```

Run with custom config:
```bash
python code/training.py --config configs/my_config.yaml
```

### Batch Experiments

```bash
# Run multiple configurations in parallel
python experiments/batch_runner.py --config_dir configs/batch/ --parallel 4
```

### Export Results for Analysis

```bash
# Export all results to Excel
python tools/export_results.py --format xlsx --output results/all_results.xlsx

# Export figures as high-res PDFs
python tools/export_figures.py --dpi 300 --format pdf
```

---

## Configuration Options

### Full Parameter List

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--episodes` | 100 | Number of training episodes |
| `--seed` | 42 | Random seed for reproducibility |
| `--w_throughput` | 0.5 | Reward weight for throughput |
| `--w_error` | 0.3 | Penalty weight for errors |
| `--w_fatigue` | 0.1 | Penalty weight for fatigue |
| `--w_equity` | 0.1 | Penalty weight for bias |
| `--learning_rate` | 0.001 | DQN learning rate |
| `--gamma` | 0.95 | Discount factor |
| `--batch_size` | 64 | Training batch size |
| `--replay_buffer` | 10000 | Experience replay buffer size |
| `--target_update` | 100 | Target network update frequency |
| `--device` | auto | 'cuda', 'cpu', or 'auto' |

---

## Troubleshooting

### Issue: Training is very slow

**Solution 1:** Use GPU acceleration
```bash
python code/training.py --device cuda
```

**Solution 2:** Reduce episodes for testing
```bash
python code/training.py --episodes 20
```

### Issue: Out of memory error

**Solution:** Reduce batch size and replay buffer
```bash
python code/training.py --batch_size 32 --replay_buffer 5000
```

### Issue: Results don't match paper

**Cause:** Random seed variation

**Solution:** Use multiple seeds and average
```bash
python experiments/baseline_comparison.py --seeds 20
```

### Issue: Figures not generating

**Solution:** Check matplotlib backend
```python
import matplotlib
matplotlib.use('Agg')  # For headless servers
```

---

## Interpreting Results

### Key Performance Indicators (KPIs)

**Normalized Throughput** (0-1)
- 1.0 = Theoretical maximum production
- 0.7-0.8 = Good performance
- <0.6 = Needs optimization

**Error Rate** (%)
- <5% = Excellent quality
- 5-10% = Acceptable
- >10% = Quality issues

**Fatigue Index** (0-1)
- <0.15 = Sustainable
- 0.15-0.30 = Moderate strain
- >0.30 = Worker exhaustion

**Cumulative Reward**
- Higher = Better multi-objective performance
- Stable values = Converged policy
- Increasing trend = Still learning

---

## Validation Checklist

After running experiments, verify:

- [ ] Training converges (reward plateaus after ~60 episodes)
- [ ] Error rate < 10% in final episodes
- [ ] Fatigue index < 0.3 throughout
- [ ] Statistical tests show p < 0.05 for comparisons
- [ ] Figures match paper qualitatively
- [ ] All output files generated successfully

---

## Performance Benchmarks

Expected runtimes on standard hardware (4-core CPU, 16GB RAM):

| Experiment | Episodes | Seeds | Runtime |
|------------|----------|-------|---------|
| Baseline training | 100 | 1 | ~5 min |
| Table 4 comparison | 100 | 20 | ~30 min |
| Cross-industry | 100 | 20 | ~2 hours |
| Monte Carlo | - | 1000 | ~3 hours |
| Pareto frontier | - | - | ~45 min |

With GPU (NVIDIA RTX 3080 or better):
- 2-3x speedup expected

---

## Next Steps

- **Modify reward weights** to test your own configurations
- **Add new industries** by editing `experiments/cross_industry_validation.py`
- **Extend the environment** in `code/environment.py`
- **Contribute improvements** via pull request

For questions: salisuauwalm@gmail.com
