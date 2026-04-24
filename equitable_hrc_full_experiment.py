"""
=============================================================================
Equitable HRC DQN — Full Experiment (Kaggle-Ready)
=============================================================================
Paper: Multi-Objective RL for Human-Robot Collaborative Task Allocation
Authors: Salisu Auwal Musa, Bashir Muhammad Ahmad
 
This notebook uses your ACTUAL environment (CementBaggingHRCEnvironment)
and implements the ACTUAL paper architecture (PyTorch 5-128-128-4 DQN).
 
Runtime: ~15-25 minutes on Kaggle CPU
RAM:     < 2 GB
 
What this produces:
  Part A — Baseline Comparison (Table 5 in paper)
    - Equitable DQN vs Productivity-Only, 20 seeds each
    - Paired t-test, Cohen's d, real statistics
    - Figure 2: Convergence curves
    - Figure 3: Box plots
 
  Part B — 5-Configuration Ablation (new Table for paper)
    - Full / No-Fatigue / No-Equity / No-Welfare / No-Quality
    - Figure B1: Component contribution bars
    - Figure B2: Ablation heatmap
 
  Part C — Printed results table to paste back for manuscript
=============================================================================
"""
 
# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
from scipy import stats
import random, time, gc, warnings
warnings.filterwarnings('ignore')
 
# ── Reproducibility ──────────────────────────────────────────────────────────
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
 
# ── Plotting style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
    'figure.dpi': 120, 'savefig.dpi': 300,
})
BLUE   = '#1B4F8A'
ORANGE = '#D45F00'
GREEN  = '#1A6B3A'
RED    = '#B22222'
PURPLE = '#5B2C8A'
GRAY   = '#555555'
 
# =============================================================================
# YOUR ENVIRONMENT — copied exactly from hrc_rl_simulation.py
# =============================================================================
 
class CementBaggingHRCEnvironment:
    """
    Simulates a cement bagging line with human-robot collaboration.
    State: [machine_speed, human_fatigue, error_rate, queue_length, worker_skill]
    Actions: 0=Idle, 1=Assist, 2=TakeOver, 3=SuggestBreak
    """
    def __init__(self, num_workers=3):
        self.num_workers = num_workers
        self.current_worker = 0
        self.episode_step = 0
        self.max_steps = 500
        self.workers = np.array([
            [0.3, 0.5, 1],   # Worker 1: Low experience
            [0.2, 0.7, 5],   # Worker 2: Medium experience
            [0.1, 0.9, 10]   # Worker 3: High experience
        ])
        self.reset()
 
    def reset(self):
        self.machine_speed  = np.random.uniform(0.7, 1.0)
        self.human_fatigue  = np.random.uniform(0.2, 0.5)
        self.error_rate     = np.random.uniform(0.05, 0.15)
        self.queue_length   = np.random.uniform(0.3, 0.8)
        self.episode_step   = 0
        self.current_worker = np.random.randint(0, self.num_workers)
        return self.get_state()
 
    def get_state(self):
        worker = self.workers[self.current_worker]
        return np.array([
            self.machine_speed, self.human_fatigue, self.error_rate,
            self.queue_length,  worker[1]
        ], dtype=np.float32)
 
    def step(self, action):
        self.episode_step += 1
        self.machine_speed = np.clip(
            self.machine_speed + np.random.normal(0, 0.05), 0.5, 1.0)
 
        if action == 0:   self.human_fatigue = np.clip(self.human_fatigue + 0.08, 0, 1)
        elif action == 1: self.human_fatigue = np.clip(self.human_fatigue + 0.03, 0, 1)
        elif action == 2: self.human_fatigue = np.clip(self.human_fatigue - 0.05, 0, 1)
        elif action == 3: self.human_fatigue = np.clip(self.human_fatigue - 0.15, 0, 1)
 
        fatigue_effect = self.human_fatigue * 0.1
        if action == 1:
            self.error_rate = np.clip(self.error_rate - 0.02 + fatigue_effect, 0.01, 0.3)
        elif action == 2:
            self.error_rate = np.clip(self.error_rate - 0.05, 0.01, 0.2)
        else:
            self.error_rate = np.clip(self.error_rate + fatigue_effect, 0.01, 0.3)
 
        processing_rate = self.machine_speed * (1 - self.error_rate)
        if action == 2: processing_rate *= 1.2
        self.queue_length = np.clip(
            self.queue_length - processing_rate * 0.1 + np.random.uniform(0, 0.05), 0, 1)
 
        self.current_worker = (self.current_worker + 1) % self.num_workers
        reward = self._calculate_reward(action)
        done   = self.episode_step >= self.max_steps
        return self.get_state(), reward, done
 
    def _calculate_reward(self, action, w1=0.5, w2=0.3, w3=0.1, w4=0.1):
        throughput = self.machine_speed * (1 - self.error_rate)
        r_throughput = throughput
        r_error      = -self.error_rate
        r_fatigue    = -self.human_fatigue * 0.5
        worker       = self.workers[self.current_worker]
        skill_level  = worker[1]
        r_bias       = -0.1 if (action == 2 and skill_level < 0.6) else 0
        return float(w1*r_throughput + w2*r_error + w3*r_fatigue + w4*r_bias)
 
 
class WeightedEnv(CementBaggingHRCEnvironment):
    """Environment with configurable reward weights for ablation."""
    def __init__(self, w1=0.5, w2=0.3, w3=0.1, w4=0.1, num_workers=3):
        self.w1, self.w2, self.w3, self.w4 = w1, w2, w3, w4
        super().__init__(num_workers)
 
    def _calculate_reward(self, action,
                          w1=None, w2=None, w3=None, w4=None):
        w1 = self.w1; w2 = self.w2; w3 = self.w3; w4 = self.w4
        throughput   = self.machine_speed * (1 - self.error_rate)
        r_throughput = throughput
        r_error      = -self.error_rate
        r_fatigue    = -self.human_fatigue * 0.5
        skill_level  = self.workers[self.current_worker][1]
        r_bias       = -0.1 if (action == 2 and skill_level < 0.6) else 0
        return float(w1*r_throughput + w2*r_error + w3*r_fatigue + w4*r_bias)
 
 
# =============================================================================
# PAPER ARCHITECTURE — PyTorch DQN (5 → 128 → 128 → 4)
# He init, ReLU, target network, experience replay, epsilon-greedy
# =============================================================================
 
class QNetwork(nn.Module):
    """5 → 128 → 128 → 4 as described in paper Table 1."""
    def __init__(self, state_dim=5, action_dim=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # He initialization for ReLU layers
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
 
    def forward(self, x):
        return self.net(x)
 
 
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
 
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
 
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))
 
    def __len__(self):
        return len(self.buffer)
 
 
class DQNAgent:
    """
    Actual DQN agent matching paper Table 1 hyperparameters.
    lr=0.001, gamma=0.95, epsilon 1.0→0.01 over 60 eps, buffer=10000,
    batch=32, target update every 100 steps.
    """
    def __init__(self, state_dim=5, action_dim=4,
                 lr=0.001, gamma=0.95,
                 eps_start=1.0, eps_end=0.01, eps_decay_episodes=60,
                 buffer_capacity=10000, batch_size=32,
                 target_update_freq=100, n_episodes=100):
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.target_freq = target_update_freq
        self.step_count  = 0
 
        # Epsilon linear decay: reach eps_end by eps_decay_episodes
        self.epsilon      = eps_start
        self.eps_end      = eps_end
        self.eps_step     = (eps_start - eps_end) / eps_decay_episodes
 
        self.online_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
 
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)
 
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0)
            return self.online_net(t).argmax(dim=1).item()
 
    def push(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)
 
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s   = torch.FloatTensor(s)
        a   = torch.LongTensor(a).unsqueeze(1)
        r   = torch.FloatTensor(r).unsqueeze(1)
        ns  = torch.FloatTensor(ns)
        d   = torch.FloatTensor(d).unsqueeze(1)
 
        q_curr  = self.online_net(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_net(ns).max(1, keepdim=True)[0]
        q_target = r + self.gamma * q_next * (1 - d)
 
        loss = nn.MSELoss()(q_curr, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
        self.step_count += 1
        if self.step_count % self.target_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
 
    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_step)
 
 
# =============================================================================
# TRAINING FUNCTION
# =============================================================================
 
def train_one_run(env, n_episodes=100, seed=0, verbose=False):
    """
    Train one DQN run. Returns per-episode metrics.
    Uses your environment, paper's architecture and hyperparameters.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
 
    agent = DQNAgent(n_episodes=n_episodes)
 
    ep_rewards  = []
    ep_errors   = []
    ep_fatigue  = []
    ep_tput     = []
    action_log  = {0: 0, 1: 0, 2: 0, 3: 0}
 
    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0
        step_errors, step_fatigue, step_tput = [], [], []
 
        for _ in range(env.max_steps):
            action = agent.act(state)
            action_log[action] += 1
            next_state, reward, done = env.step(action)
            agent.push(state, action, reward, next_state, done)
            agent.learn()
            ep_reward += reward
            step_errors.append(env.error_rate)
            step_fatigue.append(env.human_fatigue)
            step_tput.append(env.machine_speed * (1 - env.error_rate))
            state = next_state
            if done:
                break
 
        agent.decay_epsilon()
        ep_rewards.append(ep_reward)
        ep_errors.append(np.mean(step_errors))
        ep_fatigue.append(np.mean(step_fatigue))
        ep_tput.append(np.mean(step_tput))
 
        if verbose and (ep + 1) % 20 == 0:
            print(f"  ep {ep+1:3d}/{n_episodes}  "
                  f"reward={ep_reward:.2f}  "
                  f"error={np.mean(step_errors):.3f}  "
                  f"fatigue={np.mean(step_fatigue):.3f}")
 
    return {
        'rewards':  np.array(ep_rewards),
        'errors':   np.array(ep_errors),
        'fatigue':  np.array(ep_fatigue),
        'tput':     np.array(ep_tput),
        'actions':  action_log,
    }
 
 
def run_config(label, w1, w2, w3, w4, n_seeds=20, n_episodes=100):
    """Run all seeds for one weight configuration."""
    print(f"  {label:<28} w=[{w1},{w2},{w3},{w4}]", flush=True)
    all_rewards, all_errors, all_fatigue, all_tput = [], [], [], []
 
    for seed in range(n_seeds):
        env    = WeightedEnv(w1=w1, w2=w2, w3=w3, w4=w4)
        result = train_one_run(env, n_episodes=n_episodes, seed=seed + GLOBAL_SEED)
        all_rewards.append(result['rewards'])
        all_errors.append(result['errors'])
        all_fatigue.append(result['fatigue'])
        all_tput.append(result['tput'])
 
    return {
        'label':   label,
        'weights': (w1, w2, w3, w4),
        'rewards': np.array(all_rewards),
        'errors':  np.array(all_errors),
        'fatigue': np.array(all_fatigue),
        'tput':    np.array(all_tput),
    }
 
 
def plateau_stats(data, last_n=20):
    """Mean ± SD over last N episodes, averaged across seeds."""
    seg = data[:, -last_n:]
    seed_means = seg.mean(axis=1)
    return seed_means.mean(), seed_means.std()
 
 
# =============================================================================
# FIGURE 2 — Convergence (Equitable DQN, 20 seeds)
# =============================================================================
 
def plot_convergence(equitable):
    ep  = np.arange(equitable['rewards'].shape[1])
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('Figure 2: DQN Training Convergence\n'
                 'Equitable Framework  w=[0.5,0.3,0.1,0.1]  ·  20 Seeds',
                 fontsize=13, fontweight='bold', y=0.99)
 
    panels = [
        ('rewards', 'Cumulative Reward',   BLUE,   False),
        ('errors',  'Error Rate',          RED,    True),
        ('fatigue', 'Worker Fatigue Index',GREEN,   True),
    ]
 
    for ax, (key, ylabel, col, lower) in zip(axes, panels):
        data = equitable[key]
        mean = data.mean(axis=0)
        lo10 = np.percentile(data, 10, axis=0)
        hi90 = np.percentile(data, 90, axis=0)
        lo25 = np.percentile(data, 25, axis=0)
        hi75 = np.percentile(data, 75, axis=0)
 
        ax.fill_between(ep, lo10, hi90, alpha=0.12, color=col, label='10–90th pct.')
        ax.fill_between(ep, lo25, hi75, alpha=0.22, color=col, label='25–75th pct.')
        ax.plot(ep, mean, color=col, lw=2.2, label='Mean (20 seeds)')
 
        # Mark convergence zone (ep 60 per paper)
        ax.axvline(60, color=GRAY, ls=':', lw=1.5)
        ax.text(62, mean[60], 'Convergence\n(ep 60)',
                fontsize=8.5, color=GRAY, va='center')
 
        direction = '↓ lower is better' if lower else '↑ higher is better'
        ax.text(0.99, 0.97, direction, transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5, color=GRAY, style='italic')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right' if lower else 'lower right',
                  fontsize=8.5, framealpha=0.85)
 
    axes[-1].set_xlabel('Training Episode')
    axes[-1].set_xlim(0, len(ep) - 1)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('Figure2_Convergence.png', bbox_inches='tight')
    plt.close()
    print("  → Figure2_Convergence.png")
 
 
# =============================================================================
# FIGURE 3 — Box Plots: Equitable vs Productivity-Only
# =============================================================================
 
def plot_boxplots(equitable, prod_only):
    eq_r  = equitable['rewards'][:, -20:].mean(axis=1)
    po_r  = prod_only['rewards'][:,  -20:].mean(axis=1)
 
    t_stat, p_val = stats.ttest_rel(eq_r, po_r)
    pooled_sd     = np.sqrt((eq_r.std()**2 + po_r.std()**2) / 2)
    cohen_d       = (eq_r.mean() - po_r.mean()) / (pooled_sd + 1e-9)
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Figure 3: Simulation Robustness — Cumulative Reward Distribution\n'
                 'Equitable DQN vs Productivity-Only Baseline  ·  20 Seeds',
                 fontsize=12, fontweight='bold')
 
    # Box plot
    ax = axes[0]
    bp = ax.boxplot([eq_r, po_r], patch_artist=True, notch=True,
                    medianprops=dict(color='white', lw=2.5),
                    whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5),
                    flierprops=dict(marker='o', ms=5, alpha=0.5))
    bp['boxes'][0].set_facecolor(BLUE);   bp['boxes'][0].set_alpha(0.75)
    bp['boxes'][1].set_facecolor(ORANGE); bp['boxes'][1].set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Equitable DQN\n(w₃=0.1, w₄=0.1)',
                        'Productivity-Only\n(w₃=0, w₄=0)'])
    ax.set_ylabel('Mean Cumulative Reward (last 20 eps)')
    ax.set_title('Reward Distribution by Framework')
    y_max = max(eq_r.max(), po_r.max()) + 1.5
    ax.annotate('', xy=(2, y_max), xytext=(1, y_max),
                arrowprops=dict(arrowstyle='-', color='black', lw=1))
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
    ax.text(1.5, y_max + 0.3,
            f'p={p_val:.5f} {sig}  d={cohen_d:.2f}',
            ha='center', fontsize=9.5)
    eq_p = mpatches.Patch(color=BLUE,   alpha=0.75,
        label=f'Equitable  μ={eq_r.mean():.2f}, σ={eq_r.std():.2f}')
    po_p = mpatches.Patch(color=ORANGE, alpha=0.75,
        label=f'Prod-Only  μ={po_r.mean():.2f}, σ={po_r.std():.2f}')
    ax.legend(handles=[eq_p, po_p], loc='lower right', fontsize=9)
 
    # Violin
    ax2 = axes[1]
    parts = ax2.violinplot([eq_r, po_r], positions=[1, 2],
                           showmedians=True, showextrema=True)
    parts['bodies'][0].set_facecolor(BLUE);   parts['bodies'][0].set_alpha(0.6)
    parts['bodies'][1].set_facecolor(ORANGE); parts['bodies'][1].set_alpha(0.6)
    for pc in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        parts[pc].set_color('black'); parts[pc].set_lw(1.5)
    jitter = np.random.default_rng(7).uniform(-0.06, 0.06, len(eq_r))
    ax2.scatter(np.ones(len(eq_r)) + jitter,   eq_r, color=BLUE,   alpha=0.6, s=30, zorder=3)
    ax2.scatter(np.ones(len(po_r))*2 + jitter, po_r, color=ORANGE, alpha=0.6, s=30, zorder=3)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Equitable DQN', 'Productivity-Only'])
    ax2.set_ylabel('Mean Cumulative Reward (last 20 eps)')
    ax2.set_title('Violin + Individual Seed Values')
 
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('Figure3_BoxPlots.png', bbox_inches='tight')
    plt.close()
    print("  → Figure3_BoxPlots.png")
    return eq_r, po_r, p_val, cohen_d
 
 
# =============================================================================
# FIGURE B1 — Ablation Bar Chart
# =============================================================================
 
def plot_ablation_bars(ablation_results):
    labels = [r['label'] for r in ablation_results]
    cols   = [BLUE, ORANGE, GREEN, RED, PURPLE]
    metrics = [
        ('rewards', 'Cumulative Reward\n(higher ↑)',  False),
        ('errors',  'Error Rate\n(lower ↓)',           True),
        ('fatigue', 'Fatigue Index\n(lower ↓)',        True),
        ('tput',    'Normalised Throughput\n(higher ↑)',False),
    ]
 
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Figure B1: Ablation Study — Component Contribution\n'
                 '5 configurations × 20 seeds. ★ = best on that metric.',
                 fontsize=13, fontweight='bold')
    axes = axes.flatten()
 
    for ax, (key, ylabel, lower) in zip(axes, metrics):
        means = [r[key][:, -20:].mean() for r in ablation_results]
        sds   = [r[key][:, -20:].mean(axis=1).std() for r in ablation_results]
 
        bars = ax.bar(range(len(labels)), means, color=cols,
                      alpha=0.83, edgecolor='white', lw=1.2)
        ax.errorbar(range(len(labels)), means, yerr=sds,
                    fmt='none', color='black', capsize=5, lw=1.5, zorder=5)
 
        winner = np.argmin(means) if lower else np.argmax(means)
        ax.text(winner, means[winner] + sds[winner]*1.4,
                '★', ha='center', fontsize=14, color='gold',
                fontweight='bold', zorder=6)
        ax.axhline(means[0], color=BLUE, ls='--', lw=1.2, alpha=0.5,
                   label='Full model reference')
        for i, (v, sd) in enumerate(zip(means, sds)):
            ax.text(i, v * 0.5, f'{v:.3f}', ha='center', va='center',
                    fontsize=8.5, color='white', fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split('\n')[0])
        ax.legend(fontsize=8)
 
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('FigureB1_Ablation_Bars.png', bbox_inches='tight')
    plt.close()
    print("  → FigureB1_Ablation_Bars.png")
 
 
# =============================================================================
# FIGURE B2 — Ablation Heatmap
# =============================================================================
 
def plot_ablation_heatmap(ablation_results):
    row_labels = [r['label'] for r in ablation_results]
    col_labels = ['Reward', 'Error Rate', 'Fatigue', 'Throughput']
    lower_better = [False, True, True, False]
 
    data = np.array([
        [r['rewards'][:,-20:].mean(), r['errors'][:,-20:].mean(),
         r['fatigue'][:,-20:].mean(), r['tput'][:,-20:].mean()]
        for r in ablation_results
    ])
 
    norm = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        mn, mx = col.min(), col.max()
        rng = mx - mn if mx > mn else 1
        norm[:, j] = (col - mn) / rng
        if lower_better[j]:
            norm[:, j] = 1 - norm[:, j]
 
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle('Figure B2: Ablation Heatmap  (green = best, red = worst  ★ = column winner)',
                 fontsize=12, fontweight='bold')
    im = ax.imshow(norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
 
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            winner = np.argmax(norm[:, j])
            txt    = f'{"★ " if i == winner else ""}{data[i,j]:.3f}'
            color  = 'white' if norm[i,j] < 0.3 or norm[i,j] > 0.7 else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=10, color=color,
                    fontweight='bold' if i == winner else 'normal')
 
    ax.set_xticks(range(len(col_labels)));  ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)));  ax.set_yticklabels(row_labels, fontsize=10)
    plt.colorbar(im, ax=ax, label='Normalised (1=best)', shrink=0.85)
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig('FigureB2_Ablation_Heatmap.png', bbox_inches='tight')
    plt.close()
    print("  → FigureB2_Ablation_Heatmap.png")
 
 
# =============================================================================
# RESULTS TABLE
# =============================================================================
 
def print_results(equitable, prod_only, ablation_results,
                  eq_r, po_r, p_val, cohen_d):
    sep = "=" * 80
 
    # --- Throughput normalisation ---
    # Paper reports normalised throughput (0–1). Our environment's tput
    # = machine_speed × (1 - error_rate), already in [0, 1].
    eq_tput_m, eq_tput_sd = plateau_stats(equitable['tput'])
    po_tput_m, po_tput_sd = plateau_stats(prod_only['tput'])
    eq_err_m,  eq_err_sd  = plateau_stats(equitable['errors'])
    po_err_m,  po_err_sd  = plateau_stats(prod_only['errors'])
    eq_fat_m,  eq_fat_sd  = plateau_stats(equitable['fatigue'])
    po_fat_m,  po_fat_sd  = plateau_stats(prod_only['fatigue'])
 
    tput_chg  = (eq_tput_m - po_tput_m) / po_tput_m * 100
    err_chg   = (eq_err_m  - po_err_m)  / po_err_m  * 100
    fat_chg   = (eq_fat_m  - po_fat_m)  / po_fat_m  * 100
    rew_chg   = (eq_r.mean() - po_r.mean()) / abs(po_r.mean()) * 100
 
    print(f"\n{sep}")
    print("RESULTS TABLE — PASTE THIS BACK FOR MANUSCRIPT FINALISATION")
    print(sep)
 
    print("\n[TABLE 5: EQUITABLE DQN vs PRODUCTIVITY-ONLY]")
    print(f"{'Metric':<25} {'Equitable DQN':>15} {'Prod-Only':>15} {'Change':>10}")
    print("-" * 70)
    print(f"{'Throughput':<25} {eq_tput_m:>15.3f} {po_tput_m:>15.3f} {tput_chg:>+9.1f}%")
    print(f"{'Error Rate':<25} {eq_err_m:>15.4f} {po_err_m:>15.4f} {err_chg:>+9.1f}%")
    print(f"{'Fatigue Index':<25} {eq_fat_m:>15.4f} {po_fat_m:>15.4f} {fat_chg:>+9.1f}%")
    print(f"{'Cumul. Reward (mean)':<25} {eq_r.mean():>15.3f} {po_r.mean():>15.3f} {rew_chg:>+9.1f}%")
    print(f"{'Reward SD':<25} {eq_r.std():>15.3f} {po_r.std():>15.3f}")
    print(f"\np-value (paired t-test): {p_val:.6f}  |  Cohen's d: {cohen_d:.3f}")
 
    print("\n[ABLATION STUDY — 5 Configurations, last 20 episodes, 20 seeds]")
    hdr = f"{'Configuration':<22} {'Reward':>8} {'±SD':>6} {'Error':>8} {'±SD':>6} {'Fatigue':>8} {'±SD':>6} {'Tput':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in ablation_results:
        rm,  rs  = plateau_stats(r['rewards'])
        em,  es  = plateau_stats(r['errors'])
        fm,  fs  = plateau_stats(r['fatigue'])
        tm,  ts  = plateau_stats(r['tput'])
        flag = " ◄ FULL" if r['label'] == 'Full Model' else ""
        print(f"{r['label']:<22} {rm:>8.3f} {rs:>6.3f} {em:>8.4f} {es:>6.4f} "
              f"{fm:>8.4f} {fs:>6.4f} {tm:>8.4f}{flag}")
 
    # Key findings
    full = ablation_results[0]
    nof  = ablation_results[1]  # No Fatigue
    noe  = ablation_results[2]  # No Equity
    now  = ablation_results[3]  # No Welfare
    full_fat = plateau_stats(full['fatigue'])[0]
    nof_fat  = plateau_stats(nof['fatigue'])[0]
    full_eq  = plateau_stats(full['errors'])[0]  # proxy for equity through error
    noe_fat  = plateau_stats(noe['fatigue'])[0]
    now_rew  = plateau_stats(now['rewards'])[0]
    full_rew = plateau_stats(full['rewards'])[0]
 
    fat_contrib = (nof_fat - full_fat) / nof_fat * 100
    rew_cost    = (now_rew - full_rew) / abs(now_rew) * 100
 
    print(f"\n[KEY FINDINGS]")
    print(f"  Removing w₃ (fatigue weight)  increases fatigue by: {fat_contrib:.1f}%")
    print(f"  Full model reward vs No-Welfare:                    {rew_cost:+.1f}%")
    print(f"  Dominant welfare component: see heatmap (FigureB2)")
 
    print(f"\n{sep}")
    import os
    figs = ['Figure2_Convergence.png', 'Figure3_BoxPlots.png',
            'FigureB1_Ablation_Bars.png', 'FigureB2_Ablation_Heatmap.png']
    print("FILES SAVED:")
    for f in figs:
        kb = os.path.getsize(f)/1024 if os.path.exists(f) else 0
        print(f"  {f:<40} {kb:>6.0f} KB")
    print(sep)
 
 
# =============================================================================
# MAIN
# =============================================================================
 
if __name__ == '__main__':
    N_SEEDS    = 20
    N_EPISODES = 100   # matches paper (Table 1: Episodes = 100)
 
    print("=" * 60)
    print("Equitable HRC DQN — Full Experiment")
    print(f"Seeds: {N_SEEDS}  |  Episodes: {N_EPISODES}")
    print("Using: YOUR CementBaggingHRCEnvironment + paper's PyTorch DQN")
    print("=" * 60)
 
    total_start = time.time()
 
    # ── Part A: Baseline comparison ──────────────────────────────────────────
    print("\n[PART A] Baseline comparison (2 configs × 20 seeds)...")
 
    equitable = run_config('Full Model (Equitable)',
                           w1=0.5, w2=0.3, w3=0.1, w4=0.1,
                           n_seeds=N_SEEDS, n_episodes=N_EPISODES)
    gc.collect()
 
    prod_only = run_config('Productivity-Only',
                           w1=0.6, w2=0.4, w3=0.0, w4=0.0,
                           n_seeds=N_SEEDS, n_episodes=N_EPISODES)
    gc.collect()
 
    print("\n[PART A] Generating figures...")
    plot_convergence(equitable)
    eq_r, po_r, p_val, cohen_d = plot_boxplots(equitable, prod_only)
 
    # ── Part B: 5-configuration ablation ─────────────────────────────────────
    print("\n[PART B] Ablation study (5 configs × 20 seeds, 100 episodes each)...")
 
    ABLATION_CONFIGS = [
        ('Full Model',    0.50, 0.30, 0.10, 0.10),
        ('No Fatigue',    0.55, 0.35, 0.00, 0.10),
        ('No Equity',     0.55, 0.35, 0.10, 0.00),
        ('No Welfare',    0.60, 0.40, 0.00, 0.00),
        ('No Quality',    0.625,0.00, 0.1875,0.1875),
    ]
 
    ablation_results = []
    for name, w1, w2, w3, w4 in ABLATION_CONFIGS:
        res = run_config(name, w1, w2, w3, w4,
                         n_seeds=N_SEEDS, n_episodes=N_EPISODES)
        ablation_results.append(res)
        gc.collect()
 
    print("\n[PART B] Generating ablation figures...")
    plot_ablation_bars(ablation_results)
    plot_ablation_heatmap(ablation_results)
 
    # ── Results table ─────────────────────────────────────────────────────────
    print_results(equitable, prod_only, ablation_results,
                  eq_r, po_r, p_val, cohen_d)
 
    elapsed = (time.time() - total_start) / 60
    print(f"\nTotal runtime: {elapsed:.1f} minutes")
    print("Done. Paste the RESULTS TABLE above back into the chat.")
 






Result
============================================================
Equitable HRC DQN — Full Experiment
Seeds: 20  |  Episodes: 100
Using: YOUR CementBaggingHRCEnvironment + paper's PyTorch DQN
============================================================

[PART A] Baseline comparison (2 configs × 20 seeds)...
  Full Model (Equitable)       w=[0.5,0.3,0.1,0.1]
  Productivity-Only            w=[0.6,0.4,0.0,0.0]

[PART A] Generating figures...
  → Figure2_Convergence.png
  → Figure3_BoxPlots.png

[PART B] Ablation study (5 configs × 20 seeds, 100 episodes each)...
  Full Model                   w=[0.5,0.3,0.1,0.1]
  No Fatigue                   w=[0.55,0.35,0.0,0.1]
  No Equity                    w=[0.55,0.35,0.1,0.0]
  No Welfare                   w=[0.6,0.4,0.0,0.0]
  No Quality                   w=[0.625,0.0,0.1875,0.1875]

[PART B] Generating ablation figures...
  → FigureB1_Ablation_Bars.png
  → FigureB2_Ablation_Heatmap.png

================================================================================
RESULTS TABLE — PASTE THIS BACK FOR MANUSCRIPT FINALISATION
================================================================================

[TABLE 5: EQUITABLE DQN vs PRODUCTIVITY-ONLY]
Metric                      Equitable DQN       Prod-Only     Change
----------------------------------------------------------------------
Throughput                          0.743           0.741      +0.3%
Error Rate                         0.0167          0.0200     -16.8%
Fatigue Index                      0.0661          0.0943     -29.9%
Cumul. Reward (mean)              181.118         218.262     -17.0%
Reward SD                           3.164           3.447

p-value (paired t-test): 0.000000  |  Cohen's d: -11.226

[ABLATION STUDY — 5 Configurations, last 20 episodes, 20 seeds]
Configuration            Reward    ±SD    Error    ±SD  Fatigue    ±SD     Tput
-------------------------------------------------------------------------------
Full Model              181.118  3.164   0.0167 0.0036   0.0661 0.0152   0.7433 ◄ FULL
No Fatigue              200.433  3.253   0.0181 0.0033   0.0844 0.0226   0.7423
No Equity               200.269  3.701   0.0159 0.0034   0.0631 0.0191   0.7441
No Welfare              218.262  3.447   0.0200 0.0033   0.0943 0.0215   0.7409
No Quality              227.891  3.835   0.0200 0.0032   0.0612 0.0174   0.7410

[KEY FINDINGS]
  Removing w₃ (fatigue weight)  increases fatigue by: 21.7%
  Full model reward vs No-Welfare:                    +17.0%
  Dominant welfare component: see heatmap (FigureB2)

================================================================================
FILES SAVED:
  Figure2_Convergence.png                     997 KB
  Figure3_BoxPlots.png                        283 KB
  FigureB1_Ablation_Bars.png                  442 KB
  FigureB2_Ablation_Heatmap.png               199 KB
================================================================================

Total runtime: 155.3 minutes
Done. Paste the RESULTS TABLE above back into the chat.