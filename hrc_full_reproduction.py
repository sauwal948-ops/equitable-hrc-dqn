"""
=============================================================================
HRC DQN — Full Paper Reproduction
=============================================================================
Paper: Multi-Objective RL for Human-Robot Collaborative Task Allocation
Authors: Salisu Auwal Musa, Bashir Muhammad Ahmad
 
Reproduces ALL remaining experimental results in the paper:
  Part A — PPO vs DQN architectural comparison (Table 6)
  Part B — Sensitivity analysis: 3 scenarios (Section 3.3)
  Part C — Cross-industry generalization: 5 sectors (Table 7)
 
Runtime: ~35–50 minutes on Kaggle CPU
RAM:     < 3 GB
 
No extra installs needed — only torch, numpy, matplotlib, scipy.
=============================================================================
"""
 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
import random, time, gc, warnings
from scipy import stats
warnings.filterwarnings('ignore')
 
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
 
# ── Colours ────────────────────────────────────────────────────────────────────
BLUE   = '#1B4F8A'
ORANGE = '#D45F00'
GREEN  = '#1A6B3A'
RED    = '#B22222'
PURPLE = '#5B2C8A'
GRAY   = '#555555'
 
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
    'figure.dpi': 120, 'savefig.dpi': 300,
})
 
# =============================================================================
# ENVIRONMENT (your exact code from hrc_rl_simulation.py)
# =============================================================================
 
class CementBaggingHRCEnvironment:
    def __init__(self, num_workers=3, w1=0.5, w2=0.3, w3=0.1, w4=0.1,
                 skill_profile='mixed'):
        self.num_workers   = num_workers
        self.w1, self.w2   = w1, w2
        self.w3, self.w4   = w3, w4
        self.episode_step  = 0
        self.max_steps     = 500
 
        # skill_profile controls workforce composition
        if skill_profile == 'mixed':
            self.workers = np.array([
                [0.3, 0.5, 1],   # Junior
                [0.2, 0.7, 5],   # Intermediate
                [0.1, 0.9, 10]   # Senior
            ])
        elif skill_profile == 'intermediate':
            self.workers = np.array([
                [0.2, 0.65, 5], [0.2, 0.70, 5], [0.2, 0.75, 5]
            ])
        elif skill_profile == 'junior':
            self.workers = np.array([
                [0.4, 0.4, 1], [0.35, 0.45, 1], [0.3, 0.5, 1]
            ])
        elif skill_profile == 'senior':
            self.workers = np.array([
                [0.1, 0.85, 10], [0.1, 0.9, 10], [0.1, 0.95, 10]
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
            self.queue_length, worker[1]
        ], dtype=np.float32)
 
    def step(self, action):
        self.episode_step += 1
        self.machine_speed = np.clip(
            self.machine_speed + np.random.normal(0, 0.05), 0.5, 1.0)
 
        if   action == 0: self.human_fatigue = np.clip(self.human_fatigue + 0.08, 0, 1)
        elif action == 1: self.human_fatigue = np.clip(self.human_fatigue + 0.03, 0, 1)
        elif action == 2: self.human_fatigue = np.clip(self.human_fatigue - 0.05, 0, 1)
        elif action == 3: self.human_fatigue = np.clip(self.human_fatigue - 0.15, 0, 1)
 
        fe = self.human_fatigue * 0.1
        if   action == 1: self.error_rate = np.clip(self.error_rate - 0.02 + fe, 0.01, 0.3)
        elif action == 2: self.error_rate = np.clip(self.error_rate - 0.05,       0.01, 0.2)
        else:             self.error_rate = np.clip(self.error_rate + fe,          0.01, 0.3)
 
        pr = self.machine_speed * (1 - self.error_rate)
        if action == 2: pr *= 1.2
        self.queue_length = np.clip(
            self.queue_length - pr * 0.1 + np.random.uniform(0, 0.05), 0, 1)
 
        self.current_worker = (self.current_worker + 1) % self.num_workers
        reward = self._reward(action)
        done   = self.episode_step >= self.max_steps
        return self.get_state(), reward, done
 
    def _reward(self, action):
        tp  = self.machine_speed * (1 - self.error_rate)
        r_t = tp
        r_e = -self.error_rate
        r_f = -self.human_fatigue * 0.5
        skill = self.workers[self.current_worker][1]
        r_b   = -0.1 if (action == 2 and skill < 0.6) else 0
        return float(self.w1*r_t + self.w2*r_e + self.w3*r_f + self.w4*r_b)
 
    def action_counts_snapshot(self):
        return {
            'machine_speed':  self.machine_speed,
            'human_fatigue':  self.human_fatigue,
            'error_rate':     self.error_rate,
            'throughput':     self.machine_speed * (1 - self.error_rate),
        }
 
 
# =============================================================================
# DQN AGENT (paper architecture: 5-128-128-4, He init, target net)
# =============================================================================
 
class QNetwork(nn.Module):
    def __init__(self, state_dim=5, action_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight, nonlinearity='relu')
                nn.init.zeros_(l.bias)
 
    def forward(self, x):
        return self.net(x)
 
 
class ReplayBuffer:
    def __init__(self, cap=10000):
        self.buf = deque(maxlen=cap)
 
    def push(self, *args):
        self.buf.append(args)
 
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int64),
                np.array(r,  dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d,  dtype=np.float32))
 
    def __len__(self):
        return len(self.buf)
 
 
class DQNAgent:
    def __init__(self, state_dim=5, action_dim=4, lr=0.001, gamma=0.95,
                 eps_start=1.0, eps_end=0.01, eps_decay_eps=60,
                 buffer=10000, batch=32, target_freq=100, n_episodes=100):
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.batch       = batch
        self.tgt_freq    = target_freq
        self.step_n      = 0
        self.epsilon     = eps_start
        self.eps_end     = eps_end
        self.eps_step    = (eps_start - eps_end) / eps_decay_eps
 
        self.online = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt    = optim.Adam(self.online.parameters(), lr=lr)
        self.buf    = ReplayBuffer(buffer)
 
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            return self.online(torch.FloatTensor(state).unsqueeze(0)).argmax(1).item()
 
    def push(self, s, a, r, ns, d):
        self.buf.push(s, a, r, ns, d)
 
    def learn(self):
        if len(self.buf) < self.batch: return
        s, a, r, ns, d = self.buf.sample(self.batch)
        s  = torch.FloatTensor(s)
        a  = torch.LongTensor(a).unsqueeze(1)
        r  = torch.FloatTensor(r).unsqueeze(1)
        ns = torch.FloatTensor(ns)
        d  = torch.FloatTensor(d).unsqueeze(1)
        q  = self.online(s).gather(1, a)
        with torch.no_grad():
            qt = self.target(ns).max(1, keepdim=True)[0]
        loss = nn.MSELoss()(q, r + self.gamma * qt * (1 - d))
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.step_n += 1
        if self.step_n % self.tgt_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
 
    def decay(self):
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_step)
 
 
# =============================================================================
# PPO AGENT (vanilla implementation, no external library needed)
# Policy: 5-128-128-4 softmax; Value: 5-128-128-1
# Hyperparams match standard PPO: clip 0.2, 4 epochs, GAE lambda 0.95
# =============================================================================
 
class ActorCritic(nn.Module):
    def __init__(self, state_dim=5, action_dim=4, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU()
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
        for l in self.modules():
            if isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight, nonlinearity='relu')
                nn.init.zeros_(l.bias)
 
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)
 
    def act(self, state):
        with torch.no_grad():
            logits, val = self.forward(torch.FloatTensor(state).unsqueeze(0))
            dist  = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), val.item()
 
    def evaluate(self, states, actions):
        logits, vals = self.forward(states)
        dist     = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, vals.squeeze(), entropy
 
 
class PPOAgent:
    def __init__(self, state_dim=5, action_dim=4, lr=0.0003,
                 gamma=0.99, clip=0.2, epochs=4, gae_lam=0.95,
                 batch=64):
        self.gamma   = gamma
        self.clip    = clip
        self.epochs  = epochs
        self.lam     = gae_lam
        self.batch   = batch
        self.ac      = ActorCritic(state_dim, action_dim)
        self.opt     = optim.Adam(self.ac.parameters(), lr=lr)
 
        # Episode buffer
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones     = [], [], []
 
    def act(self, state):
        action, log_prob, val = self.ac.act(state)
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(val)
        return action
 
    def store(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
 
    def learn(self):
        if len(self.rewards) == 0: return
 
        # GAE returns
        returns, adv = [], []
        gae = 0
        vals = self.values + [0]
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * vals[i+1] * (1 - self.dones[i]) - vals[i]
            gae   = delta + self.gamma * self.lam * (1 - self.dones[i]) * gae
            adv.insert(0, gae)
            returns.insert(0, gae + vals[i])
 
        states    = torch.FloatTensor(np.array(self.states))
        actions   = torch.LongTensor(self.actions)
        old_lps   = torch.FloatTensor(self.log_probs)
        returns_t = torch.FloatTensor(returns)
        adv_t     = torch.FloatTensor(adv)
        adv_t     = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
 
        n = len(states)
        for _ in range(self.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch):
                b = idx[start:start + self.batch]
                lp, v, ent = self.ac.evaluate(states[b], actions[b])
                ratio  = (lp - old_lps[b]).exp()
                s1     = ratio * adv_t[b]
                s2     = ratio.clamp(1 - self.clip, 1 + self.clip) * adv_t[b]
                a_loss = -torch.min(s1, s2).mean()
                c_loss = nn.MSELoss()(v, returns_t[b])
                loss   = a_loss + 0.5 * c_loss - 0.01 * ent.mean()
                self.opt.zero_grad(); loss.backward(); self.opt.step()
 
        self.states.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.values.clear(); self.dones.clear()
 
 
# =============================================================================
# SHARED TRAINING LOOP
# =============================================================================
 
def run_one_episode_dqn(env, agent, train=True):
    state = env.reset()
    ep_reward, errors, fatigues, tputs, ac = 0, [], [], [], {0:0,1:0,2:0,3:0}
    for _ in range(env.max_steps):
        action = agent.act(state)
        ac[action] += 1
        ns, reward, done = env.step(action)
        if train:
            agent.push(state, action, reward, ns, done)
            agent.learn()
        ep_reward += reward
        errors.append(env.error_rate)
        fatigues.append(env.human_fatigue)
        tputs.append(env.machine_speed * (1 - env.error_rate))
        state = ns
        if done: break
    if train: agent.decay()
    return ep_reward, np.mean(errors), np.mean(fatigues), np.mean(tputs), ac
 
 
def run_one_episode_ppo(env, agent):
    state = env.reset()
    ep_reward, errors, fatigues, tputs, ac = 0, [], [], [], {0:0,1:0,2:0,3:0}
    for _ in range(env.max_steps):
        action = agent.act(state)
        ac[action] += 1
        ns, reward, done = env.step(action)
        agent.store(reward, done)
        ep_reward += reward
        errors.append(env.error_rate)
        fatigues.append(env.human_fatigue)
        tputs.append(env.machine_speed * (1 - env.error_rate))
        state = ns
        if done: break
    agent.learn()
    return ep_reward, np.mean(errors), np.mean(fatigues), np.mean(tputs), ac
 
 
def train_config_dqn(label, w1, w2, w3, w4, n_seeds=20, n_episodes=100,
                     skill_profile='mixed'):
    print(f"  DQN  {label:<30} w=[{w1},{w2},{w3},{w4}]", flush=True)
    rewards, errors, fatigues, tputs, all_ac = [], [], [], [], []
    for seed in range(n_seeds):
        random.seed(seed + GLOBAL_SEED)
        np.random.seed(seed + GLOBAL_SEED)
        torch.manual_seed(seed + GLOBAL_SEED)
        env   = CementBaggingHRCEnvironment(w1=w1, w2=w2, w3=w3, w4=w4,
                                             skill_profile=skill_profile)
        agent = DQNAgent(n_episodes=n_episodes)
        ep_r, ep_e, ep_f, ep_t = [], [], [], []
        last_ac = {}
        for ep in range(n_episodes):
            r, e, f, t, ac = run_one_episode_dqn(env, agent)
            ep_r.append(r); ep_e.append(e); ep_f.append(f); ep_t.append(t)
            last_ac = ac
        rewards.append(ep_r); errors.append(ep_e)
        fatigues.append(ep_f); tputs.append(ep_t)
        all_ac.append(last_ac)
    return {
        'label': label, 'algo': 'DQN',
        'rewards':  np.array(rewards),
        'errors':   np.array(errors),
        'fatigues': np.array(fatigues),
        'tputs':    np.array(tputs),
        'action_counts': all_ac,
    }
 
 
def train_config_ppo(label, w1, w2, w3, w4, n_seeds=20, n_episodes=100,
                     skill_profile='mixed'):
    print(f"  PPO  {label:<30} w=[{w1},{w2},{w3},{w4}]", flush=True)
    rewards, errors, fatigues, tputs, all_ac = [], [], [], [], []
    for seed in range(n_seeds):
        random.seed(seed + GLOBAL_SEED)
        np.random.seed(seed + GLOBAL_SEED)
        torch.manual_seed(seed + GLOBAL_SEED)
        env   = CementBaggingHRCEnvironment(w1=w1, w2=w2, w3=w3, w4=w4,
                                             skill_profile=skill_profile)
        agent = PPOAgent()
        ep_r, ep_e, ep_f, ep_t = [], [], [], []
        last_ac = {}
        for ep in range(n_episodes):
            r, e, f, t, ac = run_one_episode_ppo(env, agent)
            ep_r.append(r); ep_e.append(e); ep_f.append(f); ep_t.append(t)
            last_ac = ac
        rewards.append(ep_r); errors.append(ep_e)
        fatigues.append(ep_f); tputs.append(ep_t)
        all_ac.append(last_ac)
    return {
        'label': label, 'algo': 'PPO',
        'rewards':  np.array(rewards),
        'errors':   np.array(errors),
        'fatigues': np.array(fatigues),
        'tputs':    np.array(tputs),
        'action_counts': all_ac,
    }
 
 
def plateau(data, n=20):
    """Mean and SD of last n episodes across seeds."""
    seg = data[:, -n:]
    m   = seg.mean(axis=1)
    return m.mean(), m.std()
 
 
def action_pct(all_ac, key):
    """Average % of a given action across seeds."""
    vals = []
    for ac in all_ac:
        total = sum(ac.values())
        vals.append(ac.get(key, 0) / total * 100 if total > 0 else 0)
    return np.mean(vals)
 
 
# =============================================================================
# PART A — PPO vs DQN ARCHITECTURAL COMPARISON
# =============================================================================
 
def run_part_A(n_seeds=20, n_episodes=100):
    print("\n" + "="*60)
    print("PART A: DQN vs PPO Architectural Comparison")
    print("Both run with productivity-weighted config w=[0.5,0.3,0.1,0.1]")
    print("="*60)
 
    # Both use the SAME reward weights to isolate architecture, not reward design
    dqn = train_config_dqn('DQN (equitable weights)', 0.5, 0.3, 0.1, 0.1,
                            n_seeds=n_seeds, n_episodes=n_episodes)
    ppo = train_config_ppo('PPO (equitable weights)', 0.5, 0.3, 0.1, 0.1,
                            n_seeds=n_seeds, n_episodes=n_episodes)
 
    return dqn, ppo
 
 
def plot_part_A(dqn, ppo):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Figure A: DQN vs PPO Architectural Comparison\n'
                 'Same reward weights w=[0.5,0.3,0.1,0.1] · 20 Seeds · 100 Episodes',
                 fontsize=12, fontweight='bold')
 
    ep = np.arange(dqn['rewards'].shape[1])
    metrics = [
        ('rewards',  'Cumulative Reward\n(higher = better)'),
        ('errors',   'Error Rate\n(lower = better)'),
        ('fatigues', 'Fatigue Index\n(lower = better)'),
    ]
 
    for ax, (key, ylabel) in zip(axes, metrics):
        for res, col, lbl in [(dqn, BLUE, 'DQN'), (ppo, ORANGE, 'PPO')]:
            m    = res[key].mean(axis=0)
            lo   = np.percentile(res[key], 25, axis=0)
            hi   = np.percentile(res[key], 75, axis=0)
            ax.fill_between(ep, lo, hi, alpha=0.18, color=col)
            ax.plot(ep, m, color=col, lw=2.2, label=f'{lbl} (mean)')
        ax.set_xlabel('Episode'); ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split('\n')[0])
        ax.legend(fontsize=9)
 
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('FigA_DQN_vs_PPO.png', bbox_inches='tight')
    plt.close()
    print("  → FigA_DQN_vs_PPO.png")
 
 
# =============================================================================
# PART B — SENSITIVITY ANALYSIS (3 scenarios)
# =============================================================================
 
def run_part_B(n_seeds=20, n_episodes=100):
    print("\n" + "="*60)
    print("PART B: Sensitivity Analysis — 3 Operational Scenarios")
    print("="*60)
 
    configs = [
        # label,              w1,    w2,    w3,    w4,   profile
        ('Baseline Equitable', 0.50,  0.30,  0.10,  0.10, 'mixed'),
        ('Safety-First',       0.30,  0.30,  0.50,  0.10, 'mixed'),
        ('Production-Critical',0.80,  0.15,  0.05,  0.00, 'mixed'),
    ]
 
    results = []
    for label, w1, w2, w3, w4, profile in configs:
        r = train_config_dqn(label, w1, w2, w3, w4,
                              n_seeds=n_seeds, n_episodes=n_episodes,
                              skill_profile=profile)
        results.append(r)
        gc.collect()
 
    # ── Skill Generalisation: correct implementation ────────────────────────
    # Step 1: Train fully on Intermediate profile (paper description)
    # Step 2: Freeze weights — no further learning
    # Step 3: Evaluate frozen agent on Junior and Senior profiles
    # This tests whether the agent adapts its action selection via the
    # skill_level state component even when weights are frozen.
    print("  DQN  Skill-Generaliz. — TRAIN on Intermediate profile", flush=True)
 
    junior_ac_list, senior_ac_list = [], []
    junior_fat_list, senior_fat_list = [], []
    junior_err_list, senior_err_list = [], []
 
    for seed in range(n_seeds):
        random.seed(seed + GLOBAL_SEED)
        np.random.seed(seed + GLOBAL_SEED)
        torch.manual_seed(seed + GLOBAL_SEED)
 
        # --- Train on intermediate ---
        env_train = CementBaggingHRCEnvironment(
            w1=0.5, w2=0.3, w3=0.1, w4=0.1, skill_profile='intermediate')
        agent = DQNAgent(n_episodes=n_episodes)
        for ep in range(n_episodes):
            run_one_episode_dqn(env_train, agent, train=True)
 
        # --- Freeze: set epsilon to 0 so agent always acts greedily ---
        agent.epsilon = 0.0
 
        # --- Evaluate on Junior (20 episodes, no learning) ---
        env_junior = CementBaggingHRCEnvironment(
            w1=0.5, w2=0.3, w3=0.1, w4=0.1, skill_profile='junior')
        j_ac = {0: 0, 1: 0, 2: 0, 3: 0}
        j_fat, j_err = [], []
        for ep in range(20):
            _, e, f, _, ac = run_one_episode_dqn(env_junior, agent, train=False)
            for k, v in ac.items(): j_ac[k] += v
            j_fat.append(f); j_err.append(e)
        junior_ac_list.append(j_ac)
        junior_fat_list.append(np.mean(j_fat))
        junior_err_list.append(np.mean(j_err))
 
        # --- Evaluate on Senior (20 episodes, no learning) ---
        env_senior = CementBaggingHRCEnvironment(
            w1=0.5, w2=0.3, w3=0.1, w4=0.1, skill_profile='senior')
        s_ac = {0: 0, 1: 0, 2: 0, 3: 0}
        s_fat, s_err = [], []
        for ep in range(20):
            _, e, f, _, ac = run_one_episode_dqn(env_senior, agent, train=False)
            for k, v in ac.items(): s_ac[k] += v
            s_fat.append(f); s_err.append(e)
        senior_ac_list.append(s_ac)
        senior_fat_list.append(np.mean(s_fat))
        senior_err_list.append(np.mean(s_err))
 
    skill_results = {
        'junior': {'action_counts': junior_ac_list,
                   'fatigue': np.mean(junior_fat_list),
                   'error':   np.mean(junior_err_list)},
        'senior': {'action_counts': senior_ac_list,
                   'fatigue': np.mean(senior_fat_list),
                   'error':   np.mean(senior_err_list)},
    }
    print("  → Skill generalisation complete")
    return results, skill_results
 
 
def plot_part_B(results, skill_results):
    labels = [r['label'] for r in results]
    cols   = [BLUE, GREEN, RED]
 
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Figure B: Sensitivity Analysis — Three Operational Scenarios\n'
                 '20 Seeds · 100 Episodes Each',
                 fontsize=12, fontweight='bold')
    axes = axes.flatten()
 
    metrics = [
        ('rewards',  'Cumulative Reward (↑)', False),
        ('errors',   'Error Rate (↓)',         True),
        ('fatigues', 'Fatigue Index (↓)',       True),
        ('tputs',    'Throughput (↑)',           False),
    ]
 
    for ax, (key, ylabel, lb) in zip(axes, metrics):
        means = [plateau(r[key])[0] for r in results]
        sds   = [plateau(r[key])[1] for r in results]
        bars  = ax.bar(range(len(labels)), means, color=cols,
                       alpha=0.83, edgecolor='white', lw=1.2)
        ax.errorbar(range(len(labels)), means, yerr=sds,
                    fmt='none', color='black', capsize=5, lw=1.5, zorder=5)
        ax.axhline(means[0], color=BLUE, ls='--', lw=1.2, alpha=0.5,
                   label='Baseline reference')
        for i, (v, sd) in enumerate(zip(means, sds)):
            ax.text(i, v * 0.5, f'{v:.3f}', ha='center', va='center',
                    fontsize=8.5, color='white', fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=8.5)
        ax.set_ylabel(ylabel); ax.set_title(ylabel.split('(')[0].strip())
        ax.legend(fontsize=8)
 
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('FigB_Sensitivity_Analysis.png', bbox_inches='tight')
    plt.close()
    print("  → FigB_Sensitivity_Analysis.png")
 
    # Action distribution radar
    action_names = ['Idle', 'Assist', 'TakeOver', 'SuggestBreak']
    fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    fig2.suptitle('Figure B2: Action Distribution — Sensitivity Scenarios',
                  fontsize=12, fontweight='bold')
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist()
    angles += angles[:1]
 
    for res, col in zip(results, cols):
        vals = [action_pct(res['action_counts'], k) for k in range(4)]
        vals += vals[:1]
        ax2.plot(angles, vals, color=col, lw=2.5, label=res['label'])
        ax2.fill(angles, vals, color=col, alpha=0.1)
 
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(action_names, fontsize=10)
    ax2.set_yticklabels([])
    ax2.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=9)
    plt.tight_layout()
    plt.savefig('FigB2_Action_Radar.png', bbox_inches='tight')
    plt.close()
    print("  → FigB2_Action_Radar.png")
 
 
# =============================================================================
# PART C — CROSS-INDUSTRY GENERALIZATION (5 sectors)
# =============================================================================
 
# Each industry is modelled as a parameterised variant of the environment.
# Differences: machine speed range, initial fatigue range, error range,
# max steps (workload intensity), worker count.
 
INDUSTRY_CONFIGS = {
    'Cement (Baseline)': {
        'w1': 0.50, 'w2': 0.30, 'w3': 0.10, 'w4': 0.10,
        'speed_range': (0.7, 1.0), 'fatigue_range': (0.2, 0.5),
        'error_range': (0.05, 0.15), 'max_steps': 500,
    },
    'Electronics Assembly': {
        'w1': 0.45, 'w2': 0.40, 'w3': 0.08, 'w4': 0.07,  # precision-heavy
        'speed_range': (0.8, 1.0), 'fatigue_range': (0.1, 0.3),
        'error_range': (0.02, 0.10), 'max_steps': 500,
    },
    'Food Processing': {
        'w1': 0.40, 'w2': 0.25, 'w3': 0.20, 'w4': 0.15,  # physical-heavy
        'speed_range': (0.6, 0.9), 'fatigue_range': (0.3, 0.6),
        'error_range': (0.05, 0.20), 'max_steps': 500,
    },
    'Textile Mfg': {
        'w1': 0.50, 'w2': 0.35, 'w3': 0.08, 'w4': 0.07,
        'speed_range': (0.7, 1.0), 'fatigue_range': (0.2, 0.45),
        'error_range': (0.04, 0.12), 'max_steps': 500,
    },
    'Automotive Parts': {
        'w1': 0.45, 'w2': 0.30, 'w3': 0.15, 'w4': 0.10,
        'speed_range': (0.65, 0.95), 'fatigue_range': (0.25, 0.55),
        'error_range': (0.04, 0.14), 'max_steps': 500,
    },
}
 
 
class IndustryEnv(CementBaggingHRCEnvironment):
    """Environment with industry-specific parameterisation."""
    def __init__(self, config):
        self.speed_range   = config['speed_range']
        self.fatigue_range = config['fatigue_range']
        self.error_range   = config['error_range']
        super().__init__(
            w1=config['w1'], w2=config['w2'],
            w3=config['w3'], w4=config['w4']
        )
        self.max_steps = config['max_steps']
 
    def reset(self):
        self.machine_speed  = np.random.uniform(*self.speed_range)
        self.human_fatigue  = np.random.uniform(*self.fatigue_range)
        self.error_rate     = np.random.uniform(*self.error_range)
        self.queue_length   = np.random.uniform(0.3, 0.8)
        self.episode_step   = 0
        self.current_worker = np.random.randint(0, self.num_workers)
        return self.get_state()
 
 
def run_part_C(n_seeds=20, n_episodes=100):
    print("\n" + "="*60)
    print("PART C: Cross-Industry Generalization (5 sectors)")
    print("="*60)
 
    # Run productivity-only baseline for each industry to compute % changes
    industry_results = {}
    for name, cfg in INDUSTRY_CONFIGS.items():
        print(f"\n  Industry: {name}")
 
        # Equitable
        eq_r, eq_e, eq_f, eq_t = [], [], [], []
        for seed in range(n_seeds):
            random.seed(seed + GLOBAL_SEED)
            np.random.seed(seed + GLOBAL_SEED)
            torch.manual_seed(seed + GLOBAL_SEED)
            env   = IndustryEnv(cfg)
            agent = DQNAgent(n_episodes=n_episodes)
            ep_r, ep_e, ep_f, ep_t = [], [], [], []
            for ep in range(n_episodes):
                r, e, f, t, _ = run_one_episode_dqn(env, agent)
                ep_r.append(r); ep_e.append(e); ep_f.append(f); ep_t.append(t)
            eq_r.append(ep_r); eq_e.append(ep_e); eq_f.append(ep_f); eq_t.append(ep_t)
 
        # Productivity-only baseline for same industry
        base_cfg = {**cfg, 'w3': 0.0, 'w4': 0.0,
                    'w1': cfg['w1'] + cfg['w3']/2 + cfg['w4']/2,
                    'w2': cfg['w2'] + cfg['w3']/2 + cfg['w4']/2}
        po_r, po_e, po_f, po_t = [], [], [], []
        for seed in range(n_seeds):
            random.seed(seed + GLOBAL_SEED + 500)
            np.random.seed(seed + GLOBAL_SEED + 500)
            torch.manual_seed(seed + GLOBAL_SEED + 500)
            env   = IndustryEnv(base_cfg)
            agent = DQNAgent(n_episodes=n_episodes)
            ep_r, ep_e, ep_f, ep_t = [], [], [], []
            for ep in range(n_episodes):
                r, e, f, t, _ = run_one_episode_dqn(env, agent)
                ep_r.append(r); ep_e.append(e); ep_f.append(f); ep_t.append(t)
            po_r.append(ep_r); po_e.append(ep_e); po_f.append(ep_f); po_t.append(ep_t)
 
        industry_results[name] = {
            'eq':  {'r': np.array(eq_r), 'e': np.array(eq_e),
                    'f': np.array(eq_f), 't': np.array(eq_t)},
            'po':  {'r': np.array(po_r), 'e': np.array(po_e),
                    'f': np.array(po_f), 't': np.array(po_t)},
        }
        gc.collect()
 
    return industry_results
 
 
def plot_part_C(industry_results):
    names = list(industry_results.keys())
    cols  = [BLUE, PURPLE, GREEN, ORANGE, RED]
 
    # Compute metrics
    tputs    = [plateau(industry_results[n]['eq']['t'])[0] for n in names]
    err_eq   = [plateau(industry_results[n]['eq']['e'])[0] for n in names]
    err_po   = [plateau(industry_results[n]['po']['e'])[0] for n in names]
    fat_eq   = [plateau(industry_results[n]['eq']['f'])[0] for n in names]
    fat_po   = [plateau(industry_results[n]['po']['f'])[0] for n in names]
    err_red  = [(po - eq) / po * 100 for eq, po in zip(err_eq, err_po)]
    fat_red  = [(po - eq) / po * 100 for eq, po in zip(fat_eq, fat_po)]
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Figure C: Cross-Industry Generalization\n'
                 'Equitable DQN across 5 Manufacturing Sectors · 20 Seeds Each',
                 fontsize=12, fontweight='bold')
 
    # Throughput
    x = np.arange(len(names))
    bars = axes[0].bar(x, tputs, color=cols, alpha=0.83, edgecolor='white')
    axes[0].axhline(np.mean(tputs), color=GRAY, ls='--', lw=1.5,
                    label=f'Mean = {np.mean(tputs):.3f}')
    for bar, v in zip(bars, tputs):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.005,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=8.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8.5)
    axes[0].set_ylabel('Normalised Throughput')
    axes[0].set_title('Throughput by Sector')
    axes[0].legend(fontsize=8.5)
 
    # Error & Fatigue reduction
    w = 0.38
    bars1 = axes[1].bar(x - w/2, err_red, w, color=cols, alpha=0.83, edgecolor='white',
                        label='Error Reduction (%)')
    bars2 = axes[1].bar(x + w/2, fat_red, w, color=cols, alpha=0.45, edgecolor='white',
                        label='Fatigue Reduction (%)')
    axes[1].axhline(np.mean(err_red), color=RED,   ls='--', lw=1.2, alpha=0.7,
                    label=f'Pooled err {np.mean(err_red):.1f}%')
    axes[1].axhline(np.mean(fat_red), color=GREEN, ls='--', lw=1.2, alpha=0.7,
                    label=f'Pooled fat {np.mean(fat_red):.1f}%')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8.5)
    axes[1].set_ylabel('Percentage Reduction (%)')
    axes[1].set_title('Error & Fatigue Reduction vs Baseline')
    axes[1].legend(fontsize=7.5)
 
    # Radar
    axes[2].remove()
    ax_r = fig.add_subplot(1, 3, 3, projection='polar')
    cats   = ['Error\nReduc.', 'Fatigue\nReduc.', 'Throughput', 'Stability']
    N      = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
 
    def norm_v(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn + 1e-9) for v in vals]
 
    stabs = [1 / (plateau(industry_results[n]['eq']['r'])[1] + 1e-3) for n in names]
    all_m = [norm_v(err_red), norm_v(fat_red), norm_v(tputs), norm_v(stabs)]
 
    for i, (name, col) in enumerate(zip(names, cols)):
        vals  = [all_m[k][i] for k in range(N)]
        vals += vals[:1]
        ax_r.plot(angles, vals, color=col, lw=2, label=name.split(' ')[0])
        ax_r.fill(angles, vals, color=col, alpha=0.10)
 
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(cats, fontsize=9)
    ax_r.set_yticklabels([])
    ax_r.set_title('Sector Profile\n(normalised)', pad=18, fontsize=10)
    ax_r.legend(loc='upper right', bbox_to_anchor=(1.55, 1.15), fontsize=8.5)
 
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('FigC_Cross_Industry.png', bbox_inches='tight')
    plt.close()
    print("  → FigC_Cross_Industry.png")
 
    return tputs, err_red, fat_red
 
 
# =============================================================================
# PRINT FULL RESULTS TABLE
# =============================================================================
 
def print_all_results(dqn, ppo, sensitivity_results, skill_results,
                      industry_results, tputs, err_red, fat_red):
    sep = "=" * 80
    print(f"\n{sep}")
    print("FULL RESULTS TABLE — PASTE THIS BACK FOR MANUSCRIPT FINALISATION")
    print(sep)
 
    # --- Table 6: DQN vs PPO ---
    print("\n[TABLE 6: DQN vs PPO ARCHITECTURAL COMPARISON]")
    print("(Both use equitable weights w=[0.5,0.3,0.1,0.1], 20 seeds)")
    print(f"{'Metric':<25} {'DQN':>12} {'PPO':>12} {'Winner':>10}")
    print("-" * 62)
 
    metrics_AB = [
        ('Throughput',     'tputs',   False),
        ('Error Rate',     'errors',  True),
        ('Fatigue Index',  'fatigues',True),
        ('Cumul. Reward',  'rewards', False),
    ]
    for label, key, lb in metrics_AB:
        dm, ds = plateau(dqn[key])
        pm, ps = plateau(ppo[key])
        winner = 'DQN' if (dm < pm) == lb else 'PPO'
        pct    = (pm - dm) / abs(pm) * 100
        print(f"{label:<25} {dm:>8.4f}±{ds:.3f} {pm:>8.4f}±{ps:.3f}  "
              f"{winner} ({pct:+.1f}%)")
 
    # Action distributions
    print(f"\n  DQN Action Distribution (mean across seeds):")
    for k, name in enumerate(['Idle', 'Assist', 'TakeOver', 'SuggestBreak']):
        print(f"    {name:<15} {action_pct(dqn['action_counts'], k):.1f}%")
    print(f"  PPO Action Distribution (mean across seeds):")
    for k, name in enumerate(['Idle', 'Assist', 'TakeOver', 'SuggestBreak']):
        print(f"    {name:<15} {action_pct(ppo['action_counts'], k):.1f}%")
 
    # --- Sensitivity Analysis ---
    print(f"\n[SENSITIVITY ANALYSIS — 3 OPERATIONAL SCENARIOS]")
    base   = sensitivity_results[0]
    safety = sensitivity_results[1]
    prod   = sensitivity_results[2]
 
    base_tput = plateau(base['tputs'])[0]
    base_fat  = plateau(base['fatigues'])[0]
 
    sf_tput = plateau(safety['tputs'])[0]
    sf_fat  = plateau(safety['fatigues'])[0]
    pc_tput = plateau(prod['tputs'])[0]
    pc_fat  = plateau(prod['fatigues'])[0]
 
    sf_sb_pct  = action_pct(safety['action_counts'], 3)
    sf_to_pct  = action_pct(safety['action_counts'], 2)
    pc_idle_pct= action_pct(prod['action_counts'],   0)
    pc_sb_pct  = action_pct(prod['action_counts'],   3)
 
    print(f"\n  Scenario A — Safety-First (w=[0.3,0.3,0.5,0.1]):")
    print(f"    SuggestBreak frequency : {sf_sb_pct:.1f}%  (baseline: "
          f"{action_pct(base['action_counts'], 3):.1f}%)")
    print(f"    TakeOver frequency     : {sf_to_pct:.1f}%  (baseline: "
          f"{action_pct(base['action_counts'], 2):.1f}%)")
    print(f"    Throughput             : {sf_tput:.3f}  (baseline: {base_tput:.3f}, "
          f"Δ={((sf_tput-base_tput)/base_tput*100):+.1f}%)")
    print(f"    Fatigue Index          : {sf_fat:.4f}  (baseline: {base_fat:.4f}, "
          f"Δ={((sf_fat-base_fat)/base_fat*100):+.1f}%)")
 
    print(f"\n  Scenario B — Production-Critical (w=[0.8,0.15,0.05,0.0]):")
    print(f"    Idle frequency         : {pc_idle_pct:.1f}%  (baseline: "
          f"{action_pct(base['action_counts'], 0):.1f}%)")
    print(f"    SuggestBreak frequency : {pc_sb_pct:.1f}%  (baseline: "
          f"{action_pct(base['action_counts'], 3):.1f}%)")
    print(f"    Throughput             : {pc_tput:.3f}  (baseline: {base_tput:.3f}, "
          f"Δ={((pc_tput-base_tput)/base_tput*100):+.1f}%)")
    print(f"    Fatigue Index          : {pc_fat:.4f}  (baseline: {base_fat:.4f}, "
          f"Δ={((pc_fat-base_fat)/base_fat*100):+.1f}%)")
 
    sk_j = skill_results['junior']
    sk_s = skill_results['senior']
    j_assist_pct = action_pct(sk_j['action_counts'], 1)
    s_assist_pct = action_pct(sk_s['action_counts'], 1)
    j_sb_pct     = action_pct(sk_j['action_counts'], 3)
    s_sb_pct     = action_pct(sk_s['action_counts'], 3)
    assist_diff  = (j_assist_pct - s_assist_pct) / (s_assist_pct + 1e-9) * 100
    sb_diff      = (j_sb_pct - s_sb_pct) / (s_sb_pct + 1e-9) * 100
 
    print(f"\n  Scenario C — Skill-Level Generalisation:")
    print(f"    Trained on: Intermediate profile (weights frozen for evaluation)")
    print(f"    Assist % — Junior workers  : {j_assist_pct:.1f}%")
    print(f"    Assist % — Senior workers  : {s_assist_pct:.1f}%")
    print(f"    Junior receives {assist_diff:+.0f}% more Assist than Senior")
    print(f"    SuggestBreak % — Junior    : {j_sb_pct:.1f}%")
    print(f"    SuggestBreak % — Senior    : {s_sb_pct:.1f}%")
    print(f"    Junior receives {sb_diff:+.0f}% more SuggestBreak than Senior")
    print(f"    Fatigue — Junior           : {sk_j['fatigue']:.4f}")
    print(f"    Fatigue — Senior           : {sk_s['fatigue']:.4f}")
    print(f"    (Agent adapts actions via skill_level in state — weights frozen)")
 
    # --- Cross-industry ---
    names = list(industry_results.keys())
    print(f"\n[TABLE 7: CROSS-INDUSTRY GENERALIZATION]")
    print(f"{'Industry':<24} {'Throughput':>12} {'Err Reduc':>12} {'Fat Reduc':>12}")
    print("-" * 62)
    for name, tp, er, fr in zip(names, tputs, err_red, fat_red):
        print(f"{name:<24} {tp:>12.3f} {er:>11.1f}% {fr:>11.1f}%")
    print(f"\n  Pooled error reduction  : {np.mean(err_red):.1f}%  "
          f"(95% CI: {np.mean(err_red)-1.96*np.std(err_red)/np.sqrt(5):.1f}–"
          f"{np.mean(err_red)+1.96*np.std(err_red)/np.sqrt(5):.1f}%)")
    print(f"  Pooled fatigue reduction: {np.mean(fat_red):.1f}%  "
          f"(95% CI: {np.mean(fat_red)-1.96*np.std(fat_red)/np.sqrt(5):.1f}–"
          f"{np.mean(fat_red)+1.96*np.std(fat_red)/np.sqrt(5):.1f}%)")
 
    import os
    print(f"\n{sep}")
    print("FILES SAVED:")
    files = ['FigA_DQN_vs_PPO.png', 'FigB_Sensitivity_Analysis.png',
             'FigB2_Action_Radar.png', 'FigC_Cross_Industry.png']
    for f in files:
        kb = os.path.getsize(f)/1024 if os.path.exists(f) else 0
        print(f"  {f:<40} {kb:>6.0f} KB")
    print(sep)
 
 
# =============================================================================
# MAIN
# =============================================================================
 
if __name__ == '__main__':
    N_SEEDS    = 20
    N_EPISODES = 100
 
    print("="*60)
    print("HRC Full Reproduction — PPO / Sensitivity / Cross-Industry")
    print(f"Seeds: {N_SEEDS}  |  Episodes: {N_EPISODES}")
    print("="*60)
    t0 = time.time()
 
    # Part A
    dqn, ppo = run_part_A(N_SEEDS, N_EPISODES)
    plot_part_A(dqn, ppo)
    gc.collect()
 
    # Part B
    sensitivity_results, skill_results = run_part_B(N_SEEDS, N_EPISODES)
    plot_part_B(sensitivity_results, skill_results)
    gc.collect()
 
    # Part C
    industry_results = run_part_C(N_SEEDS, N_EPISODES)
    tputs, err_red, fat_red = plot_part_C(industry_results)
    gc.collect()
 
    # Full results table
    print_all_results(dqn, ppo, sensitivity_results, skill_results,
                      industry_results, tputs, err_red, fat_red)
 
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} minutes")
    print("Paste the FULL RESULTS TABLE back into the chat.")
 


Result

============================================================
HRC Full Reproduction — PPO / Sensitivity / Cross-Industry
Seeds: 20  |  Episodes: 100
============================================================

============================================================
PART A: DQN vs PPO Architectural Comparison
Both run with productivity-weighted config w=[0.5,0.3,0.1,0.1]
============================================================
  DQN  DQN (equitable weights)        w=[0.5,0.3,0.1,0.1]
  PPO  PPO (equitable weights)        w=[0.5,0.3,0.1,0.1]
  → FigA_DQN_vs_PPO.png

============================================================
PART B: Sensitivity Analysis — 3 Operational Scenarios
============================================================
  DQN  Baseline Equitable             w=[0.5,0.3,0.1,0.1]
  DQN  Safety-First                   w=[0.3,0.3,0.5,0.1]
  DQN  Production-Critical            w=[0.8,0.15,0.05,0.0]
  DQN  Skill-Generaliz. — TRAIN on Intermediate profile
  → Skill generalisation complete
  → FigB_Sensitivity_Analysis.png
  → FigB2_Action_Radar.png

============================================================
PART C: Cross-Industry Generalization (5 sectors)
============================================================

  Industry: Cement (Baseline)

  Industry: Electronics Assembly

  Industry: Food Processing

  Industry: Textile Mfg

  Industry: Automotive Parts
  → FigC_Cross_Industry.png

================================================================================
FULL RESULTS TABLE — PASTE THIS BACK FOR MANUSCRIPT FINALISATION
================================================================================

[TABLE 6: DQN vs PPO ARCHITECTURAL COMPARISON]
(Both use equitable weights w=[0.5,0.3,0.1,0.1], 20 seeds)
Metric                             DQN          PPO     Winner
--------------------------------------------------------------
Throughput                  0.7416±0.012   0.7372±0.009  DQN (-0.6%)
Error Rate                  0.0165±0.002   0.0209±0.010  DQN (+21.1%)
Fatigue Index               0.0599±0.012   0.0621±0.050  DQN (+3.6%)
Cumul. Reward             180.8969±2.995 178.9402±4.448  DQN (-1.1%)

  DQN Action Distribution (mean across seeds):
    Idle            6.2%
    Assist          23.6%
    TakeOver        39.0%
    SuggestBreak    31.3%
  PPO Action Distribution (mean across seeds):
    Idle            12.0%
    Assist          19.6%
    TakeOver        41.2%
    SuggestBreak    27.2%

[SENSITIVITY ANALYSIS — 3 OPERATIONAL SCENARIOS]

  Scenario A — Safety-First (w=[0.3,0.3,0.5,0.1]):
    SuggestBreak frequency : 41.8%  (baseline: 31.3%)
    TakeOver frequency     : 35.6%  (baseline: 39.0%)
    Throughput             : 0.745  (baseline: 0.742, Δ=+0.4%)
    Fatigue Index          : 0.0188  (baseline: 0.0599, Δ=-68.5%)

  Scenario B — Production-Critical (w=[0.8,0.15,0.05,0.0]):
    Idle frequency         : 8.6%  (baseline: 6.2%)
    SuggestBreak frequency : 26.7%  (baseline: 31.3%)
    Throughput             : 0.739  (baseline: 0.742, Δ=-0.4%)
    Fatigue Index          : 0.0827  (baseline: 0.0599, Δ=+38.1%)

  Scenario C — Skill-Level Generalisation:
    Trained on: Intermediate profile (weights frozen for evaluation)
    Assist % — Junior workers  : 21.4%
    Assist % — Senior workers  : 25.1%
    Junior receives -14% more Assist than Senior
    SuggestBreak % — Junior    : 41.5%
    SuggestBreak % — Senior    : 34.8%
    Junior receives +19% more SuggestBreak than Senior
    Fatigue — Junior           : 0.0849
    Fatigue — Senior           : 0.1424
    (Agent adapts actions via skill_level in state — weights frozen)

[TABLE 7: CROSS-INDUSTRY GENERALIZATION]
Industry                   Throughput    Err Reduc    Fat Reduc
--------------------------------------------------------------
Cement (Baseline)               0.742        11.0%        31.5%
Electronics Assembly            0.744         7.9%        31.9%
Food Processing                 0.738        16.1%        48.4%
Textile Mfg                     0.742         5.6%        21.9%
Automotive Parts                0.739        13.3%        42.5%

  Pooled error reduction  : 10.8%  (95% CI: 7.5–14.1%)
  Pooled fatigue reduction: 35.2%  (95% CI: 27.1–43.4%)

================================================================================
FILES SAVED:
  FigA_DQN_vs_PPO.png                         643 KB
  FigB_Sensitivity_Analysis.png               371 KB
  FigB2_Action_Radar.png                      390 KB
  FigC_Cross_Industry.png                     557 KB
================================================================================

Total runtime: 537.1 minutes
Paste the FULL RESULTS TABLE back into the chat.






Code for figure C cell 1

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE   = '#1B4F8A'
PURPLE = '#5B2C8A'
GREEN  = '#1A6B3A'
ORANGE = '#D45F00'
RED    = '#B22222'

names   = ['Cement\n(Baseline)', 'Electronics\nAssembly', 'Food\nProcessing', 'Textile\nMfg', 'Automotive\nParts']
tputs   = [0.742, 0.744, 0.738, 0.742, 0.739]
err_red = [11.0,  7.9,  16.1,  5.6,  13.3]
fat_red = [31.5, 31.9,  48.4, 21.9,  42.5]
cols    = [BLUE, PURPLE, GREEN, ORANGE, RED]
x       = np.arange(len(names))


Code for figure C cell 2

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Figure C: Cross-Industry Generalization\nEquitable DQN across 5 Manufacturing Sectors · 20 Seeds Each', fontsize=12, fontweight='bold')

bars = axes[0].bar(x, tputs, color=cols, alpha=0.83, edgecolor='white')
axes[0].axhline(np.mean(tputs), color='gray', ls='--', lw=1.5, label=f'Mean={np.mean(tputs):.3f}')
axes[0].set_ylim(0.70, 0.76)
for bar, v in zip(bars, tputs):
    axes[0].text(bar.get_x()+bar.get_width()/2, v+0.0005, f'{v:.3f}', ha='center', fontsize=9)
axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=8.5)
axes[0].set_ylabel('Normalised Throughput'); axes[0].set_title('Throughput by Sector'); axes[0].legend(fontsize=8.5)

w = 0.38
axes[1].bar(x-w/2, err_red, w, color=cols, alpha=0.83, edgecolor='white', label='Error Reduction (%)')
axes[1].bar(x+w/2, fat_red, w, color=cols, alpha=0.45, edgecolor='white', label='Fatigue Reduction (%)')
axes[1].axhline(np.mean(err_red), color=RED, ls='--', lw=1.2, alpha=0.7, label=f'Pooled err {np.mean(err_red):.1f}%')
axes[1].axhline(np.mean(fat_red), color=GREEN, ls='--', lw=1.2, alpha=0.7, label=f'Pooled fat {np.mean(fat_red):.1f}%')
axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=8.5)
axes[1].set_ylabel('Percentage Reduction (%)'); axes[1].set_title('Error & Fatigue Reduction vs Baseline'); axes[1].legend(fontsize=7.5)

axes[2].remove()
ax_r = fig.add_subplot(1, 3, 3, projection='polar')
cats = ['Error\nReduc.', 'Fatigue\nReduc.', 'Throughput']
angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist() + [0]
def norm(vals):
    mn, mx = min(vals), max(vals)
    return [(v-mn)/(mx-mn+1e-9) for v in vals]
all_m = [norm(err_red), norm(fat_red), norm(tputs)]
for i, (name, col) in enumerate(zip(names, cols)):
    vals = [all_m[k][i] for k in range(3)] + [all_m[0][i]]
    ax_r.plot(angles, vals, color=col, lw=2, label=name.split('\n')[0])
    ax_r.fill(angles, vals, color=col, alpha=0.10)
ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(cats, fontsize=9); ax_r.set_yticklabels([])
ax_r.set_title('Sector Profile\n(normalised)', pad=18, fontsize=10)
ax_r.legend(loc='upper right', bbox_to_anchor=(1.55, 1.15), fontsize=8.5)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('/kaggle/working/FigC_NEW.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: FigC_NEW.png")



Result

Saved: FigC_NEW.png




