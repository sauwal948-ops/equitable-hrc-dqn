"""
Deep Q-Network (DQN) Simulation for HRC Task Allocation
Context: Cement bagging line in a resource-constrained manufacturing setting
Demonstrates equitable task allocation between human workers and collaborative robots
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# ============================================================================
# ENVIRONMENT: Cement Bagging Line HRC System
# ============================================================================

class CementBaggingHRCEnvironment:
    """
    Simulates a cement bagging line with human-robot collaboration.
    
    State: [machine_speed, human_fatigue, error_rate, queue_length, worker_id]
    Actions: 0=Idle, 1=Assist, 2=TakeOver, 3=SuggestBreak
    """
    
    def __init__(self, num_workers=3):
        self.num_workers = num_workers
        self.current_worker = 0
        self.episode_step = 0
        self.max_steps = 500
        
        # Worker states: [fatigue, skill_level, experience]
        self.workers = np.array([
            [0.3, 0.5, 1],  # Worker 1: Low experience
            [0.2, 0.7, 5],  # Worker 2: Medium experience
            [0.1, 0.9, 10]  # Worker 3: High experience
        ])
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.machine_speed = np.random.uniform(0.7, 1.0)
        self.human_fatigue = np.random.uniform(0.2, 0.5)
        self.error_rate = np.random.uniform(0.05, 0.15)
        self.queue_length = np.random.uniform(0.3, 0.8)
        self.episode_step = 0
        self.current_worker = np.random.randint(0, self.num_workers)
        
        return self.get_state()
    
    def get_state(self):
        """Return current state vector"""
        worker = self.workers[self.current_worker]
        return np.array([
            self.machine_speed,
            self.human_fatigue,
            self.error_rate,
            self.queue_length,
            worker[1]  # skill_level
        ], dtype=np.float32)
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done)
        
        Actions:
        0 = Idle (no intervention)
        1 = Assist (robot helps human)
        2 = TakeOver (robot takes full task)
        3 = SuggestBreak (human takes break)
        """
        self.episode_step += 1
        
        # Update machine dynamics
        self.machine_speed = np.clip(self.machine_speed + np.random.normal(0, 0.05), 0.5, 1.0)
        
        # Update human fatigue based on action
        if action == 0:  # Idle
            self.human_fatigue = np.clip(self.human_fatigue + 0.08, 0, 1)
        elif action == 1:  # Assist
            self.human_fatigue = np.clip(self.human_fatigue + 0.03, 0, 1)
        elif action == 2:  # TakeOver
            self.human_fatigue = np.clip(self.human_fatigue - 0.05, 0, 1)
        elif action == 3:  # SuggestBreak
            self.human_fatigue = np.clip(self.human_fatigue - 0.15, 0, 1)
        
        # Update error rate based on fatigue and action
        fatigue_effect = self.human_fatigue * 0.1
        if action == 1:  # Assist reduces errors
            self.error_rate = np.clip(self.error_rate - 0.02 + fatigue_effect, 0.01, 0.3)
        elif action == 2:  # TakeOver minimizes errors
            self.error_rate = np.clip(self.error_rate - 0.05, 0.01, 0.2)
        else:
            self.error_rate = np.clip(self.error_rate + fatigue_effect, 0.01, 0.3)
        
        # Update queue length
        processing_rate = self.machine_speed * (1 - self.error_rate)
        if action == 2:  # TakeOver increases processing
            processing_rate *= 1.2
        self.queue_length = np.clip(self.queue_length - processing_rate * 0.1 + np.random.uniform(0, 0.05), 0, 1)
        
        # Rotate worker
        self.current_worker = (self.current_worker + 1) % self.num_workers
        
        # Calculate reward (composite: productivity + equity + worker well-being)
        reward = self._calculate_reward(action)
        
        done = self.episode_step >= self.max_steps
        
        return self.get_state(), reward, done
    
    def _calculate_reward(self, action):
        """
        Composite reward function balancing:
        - α (0.5): Throughput maximization
        - β (0.3): Error minimization
        - γ (0.1): Fatigue reduction
        - δ (0.1): Equitable task distribution (no bias)
        """
        # Throughput reward
        throughput = self.machine_speed * (1 - self.error_rate)
        r_throughput = throughput
        
        # Error penalty
        r_error = -self.error_rate
        
        # Fatigue penalty
        r_fatigue = -self.human_fatigue * 0.5
        
        # Bias penalty (penalize overuse of TakeOver on low-experience workers)
        worker = self.workers[self.current_worker]
        skill_level = worker[1]
        if action == 2 and skill_level < 0.6:  # TakeOver on low-skill worker
            r_bias = -0.1
        else:
            r_bias = 0
        
        # Composite reward
        reward = (0.5 * r_throughput + 0.3 * r_error + 0.1 * r_fatigue + 0.1 * r_bias)
        
        return float(reward)


# ============================================================================
# DEEP Q-NETWORK AGENT
# ============================================================================

class DQNAgent:
    """Simple DQN agent for HRC task allocation"""
    
    def __init__(self, state_size=5, action_size=4, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Simple Q-table (for lightweight simulation)
        self.q_table = {}
        
        # Experience replay
        self.memory = deque(maxlen=1000)
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete for Q-table"""
        # Discretize each dimension into 5 bins
        bins = 5
        discrete_state = tuple(np.digitize(state[i], np.linspace(0, 1, bins)) 
                               for i in range(len(state)))
        return discrete_state
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        discrete_state = self._discretize_state(state)
        
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_size)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[discrete_state])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Experience replay for Q-learning"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            discrete_state = self._discretize_state(state)
            discrete_next_state = self._discretize_state(next_state)
            
            if discrete_state not in self.q_table:
                self.q_table[discrete_state] = np.zeros(self.action_size)
            if discrete_next_state not in self.q_table:
                self.q_table[discrete_next_state] = np.zeros(self.action_size)
            
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.q_table[discrete_next_state])
            
            self.q_table[discrete_state][action] = (
                self.q_table[discrete_state][action] +
                self.learning_rate * (target - self.q_table[discrete_state][action])
            )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_hrc_agent(episodes=100):
    """Train DQN agent for HRC task allocation"""
    
    env = CementBaggingHRCEnvironment(num_workers=3)
    agent = DQNAgent(state_size=5, action_size=4)
    
    # Tracking metrics
    episode_rewards = []
    episode_errors = []
    episode_fatigue = []
    episode_throughput = []
    
    action_names = {0: "Idle", 1: "Assist", 2: "TakeOver", 3: "SuggestBreak"}
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            action = agent.act(state)
            action_counts[action] += 1
            
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.replay(batch_size=32)
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_errors.append(env.error_rate)
        episode_fatigue.append(env.human_fatigue)
        episode_throughput.append(env.machine_speed * (1 - env.error_rate))
        
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{episodes} | Reward: {episode_reward:.3f} | "
                  f"Error Rate: {env.error_rate:.3f} | Fatigue: {env.human_fatigue:.3f}")
    
    return {
        'agent': agent,
        'env': env,
        'rewards': episode_rewards,
        'errors': episode_errors,
        'fatigue': episode_fatigue,
        'throughput': episode_throughput,
        'action_counts': action_counts
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HRC TASK ALLOCATION - LIGHTWEIGHT DQN SIMULATION")
    print("=" * 70)
    print("\nTraining DQN agent for equitable task allocation...")
    print("Environment: Cement bagging line with 3 workers (varying experience levels)")
    print("Objective: Balance productivity, error reduction, fatigue management, and equity\n")
    
    results = train_hrc_agent(episodes=100)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("=" * 70)
    
    # Summary statistics
    avg_reward = np.mean(results['rewards'][-20:])
    avg_error = np.mean(results['errors'][-20:])
    avg_fatigue = np.mean(results['fatigue'][-20:])
    avg_throughput = np.mean(results['throughput'][-20:])
    
    print(f"\nFinal Episode Metrics (last 20 episodes average):")
    print(f"  Average Reward:     {avg_reward:.4f}")
    print(f"  Average Error Rate: {avg_error:.4f} ({avg_error*100:.2f}%)")
    print(f"  Average Fatigue:    {avg_fatigue:.4f}")
    print(f"  Average Throughput: {avg_throughput:.4f}")
    
    print(f"\nAction Distribution (total {sum(results['action_counts'].values())} actions):")
    for action, count in results['action_counts'].items():
        pct = (count / sum(results['action_counts'].values())) * 100
        print(f"  {['Idle', 'Assist', 'TakeOver', 'SuggestBreak'][action]}: {count} ({pct:.1f}%)")
    
    # Save results for visualization
    np.save('/home/ubuntu/hrc_simulation_results.npy', results, allow_pickle=True)
    print("\n✓ Simulation results saved to /home/ubuntu/hrc_simulation_results.npy")

