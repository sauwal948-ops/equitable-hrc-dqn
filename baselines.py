# baselines.py
import numpy as np

class RuleBasedPolicy:
    """
    Simple threshold-based policy using domain knowledge
    
    Rules:
    1. If worker is very fatigued OR hasn't had break in 2+ hours → SuggestBreak
    2. Else if error rate is high → Assist  
    3. Else if task queue is long → TakeOver
    4. Else → Idle (let worker work independently)
    """
    
    def __init__(self, config=None):
        """
        Initialize with thresholds (you can tune these)
        """
        self.fatigue_threshold = 0.20      # Suggest break if fatigue > 0.20
        self.error_threshold = 0.10        # Assist if error rate > 10%
        self.queue_threshold = 12          # Take over if queue > 12 tasks
        self.break_interval = 120          # Suggest break after 120 minutes (2 hours)
        
        # For tracking (optional - helps with analysis)
        self.name = "RuleBased"
    
    def select_action(self, state):
        """
        Select action based on rules
        
        Args:
            state: dict or array with keys/indices:
                - task_queue_length
                - worker_fatigue  
                - error_rate
                - time_since_break
                - robot_availability (optional)
        
        Returns:
            action: int in [0, 1, 2, 3, 4]
                0 = Idle
                1 = Assist
                2 = TakeOver
                3 = SuggestBreak
                4 = Training (we'll keep this as 0 for simplicity)
        """
        
        # Extract state variables (adjust based on YOUR state representation)
        if isinstance(state, dict):
            fatigue = state['worker_fatigue']
            error_rate = state['error_rate']
            queue_length = state['task_queue_length']
            time_since_break = state['time_since_break']
        else:
            # If state is a numpy array [queue, fatigue, error, time, robot_avail]
            queue_length = state[0]
            fatigue = state[1]
            error_rate = state[2]
            time_since_break = state[3]
        
        # Rule 1: Safety/welfare first - suggest break if needed
        if fatigue > self.fatigue_threshold:
            return 3  # SuggestBreak
        
        if time_since_break > self.break_interval:
            return 3  # SuggestBreak
        
        # Rule 2: Quality control - assist if errors are high
        if error_rate > self.error_threshold:
            return 1  # Assist
        
        # Rule 3: Productivity - take over if backlog building up
        if queue_length > self.queue_threshold:
            return 2  # TakeOver
        
        # Rule 4: Default - let worker work independently
        return 0  # Idle
    
    def train(self, env, n_episodes):
        """
        Rule-based doesn't need training, but we provide this method
        for compatibility with your existing code
        """
        print(f"{self.name}: No training needed (rule-based policy)")
        pass
    
    def save(self, path):
        """For compatibility"""
        pass
    
    def load(self, path):
        """For compatibility"""
        pass


class GreedyProductivityPolicy:
    """
    Baseline that maximizes throughput with minimal safety constraints
    
    This represents what a factory might do without equity concerns:
    - Only care about getting bags out the door
    - Minimum safety (prevent injury)
    - No fairness considerations
    """
    
    def __init__(self, config=None):
        # Hard safety limits (absolute minimum)
        self.max_fatigue = 0.30        # Don't let fatigue exceed 30% (unsafe)
        self.max_time_without_break = 180  # Force break after 3 hours
        
        # Productivity thresholds
        self.queue_takeover_threshold = 8   # Take over quickly if backlog
        self.error_assist_threshold = 0.12  # Only assist if errors very bad
        
        self.name = "GreedyProductivity"
    
    def select_action(self, state):
        """Greedy policy focused on throughput"""
        
        # Extract state
        if isinstance(state, dict):
            fatigue = state['worker_fatigue']
            error_rate = state['error_rate']
            queue_length = state['task_queue_length']
            time_since_break = state['time_since_break']
        else:
            queue_length = state[0]
            fatigue = state[1]
            error_rate = state[2]
            time_since_break = state[3]
        
        # Safety constraints (minimum required)
        if fatigue > self.max_fatigue or time_since_break > self.max_time_without_break:
            return 3  # SuggestBreak (forced for safety)
        
        # Otherwise, maximize throughput
        
        # Priority 1: Clear backlog (throughput)
        if queue_length > self.queue_takeover_threshold:
            return 2  # TakeOver - robot is faster
        
        # Priority 2: Prevent defects if really bad
        if error_rate > self.error_assist_threshold:
            return 1  # Assist
        
        # Priority 3: Default to idle (worker works alone - cheapest)
        return 0  # Idle


class ProductivityOnlyDQN:
    """
    Your DQN architecture but with reward function that ignores equity and fatigue
    
    This is NOT a new implementation - it's your existing DQN with modified reward
    """
    
    def __init__(self, dqn_class, state_dim, action_dim, config=None):
        """
        Args:
            dqn_class: Your existing DQN class (we'll reuse it)
            state_dim: State space dimension
            action_dim: Action space dimension  
            config: Hyperparameters
        """
        # Use your existing DQN implementation
        self.agent = dqn_class(state_dim, action_dim, config)
        self.name = "ProductivityDQN"
        
        # Override reward function to remove equity and fatigue terms
        self.use_equity_reward = False
        self.use_fatigue_reward = False
    
    def compute_reward(self, state, action, next_state, info):
        """
        Productivity-only reward (NO equity, NO fatigue)
        
        Compare to your original reward:
        Original: r = w1*throughput + w2*error + w3*fatigue + w4*equity
        This:     r = w1*throughput + w2*error
        """
        
        # Throughput component (positive reward for bags produced)
        throughput = info.get('bags_produced', 0)
        throughput_reward = throughput * 1.0
        
        # Error penalty (negative reward for defects)
        error_rate = next_state.get('error_rate', 0) if isinstance(next_state, dict) else next_state[2]
        error_penalty = -error_rate * 10.0
        
        # NO fatigue term
        # NO equity term
        
        total_reward = throughput_reward + error_penalty
        
        return total_reward
    
    def select_action(self, state, epsilon=0.0):
        """Use DQN's action selection"""
        return self.agent.select_action(state, epsilon)
    
    def train(self, env, n_episodes):
        """Train using productivity-only reward"""
        print(f"{self.name}: Training with productivity-only reward...")
        
        # You'll need to modify your training loop to use self.compute_reward
        # instead of the equity-aware reward
        # This is the trickiest part - see detailed explanation below
        
        pass  # Implementation depends on your training code structure
    
    def save(self, path):
        self.agent.save(path)
    
    def load(self, path):
        self.agent.load(path)


# ============================================
# Optional: Constraint Programming Baseline
# (More advanced - skip if you're short on time)
# ============================================

class ConstraintProgrammingScheduler:
    """
    Uses Google OR-Tools to solve task allocation as optimization problem
    
    Formulation:
    - Minimize: makespan (total time to complete tasks)
    - Subject to: fatigue < 0.25, error_rate < 0.12
    
    This requires installing: pip install ortools
    """
    
    def __init__(self, config=None):
        try:
            from ortools.sat.python import cp_model
            self.cp_model = cp_model
            self.name = "CPScheduler"
        except ImportError:
            raise ImportError("Install ortools: pip install ortools")
    
    def select_action(self, state):
        """
        Solve constrained optimization problem at each time step
        
        Note: This is a simplified version. Full CP approach would plan
        over a horizon (next 50 steps), but that's more complex.
        
        For simplicity, we use CP-inspired heuristics:
        - Respect constraints strictly
        - Optimize local throughput
        """
        
        # Extract state
        if isinstance(state, dict):
            fatigue = state['worker_fatigue']
            error_rate = state['error_rate']
            queue_length = state['task_queue_length']
            time_since_break = state['time_since_break']
        else:
            queue_length = state[0]
            fatigue = state[1]
            error_rate = state[2]
            time_since_break = state[3]
        
        # Hard constraints (CP philosophy: constraints are STRICT)
        if fatigue >= 0.25:  # Fatigue constraint
            return 3  # SuggestBreak (REQUIRED)
        
        if error_rate >= 0.12:  # Error constraint  
            return 1  # Assist (REQUIRED)
        
        # If constraints satisfied, optimize for makespan (minimize completion time)
        # Greedy choice: TakeOver maximizes immediate throughput
        if queue_length > 10:
            return 2  # TakeOver
        
        return 0  # Idle
    
    def train(self, env, n_episodes):
        """CP doesn't need training"""
        print(f"{self.name}: No training needed (optimization-based)")
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
