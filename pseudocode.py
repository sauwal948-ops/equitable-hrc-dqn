Algorithm S1: Training Pseudocode
# Initialize
Q_network = DQN(state_dim=5, action_dim=4, hidden=[128, 128])
Target_network = copy.deepcopy(Q_network)
Replay_buffer = ReplayBuffer(capacity=10000)
epsilon = 1.0

# Training loop
for episode in range(100):
    state = env.reset()
    episode_reward = 0
    
    for step in range(500):  # Max 500 steps per episode
        # Action selection (epsilon-greedy)
        if random() < epsilon:
            action = random_action()
        else:
            Q_values = Q_network(state)
            action = argmax(Q_values)
        
        # Environment interaction
        next_state, reward, done = env.step(action)
        Replay_buffer.store(state, action, reward, next_state, done)
        episode_reward += reward
        
        # Learning step
        if len(Replay_buffer) > 64:
            batch = Replay_buffer.sample(batch_size=64)
            loss = compute_loss(Q_network, Target_network, batch)
            Q_network.backward(loss)
            Q_network.update(learning_rate=0.001)
        
        # Target network update
        if step % 100 == 0:
            Target_network = copy.deepcopy(Q_network)
        
        state = next_state
        if done: break
    
    # Epsilon decay
    epsilon = max(0.01, epsilon - 0.0165)  # Linear decay to 0.01 by episode 60
    
    print(f"Episode {episode}: Reward = {episode_reward}")
