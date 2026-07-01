import numpy as np
import random

class SARSAAgent:
    def __init__(self, n_actions=6, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.997):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def discretize(self, state):
        x, y, on_floor, apple_dx, apple_dy, snail_dist = state
        # Bin sizes — tune these based on your level's scale
        
        return (
            int(x // 50),
            int(y // 50),
            int(on_floor),
            int(apple_dx // 50),
            int(apple_dy // 50),
            int(snail_dist // 50),
        )
    
    def get_q(self, state_key, action):
        return self.q_table.get((state_key, action), 0.0)

    def select_action(self, state):
        state_key = self.discretize(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_values = [self.get_q(state_key, a) for a in range(self.n_actions)]
        return int(np.argmax(q_values))
    
    def update(self, state, action, reward, next_state, next_action, done):
        s_key = self.discretize(state)
        s_next_key = self.discretize(next_state)

        current_q = self.get_q(s_key, action)
        next_q = 0.0 if done else self.get_q(s_next_key, next_action)

        # SARSA update — uses the ACTUAL next action, not max
        target = reward + self.gamma * next_q
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[(s_key, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)