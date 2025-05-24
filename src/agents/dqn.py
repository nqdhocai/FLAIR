import random
import numpy as np
from collections import deque
from typing import Tuple, List, Any

import torch
import torch.nn as nn
import torch.optim as optim

# Import unified training system
from ..trainer import train_agent, plot_training_stats

# ----------------------------------------
# 1) Q-Network
# ----------------------------------------
class QNet(nn.Module):
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),      nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),       nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ----------------------------------------
# 2) Replay Buffer
# ----------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, transition: Tuple):
        self.buf.append(transition)

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)

# ----------------------------------------
# 3) DQN Agent
# ----------------------------------------
class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        buffer_size: int = 10000,
        batch_size: int = 64,
        epsilon: float = 0.1,
        target_update_freq: int = 1000
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNet(state_dim, num_actions).to(self.device)
        self.target_net = QNet(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.buffer = ReplayBuffer(buffer_size)
        self.learn_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.q_net.net[-1].out_features)
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(st)
        return int(q_vals.argmax(dim=1).item())

    def store(self, transition: Tuple[Any, ...]):
        # transition = (state, action, reward, next_state)
        self.buffer.push(transition)

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)        
        states, actions, rewards, next_states = zip(*batch)

        # Convert to tensors efficiently
        st = torch.from_numpy(np.stack(states)).float().to(self.device)
        ns = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        a  = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

        # current Q
        q_pred = self.q_net(st).gather(1, a)

        # target Q
        next_q = self.target_net(ns).max(dim=1, keepdim=True)[0].detach()
        q_target = r + self.gamma * next_q

        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target net
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

# ----------------------------------------
# Convenience functions for backward compatibility
# ----------------------------------------
def train_dqn(env, agent: DQNAgent, epochs: int = 25) -> Tuple[List[float], List[float]]:
    """Legacy function - use train_agent from train.py instead."""
    return train_agent(env, agent, epochs)

def plot_dqn_stats(rewards: List[float], losses: List[float]):
    """Legacy function - use plot_training_stats from train.py instead."""
    plot_training_stats(rewards, losses, "DQN")

# ----------------------------------------
# Example Usage
# ----------------------------------------
"""
# Modern usage (recommended):
from ..train import train_agent, plot_training_stats

rewards, losses = train_agent(env, dqn_agent, epochs=50)
plot_training_stats(rewards, losses, "DQN")

# Legacy usage (still works):
rewards, losses = train_dqn(env, dqn_agent, epochs=50)
plot_dqn_stats(rewards, losses)
"""