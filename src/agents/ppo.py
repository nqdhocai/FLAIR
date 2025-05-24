import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Any

# Import unified training system
from ..trainer import train_agent, plot_training_stats

# ----------------------------------------
# 1) ActorCritic Network
# ----------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),      nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),       nn.ReLU()
        )
        self.actor  = nn.Linear(64, num_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# ----------------------------------------
# 2) PPOAgent
# ----------------------------------------
class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        entropy_coeff: float = 0.01,
    ):
        self.net = ActorCritic(state_dim, action_dim)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coeff = entropy_coeff
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # episode buffer
        self.states: List[np.ndarray]    = []
        self.actions: List[int]          = []
        self.log_probs: List[float]      = []
        self.rewards: List[float]        = []
        self.values: List[float]         = []
        self.dones: List[bool]           = []

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Return (action, log_prob, entropy)."""
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value = self.net(st)
        probs = torch.softmax(logits, dim=-1)
        dist  = torch.distributions.Categorical(probs)
        a      = dist.sample()
        return (
            a.item(),
            dist.log_prob(a).item(),
            dist.entropy().item(),
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        logp: float,
        reward: float,
        value: float,
        done: bool
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(logp)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute discounted returns and advantages (GAE not used)."""
        returns = []
        R = 0.0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + self.gamma * R * (1.0 - done)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        values  = torch.FloatTensor(self.values).to(self.device)
        advs    = returns - values
        advs    = (advs - advs.mean()) / (advs.std() + 1e-8)
        return returns, advs

    def update(self, ppo_epochs: int = 4, batch_size: int = 128) -> float:
        """Perform PPO update over collected episode."""
        # prepare data
        states    = torch.FloatTensor(self.states).to(self.device)
        actions   = torch.LongTensor(self.actions).to(self.device)
        old_logps = torch.FloatTensor(self.log_probs).to(self.device)
        returns, advantages = self.compute_returns_and_advantages()

        # dataloader
        dataset = torch.utils.data.TensorDataset(states, actions, old_logps, returns, advantages)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        for _ in range(ppo_epochs):
            for st, ac, old_lp, ret, adv in loader:
                logits, vals = self.net(st)
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(ac)
                entropy = dist.entropy().mean()

                ratio = (new_lp - old_lp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(vals, ret)
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item()

        # clear buffer
        num_updates = ppo_epochs * len(loader)
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

        return total_loss / max(1, num_updates)


# ----------------------------------------
# Convenience functions for backward compatibility
# ----------------------------------------
def train_ppo(
    env,
    agent: PPOAgent,
    epochs: int = 25,
    ppo_epochs: int = 4,
    batch_size: int = 128
) -> Tuple[List[float], List[float]]:
    """Legacy function - use train_agent from train.py instead."""
    return train_agent(
        env, agent, epochs, 
        ppo_epochs=ppo_epochs, 
        batch_size=batch_size
    )

def plot_stats(
    rewards: List[float],
    losses: List[float],
    action_counts: Optional[np.ndarray] = None,
    k_candidates: Optional[List[int]] = None
):
    """Legacy function - use plot_training_stats from train.py instead."""
    if action_counts is not None and k_candidates is not None:
        # Create extended plot with action distribution
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Reward curve
        ax = axes[0]
        ax.plot(rewards, color='tab:blue')
        ax.set_title("PPO - Rewards per Epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("Reward")
        
        # Loss curve
        ax = axes[1]
        ax.plot(losses, color='tab:red')
        ax.set_title("PPO - Loss per Epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        
        # Action distribution
        ax = axes[2]
        ax.bar(range(len(action_counts)), action_counts, color='tab:green')
        ax.set_xticks(range(len(k_candidates)))
        ax.set_xticklabels(k_candidates)
        ax.set_title("Action Distribution"); ax.set_xlabel("Action"); ax.set_ylabel("Count")
        
        plt.tight_layout()
        plt.show()
    else:
        plot_training_stats(rewards, losses, "PPO")


# ----------------------------------------
# Example Usage
# ----------------------------------------
"""
# Modern usage (recommended):
from ..train import train_agent, plot_training_stats

rewards, losses = train_agent(
    env, ppo_agent, 
    epochs=30, 
    ppo_epochs=4, 
    batch_size=128
)
plot_training_stats(rewards, losses, "PPO")

# Legacy usage (still works):
rewards, losses = train_ppo(env, ppo_agent, epochs=30, ppo_epochs=4, batch_size=128)
plot_stats(rewards, losses)
"""