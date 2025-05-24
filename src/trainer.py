import torch
from typing import List, Tuple
from tqdm import trange
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# ----------------------------------------
# Base Training Interface
# ----------------------------------------
class BaseTrainer(ABC):
    """Base trainer class that defines the common training interface."""
    
    def __init__(
        self,
        env,
        agent,
        epochs: int = 25,
        early_stopping_patience: int = 8,
        **kwargs
    ):
        self.env = env
        self.agent = agent
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.kwargs = kwargs
        
    @abstractmethod
    def run_episode(self) -> Tuple[float, int]:
        """Run one episode and return (total_reward, total_steps)."""
        pass
    
    @abstractmethod
    def update_agent(self, steps: int) -> float:
        """Update the agent and return average loss."""
        pass
    
    def train(self) -> Tuple[List[float], List[float]]:
        """Main training loop."""
        rewards: List[float] = []
        losses: List[float] = []
        
        pbar = trange(self.epochs, desc=f"Training {self.agent.__class__.__name__}", unit="epoch")
        best_reward = -float("inf")
        no_improve = 0
        
        for epoch in pbar:
            total_reward, total_steps = self.run_episode()
            avg_reward = total_reward / max(1, total_steps)
            avg_loss = self.update_agent(total_steps)
            
            rewards.append(avg_reward)
            losses.append(avg_loss)
            
            pbar.set_postfix({
                "reward": f"{avg_reward:.3f}",
                "loss": f"{avg_loss:.4f}"
            })
            
            # Early stopping logic
            if avg_reward > best_reward:
                best_reward = avg_reward
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= self.early_stopping_patience:
                print(f"Early stopping: no reward improvement in {self.early_stopping_patience} epochs.")
                break
                
        return rewards, losses


# ----------------------------------------
# DQN Trainer
# ----------------------------------------
class DQNTrainer(BaseTrainer):
    """Trainer for DQN and Double DQN agents."""
    
    def run_episode(self) -> Tuple[float, int]:
        """Run episode for DQN-based agents."""
        total_reward = 0.0
        total_steps = 0
        
        for _ in range(len(self.env.dataset)):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Store transition (DQN stores s,a,r,s',done or s,a,r,s' depending on agent)
                if hasattr(self.agent, 'store'):
                    if 'Double' in self.agent.__class__.__name__:
                        self.agent.store((state, action, reward, next_state, done))
                    else:
                        self.agent.store((state, action, reward, next_state))
                
                total_reward += reward
                total_steps += 1
                state = next_state
                
        return total_reward, total_steps
    
    def update_agent(self, steps: int) -> float:
        """Update DQN agent."""
        total_loss = 0.0
        for _ in range(steps):
            total_loss += self.agent.update()
        return total_loss / max(1, steps)


# ----------------------------------------
# PPO Trainer
# ----------------------------------------
class PPOTrainer(BaseTrainer):
    """Trainer for PPO agents."""
    
    def __init__(self, *args, ppo_epochs: int = 4, batch_size: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
    
    def run_episode(self) -> Tuple[float, int]:
        """Run episode for PPO agent."""
        total_reward = 0.0
        total_steps = 0
        
        for _ in range(len(self.env.dataset)):
            state = self.env.reset()
            total_steps += 1
            done = False
            
            while not done:
                # Get value for storing transition
                with torch.no_grad():
                    _, value = self.agent.net(
                        torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                    )
                
                action, log_prob, _ = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.agent.store_transition(
                    state, action, log_prob, reward, value.item(), done
                )
                
                total_reward += reward
                state = next_state
                
        return total_reward, total_steps
    
    def update_agent(self, steps: int) -> float:
        """Update PPO agent."""
        return self.agent.update(
            ppo_epochs=self.ppo_epochs,
            batch_size=self.batch_size
        )


# ----------------------------------------
# Training Factory
# ----------------------------------------
def _create_trainer(
    env,
    agent,
    epochs: int = 25,
    early_stopping_patience: int = 8,
    **kwargs
) -> BaseTrainer:
    """Factory function to create appropriate trainer for agent type."""
    
    agent_class_name = agent.__class__.__name__
    
    if 'PPO' in agent_class_name:
        return PPOTrainer(
            env=env,
            agent=agent,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            **kwargs
        )
    elif 'DQN' in agent_class_name:
        return DQNTrainer(
            env=env,
            agent=agent,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_class_name}")


# ----------------------------------------
# Unified Training Function
# ----------------------------------------
def train_agent(
    env,
    agent,
    epochs: int = 25,
    early_stopping_patience: int = 8,
    **kwargs
) -> Tuple[List[float], List[float]]:
    """
    Unified training function that works with all agent types.
    
    Args:
        env: Environment to train on
        agent: Agent to train (DQN, DoubleDQN, or PPO)
        epochs: Number of training epochs
        early_stopping_patience: Number of epochs without improvement before stopping
        **kwargs: Additional parameters (e.g., ppo_epochs, batch_size for PPO)
    
    Returns:
        Tuple of (rewards, losses) lists
    """
    trainer = _create_trainer(
        env=env,
        agent=agent,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        **kwargs
    )
    
    return trainer.train()


# ----------------------------------------
# Unified Plotting Function
# ----------------------------------------
def plot_training_stats(
    rewards: List[float],
    losses: List[float],
    agent_name: str = "Agent",
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot training statistics for any agent type.
    
    Args:
        rewards: List of rewards per epoch
        losses: List of losses per epoch
        agent_name: Name of the agent for plot title
        figsize: Figure size tuple
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Reward plot
    ax1.plot(rewards, color='tab:blue', linewidth=2)
    ax1.set_title(f"{agent_name} - Rewards per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Reward")
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(losses, color='tab:red', linewidth=2)
    ax2.set_title(f"{agent_name} - Loss per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Loss")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ----------------------------------------
# Example Usage
# ----------------------------------------
"""
# Example usage for any agent type:

# For DQN:
rewards, losses = train_agent(env, dqn_agent, epochs=50)
plot_training_stats(rewards, losses, "DQN")

# For Double DQN:
rewards, losses = train_agent(env, ddqn_agent, epochs=50)
plot_training_stats(rewards, losses, "Double DQN")

# For PPO:
rewards, losses = train_agent(
    env, ppo_agent, 
    epochs=30, 
    ppo_epochs=4, 
    batch_size=128
)
plot_training_stats(rewards, losses, "PPO")
"""