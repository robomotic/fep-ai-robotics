"""
Deep Q-Network (DQN) Agent for CartPole.

This module implements the DQN algorithm with experience replay and target network,
representing a significant advancement from tabular methods to deep reinforcement learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import random
from typing import List, Tuple, Optional


class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture.
    
    A simple fully connected neural network that maps states to Q-values.
    """
    
    def __init__(self, state_size: int = 4, action_size: int = 2, hidden_sizes: List[int] = [128, 128]):
        """
        Initialize the network.
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            hidden_sizes: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, terminated: bool) -> None:
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, terminated))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated = zip(*batch)
        
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.BoolTensor(terminated)
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    
    This represents the foundational deep RL algorithm that kicked off
    the deep reinforcement learning revolution.
    """
    
    def __init__(
        self,
        env: gym.Env,
        hidden_sizes: List[int] = [128, 128],
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            env: CartPole environment
            hidden_sizes: Hidden layer sizes for the network
            learning_rate: Learning rate for the optimizer
            discount_factor: Discount factor for future rewards
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Minimum epsilon value
            epsilon_decay: Decay rate for epsilon
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to run on ('cpu' or 'cuda')
        """
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_network = DQNNetwork(self.state_size, self.action_size, hidden_sizes).to(self.device)
        self.target_network = DQNNetwork(self.state_size, self.action_size, hidden_sizes).to(self.device)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        self.losses = []
    
    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        return max(self.epsilon_end, self.epsilon_start * (self.epsilon_decay ** self.episode_count))
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.get_epsilon():
            # Explore: random action
            return self.env.action_space.sample()
        else:
            # Exploit: best action according to Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update_target_network(self) -> None:
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def optimize_model(self) -> float:
        """
        Perform one step of optimization on the Q-network.
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, terminated = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminated = terminated.to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~terminated)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(
        self, 
        episodes: int = 1000, 
        max_steps: int = 500,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the DQN agent.
        
        Args:
            episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
            
        Returns:
            List of episode rewards
        """
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            episode_loss = 0
            
            for step in range(max_steps):
                # Select action
                action = self.select_action(state, training=True)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Store experience
                self.replay_buffer.push(state, action, reward, next_state, terminated)
                
                # Optimize model
                loss = self.optimize_model()
                episode_loss += loss
                
                state = next_state
                total_reward += reward
                self.total_steps += 1
                
                # Update target network
                if self.total_steps % self.target_update_freq == 0:
                    self.update_target_network()
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            self.episode_count += 1
            self.losses.append(episode_loss / (step + 1))
            
            # Print progress
            if verbose and (episode % 100 == 0 or episode == episodes - 1):
                avg_reward = np.mean(episode_rewards[-100:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                epsilon = self.get_epsilon()
                print(f"Episode {episode:4d}, Avg Reward: {avg_reward:6.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}, Steps: {step + 1}")
        
        return episode_rewards
    
    def test(self, episodes: int = 10, max_steps: int = 500, render: bool = False) -> List[float]:
        """
        Test the trained agent.
        
        Args:
            episodes: Number of test episodes
            max_steps: Maximum steps per episode
            render: Whether to render the environment
            
        Returns:
            List of test episode rewards
        """
        test_rewards = []
        
        # Set network to evaluation mode
        self.q_network.eval()
        
        for episode in range(episodes):
            if render:
                env = gym.make('CartPole-v1', render_mode='human')
            else:
                env = self.env
                
            state, _ = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                state = next_state
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            test_rewards.append(total_reward)
            print(f"Test Episode {episode + 1}: {total_reward} steps")
            
            if render:
                env.close()
        
        # Set network back to training mode
        self.q_network.train()
        
        avg_test_reward = np.mean(test_rewards)
        print(f"Average test reward: {avg_test_reward:.2f}")
        
        return test_rewards
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps
        }, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']
        print(f"Model loaded from: {filepath}")