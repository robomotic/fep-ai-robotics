"""
Q-Learning Agent for CartPole.

This module implements a classic Q-learning agent using tabular Q-values
with state discretization for the continuous CartPole environment.
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Optional
from .utils import discretize_state, epsilon_decay_schedule, learning_rate_schedule


class QLearningAgent:
    """
    Q-Learning agent with epsilon-greedy exploration and state discretization.
    
    This is the classic tabular Q-learning algorithm applied to CartPole.
    Since CartPole has continuous states, we discretize them into bins.
    """
    
    def __init__(
        self,
        env: gym.Env,
        bins: Tuple[int, int, int, int] = (10, 10, 10, 10),
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            env: CartPole environment
            bins: Number of bins for state discretization (cart_pos, cart_vel, pole_angle, pole_vel)
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Minimum epsilon value
            epsilon_decay: Decay rate for epsilon
        """
        self.env = env
        self.bins = bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table
        self.q_table = np.zeros(bins + (env.action_space.n,))
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        
    def get_state_index(self, state: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert continuous state to discrete state indices."""
        return discretize_state(state, self.bins)
    
    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        return epsilon_decay_schedule(
            self.episode_count, 
            self.epsilon_start, 
            self.epsilon_end, 
            self.epsilon_decay
        )
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action (0 or 1)
        """
        state_idx = self.get_state_index(state)
        epsilon = self.get_epsilon()
        
        if np.random.random() < epsilon:
            # Explore: random action
            return self.env.action_space.sample()
        else:
            # Exploit: best action according to Q-table
            return np.argmax(self.q_table[state_idx])
    
    def update_q_value(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        terminated: bool
    ) -> None:
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            terminated: Whether episode terminated
        """
        state_idx = self.get_state_index(state)
        next_state_idx = self.get_state_index(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_idx + (action,)]
        
        # Calculate target Q-value
        if terminated:
            target_q = reward
        else:
            # Q-learning: max over next state actions
            next_max_q = np.max(self.q_table[next_state_idx])
            target_q = reward + self.discount_factor * next_max_q
        
        # Q-learning update
        td_error = target_q - current_q
        self.q_table[state_idx + (action,)] += self.learning_rate * td_error
        
        self.total_steps += 1
    
    def train(
        self, 
        episodes: int = 1000, 
        max_steps: int = 500,
        verbose: bool = True
    ) -> list:
        """
        Train the Q-learning agent.
        
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
            
            for step in range(max_steps):
                # Select action
                action = self.select_action(state)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Update Q-table
                self.update_q_value(state, action, reward, next_state, terminated)
                
                state = next_state
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            self.episode_count += 1
            
            # Print progress
            if verbose and (episode % 100 == 0 or episode == episodes - 1):
                avg_reward = np.mean(episode_rewards[-100:])
                epsilon = self.get_epsilon()
                print(f"Episode {episode:4d}, Avg Reward: {avg_reward:6.2f}, "
                      f"Epsilon: {epsilon:.3f}, Steps: {step + 1}")
        
        return episode_rewards
    
    def test(self, episodes: int = 10, max_steps: int = 500, render: bool = False) -> list:
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
        
        # Use greedy policy (no exploration)
        original_epsilon = self.epsilon_end
        self.epsilon_end = 0.0
        
        for episode in range(episodes):
            if render:
                env = gym.make('CartPole-v1', render_mode='human')
            else:
                env = self.env
                
            state, _ = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                state = next_state
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            test_rewards.append(total_reward)
            print(f"Test Episode {episode + 1}: {total_reward} steps")
            
            if render:
                env.close()
        
        # Restore original epsilon
        self.epsilon_end = original_epsilon
        
        avg_test_reward = np.mean(test_rewards)
        print(f"Average test reward: {avg_test_reward:.2f}")
        
        return test_rewards
    
    def get_policy_visualization(self) -> np.ndarray:
        """
        Get a visualization of the learned policy.
        
        Returns:
            Array showing the policy for different states
        """
        # Simplified visualization for 2D slice of state space
        policy = np.zeros((self.bins[0], self.bins[2]))  # cart_pos vs pole_angle
        
        for i in range(self.bins[0]):
            for j in range(self.bins[2]):
                # Use middle values for cart_vel and pole_vel
                state_idx = (i, self.bins[1] // 2, j, self.bins[3] // 2)
                policy[i, j] = np.argmax(self.q_table[state_idx])
        
        return policy
    
    def save_model(self, filepath: str) -> None:
        """Save the Q-table to a file."""
        np.save(filepath, self.q_table)
        print(f"Q-table saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load Q-table from a file."""
        self.q_table = np.load(filepath)
        print(f"Q-table loaded from: {filepath}")