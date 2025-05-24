"""
Utility functions for RL CartPole agents.

This module provides common utilities used by different RL algorithms,
including state discretization, scheduling functions, and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import gymnasium as gym


def discretize_state(state: np.ndarray, bins: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Discretize continuous state space into discrete bins.
    
    CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    
    Args:
        state: Continuous state vector
        bins: Number of bins for each state dimension
        
    Returns:
        Tuple of discrete state indices
    """
    # Define state space bounds based on CartPole environment
    bounds = [
        (-2.4, 2.4),     # cart position
        (-3.0, 3.0),     # cart velocity (approximation)
        (-0.2095, 0.2095),  # pole angle (12 degrees in radians)
        (-2.0, 2.0)      # pole angular velocity (approximation)
    ]
    
    state_indices = []
    for i, (low, high) in enumerate(bounds):
        # Clip state to bounds
        clipped_state = np.clip(state[i], low, high)
        # Discretize
        bin_size = (high - low) / bins[i]
        state_idx = int((clipped_state - low) / bin_size)
        # Ensure index is within bounds
        state_idx = min(state_idx, bins[i] - 1)
        state_indices.append(state_idx)
    
    return tuple(state_indices)


def epsilon_decay_schedule(episode: int, epsilon_start: float, epsilon_end: float, 
                          epsilon_decay: float) -> float:
    """
    Calculate epsilon value using exponential decay schedule.
    
    Args:
        episode: Current episode number
        epsilon_start: Initial epsilon value
        epsilon_end: Minimum epsilon value
        epsilon_decay: Decay rate
        
    Returns:
        Current epsilon value
    """
    return max(epsilon_end, epsilon_start * (epsilon_decay ** episode))


def linear_decay_schedule(episode: int, start_value: float, end_value: float, 
                         decay_episodes: int) -> float:
    """
    Calculate value using linear decay schedule.
    
    Args:
        episode: Current episode number
        start_value: Initial value
        end_value: Final value
        decay_episodes: Number of episodes over which to decay
        
    Returns:
        Current value
    """
    if episode >= decay_episodes:
        return end_value
    
    decay_rate = (start_value - end_value) / decay_episodes
    return start_value - decay_rate * episode


def learning_rate_schedule(episode: int, initial_lr: float, min_lr: float = 0.01, 
                          decay_rate: float = 0.99) -> float:
    """
    Calculate learning rate using exponential decay schedule.
    
    Args:
        episode: Current episode number
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate
        decay_rate: Decay rate
        
    Returns:
        Current learning rate
    """
    return max(min_lr, initial_lr * (decay_rate ** episode))


def plot_training_results(episode_rewards: List[float], window_size: int = 100, 
                         title: str = "Training Results", save_path: Optional[str] = None) -> None:
    """
    Plot training results with moving average.
    
    Args:
        episode_rewards: List of episode rewards
        window_size: Window size for moving average
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    episodes = range(len(episode_rewards))
    
    # Calculate moving average
    moving_avg = []
    for i in range(len(episode_rewards)):
        start_idx = max(0, i - window_size + 1)
        moving_avg.append(np.mean(episode_rewards[start_idx:i + 1]))
    
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    plt.plot(episodes, moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{title} - Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot histogram of rewards
    plt.subplot(1, 2, 2)
    plt.hist(episode_rewards, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title(f'{title} - Reward Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def compare_algorithms(results_dict: dict, window_size: int = 100, 
                      save_path: Optional[str] = None) -> None:
    """
    Compare performance of different RL algorithms.
    
    Args:
        results_dict: Dictionary with algorithm names as keys and episode rewards as values
        window_size: Window size for moving average
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Raw episode rewards
    plt.subplot(2, 2, 1)
    for name, rewards in results_dict.items():
        episodes = range(len(rewards))
        plt.plot(episodes, rewards, alpha=0.3, label=f'{name} (raw)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards (Raw)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Moving averages
    plt.subplot(2, 2, 2)
    for name, rewards in results_dict.items():
        episodes = range(len(rewards))
        moving_avg = []
        for i in range(len(rewards)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(np.mean(rewards[start_idx:i + 1]))
        plt.plot(episodes, moving_avg, linewidth=2, label=f'{name}')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Moving Average Rewards (window={window_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Final performance comparison
    plt.subplot(2, 2, 3)
    final_performances = []
    algorithm_names = []
    for name, rewards in results_dict.items():
        # Take average of last 100 episodes
        final_perf = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        final_performances.append(final_perf)
        algorithm_names.append(name)
    
    bars = plt.bar(algorithm_names, final_performances, alpha=0.7)
    plt.ylabel('Average Reward (Last 100 Episodes)')
    plt.title('Final Performance Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_performances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Learning curves (cumulative performance)
    plt.subplot(2, 2, 4)
    for name, rewards in results_dict.items():
        episodes = range(len(rewards))
        cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
        plt.plot(episodes, cumulative_avg, linewidth=2, label=f'{name}')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Average Reward')
    plt.title('Learning Curves (Cumulative Average)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def evaluate_agent(agent, env: gym.Env, episodes: int = 100, max_steps: int = 500, 
                   render: bool = False) -> dict:
    """
    Evaluate a trained agent's performance.
    
    Args:
        agent: Trained agent with select_action method
        env: Environment to evaluate on
        episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        
    Returns:
        Dictionary with evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        if render and episode < 5:  # Only render first 5 episodes
            eval_env = gym.make('CartPole-v1', render_mode='human')
        else:
            eval_env = env
            
        state, _ = eval_env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, training=False)
            else:
                # For function-based agents
                action = agent(state)
                
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        
        if render and episode < 5:
            eval_env.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': np.mean([r >= 475 for r in episode_rewards])  # 475+ is considered success
    }


def print_evaluation_results(results: dict, algorithm_name: str) -> None:
    """
    Print formatted evaluation results.
    
    Args:
        results: Results dictionary from evaluate_agent
        algorithm_name: Name of the algorithm
    """
    print(f"\n{algorithm_name} Evaluation Results:")
    print("=" * 40)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Reward Range: [{results['min_reward']:.0f}, {results['max_reward']:.0f}]")
    print(f"Mean Episode Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print("=" * 40)