"""
Utility functions for Active Inference CartPole demonstrations.

This module provides functions for plotting results, saving data,
and analyzing agent performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import json
import os
from datetime import datetime


def plot_results(
    episode_lengths: List[int],
    agent_name: str = "Active Inference Agent",
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot episode lengths over time with moving average.
    
    Args:
        episode_lengths: List of episode lengths
        agent_name: Name of the agent for the plot title
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Main plot
    episodes = range(len(episode_lengths))
    plt.subplot(2, 2, 1)
    plt.plot(episodes, episode_lengths, alpha=0.6, color='blue', label='Episode Length')
    
    # Moving average
    window_size = min(10, len(episode_lengths) // 4)
    if window_size > 1:
        moving_avg = np.convolve(
            episode_lengths, 
            np.ones(window_size) / window_size, 
            mode='valid'
        )
        plt.plot(
            range(window_size - 1, len(episode_lengths)), 
            moving_avg, 
            color='red', 
            linewidth=2, 
            label=f'Moving Average ({window_size})'
        )
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title(f'{agent_name} Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Performance statistics
    plt.subplot(2, 2, 2)
    bins = np.linspace(0, max(episode_lengths), 20)
    plt.hist(episode_lengths, bins=bins, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Episode Length')
    plt.ylabel('Frequency')
    plt.title('Episode Length Distribution')
    plt.grid(True, alpha=0.3)
    
    # Cumulative performance
    plt.subplot(2, 2, 3)
    cumulative_reward = np.cumsum(episode_lengths)
    plt.plot(episodes, cumulative_reward, color='purple', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Steps')
    plt.title('Cumulative Performance')
    plt.grid(True, alpha=0.3)
    
    # Performance metrics text
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    max_length = np.max(episode_lengths)
    min_length = np.min(episode_lengths)
    
    # Calculate success rate (episodes lasting more than 100 steps)
    success_rate = np.mean(np.array(episode_lengths) >= 100) * 100
    
    stats_text = f"""
    Performance Statistics:
    
    Mean Episode Length: {mean_length:.2f}
    Std Episode Length: {std_length:.2f}
    Max Episode Length: {max_length}
    Min Episode Length: {min_length}
    
    Success Rate (≥100 steps): {success_rate:.1f}%
    
    Total Episodes: {len(episode_lengths)}
    Total Steps: {sum(episode_lengths)}
    """
    
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()


def plot_agent_analysis(
    agent,
    episode_lengths: List[int],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot detailed analysis of agent behavior including free energies and prediction errors.
    
    Args:
        agent: The active inference agent with history
        episode_lengths: List of episode lengths
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Episode performance
    axes[0, 0].plot(episode_lengths, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Length')
    axes[0, 0].set_title('Episode Performance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Prediction errors over time
    if hasattr(agent, 'prediction_errors') and agent.prediction_errors:
        axes[0, 1].plot(agent.prediction_errors, color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Prediction Error')
        axes[0, 1].set_title('Prediction Error Over Time')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Action distribution
    if hasattr(agent, 'actions_taken') and agent.actions_taken:
        action_counts = np.bincount(agent.actions_taken, minlength=2)
        axes[0, 2].bar(['Left', 'Right'], action_counts, color=['orange', 'green'])
        axes[0, 2].set_ylabel('Action Count')
        axes[0, 2].set_title('Action Distribution')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Free energy components (if available)
    if hasattr(agent, 'free_energies') and agent.free_energies:
        free_energies = np.array(agent.free_energies)
        if free_energies.size > 0:
            axes[1, 0].plot(free_energies[:, 0], label='Left Action', alpha=0.7)
            axes[1, 0].plot(free_energies[:, 1], label='Right Action', alpha=0.7)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Expected Free Energy')
            axes[1, 0].set_title('Expected Free Energy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    # Belief evolution (for enhanced agent)
    if hasattr(agent, 'belief_history') and agent.belief_history:
        belief_means = np.array([b['mean'] for b in agent.belief_history])
        for i, label in enumerate(['Cart Pos', 'Cart Vel', 'Pole Angle', 'Pole Vel']):
            axes[1, 1].plot(belief_means[:, i], label=label, alpha=0.7)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Belief Mean')
        axes[1, 1].set_title('Belief State Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Epistemic vs Pragmatic values (for enhanced agent)
    if (hasattr(agent, 'epistemic_values') and agent.epistemic_values and
        hasattr(agent, 'pragmatic_values') and agent.pragmatic_values):
        axes[1, 2].plot(agent.epistemic_values, label='Epistemic', alpha=0.7)
        axes[1, 2].plot(agent.pragmatic_values, label='Pragmatic', alpha=0.7)
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].set_title('Epistemic vs Pragmatic Value')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Analysis plot saved to: {save_path}")
    
    if show_plot:
        plt.show()


def save_results(
    episode_lengths: List[int],
    agent,
    experiment_name: str,
    save_dir: str = "results"
) -> str:
    """
    Save experimental results to JSON file.
    
    Args:
        episode_lengths: List of episode lengths
        agent: The active inference agent
        experiment_name: Name of the experiment
        save_dir: Directory to save results
        
    Returns:
        Path to the saved file
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare results dictionary
    results = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'episode_lengths': episode_lengths,
        'statistics': {
            'mean_length': float(np.mean(episode_lengths)),
            'std_length': float(np.std(episode_lengths)),
            'max_length': int(np.max(episode_lengths)),
            'min_length': int(np.min(episode_lengths)),
            'success_rate': float(np.mean(np.array(episode_lengths) >= 100) * 100),
            'total_episodes': len(episode_lengths),
            'total_steps': sum(episode_lengths)
        },
        'agent_config': {
            'type': type(agent).__name__,
            'learning_rate': getattr(agent, 'learning_rate', None),
            'state_dim': getattr(agent, 'state_dim', None),
            'action_dim': getattr(agent, 'action_dim', None)
        }
    }
    
    # Add agent-specific data
    if hasattr(agent, 'prediction_errors'):
        results['prediction_errors'] = agent.prediction_errors
    
    if hasattr(agent, 'actions_taken'):
        results['actions_taken'] = agent.actions_taken
    
    if hasattr(agent, 'free_energies'):
        results['free_energies'] = [fe.tolist() if hasattr(fe, 'tolist') else fe 
                                   for fe in agent.free_energies]
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath


def compare_agents(
    results_dict: Dict[str, List[int]],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Compare performance of different agents.
    
    Args:
        results_dict: Dictionary mapping agent names to episode lengths
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Performance comparison
    plt.subplot(2, 2, 1)
    for agent_name, episode_lengths in results_dict.items():
        episodes = range(len(episode_lengths))
        plt.plot(episodes, episode_lengths, alpha=0.7, label=agent_name)
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Agent Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot comparison
    plt.subplot(2, 2, 2)
    data = [episode_lengths for episode_lengths in results_dict.values()]
    labels = list(results_dict.keys())
    plt.boxplot(data, labels=labels)
    plt.ylabel('Episode Length')
    plt.title('Performance Distribution')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Success rate comparison
    plt.subplot(2, 2, 3)
    success_rates = [np.mean(np.array(lengths) >= 100) * 100 
                    for lengths in results_dict.values()]
    bars = plt.bar(labels, success_rates, alpha=0.7)
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Comparison (≥100 steps)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # Statistics table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    stats_data = []
    for agent_name, episode_lengths in results_dict.items():
        stats_data.append([
            agent_name,
            f"{np.mean(episode_lengths):.1f}",
            f"{np.std(episode_lengths):.1f}",
            f"{np.max(episode_lengths)}",
            f"{np.mean(np.array(episode_lengths) >= 100) * 100:.1f}%"
        ])
    
    headers = ['Agent', 'Mean', 'Std', 'Max', 'Success Rate']
    
    # Create table
    table = plt.table(cellText=stats_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    if show_plot:
        plt.show()


def load_results(filepath: str) -> Dict:
    """
    Load experimental results from JSON file.
    
    Args:
        filepath: Path to the results file
        
    Returns:
        Dictionary containing experimental results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded results from: {filepath}")
    print(f"Experiment: {results['experiment_name']}")
    print(f"Episodes: {results['statistics']['total_episodes']}")
    print(f"Mean length: {results['statistics']['mean_length']:.2f}")
    
    return results