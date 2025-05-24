"""
Main script for running Active Inference CartPole demonstrations.

This script provides examples of how to use both the basic and enhanced
active inference agents for the CartPole control problem.
"""

import gymnasium as gym
import numpy as np
import argparse
import sys
import os
from typing import List, Dict

# Add the package to the path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import ActiveInferenceCartPole, ActiveInferenceCartPoleWithBelief
from utils import plot_results, plot_agent_analysis, save_results, compare_agents


def run_basic_agent(
    episodes: int = 100,
    max_steps: int = 500,
    learning_rate: float = 0.01,
    render: bool = False,
    save_results_flag: bool = True
) -> tuple:
    """
    Run the basic Active Inference agent.
    
    Args:
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        learning_rate: Learning rate for model updates
        render: Whether to render the environment
        save_results_flag: Whether to save results
        
    Returns:
        Tuple of (agent, episode_lengths)
    """
    print(f"Running Basic Active Inference Agent for {episodes} episodes...")
    
    # Create environment
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    agent = ActiveInferenceCartPole(env, learning_rate=learning_rate)
    
    episode_lengths = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        agent.reset_history()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Agent selects action based on active inference
            action, action_probs = agent.action_selection(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Update agent's beliefs
            prediction_error = agent.update_beliefs(state, action, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_lengths.append(steps)
        
        if episode % 20 == 0 or episode == episodes - 1:
            avg_length = np.mean(episode_lengths[-20:])
            print(f"Episode {episode:3d}, Recent Avg Length: {avg_length:6.2f}, "
                  f"Last Episode: {steps:3d} steps")
    
    env.close()
    
    # Save results if requested
    if save_results_flag:
        save_results(episode_lengths, agent, "basic_active_inference")
    
    return agent, episode_lengths


def run_enhanced_agent(
    episodes: int = 100,
    max_steps: int = 500,
    learning_rate: float = 0.01,
    render: bool = False,
    save_results_flag: bool = True
) -> tuple:
    """
    Run the enhanced Active Inference agent with belief states.
    
    Args:
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        learning_rate: Learning rate for belief updates
        render: Whether to render the environment
        save_results_flag: Whether to save results
        
    Returns:
        Tuple of (agent, episode_lengths)
    """
    print(f"Running Enhanced Active Inference Agent for {episodes} episodes...")
    
    # Create environment
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    agent = ActiveInferenceCartPoleWithBelief(env, learning_rate=learning_rate)
    
    episode_lengths = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        agent.reset_history()
        
        # Initialize belief with first observation
        agent.belief_mean = state.copy()
        agent.belief_precision = np.eye(agent.state_dim)
        
        steps = 0
        
        for step in range(max_steps):
            # Update belief with current observation
            agent.update_belief(state)
            
            # Agent selects action based on expected free energy
            action = agent.select_action()
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            state = next_state
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_lengths.append(steps)
        
        if episode % 20 == 0 or episode == episodes - 1:
            avg_length = np.mean(episode_lengths[-20:])
            print(f"Episode {episode:3d}, Recent Avg Length: {avg_length:6.2f}, "
                  f"Last Episode: {steps:3d} steps")
    
    env.close()
    
    # Save results if requested
    if save_results_flag:
        save_results(episode_lengths, agent, "enhanced_active_inference")
    
    return agent, episode_lengths


def run_comparison_experiment(
    episodes: int = 100,
    max_steps: int = 500,
    learning_rate: float = 0.01
) -> Dict[str, List[int]]:
    """
    Run both agents and compare their performance.
    
    Args:
        episodes: Number of episodes to run for each agent
        max_steps: Maximum steps per episode
        learning_rate: Learning rate for both agents
        
    Returns:
        Dictionary mapping agent names to episode lengths
    """
    print("Running comparison experiment...")
    
    # Run basic agent
    basic_agent, basic_lengths = run_basic_agent(
        episodes=episodes,
        max_steps=max_steps,
        learning_rate=learning_rate,
        render=False,
        save_results_flag=False
    )
    
    # Run enhanced agent
    enhanced_agent, enhanced_lengths = run_enhanced_agent(
        episodes=episodes,
        max_steps=max_steps,
        learning_rate=learning_rate,
        render=False,
        save_results_flag=False
    )
    
    # Create comparison dictionary
    results_dict = {
        'Basic Active Inference': basic_lengths,
        'Enhanced Active Inference': enhanced_lengths
    }
    
    # Save comparison results
    save_results(basic_lengths, basic_agent, "comparison_basic")
    save_results(enhanced_lengths, enhanced_agent, "comparison_enhanced")
    
    return results_dict


def run_random_baseline(
    episodes: int = 100,
    max_steps: int = 500
) -> List[int]:
    """
    Run random baseline for comparison.
    
    Args:
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        List of episode lengths
    """
    print(f"Running random baseline for {episodes} episodes...")
    
    env = gym.make('CartPole-v1')
    episode_lengths = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        steps = 0
        
        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            state = next_state
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_lengths.append(steps)
        
        if episode % 20 == 0 or episode == episodes - 1:
            avg_length = np.mean(episode_lengths[-20:])
            print(f"Episode {episode:3d}, Recent Avg Length: {avg_length:6.2f}")
    
    env.close()
    return episode_lengths


def demonstrate_active_inference_principles():
    """
    Demonstrate key active inference principles with a simple example.
    """
    print("\n" + "="*60)
    print("ACTIVE INFERENCE PRINCIPLES DEMONSTRATION")
    print("="*60)
    
    # Create environment and agent
    env = gym.make('CartPole-v1')
    agent = ActiveInferenceCartPole(env, learning_rate=0.05)
    
    print("\n1. GENERATIVE MODEL:")
    print("   The agent maintains a model of how actions affect states")
    print(f"   Initial action effects (B matrix):")
    print(f"   {agent.B[:2, :]}")  # Show first 2 rows
    
    print("\n2. PREFERRED OBSERVATIONS (GOALS):")
    print(f"   Agent prefers: {agent.preferred_obs}")
    print("   [cart centered, no velocity, pole upright, no angular velocity]")
    
    print("\n3. EXPECTED FREE ENERGY MINIMIZATION:")
    
    # Simulate one step
    state, _ = env.reset()
    print(f"   Current state: [{state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}, {state[3]:.3f}]")
    
    # Evaluate both actions
    action_vector_left = np.array([1.0, 0.0])
    action_vector_right = np.array([0.0, 1.0])
    
    G_left = agent.expected_free_energy(state, action_vector_left)
    G_right = agent.expected_free_energy(state, action_vector_right)
    
    print(f"   Expected Free Energy - Left:  {G_left:.3f}")
    print(f"   Expected Free Energy - Right: {G_right:.3f}")
    print(f"   Agent prefers: {'Left' if G_left < G_right else 'Right'}")
    
    # Take action and update
    action, probs = agent.action_selection(state)
    next_state, _, terminated, truncated, _ = env.step(action)
    prediction_error = agent.update_beliefs(state, action, next_state)
    
    print(f"\n4. LEARNING FROM PREDICTION ERROR:")
    print(f"   Prediction error magnitude: {np.linalg.norm(prediction_error):.3f}")
    print(f"   Model updated based on this error")
    
    env.close()


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Active Inference CartPole Demonstrations"
    )
    parser.add_argument(
        '--mode', 
        choices=['basic', 'enhanced', 'compare', 'demo', 'all'],
        default='demo',
        help='Which demonstration to run'
    )
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=100,
        help='Number of episodes to run'
    )
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=0.01,
        help='Learning rate for the agents'
    )
    parser.add_argument(
        '--render', 
        action='store_true',
        help='Render the environment during training'
    )
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help='Skip plotting results'
    )
    
    args = parser.parse_args()
    
    print("Active Inference CartPole Demonstration")
    print("="*50)
    
    if args.mode == 'demo':
        # Run principles demonstration
        demonstrate_active_inference_principles()
        
        # Run a short basic experiment
        print(f"\nRunning basic agent for {min(50, args.episodes)} episodes...")
        agent, episode_lengths = run_basic_agent(
            episodes=min(50, args.episodes),
            learning_rate=args.learning_rate,
            render=args.render
        )
        
        if not args.no_plots:
            plot_results(episode_lengths, "Basic Active Inference (Demo)")
    
    elif args.mode == 'basic':
        agent, episode_lengths = run_basic_agent(
            episodes=args.episodes,
            learning_rate=args.learning_rate,
            render=args.render
        )
        
        if not args.no_plots:
            plot_results(episode_lengths, "Basic Active Inference")
            plot_agent_analysis(agent, episode_lengths)
    
    elif args.mode == 'enhanced':
        agent, episode_lengths = run_enhanced_agent(
            episodes=args.episodes,
            learning_rate=args.learning_rate,
            render=args.render
        )
        
        if not args.no_plots:
            plot_results(episode_lengths, "Enhanced Active Inference")
            plot_agent_analysis(agent, episode_lengths)
    
    elif args.mode == 'compare':
        results_dict = run_comparison_experiment(
            episodes=args.episodes,
            learning_rate=args.learning_rate
        )
        
        # Add random baseline
        random_lengths = run_random_baseline(episodes=args.episodes)
        results_dict['Random Baseline'] = random_lengths
        
        if not args.no_plots:
            compare_agents(results_dict)
    
    elif args.mode == 'all':
        # Run comprehensive demonstration
        demonstrate_active_inference_principles()
        
        print(f"\nRunning comprehensive comparison with {args.episodes} episodes...")
        results_dict = run_comparison_experiment(
            episodes=args.episodes,
            learning_rate=args.learning_rate
        )
        
        # Add random baseline
        random_lengths = run_random_baseline(episodes=args.episodes)
        results_dict['Random Baseline'] = random_lengths
        
        if not args.no_plots:
            compare_agents(results_dict)
            
            # Individual analysis plots
            basic_agent, basic_lengths = run_basic_agent(
                episodes=min(50, args.episodes),
                learning_rate=args.learning_rate,
                render=False,
                save_results_flag=False
            )
            enhanced_agent, enhanced_lengths = run_enhanced_agent(
                episodes=min(50, args.episodes),
                learning_rate=args.learning_rate,
                render=False,
                save_results_flag=False
            )
            
            plot_agent_analysis(basic_agent, basic_lengths)
            plot_agent_analysis(enhanced_agent, enhanced_lengths)


if __name__ == "__main__":
    main()