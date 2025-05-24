"""
Reinforcement Learning CartPole Package

This package implements various RL algorithms for the CartPole control problem,
progressing from simple tabular methods to advanced deep learning approaches.
"""

__version__ = "0.1.0"
__author__ = "RL CartPole Team"

from .q_learning import QLearningAgent
from .sarsa import SarsaAgent
from .dqn import DQNAgent
from .double_dqn import DoubleDQNAgent
from .dueling_dqn import DuelingDQNAgent
from .utils import plot_results, compare_algorithms, discretize_state

__all__ = [
    "QLearningAgent",
    "SarsaAgent", 
    "DQNAgent",
    "DoubleDQNAgent",
    "DuelingDQNAgent",
    "plot_results",
    "compare_algorithms",
    "discretize_state"
]