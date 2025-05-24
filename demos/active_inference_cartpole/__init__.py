"""
Active Inference CartPole Package

This package implements active inference for the CartPole control problem
using the Free Energy Principle framework.
"""

__version__ = "0.1.0"
__author__ = "FEP AI Robotics Team"

from .agent import ActiveInferenceCartPole, ActiveInferenceCartPoleWithBelief
from .utils import plot_results, save_results

__all__ = [
    "ActiveInferenceCartPole",
    "ActiveInferenceCartPoleWithBelief", 
    "plot_results",
    "save_results"
]