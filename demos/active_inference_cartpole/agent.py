"""
Active Inference Agent for CartPole Control

This module implements active inference agents that use the Free Energy Principle
to control the CartPole environment in OpenAI Gymnasium.
"""

import numpy as np
import gymnasium as gym
from scipy.special import softmax
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActiveInferenceCartPole:
    """
    Basic Active Inference agent for CartPole control.
    
    This agent minimizes expected free energy by balancing accuracy 
    (achieving preferred observations) and complexity (action costs).
    """
    
    def __init__(self, env: gym.Env, learning_rate: float = 0.01):
        """
        Initialize the Active Inference agent.
        
        Args:
            env: Gymnasium CartPole environment
            learning_rate: Learning rate for model updates
        """
        self.env = env
        self.learning_rate = learning_rate
        
        # State and action dimensions
        self.state_dim = 4  # cart position, cart velocity, pole angle, pole angular velocity
        self.action_dim = 2  # left, right
        
        # Prior beliefs about preferred observations (goals)
        # We want: centered cart (pos=0), no velocity, upright pole (angle=0), no angular velocity
        self.preferred_obs = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Precision (inverse variance) for each state dimension
        # Higher values mean we care more about that dimension
        self.obs_precision = np.array([1.0, 0.1, 10.0, 1.0])  # Care most about pole angle
        
        # Transition model parameters (simplified linear dynamics)
        self.A = np.eye(self.state_dim)  # State transition matrix
        self.B = np.random.randn(self.state_dim, self.action_dim) * 0.1  # Action effects
        
        # Observation model (identity - we observe states directly)
        self.C = np.eye(self.state_dim)
        
        # Action precision (preference for minimal action)
        self.action_precision = 2.0
        
        # History tracking
        self.reset_history()
        
    def reset_history(self):
        """Reset tracking variables for a new episode."""
        self.prediction_errors = []
        self.free_energies = []
        self.actions_taken = []
        
    def generative_model(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict next state given current state and action.
        
        Args:
            state: Current state vector
            action: Action vector (one-hot encoded)
            
        Returns:
            Predicted next state
        """
        return self.A @ state + self.B @ action
    
    def observation_likelihood(self, obs: np.ndarray, predicted_obs: np.ndarray) -> float:
        """
        Calculate log likelihood of observation given prediction.
        
        Args:
            obs: Actual observation
            predicted_obs: Predicted observation
            
        Returns:
            Log likelihood
        """
        error = obs - predicted_obs
        weighted_error = error * self.obs_precision
        return -0.5 * np.sum(weighted_error ** 2)
    
    def prior_preference(self, obs: np.ndarray) -> float:
        """
        Calculate prior preference over observations (goal-directed term).
        
        Args:
            obs: Observation vector
            
        Returns:
            Log preference
        """
        error = obs - self.preferred_obs
        weighted_error = error * self.obs_precision
        return -0.5 * np.sum(weighted_error ** 2)
    
    def expected_free_energy(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Calculate expected free energy for state-action pair.
        
        Expected free energy = -Expected log likelihood - Prior preference - Action prior
        
        Args:
            state: Current state
            action: Action vector
            
        Returns:
            Expected free energy (to be minimized)
        """
        # Predict next observation
        predicted_state = self.generative_model(state, action)
        predicted_obs = self.C @ predicted_state
        
        # Expected log likelihood (accuracy term)
        accuracy = self.observation_likelihood(predicted_obs, predicted_obs)
        
        # Prior preference (goal-seeking term)
        preference = self.prior_preference(predicted_obs)
        
        # Action prior (complexity term - preference for certain actions)
        action_cost = -0.5 * self.action_precision * np.sum(action ** 2)
        
        # Expected free energy (to be minimized)
        G = -(accuracy + preference + action_cost)
        return G
    
    def action_selection(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Select action by minimizing expected free energy.
        
        Args:
            state: Current state vector
            
        Returns:
            Tuple of (selected_action, action_probabilities)
        """
        actions = []
        free_energies = []
        
        # Evaluate both possible actions
        for a in range(self.action_dim):
            action_vector = np.zeros(self.action_dim)
            action_vector[a] = 1.0
            
            G = self.expected_free_energy(state, action_vector)
            actions.append(a)
            free_energies.append(G)
        
        # Store free energies for analysis
        self.free_energies.append(free_energies.copy())
        
        # Convert to probabilities using softmax (precision parameter)
        beta = 4.0  # inverse temperature
        action_probs = softmax(-beta * np.array(free_energies))
        
        # Sample action according to probabilities
        action = np.random.choice(actions, p=action_probs)
        self.actions_taken.append(action)
        
        return action, action_probs
    
    def update_beliefs(self, state: np.ndarray, action: int, next_obs: np.ndarray) -> np.ndarray:
        """
        Update model parameters based on prediction error.
        
        Args:
            state: Previous state
            action: Action taken
            next_obs: Observed next state
            
        Returns:
            Prediction error vector
        """
        # Convert action to one-hot vector
        action_vector = np.zeros(self.action_dim)
        action_vector[action] = 1.0
        
        # Predict what we should have observed
        predicted_state = self.generative_model(state, action_vector)
        predicted_obs = self.C @ predicted_state
        
        # Calculate prediction error
        prediction_error = next_obs - predicted_obs
        self.prediction_errors.append(np.linalg.norm(prediction_error))
        
        # Update B matrix (action effects) using gradient descent
        # ∂E/∂B = prediction_error ⊗ action_vector
        self.B += self.learning_rate * np.outer(prediction_error, action_vector)
        
        return prediction_error


class ActiveInferenceCartPoleWithBelief:
    """
    Enhanced Active Inference agent with explicit belief state maintenance.
    
    This version maintains probabilistic beliefs about the state and incorporates
    epistemic value (information seeking) in addition to pragmatic value.
    """
    
    def __init__(self, env: gym.Env, learning_rate: float = 0.01):
        """
        Initialize the enhanced Active Inference agent.
        
        Args:
            env: Gymnasium CartPole environment
            learning_rate: Learning rate for belief updates
        """
        self.env = env
        self.learning_rate = learning_rate
        self.state_dim = 4
        self.action_dim = 2
        
        # Belief state (mean and precision matrix)
        self.belief_mean = np.zeros(self.state_dim)
        self.belief_precision = np.eye(self.state_dim)
        
        # Preferred observations and their precision
        self.preferred_obs = np.array([0.0, 0.0, 0.0, 0.0])
        self.preference_precision = np.diag([1.0, 0.1, 10.0, 1.0])
        
        # Generative model parameters
        self.A = np.eye(self.state_dim)
        self.B = np.zeros((self.state_dim, self.action_dim))
        
        # Initialize action effects based on CartPole physics
        self.B[1, :] = [-1, 1]      # Actions affect cart velocity
        self.B[3, :] = [-0.5, 0.5]  # Actions affect pole angular velocity
        
        # Process and observation noise
        self.process_precision = 0.9 * np.eye(self.state_dim)
        self.obs_precision = 10.0 * np.eye(self.state_dim)
        
        # History tracking
        self.reset_history()
        
    def reset_history(self):
        """Reset tracking variables for a new episode."""
        self.belief_history = []
        self.epistemic_values = []
        self.pragmatic_values = []
        
    def predict_state(self, action: int) -> np.ndarray:
        """
        Predict next state belief mean.
        
        Args:
            action: Action index
            
        Returns:
            Predicted state mean
        """
        action_vector = np.zeros(self.action_dim)
        action_vector[action] = 1.0
        
        predicted_mean = self.A @ self.belief_mean + self.B @ action_vector
        return predicted_mean
    
    def update_belief(self, obs: np.ndarray):
        """
        Update belief state given observation using Bayesian inference.
        
        Args:
            obs: New observation
        """
        # Store belief history
        self.belief_history.append({
            'mean': self.belief_mean.copy(),
            'precision': self.belief_precision.copy()
        })
        
        # Prediction step (add process noise)
        prediction_precision = self.process_precision @ self.belief_precision
        
        # Update step (incorporate observation)
        combined_precision = prediction_precision + self.obs_precision
        
        try:
            precision_inv = np.linalg.inv(combined_precision)
            combined_mean = precision_inv @ (
                prediction_precision @ self.belief_mean + 
                self.obs_precision @ obs
            )
            
            self.belief_mean = combined_mean
            self.belief_precision = combined_precision
            
        except np.linalg.LinAlgError:
            # Fallback for numerical issues
            logger.warning("Precision matrix inversion failed, using simple update")
            self.belief_mean = 0.9 * self.belief_mean + 0.1 * obs
    
    def epistemic_value(self) -> float:
        """
        Calculate epistemic value (information gain / uncertainty reduction).
        
        Returns:
            Epistemic value
        """
        try:
            # Information gain is related to uncertainty (inverse of precision)
            uncertainty = np.trace(np.linalg.inv(self.belief_precision))
            return 0.1 * uncertainty  # Scale factor for epistemic drive
        except np.linalg.LinAlgError:
            return 0.0
    
    def pragmatic_value(self, predicted_mean: np.ndarray) -> float:
        """
        Calculate pragmatic value (goal achievement).
        
        Args:
            predicted_mean: Predicted state mean
            
        Returns:
            Pragmatic value
        """
        goal_error = predicted_mean - self.preferred_obs
        return -0.5 * goal_error.T @ self.preference_precision @ goal_error
    
    def expected_free_energy_belief(self, action: int) -> float:
        """
        Calculate expected free energy using belief state.
        
        Args:
            action: Action index
            
        Returns:
            Expected free energy
        """
        predicted_mean = self.predict_state(action)
        
        # Epistemic value (information gain)
        epistemic = self.epistemic_value()
        self.epistemic_values.append(epistemic)
        
        # Pragmatic value (goal achievement)
        pragmatic = self.pragmatic_value(predicted_mean)
        self.pragmatic_values.append(pragmatic)
        
        # Expected free energy (negative because we want to maximize value)
        return -(epistemic + pragmatic)
    
    def select_action(self) -> int:
        """
        Select action minimizing expected free energy.
        
        Returns:
            Selected action
        """
        free_energies = [
            self.expected_free_energy_belief(a) for a in range(self.action_dim)
        ]
        
        # Convert to probabilities
        action_probs = softmax(-4.0 * np.array(free_energies))
        return np.random.choice(self.action_dim, p=action_probs)