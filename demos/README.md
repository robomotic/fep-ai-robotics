# Active Inference CartPole Demo

This demo implements active inference agents for the CartPole control problem using the Free Energy Principle framework. It provides both basic and enhanced implementations with comprehensive analysis tools.

## Overview

Active inference is a theoretical framework that treats perception, action, and learning as inference processes. Agents minimize "expected free energy" which balances:

- **Accuracy**: Achieving preferred observations (goals)
- **Complexity**: Minimizing surprise and action costs
- **Epistemic Value**: Seeking information to reduce uncertainty

## Features

- **Two Agent Types**:
  - `ActiveInferenceCartPole`: Basic implementation with model learning
  - `ActiveInferenceCartPoleWithBelief`: Enhanced version with explicit belief states and epistemic/pragmatic value separation

- **Comprehensive Analysis**:
  - Performance tracking and visualization
  - Free energy decomposition analysis
  - Belief state evolution (enhanced agent)
  - Comparison with baselines

- **Reproducible Experiments**: Full Poetry and UV support for consistent environments

## Quick Start

### Installation

```bash
# Navigate to the demos directory
cd demos

# Install dependencies with Poetry
poetry install

# Or install with UV (if available)
uv sync
```

### Running the Demo

```bash
# Run the default demonstration
poetry run python -m active_inference_cartpole.main

# Or use the convenient script
poetry run run-cartpole-demo

# Compare different agents
poetry run python -m active_inference_cartpole.main --mode compare --episodes 200

# Run enhanced agent only
poetry run python -m active_inference_cartpole.main --mode enhanced --episodes 100

# Run with rendering (visual)
poetry run python -m active_inference_cartpole.main --mode demo --render
```

## Usage Examples

### Basic Usage

```python
import gymnasium as gym
from active_inference_cartpole import ActiveInferenceCartPole

# Create environment and agent
env = gym.make('CartPole-v1')
agent = ActiveInferenceCartPole(env, learning_rate=0.01)

# Run episode
state, _ = env.reset()
agent.reset_history()

for step in range(500):
    action, action_probs = agent.action_selection(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    prediction_error = agent.update_beliefs(state, action, next_state)
    
    state = next_state
    if terminated or truncated:
        break
```

### Enhanced Agent with Belief States

```python
from active_inference_cartpole import ActiveInferenceCartPoleWithBelief

# Create enhanced agent
agent = ActiveInferenceCartPoleWithBelief(env, learning_rate=0.01)

# Initialize belief state
agent.belief_mean = state.copy()
agent.belief_precision = np.eye(agent.state_dim)

# Run with belief updates
for step in range(500):
    agent.update_belief(state)  # Update belief with observation
    action = agent.select_action()  # Select action based on expected free energy
    next_state, reward, terminated, truncated, _ = env.step(action)
    state = next_state
    if terminated or truncated:
        break
```

### Analysis and Visualization

```python
from active_inference_cartpole.utils import plot_results, plot_agent_analysis, save_results

# Plot performance
plot_results(episode_lengths, "My Agent")

# Detailed agent analysis
plot_agent_analysis(agent, episode_lengths)

# Save results
save_results(episode_lengths, agent, "my_experiment")
```

## Command Line Options

```bash
python -m active_inference_cartpole.main [options]

Options:
  --mode {basic,enhanced,compare,demo,all}
                        Which demonstration to run (default: demo)
  --episodes EPISODES   Number of episodes to run (default: 100)
  --learning-rate LEARNING_RATE
                        Learning rate for the agents (default: 0.01)
  --render              Render the environment during training
  --no-plots            Skip plotting results
```

## Active Inference Concepts

### Free Energy Minimization
The agent minimizes expected free energy G:
```
G = -E[log P(o|s,a)] - log P(o) - log P(a)
```
Where:
- E[log P(o|s,a)]: Expected log likelihood (accuracy)
- log P(o): Prior preference over observations (goals)
- log P(a): Action prior (complexity/cost)

### Generative Model
The agent maintains beliefs about state transitions:
```
s_{t+1} = A * s_t + B * a_t + ε
```
Where A is the transition matrix and B represents action effects.

### Belief Updates
The enhanced agent uses Bayesian inference to update beliefs:
```
P(s_t|o_{1:t}) ∝ P(o_t|s_t) * P(s_t|o_{1:t-1})
```

## Experiment Results

The demo typically shows:

1. **Learning Progression**: Agents improve over episodes by learning action effects
2. **Free Energy Decomposition**: Visualization of accuracy vs. complexity trade-offs
3. **Belief Evolution**: How the enhanced agent's beliefs change over time
4. **Comparative Performance**: Active inference vs. random baseline

## File Structure

```
active_inference_cartpole/
├── __init__.py          # Package initialization
├── agent.py             # Agent implementations
├── utils.py             # Plotting and analysis utilities
└── main.py              # Main demonstration script
```

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Statistical functions (softmax, etc.)
- `matplotlib`: Plotting and visualization
- `gymnasium`: OpenAI Gym environments
- `torch`: Deep learning framework (for potential extensions)

## Theory Background

This implementation is based on:

1. **Free Energy Principle** (Friston, 2010): Biological systems minimize free energy
2. **Active Inference** (Friston et al., 2017): Extension to action and planning
3. **Expected Free Energy** (Friston et al., 2015): Forward-looking free energy for control

## Extensions

The framework can be extended to:
- Multi-agent scenarios
- Continuous action spaces
- Hierarchical active inference
- Integration with deep learning models
- Real robotic systems

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Poetry environment is activated
2. **Display Issues**: Use `--no-plots` flag for headless environments
3. **Convergence Problems**: Try different learning rates (0.001 - 0.1)

### Performance Tips

- Start with demo mode to understand the framework
- Use enhanced agent for more sophisticated behavior
- Adjust precision parameters for different goal priorities
- Experiment with different learning rates

## Contributing

To contribute to this demo:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows the existing style
5. Submit a pull request

## License

This demo is part of the FEP AI Robotics project and follows the same license terms.