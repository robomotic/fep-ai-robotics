# FEP AI Robotics Demos

This repository contains demonstrations of the Free Energy Principle (FEP), Active Inference, and Reinforcement Learning (RL) algorithms for the CartPole control problem.

## Folder Structure

```
demos/
├── pyproject.toml              # Unified configuration for all demos
├── README.md                   # (This file)
├── setup.sh                    # Setup and run script
├── active_inference_cartpole/  # Active inference demos
│   ├── __init__.py
│   ├── agent.py
│   ├── main.py
│   └── utils.py
└── rl_cartpole/                # RL algorithms (Q-learning, SARSA, DQN, etc.)
    ├── __init__.py
    ├── dqn.py
    ├── q_learning.py
    ├── sarsa.py
    └── utils.py
```

## Quick Start

1. **Install dependencies** (from the `demos` directory):
   ```bash
   poetry install
   # or, if you have UV:
   uv sync
   ```

2. **Run Active Inference demo:**
   ```bash
   poetry run run-cartpole-demo
   # or
   poetry run python -m active_inference_cartpole.main
   ```

3. **Run RL demo (Q-learning, SARSA, DQN, etc.):**
   ```bash
   poetry run run-rl-demo
   # or
   poetry run python -m rl_cartpole.main
   ```

4. **Use the setup script for guided setup and running:**
   ```bash
   ./setup.sh
   ```


## About

- The `active_inference_cartpole` package demonstrates FEP and active inference control for CartPole.
- The `rl_cartpole` package demonstrates classic and deep RL algorithms for CartPole, including Q-learning, SARSA, and DQN.
- All dependencies and scripts are managed via a single `pyproject.toml` for reproducibility and ease of use.

## License

See [LICENSE](../LICENSE) for details.