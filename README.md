# Free Energy Principle and Active Inference for Software Robotics

A Python framework implementing the Free Energy Principle (FEP) and Active Inference for autonomous robotic systems, focusing on predictive processing and adaptive behavior.

## Table of Contents

- [Overview](#overview)
- [The Free Energy Principle](#the-free-energy-principle)
- [Active Inference](#active-inference)
- [Low Road vs High Road to Active Inference](#low-road-vs-high-road-to-active-inference)
- [Implementation Plan](#implementation-plan)
- [Core Components](#core-components)
- [Development Phases](#development-phases)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Contributing](#contributing)

## Overview

This project implements a comprehensive framework for applying the Free Energy Principle and Active Inference to software robotics. The goal is to create autonomous agents that minimize surprise through predictive modeling, belief updating, and adaptive action selection.

## The Free Energy Principle

The **Free Energy Principle (FEP)** is a unifying theory proposed by Karl Friston that explains how biological systems maintain their organization and adapt to their environment. In the context of robotics and AI, FEP provides a mathematical framework for:

### Key Concepts

- **Free Energy**: A measure of the difference between an agent's internal model and the actual state of the world
- **Surprise Minimization**: Agents act to minimize unexpected sensory inputs
- **Predictive Processing**: The brain/system constantly generates predictions about incoming sensory data
- **Bayesian Brain**: Neural processing as continuous Bayesian inference

### Mathematical Foundation

The free energy `F` is defined as:

```
F = E_q[log q(s) - log p(o,s)]
```

Where:
- `q(s)` is the agent's belief about hidden states
- `p(o,s)` is the generative model of observations and states
- Minimizing F is equivalent to maximizing evidence and minimizing KL divergence

## Active Inference

**Active Inference** extends the Free Energy Principle to include action. Rather than just passively perceiving, agents actively sample the environment to:

1. **Reduce uncertainty** about the current state of the world
2. **Achieve preferred outcomes** by acting to bring about desired states
3. **Learn and adapt** their internal models based on experience

### Core Mechanisms

- **Perception**: Updating beliefs about hidden states given observations
- **Action**: Selecting actions to minimize expected free energy
- **Learning**: Updating model parameters and structure over time
- **Precision Control**: Weighting the reliability of different information sources

## Low Road vs High Road to Active Inference

![High Road vs Low Road](https://pbs.twimg.com/media/FQdYaowWQAQi5Qm.jpg)
*Visual representation of the two approaches to implementing Active Inference*

### Low Road (Bottom-Up Approach)
The **Low Road** focuses on implementing the mathematical and computational foundations first:

**Characteristics:**
- Starts with well-defined mathematical formulations
- Implements core algorithms (variational inference, belief updating)
- Uses simplified models and controlled environments
- Emphasizes computational efficiency and numerical stability
- Gradual increase in complexity

**Advantages:**
- Solid mathematical foundation
- Easier to debug and validate
- Clear performance metrics
- Modular and extensible architecture

### High Road (Top-Down Approach)
The **High Road** starts with complex, realistic scenarios and works backward:

**Characteristics:**
- Begins with full robotic scenarios and real-world complexity
- Implements approximate solutions and heuristics
- Focuses on practical performance over theoretical purity
- Uses domain-specific knowledge and engineering solutions

**Our Strategy:**
We start with the **Low Road** to establish a robust foundation, then progressively move toward the **High Road** as we tackle more complex scenarios.

## Implementation Plan

### Phase 1: Mathematical Foundation (Months 1-2)
**Core Infrastructure**
- Variational inference algorithms
- Basic generative models (Gaussian, linear dynamics)
- Prediction error computation
- State estimation (Kalman filtering)

### Phase 2: Active Inference Core (Months 3-4)
**Action and Decision Making**
- Expected free energy calculation
- Action selection mechanisms
- Precision control and attention
- Simple sensorimotor loops

### Phase 3: Learning and Adaptation (Months 5-6)
**Adaptive Capabilities**
- Parameter learning
- Habit formation
- Structure learning
- Meta-learning components

### Phase 4: Robotics Integration (Months 7-8)
**Real-World Application**
- Robot interface abstractions
- Simulation environments
- Control and sensor interfaces
- Example applications

### Phase 5: Advanced Features (Months 9-12)
**Complex Scenarios**
- Hierarchical models
- Nonlinear dynamics
- Multi-agent systems
- Complex robotics tasks

## Core Components

### 1. Generative Models (`generative_models/`)
Mathematical models that generate predictions about sensory observations:
- **Base Model**: Abstract interface for all generative models
- **Gaussian Models**: Simple probabilistic models
- **Linear Dynamics**: State-space models with linear transitions
- **Nonlinear Dynamics**: Complex dynamical systems

### 2. Variational Inference (`variational_inference/`)
Algorithms for approximate Bayesian inference:
- **Variational Bayes**: Core VB implementation
- **Message Passing**: Belief propagation algorithms
- **Laplace Approximation**: Second-order approximations
- **Mean Field**: Factorized approximations

### 3. Prediction and Error Processing (`prediction/`)
Predictive coding and error minimization:
- **Predictive Coding**: Hierarchical prediction mechanisms
- **Prediction Error**: Error computation and propagation
- **Temporal Prediction**: Time-series forecasting
- **Hierarchical Prediction**: Multi-level predictive models

### 4. Action and Policy (`action/`)
Action selection and policy optimization:
- **Action Selection**: Decision-making mechanisms
- **Policy Optimization**: Gradient-based policy improvement
- **Expected Free Energy**: Future surprise minimization
- **Precision Control**: Attention and uncertainty weighting

### 5. State Estimation (`state_estimation/`)
Belief state tracking and filtering:
- **Kalman Filter**: Linear Gaussian filtering
- **Particle Filter**: Non-parametric filtering
- **Variational Filter**: Approximate Bayesian filtering
- **Belief Updating**: Posterior belief computation

### 6. Sensorimotor Integration (`sensorimotor/`)
Coupling between sensing and acting:
- **Sensor Models**: Observation likelihood models
- **Motor Models**: Action effect models
- **Proprioception**: Internal state sensing
- **Sensorimotor Loop**: Closed-loop control

### 7. Learning (`learning/`)
Adaptive model improvement:
- **Parameter Learning**: Online parameter updates
- **Structure Learning**: Model architecture adaptation
- **Habit Formation**: Policy learning and automation
- **Meta-Learning**: Learning to learn efficiently

### 8. Robotics Interface (`robotics/`)
Hardware and simulation interfaces:
- **Robot Interface**: Abstract robot control
- **Simulated Robot**: Virtual robot environments
- **Control Interface**: Motor command abstraction
- **Sensor Interface**: Sensor data processing

### 9. Utilities (`utils/`)
Mathematical and computational tools:
- **Math Utils**: Linear algebra, statistics
- **Distributions**: Probability distributions
- **Optimization**: Gradient descent, natural gradients
- **Visualization**: Plotting and analysis tools

## Development Phases

### Getting Started

#### Prerequisites
```bash
# Python 3.8+
# pip install numpy scipy matplotlib
# pip install torch gymnasium pybullet
```

#### Installation
```bash
git clone https://github.com/yourusername/fep-ai-robotics.git
cd fep-ai-robotics
pip install -r requirements.txt
```

#### Basic Usage
```python
from fep_robotics import ActiveInferenceAgent
from fep_robotics.environments import SimpleNavigation

# Create environment and agent
env = SimpleNavigation()
agent = ActiveInferenceAgent(env.observation_space, env.action_space)

# Run active inference loop
for episode in range(100):
    observation = env.reset()
    done = False
    
    while not done:
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        agent.update(observation, action)
```

## Examples

The `examples/` directory contains demonstrations of active inference principles:

- **Simple Navigation**: Basic spatial navigation with obstacle avoidance
- **Object Tracking**: Visual tracking using predictive models
- **Arm Reaching**: Robotic arm control with proprioceptive feedback
- **Balance Control**: Dynamic balance using sensorimotor integration
- **Multi-Agent**: Coordinated behavior in multi-agent scenarios

## Key Features

- 🧠 **Biologically Inspired**: Based on neuroscientific principles from the Free Energy Principle
- 🔄 **Predictive Processing**: Continuous prediction and error minimization through hierarchical models
- 🎯 **Goal-Directed**: Behavior emerges from minimizing expected free energy
- 🔧 **Modular Design**: Composable components for different applications and complexity levels
- 📊 **Rich Visualization**: Tools for understanding agent behavior and internal dynamics
- 🤖 **Robotics Ready**: Direct integration with robotic systems and simulation environments
- 🔬 **Research-Oriented**: Implements both discrete and continuous formulations of active inference
- 📚 **Well-Documented**: Extensive references to foundational literature and mathematical derivations

## Implementation Highlights

### Three Formulations Supported
- **Discrete State-Space**: For bandits, navigation, and neuroscience tasks
- **Continuous Time**: For robot control with connections to classical control theory  
- **Deep Active Inference**: Scalable approaches using neural networks for complex environments

### Key Algorithms Implemented
- Variational message passing and belief propagation
- Expected free energy optimization
- Predictive coding with hierarchical prediction errors
- Kalman filtering and particle filtering for state estimation
- Policy optimization through active inference

## Research Applications

This framework enables cutting-edge research in:

**Robotics and Control**
- Autonomous navigation and exploration under uncertainty
- Adaptive robotic manipulation with proprioceptive feedback
- Multi-modal sensorimotor integration and coordination
- Biologically-inspired robot control architectures

**Cognitive Science and Neuroscience**
- Computational models of predictive processing in the brain
- Understanding attention, precision, and cognitive control
- Models of perception-action loops and embodied cognition
- Hierarchical cognitive architectures and consciousness models

**Machine Learning and AI**
- Principled exploration and uncertainty quantification
- Bayesian deep learning with active sampling
- Meta-learning and continual adaptation
- Multi-agent coordination and communication

**Computational Psychiatry**
- Models of psychiatric conditions through aberrant inference
- Understanding delusions, hallucinations, and anxiety disorders
- Therapeutic interventions based on precision and belief updating

## Contributing

We welcome contributions! Please see our contributing guidelines for:

- Code style and standards
- Testing requirements
- Documentation guidelines
- Research collaboration opportunities

## References

### Core Theory and Foundations
1. **Friston, K. (2010).** The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
2. **Friston, K. (2019).** A free energy principle for a particular physics. *arXiv preprint arXiv:1906.10184*. [Karl's magisterial monograph - most comprehensive FEP description]
3. **Friston, K., et al. (2017).** Active inference: a process theory. *Neural Computation*, 29(1), 1-49.
4. **Parr, T., Pezzulo, G., & Friston, K. J. (2022).** *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.

### Tutorials and Reviews
5. **Gershman, S. J. (2019).** What does the free energy principle tell us about the brain? *arXiv preprint arXiv:1901.07945*. [Great high-level introduction]
6. **Bogacz, R. (2017).** A tutorial on the free-energy framework for modelling perception and learning. *Journal of Mathematical Psychology*, 76, 198-211. [Excellent mathematical tutorial with MATLAB code]
7. **Buckley, C. L., et al. (2017).** The free energy principle for action and perception: A mathematical review. *Journal of Mathematical Psychology*, 81, 55-79. [Comprehensive mathematical walkthrough]
8. **Smith, R., Friston, K., & Whyte, C. (2021).** A Step-by-Step Tutorial on Active Inference and its Application to Empirical Data. *PsyArXiv*. [Detailed discrete-state-space tutorial with MATLAB code]

### Discrete State-Space Active Inference
9. **Da Costa, L., et al. (2020).** Active inference on discrete state-spaces: A synthesis. *Journal of Mathematical Psychology*, 99, 102447.
10. **Friston, K., et al. (2015).** Active inference and epistemic value. *Cognitive Neuroscience*, 6(4), 187-214. [Introduces epistemic foraging behavior]
11. **Friston, K., et al. (2020).** Sophisticated Inference. *arXiv preprint arXiv:2006.04120*. [Next-stage active inference with belief-dependent decisions]

### Continuous Time and Control Theory
12. **Friston, K., Daunizeau, J., & Kiebel, S. J. (2009).** Reinforcement learning or active inference? *PLoS ONE*, 4(7), e6421. [Earliest active inference paper]
13. **Baltieri, M., & Buckley, C. L. (2019).** PID control as a process of active inference with linear generative models. *Entropy*, 21(3), 257.
14. **Baltieri, M., & Buckley, C. L. (2020).** On Kalman-Bucy filters, linear quadratic control and active inference. *arXiv preprint arXiv:2005.06269*.

### Robotics Applications
15. **Pio-Lopez, L., et al. (2016).** Active inference and robot control: a case study. *Journal of The Royal Society Interface*, 13(122), 20160616.
16. **Oliver, G., Lanillos, P., & Cheng, G. (2019).** Active inference body perception and action for humanoid robots. *arXiv preprint arXiv:1906.03022*.
17. **Baltieri, M., & Buckley, C. L. (2017).** An active inference implementation of phototaxis. *Artificial Life Conference Proceedings*, 36-43. [Active inference in plants!]

### Deep Active Inference
18. **Ueltzhoffer, K. (2018).** Deep active inference. *Biological Cybernetics*, 112(6), 547-573. [First deep neural network + active inference combination]
19. **Tschantz, A., et al. (2020).** Reinforcement Learning through Active Inference. *arXiv preprint arXiv:2002.12636*. [EFE exploration in deep RL]
20. **Millidge, B. (2020).** Deep active inference as variational policy gradients. *Journal of Mathematical Psychology*, 96, 102348.

### Mathematical Foundations
21. **Friston, K. (2008).** Variational filtering. *NeuroImage*, 41(3), 747-766. [Foundational variational inference for dynamical systems]
22. **Friston, K., Trujillo-Barreto, N., & Daunizeau, J. (2008).** DEM: a variational treatment of dynamic systems. *NeuroImage*, 41(3), 849-885. [Extends predictive coding to generalized coordinates]
23. **Parr, T., & Friston, K. J. (2019).** Generalised free energy and active inference. *Biological Cybernetics*, 113(5-6), 495-513.

### Philosophical and Critical Analyses
24. **Aguilera, M., et al. (2021).** How particular is the physics of the Free Energy Principle? *arXiv preprint arXiv:2105.11203*. [Critical analysis of FEP claims]
25. **Andrews, M. (2020).** The Math is not the Territory: Navigating the Free Energy Principle. *PhilSci Archive*.
26. **Bruineberg, J., et al. (2020).** The Emperor's New Markov Blankets. *Behavioral and Brain Sciences*. [Critical examination of Markov blanket claims]

### Computational Resources
27. **BerenMillidge/FEP_Active_Inference_Papers** - Comprehensive paper repository: https://github.com/BerenMillidge/FEP_Active_Inference_Papers

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work is inspired by the theoretical foundations laid by **Karl Friston** and the active inference research community. We gratefully acknowledge:

- **Karl Friston** and colleagues at UCL for developing the Free Energy Principle and Active Inference
- **Beren Millidge** for the comprehensive [FEP_Active_Inference_Papers repository](https://github.com/BerenMillidge/FEP_Active_Inference_Papers)
- The **Active Inference Institute** and community researchers
- Contributors to open-source implementations: **PyMDP**, **SPM**, and other active inference tools
- **Christopher Buckley**, **Giovanni Pezzulo**, **Thomas Parr**, and other key researchers advancing the field

Special recognition to the multidisciplinary community spanning neuroscience, robotics, machine learning, and philosophy that continues to develop and apply these ideas.
