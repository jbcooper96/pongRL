# Pong Reinforcement Learning Project

A comprehensive implementation of multiple reinforcement learning algorithms for training agents to play Atari Pong. This project includes implementations of Deep Q-Network (DQN), Policy Gradient, and Proximal Policy Optimization (PPO) algorithms.

## Overview

This project demonstrates different approaches to reinforcement learning by training AI agents to play the classic Atari game Pong. Each algorithm folder contains a complete implementation with training scripts, model definitions, and evaluation tools.

## Project Structure

```
pongv1/
├── DQN/                    # Deep Q-Network implementation
│   ├── agent.py           # DQN agent with experience replay
│   ├── qModel.py          # Q-network architecture
│   ├── rollout.py         # Training and evaluation script
│   ├── groupFramesWrapper.py  # Frame preprocessing wrapper
│   └── recordings/        # Video recordings of gameplay
├── policyGradient/        # Policy Gradient implementation
│   ├── agent.py           # Policy gradient agent
│   ├── policy.py          # Policy network architecture
│   ├── rollout.py         # Training and evaluation script
│   └── recordings/        # Video recordings of gameplay
├── PPO/                   # Proximal Policy Optimization implementation
│   ├── ppo.py             # Main PPO algorithm
│   ├── ppo_agent.py       # PPO agent implementation
│   ├── ppo_lstm.py        # LSTM variant of PPO
│   ├── models.py          # Neural network architectures
│   ├── rollout.py         # Training and evaluation script
│   ├── settings.py        # Device configuration
│   └── helpers.py         # Utility functions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pongv1
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Algorithms

### 1. Deep Q-Network (DQN)

A value-based method that learns to estimate Q-values for state-action pairs.

**Features:**
- Experience replay buffer
- Target network for stable training
- Frame stacking for temporal information
- Epsilon-greedy exploration

**Usage:**
```bash
cd DQN
# Train the agent
python rollout.py --train

# Test with visualization
python rollout.py --render --load

# Record gameplay videos
python rollout.py --train --record
```

### 2. Policy Gradient

A policy-based method that directly optimizes the policy using REINFORCE algorithm.

**Features:**
- Direct policy optimization
- Monte Carlo returns
- Stochastic policy with action sampling

**Usage:**
```bash
cd policyGradient
# Train the agent
python rollout.py --train

# Test the trained agent
python rollout.py --load
```

### 3. Proximal Policy Optimization (PPO)

An advanced policy gradient method with improved stability and sample efficiency.

**Features:**
- Clipped surrogate objective
- Value function estimation
- Generalized Advantage Estimation (GAE)
- Multiple epochs per batch
- Entropy regularization
- Wandb integration for experiment tracking

**Usage:**
```bash
cd PPO
# Train the agent
python rollout.py

# Test with visualization
python rollout.py --render --load

# Train with specific device
python rollout.py --device cuda

# Adjust number of environments
python rollout.py --env 16
```

## 🔧 Configuration

### PPO Hyperparameters

Key hyperparameters can be modified in [`PPO/rollout.py`](PPO/rollout.py):

- `learning_rate`: Learning rate for the optimizer (default: 1e-5)
- `batch_size`: Batch size for training (default: 40)
- `epochs`: Number of training epochs per update (default: 10)
- `entropy_coef`: Entropy coefficient for exploration (default: 0.01)
- `ENV_NUMBER`: Number of parallel environments (default: 10)

### Device Configuration

The project automatically detects and uses available hardware:
- CUDA GPU (if available)
- MPS (Apple Silicon)
- CPU (fallback)

Override device selection:
```bash
python rollout.py --device cuda  # Force CUDA
python rollout.py --device cpu   # Force CPU
python rollout.py --device mps   # Force MPS (Apple Silicon)
```

## 📊 Monitoring and Evaluation

### Weights & Biases Integration

PPO implementation includes Wandb integration for experiment tracking:

- Training metrics (rewards, loss, entropy)
- Hyperparameter logging
- Real-time visualization

### Model Checkpoints

Models are automatically saved during training:
- **DQN**: `qModel.pt`
- **Policy Gradient**: `policy.pt`
- **PPO**: `action.pt`, `value.pt`, `opt.pt`

### Video Recording

All implementations support video recording of gameplay:
- Recordings saved in respective `recordings/` folders
- Useful for visualizing agent behavior and progress

## 🎮 Game Environment

The project uses the Atari Pong environment with preprocessing:

- **Observation**: 84x84 grayscale frames
- **Frame Stacking**: 4 consecutive frames for temporal information
- **Action Space**: 6 discrete actions (NOOP, FIRE, UP, DOWN, etc.)
- **Reward**: +1 for scoring, -1 for opponent scoring, 0 otherwise

## Performance

Each algorithm has different characteristics:

- **DQN**: Stable but sample inefficient
- **Policy Gradient**: Simple but high variance
- **PPO**: Best balance of stability and efficiency

Expected training time varies by algorithm and hardware configuration.

## Development

### Adding New Algorithms

1. Create a new directory for your algorithm
2. Implement the core algorithm logic
3. Create training/evaluation scripts
4. Add model saving/loading functionality
5. Update this README

### Debugging

Common issues and solutions:

- **CUDA out of memory**: Reduce batch size or number of environments
- **Slow training**: Ensure GPU is being used, check device configuration
- **NaN losses**: Reduce learning rate or check gradient clipping

## Dependencies

- `ale-py`: Atari Learning Environment
- `gymnasium`: OpenAI Gym interface
- `opencv-python`: Image processing
- `wandb`: Experiment tracking
- `stable-baselines3`: Environment utilities(Frame stacking wrapper)
- `torch`: Deep learning framework

## Acknowledgments

- OpenAI Gym/Gymnasium for the environment interface
- Stable Baselines3 for environment utilities
- The reinforcement learning community for algorithm implementations and insights

---

**Note**: This project is for educational and research purposes. Training times and performance may vary based on hardware configuration and hyperparameter settings.