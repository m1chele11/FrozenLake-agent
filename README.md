# Q-Learning on FrozenLake

This project implements **Q-Learning** on the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment from **Gymnasium**.  
The agent learns to navigate an 8x8 frozen lake grid to reach the goal without falling into holes.

---

## Features
- Q-Learning with adjustable hyperparameters:
  - Learning rate (α)
  - Discount factor (γ)
  - Exploration rate (ε) with decay
- Option to render the environment visually with `pygame`
- Tracks rewards over episodes and plots training progress
- Saves:
  - `frozenlake.png` → training reward plot
  - `frozenlake.pkl` → learned Q-table (for reuse)

---

## Requirements
Install dependencies (inside your virtual environment):
```bash
pip install gymnasium[toy-text] matplotlib numpy
