# Wordle ML Project

A comparative study of heuristic and machine learning approaches to solving Wordle.

## Overview

This project implements and evaluates four different Wordle-solving strategies:

1. **Frequency Heuristic** — Scores words by letter frequency and positional frequency in the remaining word list
2. **Information Gain** — Selects the guess that minimizes the expected remaining word list size (entropy-based)
3. **Deep Q-Network (DQN)** — Reinforcement learning using a value-based approach
4. **Advantage Actor-Critic (A2C)** — Reinforcement learning using a policy-gradient approach

Results are benchmarked against the known optimal solution (Bertsimas & Paskov, 2024) which achieves an average of 3.421 guesses.

## Project Structure

```
wordle-ML-project/
├── README.md
├── wordle.txt                    # Word list (2,983 five-letter words)
├── engine/
│   ├── __init__.py
│   ├── wordle_env.py             # Core game engine
│   └── state_encoder.py          # Board state → vector encoding
├── solvers/
│   ├── __init__.py
│   ├── frequency_solver.py       # Solver 1: Letter frequency heuristic
│   ├── infogain_solver.py        # Solver 2: Information gain / entropy
│   ├── dqn_solver.py             # Solver 3: Deep Q-Network
│   └── a2c_solver.py             # Solver 4: Advantage Actor-Critic
├── models/
│   ├── dqn_model.pt              # Trained DQN weights (generated)
│   └── a2c_model.pt              # Trained A2C weights (generated)
├── notebooks/
│   └── evaluation.ipynb          # Training, evaluation, and comparison
└── web/                          # Interactive web demos (future)
```

## Usage

### Evaluate all solvers
Open `notebooks/evaluation.ipynb` in Jupyter or Google Colab.

### Use a solver directly
```python
from engine.wordle_env import WordleGame
from solvers.frequency_solver import FrequencySolver

game = WordleGame(target="crane")
solver = FrequencySolver("wordle.txt")

while not game.is_solved() and game.turn < 6:
    guess = solver.get_guess(game.get_state())
    feedback = game.make_guess(guess)
    solver.update(guess, feedback)
```

## References

1. Bhambri, S., Bhattacharjee, A., & Bertsekas, D. (2022). Reinforcement Learning Methods for Wordle: A POMDP/Adaptive Control Approach. arXiv:2211.10298.
2. Liu, C.-L. (2022). Using Wordle for Learning to Design and Compare Strategies. IEEE Conference on Games (CoG).
3. Bertsimas, D. & Paskov, A. (2024). An Exact and Interpretable Solution to Wordle. Operations Research, INFORMS.
4. Anderson, B.J. & Meyer, J.G. (2022). Finding the optimal human strategy for Wordle using maximum correct letter probabilities and reinforcement learning. arXiv:2202.00557.
5. Ho, A. (2022). Wordle Solving with Deep Reinforcement Learning.
