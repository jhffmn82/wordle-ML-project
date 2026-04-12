# Wordle ML Project

A comparative study of heuristic and machine learning approaches to solving Wordle.

## Overview

This project implements and evaluates five different Wordle-solving strategies:

1. **Frequency Heuristic** — Scores words by letter frequency and positional frequency
2. **Information Gain (Minimax)** — Selects the guess that minimizes worst-case remaining words
3. **Deep Q-Network (DQN)** — Neural network trained with curated exploration curriculum
4. **Tabular Q-Learning** — Learns which strategy to use at each game state
5. **Rollout** — One-step lookahead that improves the info gain heuristic

Results are benchmarked against the known optimal solution (Bertsimas & Paskov, 2024) which achieves an average of 3.421 guesses with 100% win rate.

## Project Structure

```
wordle-ML-project/
├── README.md
├── wordle.txt                        # Word list (2,983 five-letter words)
├── requirements.txt
├── engine/
│   ├── wordle_env.py                 # Core game engine
│   ├── state_encoder.py              # Board state → vector encoding (for DQN)
│   └── word_lists.py                 # Curated word sets from solver analysis
├── solvers/
│   ├── frequency_solver.py           # Solver 1: Letter frequency heuristic
│   ├── infogain_solver.py            # Solver 2: Minimax information gain
│   ├── dqn_solver.py                 # Solver 3: Deep Q-Network
│   ├── tabular_q_solver.py           # Solver 4: Tabular Q-Learning
│   └── rollout_solver.py             # Solver 5: Rollout with lookahead
├── models/
│   ├── dqn_model.pt                  # Trained DQN weights (generated)
│   └── q_table.pkl                   # Trained Q-table (generated)
└── notebooks/
    └── evaluation.ipynb              # Training, evaluation, and comparison
```

## References

1. Bhambri, S., Bhattacharjee, A., & Bertsekas, D. (2022). *Reinforcement Learning Methods for Wordle: A POMDP/Adaptive Control Approach.* arXiv:2211.10298.
2. Liu, C.-L. (2022). *Using Wordle for Learning to Design and Compare Strategies.* IEEE Conference on Games.
3. Bertsimas, D. & Paskov, A. (2024). *An Exact and Interpretable Solution to Wordle.* Operations Research, INFORMS.
4. Anderson, B.J. & Meyer, J.G. (2022). *Finding the optimal human strategy for Wordle.* arXiv:2202.00557.
5. Ho, A. (2022). *Wordle Solving with Deep Reinforcement Learning.*
