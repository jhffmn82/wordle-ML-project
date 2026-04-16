# Wordle ML: Comparing Heuristics and Machine Learning for Wordle Solving

**Justin Hoffman**  
UIUC MCS / ISU MSCS Graduate Student  
IT 448 (Graduate Machine Learning), Illinois State University, Spring 2026  

[📄 Research Paper](https://github.com/jhffmn82/wordle-ML-project/blob/main/paper/wordle_research_paper_v2.pdf) · [🌐 Live Demo](https://jhffmn82-wordle-ml.hf.space) · [📓 Evaluation Notebook](notebooks/evaluation.ipynb)

---

## Overview

This project compares six algorithmic strategies for solving Wordle, spanning classical heuristics, reinforcement learning, and deep reinforcement learning, against the provably optimal solution (Bertsimas & Paskov). All solvers are evaluated exhaustively on the official 2,315-word answer set under a closed-vocabulary setting.

The central question: when does a more sophisticated model actually help, and when does understanding the structure of the problem matter more than model complexity?

---

## Results

| Solver | Type | Win Rate | Avg Guesses | Speed |
|--------|------|----------|-------------|--------|
| Optimal (Bertsimas) | Exact DP | 100.0% | 3.421 | — |
| Rollout (POMDP) | DP / Lookahead | 100.0% | 3.477 | 182.7 it/s |
| Frequency Heuristic | Heuristic | 100.0% | 3.575 | 19.5 it/s |
| Information Gain (Minimax) | Heuristic | 100.0% | 3.644 | 2.6 it/s |
| Tabular Q-Learning | RL | 99.0% | 3.651 | 70.5 it/s |
| DQN v2 (Teacher-Guided) | Deep RL | 97.7% | 3.678 | 135.6 it/s |
| DQN v1 (Pure) | Deep RL | 67.2% | 4.582 | 114.4 it/s |
| Ho (2022) reported | Deep RL | ~98% | ~4.1 | — |

---

## Key Findings

- **Guided RL works, unguided RL does not.**  
  DQN v2 significantly outperforms DQN v1 (97.7% vs 67.2%) by replacing random exploration with a rollout-based teacher.

- **Simple heuristics are extremely strong.**  
  The original 2022 frequency solver achieves 100% win rate at 3.575 average guesses, outperforming all learned models.

- **Problem structure dominates model complexity.**  
  The rollout solver performs near-optimally using memoization and explicit lookahead.

- **The effective state space is small.**  
  Only ~331 unique states are encountered under strong play, despite the NP-hard formulation of the general problem.

---

## The Solvers

1. **Frequency Heuristic**  
   Scores words by letter frequency with 3× weighting for positional frequency.  
   100% win rate, 3.575 average guesses.

2. **Information Gain (Minimax)**  
   Selects the guess that minimizes the worst-case partition of remaining candidates.  
   100% win rate, 3.644 average guesses.

3. **DQN v1 (Pure)**  
   Deep Q-Network following Ho (2022): 417 → 512 → 512 → 130 architecture.  
   Trained with epsilon-greedy exploration.  
   Suffers from distribution shift and unstable learning.  
   67.2% win rate.

4. **DQN v2 (Teacher-Guided)**  
   Same architecture as v1, but uses the rollout solver as a teacher during training.  
   Improves exploration and stabilizes learning.  
   97.7% win rate, 3.678 average guesses.

5. **Tabular Q-Learning**  
   Learns which of five heuristic strategies to apply based on game state.  
   Only ~19 reachable states.  
   99.0% win rate.

6. **Rollout (POMDP)**  
   One-step lookahead policy improvement over the frequency heuristic.  
   Memoized cache captures the effective game tree.  
   100% win rate, 3.477 average guesses.

---

## Project Structure

```
wordle-ML-project/
├── engine/
│   ├── wordle_env.py          # Game engine, feedback, filtering
│   ├── state_encoder.py       # 417-dim state vector for DQN
│   └── word_lists.py          # Curated curriculum from solver analysis
├── solvers/
│   ├── frequency_solver.py    # Letter + positional frequency scoring
│   ├── infogain_solver.py     # Minimax partitioning
│   ├── dqn_solver.py          # 512×512 MLP, 130-dim output
│   ├── tabular_q_solver.py    # Strategy-level RL
│   └── rollout_solver.py      # Memoized rollout with cache
├── models/                    # Trained weights and caches
├── web/
│   ├── app.py                 # Flask backend
│   ├── solver_adapter.py      # Unified solver interface
│   └── templates/
│       └── index.html         # Frontend UI
├── notebooks/
│   └── evaluation.ipynb       # Training & evaluation
├── paper/
│   └── wordle_research_paper_v2.pdf
├── wordle.txt                 # Official answer list (2,315 words)
├── Dockerfile                 # Deployment (Hugging Face Spaces)
└── requirements.txt
```

---

## Web App

The live demo supports three modes:

- **Solver Assistant**  
  Play Wordle with AI assistance. Enter guesses and feedback to receive suggestions.

- **Autoplay**  
  Watch any solver play a full game step by step.

- **About**  
  Includes methodology, results, and solver comparisons.

Hosted on Hugging Face Spaces using Docker.

---

## Inspiration

This project originated from a Wordle solver I built in 2022 as a personal challenge.  
It was later extended into a formal machine learning study inspired by Andrew Ho’s work on deep reinforcement learning for Wordle.

---

## References

See full references in the research paper:  
👉 https://github.com/jhffmn82/wordle-ML-project/blob/main/paper/wordle_research_paper_v2.pdf
