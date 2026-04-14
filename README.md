---
title: Wordle ML
emoji: 🟩
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Wordle ML: Comparing Heuristics and Machine Learning for Wordle Solving

**Justin Hoffman** UIUC MCS / ISU MSCS Graduate Student
IT 448 (Graduate Machine Learning), Illinois State University, Spring 2026

[🌐 Live Demo](https://jhffmn82-wordle-ml.hf.space) · [📓 Evaluation Notebook](notebooks/evaluation.ipynb)

## Overview

This project compares six algorithmic strategies for solving Wordle, spanning classical heuristics, reinforcement learning, and deep RL, against the provably optimal solution (Bertsimas & Paskov, 2024). All solvers are evaluated against the official Wordle answer list of 2,315 words.

The central question: when does a more sophisticated method actually help, and when does understanding the problem structure matter more than model complexity?

## Results

Solver	Type	Win Rate	Avg Guesses	Speed

Optimal (Bertsimas 2024)	Exact DP	100.0%	3.421	—

Rollout (POMDP)	DP / Lookahead	100.0%	3.477	182.7 it/s

Frequency Heuristic	Heuristic	100.0%	3.575	19.5 it/s

Information Gain (Minimax)	Heuristic	100.0%	3.644	2.6 it/s

Tabular Q-Learning	RL	99.0%	3.651	70.5 it/s

DQN v2 (Teacher-Guided)	Deep RL	97.7%	3.678	135.6 it/s

DQN v1 (Pure)	Deep RL	67.2%	4.582	114.4 it/s

Ho (2022) reported	Deep RL	~98%	~4.1	—

## Key Findings

DQN v2 surpassed Ho's reported deep RL results, 3.678 avg guesses vs Ho's ~4.1, using teacher-guided exploration from the rollout solver and Double DQN.

Pure DQN fails catastrophically, DQN v1 achieved only 67.2% win rate due to distribution shift during training.

Simple heuristics remain remarkably competitive, my original 2022 frequency solver achieved 100% win rate at 3.575 avg guesses, beating every ML approach.

Problem structure matters more than model complexity, the rollout solver (a cached lookup table) achieved the best non-optimal performance. Wordle's game tree has only a few hundred unique reachable states.

## The Solvers

1. Frequency Heuristic: My original 2022 solver. Scores words by letter frequency weighted 3× for positional frequency. 100% win rate, 3.575 avg guesses.

2. Information Gain (Minimax): Picks the guess that minimizes the worst-case partition of remaining words. 100% win rate, 3.644 avg guesses.

3. DQN v1 (Pure): Deep Q-Network following Ho (2022): 417→512→512→130 architecture with 4-tier curated exploration curriculum. Training collapses due to distribution shift. 67.2% win rate.

4. DQN v2 (Teacher-Guided): Same architecture, trained with the rollout solver as a live teacher using Double DQN. +20 reward bonus for matching teacher moves. 97.7% win rate, 3.678 avg guesses.

5. Tabular Q-Learning: Following Anderson & Meyer (2022). Learns which of 5 heuristic strategies to use at each game state. Only ~19 reachable states. 99.0% win rate.

6. Rollout (POMDP): Following Bhambri et al. (2022). One-step lookahead policy improvement over the frequency heuristic. Memoized cache covers the full game tree. 100% win rate, 3.477 avg guesses.

## Project Structure

```
wordle-ML-project/
├── engine/
│   ├── wordle_env.py          # Game engine, feedback, filtering
│   ├── state_encoder.py       # 417-dim state vector for DQN
│   └── word_lists.py          # Curated curriculum from solver analysis
├── solvers/
│   ├── frequency_solver.py    # Letter freq + positional freq scoring
│   ├── infogain_solver.py     # Minimax worst-case partition
│   ├── dqn_solver.py          # 512×512 MLP, 130-dim output
│   ├── tabular_q_solver.py    # 5 strategies, ~19 reachable states
│   └── rollout_solver.py      # Memoized rollout with disk cache
├── models/                    # Trained weights and caches
├── web/
│   ├── app.py                 # Flask web server
│   ├── solver_adapter.py      # Uniform solver interface for web API
│   └── templates/
│       └── index.html         # Single-page app frontend
├── notebooks/
│   └── evaluation.ipynb       # Training & evaluation (Google Colab)
├── wordle.txt                 # 2,315 official Wordle answer words
├── Dockerfile                 # Hugging Face Spaces deployment
└── requirements.txt
```

## Web App

The live demo has three modes:

Solver Assistant: Play Wordle with AI help. Enter guesses, click tiles to set feedback colors, and get suggestions from any of the 6 solvers.

Autoplay: Enter a target word and watch a solver play the game step by step with animated tile reveals.

About: Full results, methodology, solver descriptions, and charts.

Hosted on Hugging Face Spaces via Docker.

## Inspiration

This project started with a Python Wordle solver I built in 2022 as a fun challenge from a coworker. I was also heavily inspired by Andrew Ho's "Wordle Solving with Deep Reinforcement Learning", which provided the DQN architecture (417-dim state, 130-dim output dot-producted with word encodings) that I adapted for this project.

## References

Anderson, B.J. & Meyer, J.G. (2022). Finding the optimal human strategy for Wordle. arXiv:2202.00557.

Bertsimas, D. & Paskov, A. (2024). An Exact and Interpretable Solution to Wordle. Operations Research, 72(6), 2319–2332.

Bhambri, S., Bhattacharjee, A. & Bertsekas, D.P. (2022). RL Methods for Wordle: A POMDP/Adaptive Control Approach. arXiv:2211.10298.

Ho, A. (2022). Solving Wordle with Reinforcement Learning.

Liu, C.-L. (2022). Using Wordle for Learning to Design and Compare Strategies. IEEE Conference on Games (CoG).

Lokshtanov, D. & Subercaseaux, B. (2022). Wordle is NP-Hard. arXiv:2203.16713.

Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature, 518, 529–533.
