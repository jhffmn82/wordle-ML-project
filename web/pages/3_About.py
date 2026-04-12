"""
About — project writeup, results, and charts.
"""

import streamlit as st

st.set_page_config(page_title="About", page_icon="📊", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;900&display=swap');

.section-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    margin-top: 1.5em;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 📊 About This Project")

st.markdown("""
### Overview

A comparative study of heuristic and machine learning approaches to solving Wordle,
developed as a graduate ML course project. We implement and evaluate six solvers
ranging from simple letter frequency heuristics to deep reinforcement learning,
benchmarked against the known optimal solution.

### Solvers

| Solver | Type | Win Rate | Avg Guesses |
|--------|------|----------|-------------|
| Frequency Heuristic | Heuristic | 99.8% | 3.707 |
| Information Gain | Heuristic | 100.0% | 3.728 |
| Tabular Q-Learning | RL (Learned) | 98.4% | 3.812 |
| Rollout (POMDP) | RL (Planning) | 100.0% | 3.568 |
| DQN v1 (Pure) | Deep RL | 64.4% | 4.536 |
| DQN v2 (Shaped) | Deep RL | TBD | TBD |
| **Optimal** | **Exact DP** | **100%** | **3.421** |

### Key Findings

**Understanding problem structure matters more than model complexity.** Wordle's
effective state space collapses to just 431 reachable game states under a good policy —
trivially solved by dynamic programming but intractable for deep RL's exploration-based
learning. A 543,000-parameter neural network (DQN) was our worst performer, while a
431-entry lookup table (rollout cache) achieved near-optimal results.

### References

1. Bhambri, S., Bhattacharjee, A., & Bertsekas, D. (2022). *RL Methods for Wordle: A POMDP/Adaptive Control Approach.* arXiv:2211.10298.
2. Liu, C.-L. (2022). *Using Wordle for Learning to Design and Compare Strategies.* IEEE Conference on Games.
3. Bertsimas, D. & Paskov, A. (2024). *An Exact and Interpretable Solution to Wordle.* Operations Research, INFORMS.
4. Anderson, B.J. & Meyer, J.G. (2022). *Finding the optimal human strategy for Wordle.* arXiv:2202.00557.
5. Ho, A. (2022). *Wordle Solving with Deep Reinforcement Learning.*

### Links

- [GitHub Repository](https://github.com/jhffmn82/wordle-ML-project)
- [Evaluation Notebook](https://github.com/jhffmn82/wordle-ML-project/blob/main/notebooks/evaluation.ipynb)
""")
