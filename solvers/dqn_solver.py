"""
Solver 3: Deep Q-Network (DQN) Solver

Uses a trained neural network to estimate Q-values for each possible
guess given the current board state.

Architecture:
    - Input: 417-dim state vector
    - Hidden: 2 layers of 512 neurons with ReLU
    - Output: 130-dim vector (26 letters × 5 positions)
    - Action selection: dot product output with one-hot word encodings

Training uses curated exploration:
    - Phase 1: random guesses drawn from Set A (narrowing words)
    - Phase 2: random guesses drawn from Set B (broader coverage)
    - Phase 3: random guesses from full vocabulary
    Target word is always random from the full list.

References:
    Anderson, B.J. & Meyer, J.G. (2022). arXiv:2202.00557.
    Ho, A. (2022). Wordle Solving with Deep Reinforcement Learning.
"""

import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from engine.wordle_env import load_word_list, filter_words
from engine.state_encoder import encode_state, encode_words_onehot, STATE_DIM


# --- Model Architecture ---

HIDDEN_DIM = 512
OUTPUT_DIM = 130  # 26 letters × 5 positions

if TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, state):
            return self.net(state)

    class ReplayBuffer:
        def __init__(self, capacity=100000):
            from collections import deque
            self.buffer = deque(maxlen=capacity)

        def push(self, state, action_idx, reward, next_state, done):
            self.buffer.append((state, action_idx, reward, next_state, done))

        def sample(self, batch_size):
            batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
            states, actions, rewards, next_states, dones = zip(*batch)
            return (
                torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32),
            )

        def __len__(self):
            return len(self.buffer)


# --- Solver ---

class DQNSolver:

    def __init__(self, model_path=None, word_list_path=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install: pip install torch")

        self.all_words = load_word_list(word_list_path)
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []
        self.turn = 0

        self.word_encodings = torch.tensor(
            encode_words_onehot(self.all_words), dtype=torch.float32
        )

        self.model = DQNNetwork()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            print(f"Loaded DQN model from {model_path}")
        else:
            if model_path:
                print("WARNING: No trained model loaded.")
        self.model.eval()

    def reset(self):
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []
        self.turn = 0

    def update(self, guess, feedback):
        self.guesses.append(guess.lower())
        self.feedbacks.append(feedback)
        self.turn += 1
        self.remaining = filter_words(self.remaining, guess, feedback)

    def get_guess(self, game_state=None):
        if game_state is not None:
            self._sync_from_state(game_state)

        if len(self.remaining) == 1:
            return self.remaining[0]

        state = encode_state(self.guesses, self.feedbacks, self.turn)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.model(state_tensor)
            scores = torch.matmul(output, self.word_encodings.T).squeeze(0)

        best_idx = scores.argmax().item()
        return self.all_words[best_idx]

    def _sync_from_state(self, game_state):
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []
        self.turn = 0
        for guess, feedback in zip(game_state["guesses"], game_state["feedbacks"]):
            self.update(guess, feedback)


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not installed.")
    else:
        solver = DQNSolver()
        print(f"DQN Solver: {len(solver.all_words)} words")
        print(f"Parameters: {sum(p.numel() for p in solver.model.parameters()):,}")
