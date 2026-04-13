"""
Solver: Deep Q-Network (DQN)

A neural network that maps Wordle game states to word scores.

Architecture (following Ho, 2022):
    Input:  417-dim state vector (letter-position-status history + absent flags + turn)
    Hidden: 512 → ReLU → 512 → ReLU
    Output: 130-dim vector (26 letters × 5 positions)

Action selection:
    Each vocabulary word is encoded as a 130-dim one-hot vector.  The network
    output is scored against every word encoding via dot product, and the
    highest-scoring word is selected.  Words already guessed this game are
    masked out so the model never repeats a guess.

References:
    Ho, A. (2022). Solving Wordle with Reinforcement Learning.
    Mnih, V. et al. (2015). Human-level control through deep RL. Nature 518.
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


HIDDEN_DIM = 512
OUTPUT_DIM = 130  # 26 letters × 5 positions


# ---------------------------------------------------------------------------
# Network and replay buffer (used by both solver and training code)
# ---------------------------------------------------------------------------

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
        """Simple uniform experience replay buffer."""

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


# ---------------------------------------------------------------------------
# Solver (inference)
# ---------------------------------------------------------------------------

class DQNSolver:
    """
    Wordle solver backed by a trained DQN.

    At each turn the model scores every word in the vocabulary via dot product
    with its 130-dim output.  Already-guessed words are masked to -inf so the
    model never wastes a turn repeating a guess.
    """

    def __init__(self, model_path=None, word_list_path=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install: pip install torch")

        self.all_words = load_word_list(word_list_path)
        self.word_to_idx = {w: i for i, w in enumerate(self.all_words)}

        self.word_encodings = torch.tensor(
            encode_words_onehot(self.all_words), dtype=torch.float32
        )

        self.model = DQNNetwork()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            print(f"Loaded DQN model from {model_path}")
        elif model_path:
            print("WARNING: No trained model found at", model_path)

        self.model.eval()
        self.reset()

    def reset(self):
        """Clear game state for a new game."""
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []
        self.turn = 0

    def update(self, guess, feedback):
        """Record a guess and its feedback, prune remaining candidates."""
        self.guesses.append(guess.lower())
        self.feedbacks.append(feedback)
        self.turn += 1
        self.remaining = filter_words(self.remaining, guess, feedback)

    def get_guess(self, game_state=None):
        """
        Select the best word according to the Q-network.

        If only one candidate remains, return it directly.  Otherwise score
        all words, mask out previous guesses, and take the argmax.
        """
        if game_state is not None:
            self._sync_from_state(game_state)

        if len(self.remaining) == 1:
            return self.remaining[0]

        state = encode_state(self.guesses, self.feedbacks, self.turn)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.model(state_tensor)
            scores = torch.matmul(output, self.word_encodings.T).squeeze(0)

        # Mask out already-guessed words so they are never selected
        for guess in self.guesses:
            if guess in self.word_to_idx:
                scores[self.word_to_idx[guess]] = -float("inf")

        best_idx = scores.argmax().item()
        return self.all_words[best_idx]

    def _sync_from_state(self, game_state):
        """Rebuild solver state from an external game_state dict."""
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []
        self.turn = 0
        for guess, feedback in zip(game_state["guesses"], game_state["feedbacks"]):
            self.update(guess, feedback)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not installed.")
    else:
        solver = DQNSolver()
        print(f"DQN Solver: {len(solver.all_words)} words, state_dim={STATE_DIM}")
        print(f"Parameters: {sum(p.numel() for p in solver.model.parameters()):,}")

        # Verify guess masking works
        solver.reset()
        solver.guesses = ["slate"]
        g = solver.get_guess()
        assert g != "slate", "Mask failed: model re-guessed 'slate'"
        print(f"Mask check passed (first guess after 'slate': {g})")
