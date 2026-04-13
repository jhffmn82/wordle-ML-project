"""
Solver 3: Deep Q-Network (DQN) Solver

Updated design:
- Supports optional curated opening constraints
- Turn 1 can be restricted to set1
- Turn 2 can be restricted to set2
- Later turns are unrestricted

This keeps the existing architecture and 130-dim word encoding, but prevents
degenerate opening moves such as BIDDY from dominating inference.
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


class DQNSolver:
    def __init__(self, model_path=None, word_list_path=None, set1=None, set2=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install: pip install torch")

        self.all_words = load_word_list(word_list_path)
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []
        self.turn = 0

        self.set1 = list(set1) if set1 is not None else None
        self.set2 = list(set2) if set2 is not None else None

        self.word_to_idx = {w: i for i, w in enumerate(self.all_words)}

        self.word_encodings = torch.tensor(
            encode_words_onehot(self.all_words), dtype=torch.float32
        )

        self.model = DQNNetwork()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            print(f"Loaded DQN model from {model_path}")
        elif model_path:
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

    def _allowed_words_for_turn(self):
        """
        Hard opening constraints:
        - turn 0: restrict to set1 if provided
        - turn 1: prefer set2 ∩ remaining; else set2 if provided
        - later: unrestricted
        """
        if self.turn == 0 and self.set1:
            allowed = [w for w in self.set1 if w in self.word_to_idx]
            if allowed:
                return allowed

        if self.turn == 1 and self.set2:
            allowed = [w for w in self.set2 if w in set(self.remaining)]
            if allowed:
                return allowed
            allowed = [w for w in self.set2 if w in self.word_to_idx]
            if allowed:
                return allowed

        return self.all_words

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

        allowed_words = self._allowed_words_for_turn()

        # Masked argmax over allowed words only
        best_word = allowed_words[0]
        best_score = -float("inf")
        for word in allowed_words:
            idx = self.word_to_idx[word]
            score = scores[idx].item()
            if score > best_score:
                best_score = score
                best_word = word

        return best_word

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
