"""
Solver 3: Deep Q-Network (DQN) Solver

Uses a trained neural network to estimate Q-values for each possible
guess given the current board state. The word with the highest Q-value
is selected as the next guess.

Architecture:
    - Input: 417-dim state vector (encoded board state)
    - Hidden: 2 layers of 512 neurons with ReLU
    - Output: 130-dim vector (26 letters × 5 positions)
    - Action selection: dot product output with one-hot word encodings

Training is performed in the evaluation notebook.

References:
    Anderson, B.J. & Meyer, J.G. (2022). arXiv:2202.00557.
    Ho, A. (2022). Wordle Solving with Deep Reinforcement Learning.
"""

import os
import sys
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
        """
        Deep Q-Network for Wordle.
        
        Maps a 417-dim state vector to a 130-dim output.
        The output is dotted with one-hot word encodings to get
        Q-values for each word in the vocabulary.
        """
        
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
            """
            Args:
                state: tensor of shape (batch, STATE_DIM)
            Returns:
                tensor of shape (batch, OUTPUT_DIM=130)
            """
            return self.net(state)


# --- Solver ---

class DQNSolver:
    """
    DQN-based Wordle solver.
    
    Loads a pre-trained model and uses it to select guesses.
    """
    
    def __init__(self, model_path=None, word_list_path=None):
        """
        Args:
            model_path: path to trained .pt file
            word_list_path: path to wordle.txt
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQNSolver. Install with: pip install torch")
        
        self.all_words = load_word_list(word_list_path)
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []
        self.turn = 0
        
        # Pre-compute one-hot encodings for all words
        self.word_encodings = torch.tensor(
            encode_words_onehot(self.all_words), dtype=torch.float32
        )
        
        # Load model
        self.model = DQNNetwork()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded DQN model from {model_path}")
        else:
            print("WARNING: No trained model loaded. DQN solver will produce random results.")
        
        self.model.eval()
    
    def reset(self):
        """Reset solver state for a new game."""
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []
        self.turn = 0
    
    def update(self, guess, feedback):
        """Update state with a guess and its feedback."""
        self.guesses.append(guess.lower())
        self.feedbacks.append(feedback)
        self.turn += 1
        self.remaining = filter_words(self.remaining, guess, feedback)
    
    def get_guess(self, game_state=None):
        """
        Return the best guess according to the trained DQN.
        
        Args:
            game_state: optional dict from WordleGame.get_state()
        
        Returns:
            str: the recommended 5-letter guess
        """
        if game_state is not None:
            self._sync_from_state(game_state)
        
        # If only 1 word remains, guess it
        if len(self.remaining) == 1:
            return self.remaining[0]
        
        # Encode current state
        state = encode_state(self.guesses, self.feedbacks, self.turn)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Get Q-values
        with torch.no_grad():
            output = self.model(state_tensor)  # (1, 130)
        
        # Dot with word encodings to get score for each word
        scores = torch.matmul(output, self.word_encodings.T).squeeze(0)  # (n_words,)
        
        # Mask out words that aren't in remaining list (optional: can also allow exploration)
        # For inference, we prefer words from the remaining list
        remaining_set = set(self.remaining)
        mask = torch.tensor(
            [1.0 if w in remaining_set else 0.5 for w in self.all_words],
            dtype=torch.float32
        )
        scores = scores * mask
        
        # Pick the highest scoring word
        best_idx = scores.argmax().item()
        return self.all_words[best_idx]
    
    def _sync_from_state(self, game_state):
        """Rebuild internal state from a WordleGame state dict."""
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []
        self.turn = 0
        for guess, feedback in zip(game_state["guesses"], game_state["feedbacks"]):
            self.update(guess, feedback)


# --- Replay Buffer for Training ---

if TORCH_AVAILABLE:
    class ReplayBuffer:
        """Experience replay buffer for DQN training."""
        
        def __init__(self, capacity=100000):
            from collections import deque
            self.buffer = deque(maxlen=capacity)
        
        def push(self, state, action_idx, reward, next_state, done):
            self.buffer.append((state, action_idx, reward, next_state, done))
        
        def sample(self, batch_size):
            import random
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


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not installed. Run: pip install torch")
    else:
        # Test without trained model (will be random)
        solver = DQNSolver()
        print(f"DQN Solver initialized with {len(solver.all_words)} words")
        print(f"Word encodings shape: {solver.word_encodings.shape}")
        print(f"Model parameters: {sum(p.numel() for p in solver.model.parameters()):,}")
