"""
Solver 4: Advantage Actor-Critic (A2C) Solver

Uses a trained actor-critic network to select guesses.
- Actor: outputs probability distribution over words (policy)
- Critic: estimates value of current state

Architecture:
    - Shared backbone: 417-dim → 512 → 512
    - Actor head: 512 → 130 (letter-position preferences)
    - Critic head: 512 → 1 (state value estimate)

Training is performed in the evaluation notebook.

References:
    Bhambri, S. et al. (2022). RL Methods for Wordle: A POMDP/Adaptive Control Approach.
    Ho, A. (2022). Wordle Solving with Deep Reinforcement Learning.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from engine.wordle_env import load_word_list, filter_words
from engine.state_encoder import encode_state, encode_words_onehot, STATE_DIM


# --- Model Architecture ---

HIDDEN_DIM = 512
OUTPUT_DIM = 130  # 26 letters × 5 positions

if TORCH_AVAILABLE:
    class A2CNetwork(nn.Module):
        """
        Actor-Critic network for Wordle.
        
        Shared backbone feeds into:
        - Actor head: produces letter-position scores (dotted with word encodings for policy)
        - Critic head: produces scalar state value estimate
        """
        
        def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
            super().__init__()
            
            # Shared backbone
            self.backbone = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            
            # Actor head (policy)
            self.actor = nn.Linear(hidden_dim, output_dim)
            
            # Critic head (value function)
            self.critic = nn.Linear(hidden_dim, 1)
        
        def forward(self, state):
            """
            Args:
                state: tensor of shape (batch, STATE_DIM)
            Returns:
                actor_out: tensor of shape (batch, OUTPUT_DIM=130)
                value: tensor of shape (batch, 1)
            """
            features = self.backbone(state)
            actor_out = self.actor(features)
            value = self.critic(features)
            return actor_out, value
        
        def get_action_and_value(self, state, word_encodings, valid_mask=None):
            """
            Get action probabilities and state value.
            
            Args:
                state: tensor (batch, STATE_DIM)
                word_encodings: tensor (n_words, 130)
                valid_mask: optional tensor (n_words,) for masking invalid words
            
            Returns:
                probs: tensor (batch, n_words) action probabilities
                value: tensor (batch, 1) state value
            """
            actor_out, value = self.forward(state)
            
            # Score each word: dot product of actor output with word encodings
            scores = torch.matmul(actor_out, word_encodings.T)  # (batch, n_words)
            
            # Apply valid word mask if provided
            if valid_mask is not None:
                scores = scores + (1 - valid_mask) * (-1e9)
            
            probs = F.softmax(scores, dim=-1)
            return probs, value


# --- Solver ---

class A2CSolver:
    """
    A2C-based Wordle solver.
    
    Loads a pre-trained actor-critic model and uses the actor (policy)
    to select guesses.
    """
    
    def __init__(self, model_path=None, word_list_path=None):
        """
        Args:
            model_path: path to trained .pt file
            word_list_path: path to wordle.txt
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for A2CSolver. Install with: pip install torch")
        
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
        self.model = A2CNetwork()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded A2C model from {model_path}")
        else:
            print("WARNING: No trained model loaded. A2C solver will produce random results.")
        
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
        Return the best guess according to the trained A2C policy.
        
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
        
        # Get action probabilities from actor
        with torch.no_grad():
            probs, value = self.model.get_action_and_value(
                state_tensor, self.word_encodings
            )
        
        # During inference, pick the most probable action (greedy)
        best_idx = probs.squeeze(0).argmax().item()
        return self.all_words[best_idx]
    
    def _sync_from_state(self, game_state):
        """Rebuild internal state from a WordleGame state dict."""
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []
        self.turn = 0
        for guess, feedback in zip(game_state["guesses"], game_state["feedbacks"]):
            self.update(guess, feedback)


# --- Rollout Storage for Training ---

if TORCH_AVAILABLE:
    class RolloutStorage:
        """Stores episode data for A2C training."""
        
        def __init__(self):
            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.log_probs = []
            self.dones = []
        
        def push(self, state, action, reward, value, log_prob, done):
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.dones.append(done)
        
        def compute_returns(self, gamma=0.99):
            """Compute discounted returns."""
            returns = []
            R = 0
            for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
                if done:
                    R = 0
                R = reward + gamma * R
                returns.insert(0, R)
            return torch.tensor(returns, dtype=torch.float32)
        
        def clear(self):
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.values.clear()
            self.log_probs.clear()
            self.dones.clear()
        
        def __len__(self):
            return len(self.states)


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not installed. Run: pip install torch")
    else:
        # Test without trained model (will be random)
        solver = A2CSolver()
        print(f"A2C Solver initialized with {len(solver.all_words)} words")
        print(f"Word encodings shape: {solver.word_encodings.shape}")
        print(f"Model parameters: {sum(p.numel() for p in solver.model.parameters()):,}")
