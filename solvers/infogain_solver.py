"""
Solver 2: Information Gain / Minimax Solver

Strategy: For each possible guess (all N words), simulate it against every
remaining candidate target. Group the remaining words by feedback pattern.
Pick the guess that minimizes the worst-case (largest) group size.

This is a minimax approach: minimize the maximum remaining possibilities.

OPTIMIZED: Pre-computes an NxN feedback lookup table at init (~17MB for 3k words).
After that, scoring each candidate guess is a single numpy array slice + unique.

Complexity per turn: O(N * R) where N = total words, R = remaining words.
With precomputed table, this is fast numpy operations.

References:
    Liu, C.-L. (2022). Using Wordle for Learning to Design and Compare Strategies.
    Bhambri, S. et al. (2022). RL Methods for Wordle: A POMDP/Adaptive Control Approach.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.wordle_env import load_word_list, get_feedback, filter_words


def _feedback_to_id(feedback):
    """Encode a feedback list [0-2, 0-2, 0-2, 0-2, 0-2] as a single int 0-242."""
    return feedback[0]*81 + feedback[1]*27 + feedback[2]*9 + feedback[3]*3 + feedback[4]


def _compute_feedback_fast(guess, target):
    """Compute feedback and return as single int ID."""
    feedback = [0] * 5
    target_remaining = list(target)
    
    for i in range(5):
        if guess[i] == target[i]:
            feedback[i] = 2
            target_remaining[i] = None
    
    for i in range(5):
        if feedback[i] == 2:
            continue
        if guess[i] in target_remaining:
            feedback[i] = 1
            target_remaining[target_remaining.index(guess[i])] = None
    
    return feedback[0]*81 + feedback[1]*27 + feedback[2]*9 + feedback[3]*3 + feedback[4]


class InfoGainSolver:
    """
    Minimax information gain solver with precomputed feedback table.
    """
    
    def __init__(self, word_list_path=None):
        self.all_words = load_word_list(word_list_path)
        self.n_words = len(self.all_words)
        self.word_to_idx = {w: i for i, w in enumerate(self.all_words)}
        self.remaining_mask = np.ones(self.n_words, dtype=bool)
        
        # Pre-compute the full NxN feedback table
        print(f"Pre-computing feedback table ({self.n_words} x {self.n_words})...", flush=True)
        t0 = time.time()
        self.feedback_table = self._build_feedback_table()
        print(f"  Done in {time.time()-t0:.1f}s  "
              f"({self.feedback_table.nbytes / 1024 / 1024:.1f} MB)")
    
    def _build_feedback_table(self):
        """
        Build NxN table: table[guess_idx, target_idx] = feedback_id (0-242).
        """
        n = self.n_words
        table = np.zeros((n, n), dtype=np.uint16)
        
        for i in range(n):
            g = self.all_words[i]
            for j in range(n):
                table[i, j] = _compute_feedback_fast(g, self.all_words[j])
        
        return table
    
    def reset(self):
        """Reset for a new game."""
        self.remaining_mask = np.ones(self.n_words, dtype=bool)
    
    def update(self, guess, feedback):
        """
        Filter remaining words based on guess and feedback.
        Uses the precomputed table: a word W is still valid iff
        feedback_table[guess_idx, W_idx] == observed_feedback_id.
        """
        guess = guess.lower()
        fb_id = _feedback_to_id(feedback)
        
        guess_idx = self.word_to_idx.get(guess)
        if guess_idx is not None:
            consistent = self.feedback_table[guess_idx, :] == fb_id
            self.remaining_mask &= consistent
        else:
            # Fallback for unknown words
            remaining = [self.all_words[i] for i in np.where(self.remaining_mask)[0]]
            remaining = filter_words(remaining, guess, feedback)
            remaining_set = set(remaining)
            self.remaining_mask = np.array([w in remaining_set for w in self.all_words])
    
    def get_guess(self, game_state=None):
        """
        Pick the guess that minimizes the worst-case remaining word count.
        
        Always considers ALL words as candidate guesses (not just remaining).
        For each candidate, groups remaining words by feedback pattern and
        takes the largest group as the worst case.
        """
        if game_state is not None:
            self._sync_from_state(game_state)
        
        remaining_indices = np.where(self.remaining_mask)[0]
        n_remaining = len(remaining_indices)
        
        # Trivial cases
        if n_remaining <= 1:
            return self.all_words[remaining_indices[0]]
        if n_remaining == 2:
            return self.all_words[remaining_indices[0]]
        
        # Score ALL words as candidate guesses
        best_idx = remaining_indices[0]
        best_worst_case = n_remaining + 1  # start worse than possible
        
        for gi in range(self.n_words):
            # Get feedback patterns for this guess against all remaining targets
            patterns = self.feedback_table[gi, remaining_indices]
            
            # Find the largest group (worst case)
            _, counts = np.unique(patterns, return_counts=True)
            worst_case = counts.max()
            
            # Tiebreak: prefer words that are in the remaining list
            # (if worst_case ties, a remaining word might get lucky and solve it)
            is_remaining = self.remaining_mask[gi]
            
            if worst_case < best_worst_case or (worst_case == best_worst_case and is_remaining):
                best_worst_case = worst_case
                best_idx = gi
        
        return self.all_words[best_idx]
    
    def _sync_from_state(self, game_state):
        """Rebuild internal state from a game state dict."""
        self.remaining_mask = np.ones(self.n_words, dtype=bool)
        for guess, feedback in zip(game_state["guesses"], game_state["feedbacks"]):
            self.update(guess, feedback)


def play_game(solver, target, word_list=None, verbose=False):
    """
    Play a complete game using the solver.
    
    Returns:
        int: number of guesses taken (7 if failed)
    """
    from engine.wordle_env import WordleGame
    
    game = WordleGame(target=target, word_list=solver.all_words)
    solver.reset()
    
    while not game.is_over():
        t0 = time.time()
        guess = solver.get_guess()
        guess_time = time.time() - t0
        
        feedback = game.make_guess(guess)
        solver.update(guess, feedback)
        
        if verbose:
            symbols = ["⬛", "🟨", "🟩"]
            display = " ".join(symbols[f] for f in feedback)
            remaining = int(solver.remaining_mask.sum())
            print(f"  Turn {game.turn}: {guess.upper()}  {display}  "
                  f"({remaining} left, {guess_time:.2f}s)")
        
        if game.is_solved():
            if verbose:
                print(f"  Solved in {game.turn} guesses!")
            return game.turn
    
    if verbose:
        print(f"  Failed! Target was {target.upper()}")
    return 7


if __name__ == "__main__":
    print("Initializing InfoGain solver...")
    solver = InfoGainSolver()
    
    test_words = ["crane", "slink", "nymph", "vivid", "fuzzy"]
    for word in test_words:
        print(f"\nTarget: {word.upper()}")
        t0 = time.time()
        result = play_game(solver, word, verbose=True)
        print(f"  Total: {time.time()-t0:.2f}s")
