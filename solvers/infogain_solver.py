"""
Solver 2: Information Gain / Entropy Solver

Strategy: For each candidate guess, simulate the feedback against every
possible remaining word. Group the remaining words by the feedback pattern
they would produce. Select the guess that minimizes the expected remaining
group size (equivalently, maximizes information gain / entropy).

This is computationally expensive but produces strong results.

References:
    Liu, C.-L. (2022). Using Wordle for Learning to Design and Compare Strategies.
    Bhambri, S. et al. (2022). RL Methods for Wordle: A POMDP/Adaptive Control Approach.
"""

import os
import sys
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.wordle_env import load_word_list, get_feedback, filter_words


class InfoGainSolver:
    """
    Information gain / entropy-based solver.
    
    For each possible guess, computes the expected number of remaining
    words after the guess (averaged over all possible targets).
    Picks the guess that minimizes this expectation.
    """
    
    def __init__(self, word_list_path=None):
        """
        Args:
            word_list_path: path to wordle.txt (uses default if None)
        """
        self.all_words = load_word_list(word_list_path)
        self.remaining = list(self.all_words)
    
    def reset(self):
        """Reset solver state for a new game."""
        self.remaining = list(self.all_words)
    
    def update(self, guess, feedback):
        """
        Update internal state based on a guess and its feedback.
        
        Args:
            guess: 5-letter string
            feedback: list of 5 ints (0=gray, 1=yellow, 2=green)
        """
        self.remaining = filter_words(self.remaining, guess, feedback)
    
    def get_guess(self, game_state=None):
        """
        Return the best guess based on information gain.
        
        Args:
            game_state: optional dict from WordleGame.get_state()
        
        Returns:
            str: the recommended 5-letter guess
        """
        if game_state is not None:
            self._sync_from_state(game_state)
        
        # If only 1-2 words remain, just guess one
        if len(self.remaining) <= 2:
            return self.remaining[0]
        
        # Determine candidate guesses
        # If few words remain, search all words (exploration might help)
        # Otherwise, only search remaining words (faster)
        if len(self.remaining) <= 20:
            candidates = self.all_words
        else:
            candidates = self.remaining
        
        best_word = self.remaining[0]
        best_score = float('inf')
        
        for candidate in candidates:
            score = self._expected_remaining(candidate)
            if score < best_score:
                best_score = score
                best_word = candidate
        
        return best_word
    
    def _expected_remaining(self, guess):
        """
        Compute the expected number of remaining words after a guess.
        
        For each possible target in self.remaining, compute the feedback
        that would result. Group by feedback pattern. The expected remaining
        count is the weighted average of group sizes.
        
        Also gives a bonus to words that could be the answer (in remaining list).
        """
        # Group remaining words by the feedback pattern they'd produce
        pattern_groups = defaultdict(int)
        
        for target in self.remaining:
            feedback = tuple(get_feedback(guess, target))
            pattern_groups[feedback] += 1
        
        # Expected remaining = sum(group_size^2) / total
        # (because if a group has size k, and the true target is in that group,
        # we'll have k words remaining — and probability of being in that group is k/total)
        total = len(self.remaining)
        expected = sum(count * count for count in pattern_groups.values()) / total
        
        # Small bonus for words that are themselves possible answers
        # (if we guess right, game is over!)
        if guess in self.remaining:
            expected -= 1.0
        
        return expected
    
    def _entropy(self, guess):
        """
        Compute the entropy of the feedback distribution for a guess.
        Higher entropy = more information gained = better guess.
        
        (Alternative scoring to _expected_remaining. Not used by default
        but available for comparison.)
        """
        pattern_groups = defaultdict(int)
        
        for target in self.remaining:
            feedback = tuple(get_feedback(guess, target))
            pattern_groups[feedback] += 1
        
        total = len(self.remaining)
        entropy = 0.0
        
        for count in pattern_groups.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _sync_from_state(self, game_state):
        """Rebuild internal state from a WordleGame state dict."""
        self.remaining = list(self.all_words)
        for guess, feedback in zip(game_state["guesses"], game_state["feedbacks"]):
            self.update(guess, feedback)


def play_game(solver, target, word_list=None, verbose=False):
    """
    Play a complete game using the solver.
    
    Returns:
        int: number of guesses taken (7 if failed)
    """
    from engine.wordle_env import WordleGame
    
    game = WordleGame(target=target, word_list=word_list or solver.all_words)
    solver.reset()
    
    while not game.is_over():
        guess = solver.get_guess()
        feedback = game.make_guess(guess)
        solver.update(guess, feedback)
        
        if verbose:
            symbols = ["⬛", "🟨", "🟩"]
            display = " ".join(symbols[f] for f in feedback)
            remaining = len(solver.remaining)
            print(f"  Turn {game.turn}: {guess.upper()}  {display}  ({remaining} words left)")
        
        if game.is_solved():
            if verbose:
                print(f"  Solved in {game.turn} guesses!")
            return game.turn
    
    if verbose:
        print(f"  Failed! Target was {target.upper()}")
    return 7  # Failed


if __name__ == "__main__":
    solver = InfoGainSolver()
    
    # Test on a few words
    test_words = ["crane", "slink", "nymph", "vivid", "fuzzy"]
    for word in test_words:
        print(f"\nTarget: {word.upper()}")
        result = play_game(solver, word, verbose=True)
