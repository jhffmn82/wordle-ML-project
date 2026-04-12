"""
Solver 2: Information Gain / Minimax Solver

Minimax approach: for each candidate guess (all N words), partition the
remaining words by feedback pattern. Pick the guess that minimizes the
worst-case (largest) partition.

OPTIMIZED:
  - Pre-computed NxN feedback table (one-time ~17s)
  - Cached first guess (computed once, reused every game)
  - np.bincount instead of np.unique (no sorting, O(R) per candidate)

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


def _compute_feedback_id(guess, target):
    """Compute feedback as a single int 0-242 (base-3 encoding)."""
    feedback = [0] * 5
    target_remaining = list(target)

    for i in range(5):
        if guess[i] == target[i]:
            feedback[i] = 2
            target_remaining[i] = None

    for i in range(5):
        if feedback[i] == 0 and guess[i] in target_remaining:
            feedback[i] = 1
            target_remaining[target_remaining.index(guess[i])] = None

    return feedback[0]*81 + feedback[1]*27 + feedback[2]*9 + feedback[3]*3 + feedback[4]


def _id_to_feedback(fb_id):
    """Convert feedback ID back to list of 5 ints."""
    fb = []
    for _ in range(5):
        fb.append(fb_id % 3)
        fb_id //= 3
    return fb[::-1]


class InfoGainSolver:

    def __init__(self, word_list_path=None):
        self.all_words = load_word_list(word_list_path)
        self.n_words = len(self.all_words)
        self.word_to_idx = {w: i for i, w in enumerate(self.all_words)}
        self.remaining_mask = np.ones(self.n_words, dtype=bool)

        # Build feedback table
        print(f"Building feedback table ({self.n_words}x{self.n_words})...", flush=True)
        t0 = time.time()
        self.feedback_table = self._build_feedback_table()
        print(f"  Table built in {time.time()-t0:.1f}s")

        # Cache the best opening guess (same every game)
        print("Computing best opening guess...", flush=True)
        t0 = time.time()
        all_indices = np.arange(self.n_words)
        self._first_guess_idx = self._best_guess(all_indices)
        self._first_guess = self.all_words[self._first_guess_idx]
        print(f"  Best opener: {self._first_guess.upper()} ({time.time()-t0:.1f}s)")

    def _build_feedback_table(self):
        n = self.n_words
        table = np.zeros((n, n), dtype=np.uint16)
        for i in range(n):
            g = self.all_words[i]
            for j in range(n):
                table[i, j] = _compute_feedback_id(g, self.all_words[j])
        return table

    def _best_guess(self, remaining_indices):
        """
        Find the guess that minimizes the worst-case partition size.
        Uses np.bincount for fast histogram per candidate.
        """
        n_remaining = len(remaining_indices)
        best_idx = remaining_indices[0]
        best_worst = n_remaining + 1

        for gi in range(self.n_words):
            row = self.feedback_table[gi, remaining_indices]
            counts = np.bincount(row, minlength=243)
            worst = counts.max()

            # Tiebreak: prefer remaining words (chance to solve outright)
            if worst < best_worst or (worst == best_worst and self.remaining_mask[gi]):
                best_worst = worst
                best_idx = gi

        return best_idx

    def reset(self):
        self.remaining_mask = np.ones(self.n_words, dtype=bool)

    def update(self, guess, feedback):
        guess = guess.lower()
        fb_id = feedback[0]*81 + feedback[1]*27 + feedback[2]*9 + feedback[3]*3 + feedback[4]
        guess_idx = self.word_to_idx.get(guess)
        if guess_idx is not None:
            self.remaining_mask &= (self.feedback_table[guess_idx, :] == fb_id)
        else:
            remaining = [self.all_words[i] for i in np.where(self.remaining_mask)[0]]
            remaining = filter_words(remaining, guess, feedback)
            s = set(remaining)
            self.remaining_mask = np.array([w in s for w in self.all_words])

    def get_guess(self, game_state=None):
        if game_state is not None:
            self._sync_from_state(game_state)

        remaining_indices = np.where(self.remaining_mask)[0]
        n_remaining = len(remaining_indices)

        if n_remaining <= 1:
            return self.all_words[remaining_indices[0]]
        if n_remaining == 2:
            return self.all_words[remaining_indices[0]]

        # Use cached first guess if board is fresh
        if n_remaining == self.n_words:
            return self._first_guess

        return self.all_words[self._best_guess(remaining_indices)]

    def _sync_from_state(self, game_state):
        self.remaining_mask = np.ones(self.n_words, dtype=bool)
        for guess, feedback in zip(game_state["guesses"], game_state["feedbacks"]):
            self.update(guess, feedback)


def play_game(solver, target, word_list=None, verbose=False):
    from engine.wordle_env import WordleGame
    game = WordleGame(target=target, word_list=solver.all_words)
    solver.reset()

    while not game.is_over():
        t0 = time.time()
        guess = solver.get_guess()
        dt = time.time() - t0
        feedback = game.make_guess(guess)
        solver.update(guess, feedback)

        if verbose:
            symbols = ["⬛", "🟨", "🟩"]
            display = " ".join(symbols[f] for f in feedback)
            remaining = int(solver.remaining_mask.sum())
            print(f"  Turn {game.turn}: {guess.upper()}  {display}  "
                  f"({remaining} left, {dt:.3f}s)")

        if game.is_solved():
            if verbose:
                print(f"  Solved in {game.turn}!")
            return game.turn

    if verbose:
        print(f"  Failed! Target was {target.upper()}")
    return 7


if __name__ == "__main__":
    solver = InfoGainSolver()

    test_words = ["crane", "slink", "nymph", "vivid", "fuzzy"]
    for word in test_words:
        print(f"\nTarget: {word.upper()}")
        play_game(solver, word, verbose=True)

    # Quick benchmark
    import random
    sample = random.sample(solver.all_words, 20)
    t0 = time.time()
    results = [play_game(solver, w) for w in sample]
    elapsed = time.time() - t0
    print(f"\n20-word benchmark: {elapsed:.1f}s total, {elapsed/20:.2f}s/game")
    print(f"  Avg guesses: {np.mean(results):.2f}")
