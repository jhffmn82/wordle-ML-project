"""
Solver 2: Information Gain / Minimax Solver
 
For each turn, scores every word in the vocabulary as a candidate guess
against the current remaining words. Picks the guess that minimizes the
worst-case remaining partition size.
 
No precomputed table — scores are computed on-the-fly using numba-compiled
functions for speed.
 
Complexity per turn: O(N * R) where N = vocab size, R = remaining words.
Turn 1 is cached since it's always the same.
 
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
 
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
 
 
# ---- Numba-compiled core ----
 
if NUMBA_AVAILABLE:
    @njit
    def _feedback_id(guess, target):
        """Compute feedback ID (0-242) for a guess/target pair. Both are (5,) int arrays."""
        t_remaining = np.empty(5, dtype=np.int16)
        green = np.zeros(5, dtype=np.int8)
        for k in range(5):
            t_remaining[k] = target[k]
 
        for k in range(5):
            if guess[k] == target[k]:
                green[k] = 1
                t_remaining[k] = -1
 
        fb_id = 0
        mult = 81
        for k in range(5):
            if green[k]:
                fb_id += 2 * mult
            else:
                for m in range(5):
                    if guess[k] == t_remaining[m]:
                        fb_id += mult
                        t_remaining[m] = -1
                        break
            mult //= 3
 
        return fb_id
 
    @njit
    def _find_best_guess(word_chars, remaining_indices, n_words, n_remaining):
        """
        Score all N words as candidate guesses against R remaining words.
        Returns index of the guess with smallest worst-case partition.
        """
        best_idx = remaining_indices[0]
        best_worst = n_remaining + 1
 
        for gi in range(n_words):
            # Count how many remaining words fall into each feedback pattern
            counts = np.zeros(243, dtype=np.int32)
            for j in range(n_remaining):
                fb = _feedback_id(word_chars[gi], word_chars[remaining_indices[j]])
                counts[fb] += 1
 
            # Worst case = largest partition
            worst = 0
            for k in range(243):
                if counts[k] > worst:
                    worst = counts[k]
 
            # Early exit: can't beat 1
            if worst == 1:
                return gi
 
            if worst < best_worst:
                best_worst = worst
                best_idx = gi
 
        return best_idx
 
    @njit
    def _filter_remaining(word_chars, remaining_indices, n_remaining, guess_idx, fb_id):
        """Return new remaining indices after applying feedback filter."""
        result = np.empty(n_remaining, dtype=np.int64)
        count = 0
        for j in range(n_remaining):
            ri = remaining_indices[j]
            if _feedback_id(word_chars[guess_idx], word_chars[ri]) == fb_id:
                result[count] = ri
                count += 1
        return result[:count]
 
 
class InfoGainSolver:
 
    def __init__(self, word_list_path=None):
        self.all_words = load_word_list(word_list_path)
        self.n_words = len(self.all_words)
        self.word_to_idx = {w: i for i, w in enumerate(self.all_words)}
 
        # Convert words to int array for numba
        self.word_chars = np.array(
            [[ord(c) for c in w] for w in self.all_words], dtype=np.int16
        )
 
        self.remaining_indices = np.arange(self.n_words, dtype=np.int64)
 
        if NUMBA_AVAILABLE:
            # Warm up numba (first call compiles)
            print("Compiling solver (first run only)...", flush=True)
            t0 = time.time()
            dummy = np.arange(min(10, self.n_words), dtype=np.int64)
            _find_best_guess(self.word_chars, dummy, self.n_words, len(dummy))
            print(f"  Compiled in {time.time()-t0:.1f}s")
 
            # Cache first guess
            print("Computing best opening guess...", flush=True)
            t0 = time.time()
            all_idx = np.arange(self.n_words, dtype=np.int64)
            self._first_guess_idx = _find_best_guess(
                self.word_chars, all_idx, self.n_words, self.n_words
            )
            self._first_guess = self.all_words[self._first_guess_idx]
            print(f"  Best opener: {self._first_guess.upper()} ({time.time()-t0:.1f}s)")
        else:
            print("WARNING: numba not available, solver will be slow")
            self._first_guess = None
 
    def reset(self):
        self.remaining_indices = np.arange(self.n_words, dtype=np.int64)
 
    def update(self, guess, feedback):
        guess = guess.lower()
        fb_id = feedback[0]*81 + feedback[1]*27 + feedback[2]*9 + feedback[3]*3 + feedback[4]
        guess_idx = self.word_to_idx.get(guess)
 
        if guess_idx is not None and NUMBA_AVAILABLE:
            self.remaining_indices = _filter_remaining(
                self.word_chars, self.remaining_indices,
                len(self.remaining_indices), guess_idx, fb_id
            )
        else:
            remaining = [self.all_words[i] for i in self.remaining_indices]
            remaining = filter_words(remaining, guess, feedback)
            self.remaining_indices = np.array(
                [self.word_to_idx[w] for w in remaining], dtype=np.int64
            )
 
    def get_guess(self, game_state=None):
        if game_state is not None:
            self._sync_from_state(game_state)
 
        n_remaining = len(self.remaining_indices)
 
        if n_remaining <= 1:
            return self.all_words[self.remaining_indices[0]]
        if n_remaining == 2:
            return self.all_words[self.remaining_indices[0]]
        if n_remaining == self.n_words and self._first_guess:
            return self._first_guess
 
        if NUMBA_AVAILABLE:
            idx = _find_best_guess(
                self.word_chars, self.remaining_indices,
                self.n_words, n_remaining
            )
        else:
            idx = self._find_best_guess_python(self.remaining_indices)
 
        return self.all_words[idx]
 
    def _find_best_guess_python(self, remaining_indices):
        """Fallback pure-Python scoring."""
        n_remaining = len(remaining_indices)
        best_idx = remaining_indices[0]
        best_worst = n_remaining + 1
 
        remaining_words = [self.all_words[i] for i in remaining_indices]
 
        for gi in range(self.n_words):
            guess = self.all_words[gi]
            from collections import defaultdict
            groups = defaultdict(int)
            for target in remaining_words:
                fb = tuple(get_feedback(guess, target))
                groups[fb] += 1
            worst = max(groups.values())
            if worst < best_worst:
                best_worst = worst
                best_idx = gi
 
        return best_idx
 
    def _sync_from_state(self, game_state):
        self.remaining_indices = np.arange(self.n_words, dtype=np.int64)
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
            remaining = len(solver.remaining_indices)
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
 
    # Benchmark
    import random
    sample = random.sample(solver.all_words, 50)
    t0 = time.time()
    results = [play_game(solver, w) for w in sample]
    elapsed = time.time() - t0
    print(f"\n50-word benchmark: {elapsed:.1f}s total, {elapsed/50:.3f}s/game")
    print(f"  Avg guesses: {np.mean(results):.2f}")
    print(f"  Full eval estimate: {elapsed/50 * len(solver.all_words):.0f}s")
 
