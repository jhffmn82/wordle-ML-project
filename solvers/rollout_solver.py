"""
Solver 5: Memoized Rollout (Bhambri, Bhattacharjee & Bertsekas, 2022)

Implements the rollout approach from the paper:
  1. Score all words by minimax (information gain) to get top 10 candidates
  2. For each candidate, simulate the full game forward for every remaining
     word using the frequency solver as base heuristic
  3. Pick the candidate with the lowest average total guesses

Results are memoized: each game state → best guess is cached to disk.
First run is slow, but each subsequent run is faster as the cache fills.
Eventually all states are cached and evaluation is instant.

Cache key: tuple of (guess, feedback) pairs — uniquely identifies game state.

Reference:
    Bhambri, S. et al. (2022). RL Methods for Wordle. arXiv:2211.10298.
"""

import os
import sys
import time
import pickle
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.wordle_env import WordleGame, load_word_list, get_feedback, filter_words
from solvers.frequency_solver import FrequencySolver


DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "rollout_cache.pkl"
)


class RolloutSolver:

    def __init__(self, word_list_path=None, top_k=10, cache_path=None):
        """
        Args:
            word_list_path: path to wordle.txt
            top_k: number of top candidates to rollout (paper uses 10)
            cache_path: where to save/load the memoization cache
        """
        self.all_words = load_word_list(word_list_path)
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []
        self.top_k = top_k

        # Base heuristic for forward simulation
        self._base_solver = FrequencySolver(word_list_path)

        # Cache: game_state_key -> best_guess
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self.cache = self._load_cache()
        self._cache_hits = 0
        self._cache_misses = 0

    def _load_cache(self):
        """Load cache from disk if it exists."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    cache = pickle.load(f)
                print(f"Loaded rollout cache: {len(cache)} states from {self.cache_path}")
                return cache
            except Exception as e:
                print(f"Warning: could not load cache: {e}")
        return {}

    def save_cache(self):
        """Save cache to disk."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)
        print(f"Saved rollout cache: {len(self.cache)} states to {self.cache_path}")

    def _state_key(self):
        """Build cache key from current game history."""
        return tuple((g, tuple(fb)) for g, fb in zip(self.guesses, self.feedbacks))

    def reset(self):
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []

    def update(self, guess, feedback):
        self.guesses.append(guess.lower())
        self.feedbacks.append(feedback)
        self.remaining = filter_words(self.remaining, guess, feedback)

    def get_guess(self, game_state=None):
        if game_state is not None:
            self._sync_from_state(game_state)

        n_remaining = len(self.remaining)

        if n_remaining <= 1:
            return self.remaining[0] if self.remaining else self.all_words[0]
        if n_remaining == 2:
            return self.remaining[0]

        # Check cache
        key = self._state_key()
        if key in self.cache:
            self._cache_hits += 1
            return self.cache[key]

        # Cache miss — compute via rollout
        self._cache_misses += 1
        best_guess = self._rollout(self.remaining)

        # Store in cache
        self.cache[key] = best_guess
        return best_guess

    def _rollout(self, remaining):
        """
        Full rollout: score candidates, then simulate top-K forward.

        When remaining is large (>50), use frequency scoring for speed.
        When small, use minimax for precision.
        """
        # Step 1: Get top K candidates
        if len(remaining) > 50:
            # Use frequency scoring for speed on large lists
            top_candidates = self._top_k_by_frequency(remaining)
        else:
            # Use minimax for precision on small lists
            top_candidates = self._top_k_by_minimax(remaining)

        # Step 2: Rollout each candidate
        best_word = top_candidates[0]
        best_avg = float('inf')

        for guess in top_candidates:
            total_guesses = 0

            for target in remaining:
                feedback = get_feedback(guess, target)

                if feedback == [2, 2, 2, 2, 2]:
                    total_guesses += 1
                    continue

                new_remaining = filter_words(remaining, guess, feedback)
                guesses_needed = 1 + self._simulate_forward(target, new_remaining)
                total_guesses += guesses_needed

            avg = total_guesses / len(remaining)

            if avg < best_avg:
                best_avg = avg
                best_word = guess

        return best_word

    def _top_k_by_frequency(self, remaining):
        """Get top K candidates using frequency scoring (fast)."""
        self._base_solver.reset()
        self._base_solver.remaining = list(remaining)
        lf = self._base_solver._letter_frequency(remaining)
        pf = self._base_solver._letter_frequency_place(remaining)

        scored = []
        for w in self.all_words:
            s = self._base_solver._word_value(w, lf, pf)
            scored.append((s, w))
        scored.sort(reverse=True)
        return [w for _, w in scored[:self.top_k]]

    def _top_k_by_minimax(self, remaining):
        """Get top K candidates using minimax scoring (precise)."""
        scored = []
        for guess in self.all_words:
            groups = Counter()
            for target in remaining:
                fb = tuple(get_feedback(guess, target))
                groups[fb] += 1
            worst = max(groups.values())
            is_remaining = guess in remaining
            scored.append((worst, not is_remaining, guess))
        scored.sort()
        return [g for _, _, g in scored[:self.top_k]]

    def _simulate_forward(self, target, remaining):
        """
        Simulate the rest of a game using frequency solver as base heuristic.
        Returns number of additional guesses needed.
        """
        self._base_solver.reset()
        self._base_solver.remaining = list(remaining)

        for turn in range(6):
            if len(self._base_solver.remaining) == 0:
                return 6
            guess = self._base_solver.get_guess()
            feedback = get_feedback(guess, target)

            if feedback == [2, 2, 2, 2, 2]:
                return turn + 1

            self._base_solver.update(guess, feedback)

        return 6

    def _sync_from_state(self, game_state):
        self.remaining = list(self.all_words)
        self.guesses = []
        self.feedbacks = []
        for guess, feedback in zip(game_state["guesses"], game_state["feedbacks"]):
            self.update(guess, feedback)

    def cache_stats(self):
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return "No lookups yet"
        pct = self._cache_hits / total * 100
        return (f"Cache: {len(self.cache)} states, "
                f"{self._cache_hits} hits / {self._cache_misses} misses "
                f"({pct:.1f}% hit rate)")


def play_game(solver, target, verbose=False):
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
            n = len(solver.remaining)
            key = solver._state_key()
            cached = "cache" if key in solver.cache else "computed"
            symbols = ["⬛", "🟨", "🟩"]
            display = " ".join(symbols[f] for f in feedback)
            print(f"  Turn {game.turn}: {guess.upper()}  {display}  "
                  f"({n} left, {cached}, {dt:.2f}s)")

        if game.is_solved():
            if verbose:
                print(f"  Solved in {game.turn}!")
            return game.turn

    if verbose:
        print(f"  Failed! Target was {target.upper()}")
    return 7


if __name__ == "__main__":
    solver = RolloutSolver()

    test_words = ["crane", "slink", "nymph", "vivid", "fuzzy"]
    for word in test_words:
        print(f"\nTarget: {word.upper()}")
        play_game(solver, word, verbose=True)

    print(f"\n{solver.cache_stats()}")
    solver.save_cache()
