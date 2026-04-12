"""
Solver 5: Rollout (Bhambri, Bhattacharjee & Bertsekas, 2022)

Improves the info gain heuristic with one-step lookahead:
  1. Use info gain (minimax) to get top-K candidate guesses
  2. For each candidate, simulate the rest of the game for every
     remaining word using the frequency solver as base heuristic
  3. Pick the candidate with the lowest average total guesses

Rollout only activates when remaining words <= threshold (default 20).
Above that, uses info gain directly. This keeps runtime reasonable
while improving endgame decisions where ties matter most.

Runtime: ~30 minutes for full evaluation (similar to info gain).

Reference:
    Bhambri, S. et al. (2022). RL Methods for Wordle. arXiv:2211.10298.
"""

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.wordle_env import WordleGame, load_word_list, get_feedback, filter_words
from solvers.frequency_solver import FrequencySolver


class RolloutSolver:

    def __init__(self, word_list_path=None, top_k=5, rollout_threshold=20):
        """
        Args:
            word_list_path: path to wordle.txt
            top_k: number of top candidates to rollout (default 5)
            rollout_threshold: only do rollout when remaining <= this (default 20)
        """
        self.all_words = load_word_list(word_list_path)
        self.remaining = list(self.all_words)
        self.top_k = top_k
        self.rollout_threshold = rollout_threshold

        # Base heuristic for forward simulation
        self._base_solver = FrequencySolver(word_list_path)

        # Cache first guess (info gain minimax on full list)
        print("Computing best opening guess...", flush=True)
        t0 = time.time()
        self._first_guess = self._minimax_best(self.all_words, self.all_words)
        print(f"  Best opener: {self._first_guess.upper()} ({time.time()-t0:.1f}s)")

    def reset(self):
        self.remaining = list(self.all_words)

    def update(self, guess, feedback):
        self.remaining = filter_words(self.remaining, guess, feedback)

    def get_guess(self, game_state=None):
        if game_state is not None:
            self._sync_from_state(game_state)

        n = len(self.remaining)

        if n <= 1:
            return self.remaining[0] if self.remaining else self.all_words[0]
        if n == 2:
            return self.remaining[0]
        if n == len(self.all_words):
            return self._first_guess

        # If remaining is small enough, do rollout
        if n <= self.rollout_threshold:
            return self._rollout_best(self.all_words, self.remaining)

        # Otherwise, use minimax info gain directly
        return self._minimax_best(self.all_words, self.remaining)

    def _minimax_best(self, candidates, remaining):
        """Pick the guess that minimizes worst-case partition (info gain)."""
        best_word = remaining[0]
        best_worst = len(remaining) + 1

        for guess in candidates:
            groups = Counter()
            for target in remaining:
                fb = tuple(get_feedback(guess, target))
                groups[fb] += 1
            worst = max(groups.values())

            if worst < best_worst or (worst == best_worst and guess in remaining):
                best_worst = worst
                best_word = guess
            if best_worst == 1:
                break

        return best_word

    def _rollout_best(self, candidates, remaining):
        """
        Rollout: score top-K candidates by simulating the full game forward.

        1. Score all candidates with minimax to get top K
        2. For each of the K, simulate the game for every remaining word
           using frequency solver
        3. Pick the one with lowest average total guesses
        """
        # Step 1: get top K by minimax score
        scored = []
        for guess in candidates:
            groups = Counter()
            for target in remaining:
                fb = tuple(get_feedback(guess, target))
                groups[fb] += 1
            worst = max(groups.values())
            is_remaining = guess in remaining
            scored.append((worst, not is_remaining, guess))

        scored.sort()
        top_candidates = [g for _, _, g in scored[:self.top_k]]

        # Step 2: rollout each candidate
        best_word = top_candidates[0]
        best_avg = float('inf')

        for guess in top_candidates:
            total_guesses = 0

            for target in remaining:
                # Simulate: make this guess, then play out with frequency solver
                feedback = get_feedback(guess, target)

                if feedback == [2, 2, 2, 2, 2]:
                    total_guesses += 1
                    continue

                # Filter remaining after this guess
                new_remaining = filter_words(remaining, guess, feedback)

                # Play out rest with frequency solver
                guesses_used = 1 + self._simulate_forward(target, new_remaining)
                total_guesses += guesses_used

            avg = total_guesses / len(remaining)

            if avg < best_avg:
                best_avg = avg
                best_word = guess

        return best_word

    def _simulate_forward(self, target, remaining):
        """
        Simulate the rest of a game using frequency solver.
        Returns number of additional guesses needed.
        """
        self._base_solver.reset()
        self._base_solver.remaining = list(remaining)

        for turn in range(6):  # max 6 more turns
            if len(self._base_solver.remaining) == 0:
                return 6  # failed
            guess = self._base_solver.get_guess()
            feedback = get_feedback(guess, target)

            if feedback == [2, 2, 2, 2, 2]:
                return turn + 1

            self._base_solver.update(guess, feedback)

        return 6  # failed

    def _sync_from_state(self, game_state):
        self.remaining = list(self.all_words)
        for guess, feedback in zip(game_state["guesses"], game_state["feedbacks"]):
            self.update(guess, feedback)


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
            mode = "rollout" if n <= solver.rollout_threshold else "minimax"
            symbols = ["⬛", "🟨", "🟩"]
            display = " ".join(symbols[f] for f in feedback)
            print(f"  Turn {game.turn}: {guess.upper()}  {display}  "
                  f"({n} left, {mode}, {dt:.2f}s)")

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
