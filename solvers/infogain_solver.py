"""
Solver 2: Information Gain / Minimax Solver

For each turn, scores every word as a candidate guess against the
remaining words. Picks the guess that minimizes the worst-case
(largest) partition of remaining words.

Plain Python — no numba, no precomputed table.

References:
    Liu, C.-L. (2022). Using Wordle for Learning to Design and Compare Strategies.
    Bhambri, S. et al. (2022). RL Methods for Wordle: A POMDP/Adaptive Control Approach.
"""

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.wordle_env import load_word_list, get_feedback, filter_words


class InfoGainSolver:

    def __init__(self, word_list_path=None):
        self.all_words = load_word_list(word_list_path)
        self.remaining = list(self.all_words)

        # Cache the first guess since starting state is always the same
        print("Computing best opening guess (one-time cost)...", flush=True)
        t0 = time.time()
        self._first_guess = self._best_guess(self.all_words, self.all_words)
        print(f"  Best opener: {self._first_guess.upper()} ({time.time()-t0:.1f}s)")

    def reset(self):
        self.remaining = list(self.all_words)

    def update(self, guess, feedback):
        self.remaining = filter_words(self.remaining, guess, feedback)

    def get_guess(self, game_state=None):
        if game_state is not None:
            self._sync_from_state(game_state)

        if len(self.remaining) <= 2:
            return self.remaining[0]

        # Use cached opener if board is fresh
        if len(self.remaining) == len(self.all_words):
            return self._first_guess

        return self._best_guess(self.all_words, self.remaining)

    def _best_guess(self, candidates, remaining):
        """
        Find the guess that minimizes the worst-case partition size.

        For each candidate guess:
          - Compute feedback against every remaining word
          - Group remaining words by feedback pattern
          - Worst case = size of the largest group

        Pick the candidate with the smallest worst case.
        """
        best_word = remaining[0]
        best_worst = len(remaining) + 1

        for guess in candidates:
            # Group remaining words by what feedback they'd produce
            pattern_counts = Counter()
            for target in remaining:
                fb = tuple(get_feedback(guess, target))
                pattern_counts[fb] += 1

            worst = max(pattern_counts.values())

            # Tiebreak: prefer words that could be the answer
            if worst < best_worst or (worst == best_worst and guess in remaining):
                best_worst = worst
                best_word = guess

            # Can't do better than 1
            if best_worst == 1:
                break

        return best_word

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
            symbols = ["⬛", "🟨", "🟩"]
            display = " ".join(symbols[f] for f in feedback)
            print(f"  Turn {game.turn}: {guess.upper()}  {display}  "
                  f"({len(solver.remaining)} left, {dt:.2f}s)")

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
    sample = random.sample(solver.all_words, 20)
    t0 = time.time()
    results = [play_game(solver, w) for w in sample]
    elapsed = time.time() - t0
    avg = sum(results) / len(results)
    print(f"\n20-word benchmark: {elapsed:.1f}s total, {elapsed/20:.2f}s/game")
    print(f"  Avg guesses: {avg:.2f}")
    print(f"  Full eval estimate: {elapsed/20 * len(solver.all_words) / 60:.0f} min")
