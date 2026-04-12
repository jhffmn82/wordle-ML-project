"""
Curated Word Lists for Training

Runs the frequency heuristic against every word and records which words
are used as guesses at each turn. Early-turn guesses (turns 1-2) are
pure information-gathering words. Later turns are targeted solves.

Sets:
  set_a: unique words used at turns 1-2 (narrowing vocabulary)
  set_b: unique words used at turns 1-3 (narrowing + early solves)
"""

import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.wordle_env import WordleGame, load_word_list
from solvers.frequency_solver import FrequencySolver


def build_curated_lists(word_list_path=None, verbose=True):
    """
    Run the frequency solver against every word. Track which words
    are guessed at each turn number.

    Returns:
        turn_words: dict {turn_number: set of words used at that turn}
        set_a: list of unique words used at turns 1-2
        set_b: list of unique words used at turns 1-3
    """
    words = load_word_list(word_list_path)
    solver = FrequencySolver(word_list_path)

    # Track unique words used at each turn
    turn_words = defaultdict(set)

    if verbose:
        print(f"Running frequency solver against all {len(words)} words...", flush=True)

    t0 = time.time()
    for target in words:
        solver.reset()
        game = WordleGame(target=target, word_list=words)

        while not game.is_over():
            guess = solver.get_guess()
            turn = game.turn + 1  # 1-indexed
            turn_words[turn].add(guess)
            feedback = game.make_guess(guess)
            solver.update(guess, feedback)
            if game.is_solved():
                break

    elapsed = time.time() - t0

    if verbose:
        print(f"  Done in {elapsed:.1f}s\n")
        for t in sorted(turn_words.keys()):
            print(f"  Turn {t}: {len(turn_words[t])} unique words")

    # Build sets
    set_a = sorted(turn_words[1] | turn_words[2])
    set_b = sorted(turn_words[1] | turn_words[2] | turn_words[3])

    if verbose:
        print(f"\n  Set A (turns 1-2): {len(set_a)} words")
        print(f"  Set B (turns 1-3): {len(set_b)} words")

    return turn_words, set_a, set_b


def get_sets(word_list_path=None):
    """Return (set_a, set_b) word lists."""
    _, set_a, set_b = build_curated_lists(word_list_path, verbose=False)
    return set_a, set_b


if __name__ == "__main__":
    turn_words, set_a, set_b = build_curated_lists()

    print(f"\nSet A (turns 1-2) — {len(set_a)} words:")
    for i in range(0, len(set_a), 10):
        row = set_a[i:i+10]
        print(f"  {', '.join(w.upper() for w in row)}")

    print(f"\nSet B (turns 1-3) — {len(set_b)} words (first 50):")
    for i in range(0, min(50, len(set_b)), 10):
        row = set_b[i:i+10]
        print(f"  {', '.join(w.upper() for w in row)}")
    if len(set_b) > 50:
        print(f"  ... and {len(set_b) - 50} more")
