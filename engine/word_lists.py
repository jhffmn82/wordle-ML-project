"""
Curated Word Lists for DQN Training

Builds a 4-tier cascading curriculum by tracing the frequency solver's
decision tree:

  Set 1 (~25 words): Top 5 openers + their top 5 follow-up guesses
  Set 2 (~80 words): + cascaded next picks from each follow-up
  Set 3 (~1300 words): + all remaining turn 1-3 words from full solver runs
  Set 4: Full word list

This ensures DQN sees the most informative words first during training.
"""

import os
import sys
import time
import random
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.wordle_env import WordleGame, load_word_list, get_feedback, filter_words
from solvers.frequency_solver import FrequencySolver


def build_curated_lists(word_list_path=None, sample_size=500, verbose=True):
    """
    Build 4-tier cascading word sets.

    Returns:
        set1: ~25 words (openers + direct seconds)
        set2: ~80 words (+ cascaded thirds)
        set3: ~1300 words (+ remaining A/B)
        turn_words: dict of turn -> set of words (from full solver run)
    """
    words = load_word_list(word_list_path)
    solver = FrequencySolver(word_list_path)
    random.seed(42)
    sample = random.sample(words, min(sample_size, len(words)))

    if verbose:
        print("Building curated word lists...", flush=True)
    t0 = time.time()

    # --- Step 1: Top 5 openers by frequency score (unique letters only) ---
    solver.reset()
    lf = solver._letter_frequency(words)
    pf = solver._letter_frequency_place(words)
    scored = [(solver._word_value(w, lf, pf), w) for w in words if len(set(w)) == 5]
    scored.sort(reverse=True)
    top5_first = [w for _, w in scored[:5]]

    if verbose:
        print(f"  Top 5 openers: {[w.upper() for w in top5_first]}")

    # --- Step 2: For each opener, find top 5 second guesses ---
    all_seconds = set()
    for first in top5_first:
        counter = Counter()
        for target in sample:
            solver.reset()
            game = WordleGame(target=target, word_list=words)
            fb = game.make_guess(first)
            solver.update(first, fb)
            if not game.is_solved():
                counter[solver.get_guess()] += 1
        top5 = [w for w, _ in counter.most_common(5)]
        all_seconds.update(top5)

    set1 = sorted(set(top5_first) | all_seconds)
    if verbose:
        print(f"  Set 1: {len(set1)} words (openers + seconds)")

    # --- Step 3: For each second-pick, find top 5 next guesses ---
    all_thirds = set()
    for word in sorted(all_seconds):
        counter = Counter()
        for target in sample:
            solver.reset()
            game = WordleGame(target=target, word_list=words)
            fb = game.make_guess(word)
            solver.update(word, fb)
            if not game.is_solved():
                counter[solver.get_guess()] += 1
        top5 = [w for w, _ in counter.most_common(5)]
        all_thirds.update(top5)

    set2 = sorted(set(set1) | all_thirds)
    if verbose:
        print(f"  Set 2: {len(set2)} words (+ cascaded thirds)")

    # --- Step 4: Full solver run to get all turn 1-3 words ---
    if verbose:
        print(f"  Running full solver for turn 1-3 words...", flush=True)

    turn_words = defaultdict(set)
    for target in words:
        solver.reset()
        game = WordleGame(target=target, word_list=words)
        while not game.is_over():
            guess = solver.get_guess()
            turn = game.turn + 1
            turn_words[turn].add(guess)
            feedback = game.make_guess(guess)
            solver.update(guess, feedback)
            if game.is_solved():
                break

    set_ab = turn_words[1] | turn_words[2] | turn_words[3]
    set3 = sorted(set(set2) | set_ab)

    if verbose:
        print(f"  Set 3: {len(set3)} words (+ remaining A/B)")
        print(f"  Set 4: {len(words)} words (full)")
        print(f"  Done in {time.time()-t0:.1f}s")

    return set1, set2, set3, dict(turn_words)


def get_all_sets(word_list_path=None):
    """Return (set1, set2, set3) word lists."""
    s1, s2, s3, _ = build_curated_lists(word_list_path, verbose=False)
    return s1, s2, s3


if __name__ == "__main__":
    set1, set2, set3, turn_words = build_curated_lists()

    print(f"\nSet 1 ({len(set1)} words):")
    print(f"  {[w.upper() for w in set1]}")

    print(f"\nSet 2 ({len(set2)} words, first 30):")
    print(f"  {[w.upper() for w in set2[:30]]}")

    print(f"\nTurn usage:")
    for t in sorted(turn_words.keys()):
        print(f"  Turn {t}: {len(turn_words[t])} unique words")
