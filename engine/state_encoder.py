"""
State Encoder for Learned Models

Converts Wordle game state (guesses + feedbacks) into a numerical vector
suitable for neural network input.

Reference / motivation:
    Ho, A. (2022). Wordle Solving with Deep Reinforcement Learning.
    Anderson, B.J. & Meyer, J.G. (2022). arXiv:2202.00557.

Design note:
    Ho's published implementation uses a history-based encoding with:
        - 26 letters × 5 positions × 3 statuses = 390
        - 26 global absent-letter features
        - 1 turn feature
        - total = 417

    This implementation is DIFFERENT on purpose.

    Instead of encoding the full guess history, this encoder collapses the
    board into a more compact constraint-state representation:

        For each letter:
            - present in word?                      (1)
            - absent from word?                     (1)
            - green at positions 1..5              (5)
            - forbidden at positions 1..5          (5)

        Total per letter = 12 features
        Total state dim = 26 × 12 + 1 turn = 313

    Why this differs from Ho:
        - Ho's encoding preserves more raw board-history information
        - This version encodes the CURRENT logical constraint state instead
        - It is lower-dimensional and less redundant
        - It keeps the same 130-dim word encoding used for action scoring

    Important limitation:
        This encoding still does not fully capture repeated-letter count
        constraints (min/max multiplicity), but it is a cleaner state than
        the original 417-dim history encoding.
"""

import numpy as np


# Constants
NUM_LETTERS = 26
WORD_LENGTH = 5

# Per-letter blocks:
#   [0]   present
#   [1]   absent
#   [2:7] green at positions 0..4
#   [7:12] forbidden at positions 0..4
FEATURES_PER_LETTER = 12

STATE_DIM = NUM_LETTERS * FEATURES_PER_LETTER + 1  # 313


def encode_state(guesses, feedbacks, turn):
    """
    Encode the current Wordle state as a compact constraint vector.

    Args:
        guesses: list[str]
            Previous guesses.
        feedbacks: list[list[int]]
            Feedback per guess, using 0=gray, 1=yellow, 2=green.
        turn: int
            Current turn number (0-5).

    Returns:
        np.ndarray of shape (STATE_DIM,) = (313,)

    Layout:
        For each letter a-z, a 12-dim block:
            [0]    present somewhere in word
            [1]    absent from word
            [2:7]  green at positions 0..4
            [7:12] forbidden at positions 0..4

        Final feature:
            [312]  normalized turn number in [0, 1]

    Notes:
        - "present" means the letter has appeared as yellow or green
        - "absent" means the letter has appeared gray and has never appeared
          yellow/green in any guess
        - "forbidden at pos i" means the letter is known NOT to be at pos i
          (typically from yellow feedback; gray can also imply this locally)
        - Unlike Ho's 417-dim encoding, this is NOT history-preserving; it
          represents only the current accumulated constraints
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)

    present_letters = set()
    gray_letters = set()

    # Per-letter constraint tracking
    green_positions = {chr(ord('a') + i): set() for i in range(NUM_LETTERS)}
    forbidden_positions = {chr(ord('a') + i): set() for i in range(NUM_LETTERS)}

    # First pass: collect raw evidence from all guesses
    for guess, feedback in zip(guesses, feedbacks):
        guess = guess.lower()

        for pos, (letter, fb) in enumerate(zip(guess, feedback)):
            if fb == 2:  # green
                present_letters.add(letter)
                green_positions[letter].add(pos)

            elif fb == 1:  # yellow
                present_letters.add(letter)
                forbidden_positions[letter].add(pos)

            elif fb == 0:  # gray
                gray_letters.add(letter)
                # A gray tile also means "not here" for this occurrence.
                # This is not sufficient to prove global absence if the letter
                # appears yellow/green elsewhere, but it is still local evidence.
                forbidden_positions[letter].add(pos)

    truly_absent = gray_letters - present_letters

    # Fill state vector
    for letter_idx in range(NUM_LETTERS):
        letter = chr(ord('a') + letter_idx)
        base = letter_idx * FEATURES_PER_LETTER

        # [0] present
        if letter in present_letters:
            state[base + 0] = 1.0

        # [1] absent
        if letter in truly_absent:
            state[base + 1] = 1.0

        # [2:7] green positions
        for pos in green_positions[letter]:
            state[base + 2 + pos] = 1.0

        # [7:12] forbidden positions
        for pos in forbidden_positions[letter]:
            state[base + 7 + pos] = 1.0

    # Final feature: normalized turn number
    state[-1] = turn / 6.0

    return state


def encode_words_onehot(word_list):
    """
    Create one-hot encodings for all words in the vocabulary.

    This stays the same as before:
        5 positions × 26 letters = 130 features per word

    Args:
        word_list: list[str]
            List of 5-letter words.

    Returns:
        np.ndarray of shape (len(word_list), 130)
    """
    n_words = len(word_list)
    encodings = np.zeros((n_words, WORD_LENGTH * NUM_LETTERS), dtype=np.float32)

    for i, word in enumerate(word_list):
        for pos, letter in enumerate(word.lower()):
            letter_idx = ord(letter) - ord('a')
            encodings[i, pos * NUM_LETTERS + letter_idx] = 1.0

    return encodings


if __name__ == "__main__":
    # Test encoding
    state = encode_state(
        guesses=["slate", "crane"],
        feedbacks=[[0, 0, 1, 0, 2], [2, 2, 2, 2, 2]],
        turn=2
    )
    print(f"State vector shape: {state.shape}")
    print(f"State dim: {STATE_DIM}")
    print(f"Non-zero entries: {np.count_nonzero(state)}")

    # Test word encoding
    test_words = ["crane", "slate", "salet"]
    onehot = encode_words_onehot(test_words)
    print(f"\nWord encodings shape: {onehot.shape}")
    print(f"Non-zero per word: {onehot.sum(axis=1)}")  # Should be 5 each
