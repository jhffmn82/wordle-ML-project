"""
State Encoder for Learned Models (Ho-style 417-dim)

Converts Wordle game state (guesses + feedbacks) into a numerical vector
suitable for neural network input.

Reference:
    Ho, A. (2022). Wordle Solving with Deep Reinforcement Learning.
    Anderson, B.J. & Meyer, J.G. (2022). arXiv:2202.00557.

Encoding layout (417 dimensions):
    ---------------------------------------------------------------
    [0..389]   Letter-position-status history  (26 × 5 × 3 = 390)
               For each letter (a-z):
                 For each position (0-4):
                   [0] was this letter gray   at this position in any guess?
                   [1] was this letter yellow at this position in any guess?
                   [2] was this letter green  at this position in any guess?

    [390..415] Global absent flags             (26)
               1.0 if the letter appeared gray and NEVER appeared
               yellow/green in any guess (truly absent from the word).

    [416]      Normalized turn number           (1)
               turn / 6.0, in [0, 1].
    ---------------------------------------------------------------
    Total = 390 + 26 + 1 = 417

    This preserves per-guess board history, which gives the DQN richer
    gradient signal than a collapsed constraint representation.  The
    tradeoff is a modestly larger input (417 vs 313) and some redundancy,
    but the network can learn to ignore what it doesn't need.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LETTERS = 26
WORD_LENGTH = 5
NUM_STATUSES = 3          # gray=0, yellow=1, green=2
MAX_GUESSES = 6

# Subvector sizes
_HISTORY_DIM = NUM_LETTERS * WORD_LENGTH * NUM_STATUSES   # 390
_ABSENT_DIM = NUM_LETTERS                                  # 26
_TURN_DIM = 1                                              # 1

STATE_DIM = _HISTORY_DIM + _ABSENT_DIM + _TURN_DIM         # 417


# ---------------------------------------------------------------------------
# Primary encoder
# ---------------------------------------------------------------------------
def encode_state(guesses, feedbacks, turn):
    """
    Encode the current Wordle state as a 417-dim history vector.

    Args:
        guesses: list[str]
            Previous guesses (each a 5-letter word).
        feedbacks: list[list[int]]
            Feedback per guess, using 0=gray, 1=yellow, 2=green.
        turn: int
            Current turn number (0-5).

    Returns:
        np.ndarray of shape (STATE_DIM,) = (417,)
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)

    present_letters = set()   # ever seen yellow or green
    gray_letters = set()      # ever seen gray

    for guess, feedback in zip(guesses, feedbacks):
        guess = guess.lower()

        for pos, (letter, fb) in enumerate(zip(guess, feedback)):
            letter_idx = ord(letter) - ord('a')

            # History block: letter_idx * 15 + pos * 3 + status
            hist_offset = letter_idx * (WORD_LENGTH * NUM_STATUSES) \
                          + pos * NUM_STATUSES \
                          + fb                       # fb is already 0/1/2
            state[hist_offset] = 1.0

            if fb == 2 or fb == 1:
                present_letters.add(letter)
            elif fb == 0:
                gray_letters.add(letter)

    # Global absent flags
    truly_absent = gray_letters - present_letters
    for letter in truly_absent:
        letter_idx = ord(letter) - ord('a')
        state[_HISTORY_DIM + letter_idx] = 1.0

    # Normalized turn
    state[-1] = turn / float(MAX_GUESSES)

    return state


# ---------------------------------------------------------------------------
# Guess masking helper  (NEW — prevents repeat-guess loops)
# ---------------------------------------------------------------------------
def build_guess_mask(word_list, guesses):
    """
    Return a boolean mask over the vocabulary that is True for words
    that have NOT been guessed yet.  Use this to zero out Q-values for
    already-guessed words before taking argmax.

    Args:
        word_list: list[str]
            Full vocabulary (same list used to build word encodings).
        guesses: list[str]
            Words already guessed this game.

    Returns:
        np.ndarray of shape (len(word_list),), dtype bool
            True  = eligible (not yet guessed)
            False = already guessed (mask out)

    Usage in action selection:
        q_values = model(state_tensor)            # shape (vocab_size,)
        mask = build_guess_mask(word_list, guesses)
        q_values[~mask] = -float('inf')           # kill repeats
        action = q_values.argmax()
    """
    guessed_set = {g.lower() for g in guesses}
    return np.array([w.lower() not in guessed_set for w in word_list],
                    dtype=bool)


# ---------------------------------------------------------------------------
# Word encoder (unchanged — 130 dim)
# ---------------------------------------------------------------------------
def encode_words_onehot(word_list):
    """
    Create one-hot encodings for all words in the vocabulary.

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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- State encoding ----
    state = encode_state(
        guesses=["slate", "crane"],
        feedbacks=[[0, 0, 1, 0, 2], [2, 2, 2, 2, 2]],
        turn=2,
    )
    print(f"State vector shape: {state.shape}")
    print(f"State dim (expected 417): {STATE_DIM}")
    print(f"Non-zero entries: {np.count_nonzero(state)}")

    # Verify specific bits:
    #   "slate" feedback [0,0,1,0,2]
    #     s(pos0)=gray  -> idx for s=18, pos=0, status=0 => 18*15 + 0*3 + 0 = 270
    #     a(pos2)=yellow-> idx for a=0,  pos=2, status=1 => 0*15  + 2*3 + 1 = 7
    #     e(pos4)=green -> idx for e=4,  pos=4, status=2 => 4*15  + 4*3 + 2 = 74
    assert state[270] == 1.0, "s gray at pos 0"
    assert state[7] == 1.0,   "a yellow at pos 2"
    assert state[74] == 1.0,  "e green at pos 4"
    print("Spot checks passed.")

    # ---- Guess mask ----
    vocab = ["slate", "crane", "motel", "slink"]
    mask = build_guess_mask(vocab, ["slate", "crane"])
    print(f"\nGuess mask (after guessing slate, crane): {mask}")
    assert mask.tolist() == [False, False, True, True]
    print("Mask check passed.")

    # ---- Word encoding (unchanged) ----
    test_words = ["crane", "slate", "salet"]
    onehot = encode_words_onehot(test_words)
    print(f"\nWord encodings shape: {onehot.shape}")
    print(f"Non-zero per word: {onehot.sum(axis=1)}")
