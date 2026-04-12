"""
State Encoder for Learned Models

Converts Wordle game state (guesses + feedbacks) into a numerical vector
suitable for neural network input.

Based on Andrew Ho's encoding scheme:
- 26 letters × 5 positions × 3 states (green/yellow/gray) = 390 features
- Plus 26 features for global letter status (confirmed absent)
- Plus 1 feature for turn number
- Total: 417 features

Reference:
    Ho, A. (2022). Wordle Solving with Deep Reinforcement Learning.
    Anderson, B.J. & Meyer, J.G. (2022). arXiv:2202.00557.
"""

import numpy as np


# Constants
NUM_LETTERS = 26
WORD_LENGTH = 5
STATE_DIM = NUM_LETTERS * WORD_LENGTH * 3 + NUM_LETTERS + 1  # 417


def encode_state(guesses, feedbacks, turn):
    """
    Encode the current game state as a numerical vector.
    
    Args:
        guesses: list of guessed words (strings)
        feedbacks: list of feedback lists (each is 5 ints: 0/1/2)
        turn: current turn number (0-5)
    
    Returns:
        numpy array of shape (STATE_DIM,) = (417,)
    
    Encoding layout:
        [0:390]   - For each letter (26) × position (5) × status (3):
                    status 0: confirmed NOT here (gray at this position)
                    status 1: confirmed IN word but NOT here (yellow at this position)  
                    status 2: confirmed HERE (green at this position)
        [390:416] - Global: letter confirmed completely absent (gray + never yellow/green)
        [416]     - Turn number (0-5)
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)
    
    # Track what we know about each letter
    green_positions = {}    # letter -> set of positions confirmed green
    yellow_positions = {}   # letter -> set of positions confirmed yellow
    gray_letters = set()    # letters confirmed absent
    present_letters = set() # letters known to be in word (green or yellow)
    
    for guess, feedback in zip(guesses, feedbacks):
        for pos, (letter, fb) in enumerate(zip(guess.lower(), feedback)):
            letter_idx = ord(letter) - ord('a')
            
            if fb == 2:  # Green
                idx = (letter_idx * WORD_LENGTH + pos) * 3 + 2
                state[idx] = 1.0
                present_letters.add(letter)
                if letter not in green_positions:
                    green_positions[letter] = set()
                green_positions[letter].add(pos)
                
            elif fb == 1:  # Yellow
                idx = (letter_idx * WORD_LENGTH + pos) * 3 + 1
                state[idx] = 1.0
                present_letters.add(letter)
                if letter not in yellow_positions:
                    yellow_positions[letter] = set()
                yellow_positions[letter].add(pos)
                
            elif fb == 0:  # Gray
                idx = (letter_idx * WORD_LENGTH + pos) * 3 + 0
                state[idx] = 1.0
                gray_letters.add(letter)
    
    # Global absent letters (gray and never seen as yellow/green)
    truly_absent = gray_letters - present_letters
    for letter in truly_absent:
        letter_idx = ord(letter) - ord('a')
        state[390 + letter_idx] = 1.0
    
    # Turn number
    state[416] = turn / 6.0  # Normalize to [0, 1]
    
    return state


def encode_words_onehot(word_list):
    """
    Create one-hot encodings for all words in the vocabulary.
    
    Args:
        word_list: list of 5-letter words
    
    Returns:
        numpy array of shape (len(word_list), 130)
        Each word encoded as 5 positions × 26 letters = 130 binary features
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
