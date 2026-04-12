"""
Wordle Game Engine

Core game logic shared by all solvers. Handles guess evaluation,
feedback generation, and word list filtering.

Feedback codes:
    0 = gray  (letter not in word)
    1 = yellow (letter in word, wrong position)
    2 = green  (letter in word, correct position)
"""

import os
import random


def load_word_list(filepath=None):
    """Load and return the word list as a list of lowercase 5-letter strings."""
    if filepath is None:
        # Look for wordle.txt relative to project root
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base, "wordle.txt")
    
    with open(filepath, "r") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    
    # Validate
    words = [w for w in words if len(w) == 5 and w.isalpha()]
    return words


def get_feedback(guess, target):
    """
    Compare a guess against the target word and return feedback.
    
    Args:
        guess: 5-letter string (the guessed word)
        target: 5-letter string (the hidden word)
    
    Returns:
        list of 5 ints: 0=gray, 1=yellow, 2=green
    
    Handles duplicate letters correctly per Wordle rules:
    - Green matches are assigned first
    - Yellow matches consume remaining unmatched target letters
    """
    guess = guess.lower()
    target = target.lower()
    feedback = [0] * 5
    target_remaining = list(target)
    
    # Pass 1: Mark greens
    for i in range(5):
        if guess[i] == target[i]:
            feedback[i] = 2
            target_remaining[i] = None  # consumed
    
    # Pass 2: Mark yellows
    for i in range(5):
        if feedback[i] == 2:
            continue
        if guess[i] in target_remaining:
            feedback[i] = 1
            # Consume the first available match
            target_remaining[target_remaining.index(guess[i])] = None
    
    return feedback


def filter_words(words, guess, feedback):
    """
    Filter word list based on guess and feedback.
    
    Args:
        words: list of candidate words
        guess: the guessed word
        feedback: list of 5 ints (0=gray, 1=yellow, 2=green)
    
    Returns:
        list of words consistent with the feedback
    """
    result = []
    for word in result_candidates(words, guess, feedback):
        result.append(word)
    return result


def result_candidates(words, guess, feedback):
    """Generator that yields words consistent with guess and feedback."""
    for word in words:
        if is_consistent(word, guess, feedback):
            yield word


def is_consistent(candidate, guess, feedback):
    """
    Check if a candidate word is consistent with a guess and its feedback.
    
    A candidate is consistent if, were it the target, the guess would
    produce the same feedback.
    """
    return get_feedback(guess, candidate) == feedback


class WordleGame:
    """
    A single game of Wordle.
    
    Usage:
        game = WordleGame(target="crane")
        feedback = game.make_guess("slate")
        # feedback = [0, 0, 1, 0, 2]
    """
    
    def __init__(self, target=None, word_list=None):
        """
        Initialize a game.
        
        Args:
            target: the hidden word (random if None)
            word_list: list of valid words (loads default if None)
        """
        if word_list is None:
            word_list = load_word_list()
        
        self.word_list = word_list
        self.target = target.lower() if target else random.choice(word_list)
        self.guesses = []       # list of guessed words
        self.feedbacks = []     # list of feedback for each guess
        self.turn = 0
        self.max_turns = 6
        self._solved = False
    
    def make_guess(self, guess):
        """
        Make a guess and return feedback.
        
        Args:
            guess: 5-letter string
        
        Returns:
            list of 5 ints (0=gray, 1=yellow, 2=green)
        
        Raises:
            ValueError: if game is already over
        """
        guess = guess.lower()
        
        if self._solved:
            raise ValueError("Game already solved!")
        if self.turn >= self.max_turns:
            raise ValueError("No guesses remaining!")
        if len(guess) != 5:
            raise ValueError(f"Guess must be 5 letters, got {len(guess)}")
        
        feedback = get_feedback(guess, self.target)
        self.guesses.append(guess)
        self.feedbacks.append(feedback)
        self.turn += 1
        
        if feedback == [2, 2, 2, 2, 2]:
            self._solved = True
        
        return feedback
    
    def is_solved(self):
        """Return True if the word has been guessed correctly."""
        return self._solved
    
    def is_over(self):
        """Return True if the game is over (solved or out of guesses)."""
        return self._solved or self.turn >= self.max_turns
    
    def get_state(self):
        """
        Return the current game state as a dictionary.
        
        Returns:
            dict with keys:
                - guesses: list of guessed words
                - feedbacks: list of feedback lists
                - turn: current turn number (0-5)
                - remaining: number of possible words remaining
        """
        return {
            "guesses": list(self.guesses),
            "feedbacks": list(self.feedbacks),
            "turn": self.turn,
        }
    
    def get_remaining_words(self):
        """Return words consistent with all guesses and feedbacks so far."""
        remaining = list(self.word_list)
        for guess, feedback in zip(self.guesses, self.feedbacks):
            remaining = filter_words(remaining, guess, feedback)
        return remaining
    
    def reset(self, target=None):
        """Reset the game with a new target word."""
        self.target = target.lower() if target else random.choice(self.word_list)
        self.guesses = []
        self.feedbacks = []
        self.turn = 0
        self._solved = False
        return self.get_state()


if __name__ == "__main__":
    # Quick test
    words = load_word_list()
    print(f"Loaded {len(words)} words")
    
    game = WordleGame(target="crane", word_list=words)
    
    test_guesses = ["slate", "crane"]
    for g in test_guesses:
        fb = game.make_guess(g)
        symbols = ["⬛", "🟨", "🟩"]
        display = " ".join(symbols[f] for f in fb)
        print(f"  {g.upper()}  {display}")
        if game.is_solved():
            print(f"  Solved in {game.turn} guesses!")
            break
