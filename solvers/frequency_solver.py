"""
Solver 1: Frequency Heuristic

Ported from wordmaster_master2.py by jhoffman.

Strategy: Score each candidate word by:
    - How many remaining words contain each of its unique letters (frequency)
    - How often each letter appears at each specific position (positional frequency)
    - Final score = frequency_score + 3 * positional_score

When more than 2 words remain, also considers "exploration" words from the
full word list that might help narrow down the answer.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.wordle_env import load_word_list, get_feedback, filter_words


class FrequencySolver:
    """
    Letter frequency + positional frequency heuristic solver.
    """
    
    def __init__(self, word_list_path=None):
        """
        Args:
            word_list_path: path to wordle.txt (uses default if None)
        """
        self.all_words = load_word_list(word_list_path)
        self.remaining = list(self.all_words)
        self.known_green = [''] * 5     # confirmed letter at each position
        self.known_yellow = [''] * 5    # letters known NOT at each position but in word
    
    def reset(self):
        """Reset solver state for a new game."""
        self.remaining = list(self.all_words)
        self.known_green = [''] * 5
        self.known_yellow = [''] * 5
    
    def update(self, guess, feedback):
        """
        Update internal state based on a guess and its feedback.
        
        Args:
            guess: 5-letter string
            feedback: list of 5 ints (0=gray, 1=yellow, 2=green)
        """
        guess = guess.lower()
        
        for i in range(5):
            if feedback[i] == 2:  # green
                self.known_green[i] = guess[i]
            elif feedback[i] == 1:  # yellow
                if guess[i] not in self.known_yellow[i]:
                    self.known_yellow[i] += guess[i]
        
        # Filter remaining words
        self.remaining = filter_words(self.remaining, guess, feedback)
    
    def get_guess(self, game_state=None):
        """
        Return the best guess based on current knowledge.
        
        Args:
            game_state: optional dict from WordleGame.get_state()
                        (if provided, syncs internal state from it)
        
        Returns:
            str: the recommended 5-letter guess
        """
        if game_state is not None:
            self._sync_from_state(game_state)
        
        if len(self.remaining) <= 2:
            return self.remaining[0]
        
        # Compute letter frequencies in remaining words
        letter_freq = self._letter_frequency(self.remaining)
        positional_freq = self._letter_frequency_place(self.remaining)
        
        # Score remaining words (candidates that could be the answer)
        best_word = ''
        best_value = -1
        
        for word in self.remaining:
            value = self._word_value(word, letter_freq, positional_freq)
            if value > best_value:
                best_value = value
                best_word = word
        
        # Also check exploration words from the full list
        # (words that aren't possible answers but help narrow things down)
        if len(self.remaining) > 2:
            for word in self.all_words:
                value = self._word_value(word, letter_freq, positional_freq)
                if value > best_value:
                    best_value = value
                    best_word = word
        
        if best_word == '' and len(self.remaining) > 0:
            return self.remaining[0]
        
        return best_word
    
    def _sync_from_state(self, game_state):
        """Rebuild internal state from a WordleGame state dict."""
        self.remaining = list(self.all_words)
        self.known_green = [''] * 5
        self.known_yellow = [''] * 5
        
        for guess, feedback in zip(game_state["guesses"], game_state["feedbacks"]):
            self.update(guess, feedback)
    
    def _letter_frequency(self, words):
        """
        Count how many words contain each letter (ignoring known green letters).
        
        Returns:
            list of 26 ints (frequency for a-z)
        """
        freq = [0] * 26
        for word in words:
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                if letter in word and letter not in self.known_green:
                    freq[ord(letter) - ord('a')] += 1
        return freq
    
    def _letter_frequency_place(self, words):
        """
        Count how often each letter appears at each position
        (only for positions not yet solved green).
        
        Returns:
            list of 26 lists, each with 5 ints
        """
        freq = [[0] * 5 for _ in range(26)]
        for word in words:
            for pos in range(5):
                if self.known_green[pos] == '':
                    letter_idx = ord(word[pos]) - ord('a')
                    freq[letter_idx][pos] += 1
        return freq
    
    def _word_value(self, word, letter_freq, positional_freq):
        """
        Score a word based on letter frequency and positional frequency.
        
        Uses unique letters only (no double-counting).
        Score = sum(letter_freq) + 3 * sum(positional_freq)
        """
        unique_letters = set(word)
        
        # Letter frequency score
        freq_score = sum(
            letter_freq[ord(c) - ord('a')]
            for c in unique_letters
        )
        
        # Positional frequency score (only for letters not in known_green)
        pos_score = sum(
            positional_freq[ord(c) - ord('a')][word.index(c)]
            for c in unique_letters
            if c not in self.known_green
        )
        
        return freq_score + 3 * pos_score


def play_game(solver, target, word_list=None, verbose=False):
    """
    Play a complete game using the solver.
    
    Args:
        solver: a solver instance with get_guess() and update() methods
        target: the hidden word
        word_list: word list (uses solver's if None)
        verbose: print each guess
    
    Returns:
        int: number of guesses taken (7 if failed)
    """
    from engine.wordle_env import WordleGame
    
    game = WordleGame(target=target, word_list=word_list or solver.all_words)
    solver.reset()
    
    while not game.is_over():
        guess = solver.get_guess()
        feedback = game.make_guess(guess)
        solver.update(guess, feedback)
        
        if verbose:
            symbols = ["⬛", "🟨", "🟩"]
            display = " ".join(symbols[f] for f in feedback)
            print(f"  Turn {game.turn}: {guess.upper()}  {display}")
        
        if game.is_solved():
            if verbose:
                print(f"  Solved in {game.turn} guesses!")
            return game.turn
    
    if verbose:
        print(f"  Failed! Target was {target.upper()}")
    return 7  # Failed


if __name__ == "__main__":
    solver = FrequencySolver()
    
    # Test on a few words
    test_words = ["crane", "slink", "nymph", "vivid", "fuzzy"]
    for word in test_words:
        print(f"\nTarget: {word.upper()}")
        result = play_game(solver, word, verbose=True)
