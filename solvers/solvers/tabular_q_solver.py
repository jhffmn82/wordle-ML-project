"""
Solver 4: Tabular Q-Learning (Anderson & Meyer, 2022)

The model doesn't pick words directly — it learns which STRATEGY to use
at each game state. The state is simply (# greens, # yellows) from the
last guess, giving ~30 possible states.

5 strategies:
  0: Random from remaining words
  1: Pick from curated narrowing list (Set A)
  2: Best by letter frequency among remaining
  3: Smart (full filter using green/yellow/gray + frequency pick)
  4: Exclude-only (remove grays only, then frequency pick)

Reference:
    Anderson, B.J. & Meyer, J.G. (2022). arXiv:2202.00557.
"""

import os
import sys
import random
import pickle
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.wordle_env import WordleGame, load_word_list, get_feedback, filter_words


# ---- Strategy functions ----

def strategy_random(remaining, all_words, green_pos, yellow_pos, gray_letters, curated=None):
    """Pick a random word from remaining."""
    if remaining:
        return random.choice(remaining)
    return random.choice(all_words)


def strategy_curated(remaining, all_words, green_pos, yellow_pos, gray_letters, curated=None):
    """Pick from the curated narrowing word list (Set A). Skip words already eliminated."""
    if curated:
        valid = [w for w in curated if w in set(remaining)]
        if valid:
            return random.choice(valid)
        # If no curated words remain, pick any curated word for info gathering
        return random.choice(curated)
    return strategy_random(remaining, all_words, green_pos, yellow_pos, gray_letters)


def strategy_frequency(remaining, all_words, green_pos, yellow_pos, gray_letters, curated=None):
    """Pick the best word by letter frequency among remaining."""
    if not remaining:
        return random.choice(all_words)
    return _best_by_frequency(remaining, remaining)


def strategy_smart(remaining, all_words, green_pos, yellow_pos, gray_letters, curated=None):
    """Use all info (greens, yellows, grays) to filter, then pick by frequency."""
    if not remaining:
        return random.choice(all_words)
    if len(remaining) <= 2:
        return remaining[0]
    return _best_by_frequency(remaining, remaining)


def strategy_exclude(remaining, all_words, green_pos, yellow_pos, gray_letters, curated=None):
    """Only use gray info to exclude, then pick by frequency. Ignores yellow positions."""
    if not remaining:
        return random.choice(all_words)
    # Filter using only gray letters
    filtered = [w for w in all_words if not any(c in gray_letters for c in w)]
    if not filtered:
        filtered = remaining
    return _best_by_frequency(filtered, remaining)


def _best_by_frequency(candidates, reference_list):
    """Score by letter frequency in reference list, pick highest."""
    freq = Counter()
    for word in reference_list:
        for c in set(word):
            freq[c] += 1

    best_word = candidates[0]
    best_score = -1
    for word in candidates:
        score = sum(freq[c] for c in set(word))
        if score > best_score:
            best_score = score
            best_word = word
    return best_word


STRATEGIES = [strategy_random, strategy_curated, strategy_frequency, strategy_smart, strategy_exclude]
STRATEGY_NAMES = ["random", "curated", "frequency", "smart", "exclude"]


# ---- Solver ----

class TabularQSolver:

    def __init__(self, q_table_path=None, word_list_path=None, curated_words=None):
        self.all_words = load_word_list(word_list_path)
        self.remaining = list(self.all_words)
        self.curated = curated_words or []
        self.green_pos = [''] * 5
        self.yellow_pos = [''] * 5
        self.gray_letters = set()
        self.last_feedback = (0, 0)

        # Q-table: state -> array of 5 values
        self.q_table = defaultdict(lambda: np.zeros(5))

        if q_table_path and os.path.exists(q_table_path):
            with open(q_table_path, 'rb') as f:
                saved = pickle.load(f)
                self.q_table.update(saved)
            print(f"Loaded Q-table from {q_table_path} ({len(saved)} states)")
        else:
            if q_table_path:
                print("WARNING: No Q-table loaded. Solver will use random strategy selection.")

    def reset(self):
        self.remaining = list(self.all_words)
        self.green_pos = [''] * 5
        self.yellow_pos = [''] * 5
        self.gray_letters = set()
        self.last_feedback = (0, 0)

    def update(self, guess, feedback):
        guess = guess.lower()
        greens = sum(1 for f in feedback if f == 2)
        yellows = sum(1 for f in feedback if f == 1)
        self.last_feedback = (greens, yellows)

        for i in range(5):
            if feedback[i] == 2:
                self.green_pos[i] = guess[i]
            elif feedback[i] == 1:
                if guess[i] not in self.yellow_pos[i]:
                    self.yellow_pos[i] += guess[i]
            elif feedback[i] == 0:
                self.gray_letters.add(guess[i])

        self.remaining = filter_words(self.remaining, guess, feedback)

    def get_guess(self, game_state=None):
        if game_state is not None:
            self._sync_from_state(game_state)

        if len(self.remaining) <= 1:
            return self.remaining[0] if self.remaining else random.choice(self.all_words)

        # Pick the best strategy for current state
        state = self.last_feedback
        q_values = self.q_table[state]
        best_action = int(np.argmax(q_values))

        # Execute the chosen strategy
        strategy_fn = STRATEGIES[best_action]
        return strategy_fn(
            self.remaining, self.all_words,
            self.green_pos, self.yellow_pos, self.gray_letters,
            self.curated
        )

    def _sync_from_state(self, game_state):
        self.reset()
        for guess, feedback in zip(game_state["guesses"], game_state["feedbacks"]):
            self.update(guess, feedback)


# ---- Training ----

def compute_reward(feedback, solved, failed):
    reward = 0.0
    for fb in feedback:
        if fb == 2: reward += 5.0
        elif fb == 1: reward += 2.0
    if solved: reward += 25.0
    elif failed: reward -= 15.0
    return reward


def train_tabular_q(num_episodes=10000, alpha=0.02, gamma=0.05,
                    epsilon=0.3, curated_words=None, word_list_path=None,
                    log_interval=1000):
    """
    Train tabular Q-learning for Wordle strategy selection.

    Returns:
        q_table: dict mapping (greens, yellows) -> array of 5 Q-values
        history: training history dict
    """
    all_words = load_word_list(word_list_path)
    q_table = defaultdict(lambda: np.zeros(5))
    curated = curated_words or []

    history = {"episode": [], "win_rate": [], "avg_guesses": []}
    recent_wins = []
    recent_guesses = []

    for episode in range(num_episodes):
        target = random.choice(all_words)
        game = WordleGame(target=target, word_list=all_words)

        remaining = list(all_words)
        green_pos = [''] * 5
        yellow_pos = [''] * 5
        gray_letters = set()
        last_state = (0, 0)

        while not game.is_over():
            state = last_state

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                action = int(np.argmax(q_table[state]))

            # Execute strategy
            strategy_fn = STRATEGIES[action]
            guess = strategy_fn(remaining, all_words, green_pos, yellow_pos, gray_letters, curated)

            feedback = game.make_guess(guess)

            # Update game knowledge
            greens = sum(1 for f in feedback if f == 2)
            yellows = sum(1 for f in feedback if f == 1)
            next_state = (greens, yellows)

            for i in range(5):
                if feedback[i] == 2:
                    green_pos[i] = guess[i]
                elif feedback[i] == 1:
                    if guess[i] not in yellow_pos[i]:
                        yellow_pos[i] += guess[i]
                elif feedback[i] == 0:
                    gray_letters.add(guess[i])

            remaining = filter_words(remaining, guess, feedback)

            solved = game.is_solved()
            failed = game.turn >= 6 and not solved
            reward = compute_reward(feedback, solved, failed)
            done = solved or failed

            # Q-learning update
            old_q = q_table[state][action]
            if done:
                q_table[state][action] = old_q + alpha * (reward - old_q)
            else:
                best_next = np.max(q_table[next_state])
                q_table[state][action] = old_q + alpha * (reward + gamma * best_next - old_q)

            last_state = next_state

            if done:
                break

        recent_wins.append(1 if game.is_solved() else 0)
        recent_guesses.append(game.turn if game.is_solved() else 7)
        if len(recent_wins) > 1000:
            recent_wins.pop(0)
            recent_guesses.pop(0)

        if (episode + 1) % log_interval == 0:
            win_rate = sum(recent_wins) / len(recent_wins)
            avg_guess = sum(recent_guesses) / len(recent_guesses)
            history["episode"].append(episode + 1)
            history["win_rate"].append(win_rate)
            history["avg_guesses"].append(avg_guess)
            print(f"  Episode {episode+1}: win_rate={win_rate:.3f}, avg_guesses={avg_guess:.2f}")

    return dict(q_table), history


def play_game(solver, target, verbose=False):
    from engine.wordle_env import WordleGame
    game = WordleGame(target=target, word_list=solver.all_words)
    solver.reset()

    while not game.is_over():
        guess = solver.get_guess()
        feedback = game.make_guess(guess)
        solver.update(guess, feedback)

        if verbose:
            symbols = ["⬛", "🟨", "🟩"]
            display = " ".join(symbols[f] for f in feedback)
            state = solver.last_feedback
            action = int(np.argmax(solver.q_table[state]))
            print(f"  Turn {game.turn}: {guess.upper()}  {display}  "
                  f"({len(solver.remaining)} left, strategy={STRATEGY_NAMES[action]})")

        if game.is_solved():
            if verbose:
                print(f"  Solved in {game.turn}!")
            return game.turn

    if verbose:
        print(f"  Failed! Target was {target.upper()}")
    return 7


if __name__ == "__main__":
    print("Training tabular Q-learning...")
    q_table, history = train_tabular_q(num_episodes=10000, epsilon=0.3)

    print(f"\nLearned policy ({len(q_table)} states):")
    print(f"{'State':>12}  {'Best Strategy':>15}  Q-values")
    print("-" * 60)
    for state in sorted(q_table.keys()):
        best = int(np.argmax(q_table[state]))
        qvals = ", ".join(f"{v:.1f}" for v in q_table[state])
        print(f"  ({state[0]}g, {state[1]}y)  {STRATEGY_NAMES[best]:>15}  [{qvals}]")

    # Test
    solver = TabularQSolver()
    solver.q_table.update(q_table)

    test_words = ["crane", "slink", "nymph", "vivid", "fuzzy"]
    for word in test_words:
        print(f"\nTarget: {word.upper()}")
        play_game(solver, word, verbose=True)
