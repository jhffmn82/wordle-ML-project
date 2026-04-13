"""
Solver adapter layer — uniform interface between Flask API and Wordle solvers.

Actual solver interface (from notebook):
    solver.reset()                      — clear state for new game
    solver.get_guess() -> str           — return next guess
    solver.update(guess, feedback)      — process feedback (list of ints: 2=green, 1=yellow, 0=gray)

This adapter:
    - Loads each solver + model weights on startup
    - Manages per-session solver instances (reset/update lifecycle)
    - Converts between web feedback format ('g','y','x') and solver format ([2,1,0])
    - Provides suggest + autoplay endpoints
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Word list
# ---------------------------------------------------------------------------

def load_word_list():
    path = os.path.join(PROJECT_ROOT, "wordle.txt")
    if os.path.exists(path):
        with open(path) as f:
            return [w.strip().lower() for w in f if w.strip()]
    try:
        from engine.wordle_env import load_word_list as _load
        return _load()
    except ImportError:
        return []

WORD_LIST = load_word_list()

# ---------------------------------------------------------------------------
# Feedback helpers
# ---------------------------------------------------------------------------

def feedback_str_to_ints(fb_str: str) -> list:
    """Convert 'ggyxx' -> [2, 2, 1, 0, 0]"""
    mapping = {'g': 2, 'y': 1, 'x': 0}
    return [mapping[c] for c in fb_str.lower()]

def feedback_ints_to_str(fb_ints: list) -> str:
    """Convert [2, 2, 1, 0, 0] -> 'ggyxx'"""
    mapping = {2: 'g', 1: 'y', 0: 'x'}
    return ''.join(mapping[i] for i in fb_ints)

def compute_feedback(guess: str, target: str) -> list:
    """Return feedback as int list: 2=green, 1=yellow, 0=gray."""
    result = [0] * 5
    target_chars = list(target)

    for i in range(5):
        if guess[i] == target_chars[i]:
            result[i] = 2
            target_chars[i] = None

    for i in range(5):
        if result[i] == 2:
            continue
        if guess[i] in target_chars:
            result[i] = 1
            target_chars[target_chars.index(guess[i])] = None

    return result


# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------

SOLVER_REGISTRY = {}
_initialized = False

def _create_solver(solver_id: str):
    """Create a fresh solver instance by ID."""
    info = SOLVER_REGISTRY.get(solver_id)
    if not info:
        return None
    return info["factory"]()


def _try_load_solvers():
    """Attempt to import and register each solver. Fails gracefully."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    # ---- Frequency Heuristic ----
    try:
        from solvers.frequency_solver import FrequencySolver
        SOLVER_REGISTRY["frequency"] = {
            "name": "Frequency Heuristic",
            "type": "Heuristic",
            "factory": lambda: FrequencySolver(),
        }
        print("[solver_adapter] Loaded: Frequency Heuristic")
    except Exception as e:
        print(f"[solver_adapter] Could not load frequency solver: {e}")

    # ---- Information Gain / Minimax ----
    try:
        from solvers.infogain_solver import InfoGainSolver
        SOLVER_REGISTRY["infogain"] = {
            "name": "Information Gain (Minimax)",
            "type": "Heuristic",
            "factory": lambda: InfoGainSolver(),
        }
        print("[solver_adapter] Loaded: Information Gain")
    except Exception as e:
        print(f"[solver_adapter] Could not load infogain solver: {e}")

    # ---- Tabular Q-Learning ----
    try:
        import pickle
        from solvers.tabular_q_solver import TabularQSolver
        q_path = os.path.join(PROJECT_ROOT, "models", "q_table.pkl")
        set3_path = os.path.join(PROJECT_ROOT, "models", "curated_set3.pkl")

        if not os.path.exists(q_path):
            raise FileNotFoundError("q_table.pkl not found")

        # Load cached curated words (saves ~400s vs rebuild)
        if os.path.exists(set3_path):
            with open(set3_path, "rb") as f:
                _set3 = pickle.load(f)
            print(f"[solver_adapter] Loaded cached curated set3: {len(_set3)} words")
        else:
            print("[solver_adapter] curated_set3.pkl not found, rebuilding (slow)...")
            from engine.word_lists import build_curated_lists
            _set1, _set2, _set3, _ = build_curated_lists()
            # Cache for next time
            with open(set3_path, "wb") as f:
                pickle.dump(_set3, f)
            print(f"[solver_adapter] Built and cached set3: {len(_set3)} words")

        SOLVER_REGISTRY["tabular_q"] = {
            "name": "Tabular Q-Learning",
            "type": "RL",
            "factory": lambda: TabularQSolver(q_table_path=q_path, curated_words=_set3),
        }
        print("[solver_adapter] Loaded: Tabular Q-Learning")
    except Exception as e:
        print(f"[solver_adapter] Could not load tabular Q solver: {e}")

    # ---- Rollout (POMDP) ----
    try:
        from solvers.rollout_solver import RolloutSolver
        SOLVER_REGISTRY["rollout"] = {
            "name": "Rollout (POMDP)",
            "type": "RL / Planning",
            "factory": lambda: RolloutSolver(top_k=10),
        }
        print("[solver_adapter] Loaded: Rollout (POMDP)")
    except Exception as e:
        print(f"[solver_adapter] Could not load rollout solver: {e}")

    # ---- DQN v1 ----
    try:
        from solvers.dqn_solver import DQNSolver
        v1_path = os.path.join(PROJECT_ROOT, "models", "dqn_model.pt")
        if os.path.exists(v1_path):
            SOLVER_REGISTRY["dqn_v1"] = {
                "name": "DQN v1 (Pure)",
                "type": "Deep RL",
                "factory": lambda: DQNSolver(model_path=v1_path),
            }
            print("[solver_adapter] Loaded: DQN v1")
    except Exception as e:
        print(f"[solver_adapter] Could not load DQN v1 solver: {e}")

    # ---- DQN v2 ----
    try:
        from solvers.dqn_solver import DQNSolver
        v2_path = os.path.join(PROJECT_ROOT, "models", "dqn_v2_model.pt")
        if os.path.exists(v2_path):
            SOLVER_REGISTRY["dqn_v2"] = {
                "name": "DQN v2 (Reward Shaped)",
                "type": "Deep RL",
                "factory": lambda: DQNSolver(model_path=v2_path),
            }
            print("[solver_adapter] Loaded: DQN v2")
    except Exception as e:
        print(f"[solver_adapter] Could not load DQN v2 solver: {e}")

    if not SOLVER_REGISTRY:
        print("[solver_adapter] WARNING: No solvers loaded — running in demo mode.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_solvers() -> list:
    """Return list of {id, name, type, available} for all solvers."""
    _try_load_solvers()
    ALL_SOLVERS = [
        {"id": "frequency",  "name": "Frequency Heuristic",       "type": "Heuristic"},
        {"id": "infogain",   "name": "Information Gain (Minimax)", "type": "Heuristic"},
        {"id": "tabular_q",  "name": "Tabular Q-Learning",        "type": "RL"},
        {"id": "rollout",    "name": "Rollout (POMDP)",            "type": "RL / Planning"},
        {"id": "dqn_v1",     "name": "DQN v1 (Pure)",             "type": "Deep RL"},
        {"id": "dqn_v2",     "name": "DQN v2 (Reward Shaped)",    "type": "Deep RL"},
    ]
    for s in ALL_SOLVERS:
        s["available"] = s["id"] in SOLVER_REGISTRY
    return ALL_SOLVERS


def suggest_guess(solver_id: str, game_history: list) -> dict:
    """
    Get the solver's suggested next guess.

    Args:
        solver_id: key from SOLVER_REGISTRY
        game_history: list of {"guess": str, "feedback": str} dicts
                      feedback is 5 chars of g/y/x

    Returns:
        {"guess": str, "solver": str} or {"error": str}
    """
    _try_load_solvers()

    if solver_id not in SOLVER_REGISTRY:
        return {"error": f"Solver '{solver_id}' not available. Deploy with solver files to enable."}

    # Create a fresh solver and replay the game history
    solver = _create_solver(solver_id)
    if solver is None:
        return {"error": f"Could not create solver '{solver_id}'."}

    solver.reset()

    for entry in game_history:
        guess = entry["guess"].lower()
        fb_str = entry["feedback"].lower()
        fb_ints = feedback_str_to_ints(fb_str)
        # Feed the solver the guess and its feedback
        # We need to set up the solver as if it made this guess
        solver.update(guess, fb_ints)

    try:
        guess = solver.get_guess()
        return {"guess": guess.upper(), "solver": SOLVER_REGISTRY[solver_id]["name"]}
    except Exception as e:
        return {"error": f"Solver error: {str(e)}"}


def autoplay(solver_id: str, target: str) -> dict:
    """
    Run a full game: solver vs target word.

    Returns:
        {
            "target": str,
            "solver": str,
            "turns": [{"guess": str, "feedback": str, "remaining": int}],
            "won": bool,
            "num_guesses": int
        }
    """
    _try_load_solvers()

    target = target.lower().strip()
    if target not in WORD_LIST:
        return {"error": f"'{target.upper()}' is not in the word list."}

    if solver_id not in SOLVER_REGISTRY:
        return {"error": f"Solver '{solver_id}' not available."}

    solver = _create_solver(solver_id)
    solver.reset()
    turns = []

    for turn in range(6):
        try:
            guess = solver.get_guess().lower()
        except Exception as e:
            return {"error": f"Solver crashed on turn {turn+1}: {str(e)}"}

        fb_ints = compute_feedback(guess, target)
        fb_str = feedback_ints_to_str(fb_ints)
        solver.update(guess, fb_ints)

        turns.append({
            "guess": guess.upper(),
            "feedback": fb_str,
        })

        if fb_ints == [2, 2, 2, 2, 2]:
            return {
                "target": target.upper(),
                "solver": SOLVER_REGISTRY[solver_id]["name"],
                "turns": turns,
                "won": True,
                "num_guesses": turn + 1,
            }

    return {
        "target": target.upper(),
        "solver": SOLVER_REGISTRY[solver_id]["name"],
        "turns": turns,
        "won": False,
        "num_guesses": 6,
    }
