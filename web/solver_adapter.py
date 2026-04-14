"""
Solver adapter — uniform interface between Flask API and Wordle solvers.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_word_list():
    from engine.wordle_env import load_word_list as _load
    return _load()

WORD_LIST = []

def _ensure_word_list():
    global WORD_LIST
    if not WORD_LIST:
        WORD_LIST = load_word_list()
    return WORD_LIST


def feedback_str_to_ints(fb_str):
    return [{'g': 2, 'y': 1, 'x': 0}[c] for c in fb_str.lower()]

def feedback_ints_to_str(fb_ints):
    return ''.join({2: 'g', 1: 'y', 0: 'x'}[i] for i in fb_ints)

def compute_feedback(guess, target):
    from engine.wordle_env import get_feedback
    return get_feedback(guess.lower(), target.lower())


SOLVERS = {}
_initialized = False

def _try_load_solvers():
    global _initialized
    if _initialized:
        return
    _initialized = True

    print("[adapter] Loading solvers...", flush=True)

    try:
        from solvers.frequency_solver import FrequencySolver
        SOLVERS["frequency"] = {
            "name": "Frequency Heuristic",
            "type": "Heuristic",
            "instance": FrequencySolver(),
        }
        print("[adapter] ✓ Frequency Heuristic")
    except Exception as e:
        print(f"[adapter] ✗ Frequency: {e}")

    try:
        from solvers.infogain_solver import InfoGainSolver
        SOLVERS["infogain"] = {
            "name": "Information Gain (Minimax)",
            "type": "Heuristic",
            "instance": InfoGainSolver(),
        }
        print("[adapter] ✓ Information Gain")
    except Exception as e:
        print(f"[adapter] ✗ InfoGain: {e}")

    try:
        from solvers.tabular_q_solver import TabularQSolver
        q_path = os.path.join(PROJECT_ROOT, "models", "q_table.pkl")
        SOLVERS["tabular_q"] = {
            "name": "Tabular Q-Learning",
            "type": "RL",
            "instance": TabularQSolver(q_table_path=q_path),
        }
        print("[adapter] ✓ Tabular Q-Learning")
    except Exception as e:
        print(f"[adapter] ✗ Tabular Q: {e}")

    try:
        from solvers.rollout_solver import RolloutSolver
        SOLVERS["rollout"] = {
            "name": "Rollout (POMDP)",
            "type": "RL / Planning",
            "instance": RolloutSolver(top_k=10),
        }
        print("[adapter] ✓ Rollout (POMDP)")
    except Exception as e:
        print(f"[adapter] ✗ Rollout: {e}")

    try:
        from solvers.dqn_solver import DQNSolver
        v1_path = os.path.join(PROJECT_ROOT, "models", "dqn_model.pt")
        if os.path.exists(v1_path):
            SOLVERS["dqn_v1"] = {
                "name": "DQN v1 (Pure)",
                "type": "Deep RL",
                "instance": DQNSolver(model_path=v1_path),
            }
            print("[adapter] ✓ DQN v1")
    except Exception as e:
        print(f"[adapter] ✗ DQN v1: {e}")

    try:
        from solvers.dqn_solver import DQNSolver
        v2_path = os.path.join(PROJECT_ROOT, "models", "dqn_v2_model.pt")
        if os.path.exists(v2_path):
            SOLVERS["dqn_v2"] = {
                "name": "DQN v2 (Reward Shaped)",
                "type": "Deep RL",
                "instance": DQNSolver(model_path=v2_path),
            }
            print("[adapter] ✓ DQN v2")
    except Exception as e:
        print(f"[adapter] ✗ DQN v2: {e}")

    print(f"[adapter] {len(SOLVERS)} solver(s) ready.", flush=True)


ALL_SOLVER_INFO = [
    {"id": "frequency",  "name": "Frequency Heuristic",       "type": "Heuristic"},
    {"id": "infogain",   "name": "Information Gain (Minimax)", "type": "Heuristic"},
    {"id": "tabular_q",  "name": "Tabular Q-Learning",        "type": "RL"},
    {"id": "rollout",    "name": "Rollout (POMDP)",            "type": "RL / Planning"},
    {"id": "dqn_v1",     "name": "DQN v1 (Pure)",             "type": "Deep RL"},
    {"id": "dqn_v2",     "name": "DQN v2 (Reward Shaped)",    "type": "Deep RL"},
]


def list_solvers():
    _try_load_solvers()
    result = []
    for s in ALL_SOLVER_INFO:
        entry = dict(s)
        entry["available"] = s["id"] in SOLVERS
        result.append(entry)
    return result


def suggest_guess(solver_id, game_history):
    _try_load_solvers()
    if solver_id not in SOLVERS:
        return {"error": f"Solver '{solver_id}' not available."}

    solver = SOLVERS[solver_id]["instance"]
    solver.reset()

    for entry in game_history:
        guess = entry["guess"].lower()
        fb_ints = feedback_str_to_ints(entry["feedback"])
        solver.update(guess, fb_ints)

    try:
        guess = solver.get_guess()
        return {"guess": guess.upper(), "solver": SOLVERS[solver_id]["name"]}
    except Exception as e:
        return {"error": f"Solver error: {str(e)}"}


def autoplay(solver_id, target):
    _try_load_solvers()
    _ensure_word_list()

    target = target.lower().strip()
    if target not in WORD_LIST:
        return {"error": f"'{target.upper()}' is not in the word list."}
    if solver_id not in SOLVERS:
        return {"error": f"Solver '{solver_id}' not available."}

    solver = SOLVERS[solver_id]["instance"]
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

        turns.app