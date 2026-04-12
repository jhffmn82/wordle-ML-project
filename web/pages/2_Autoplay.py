"""
Autoplay — watch a solver play Wordle against any target word.
"""

import streamlit as st
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from engine.wordle_env import WordleGame, load_word_list
import random

st.set_page_config(page_title="Autoplay", page_icon="▶️", layout="centered")

@st.cache_resource
def get_words():
    return load_word_list(os.path.join(project_root, "wordle.txt"))

WORDS = get_words()
WORD_SET = set(WORDS)

@st.cache_resource
def load_solvers():
    solvers = {}
    from solvers.frequency_solver import FrequencySolver
    solvers["Frequency Heuristic"] = lambda: FrequencySolver(os.path.join(project_root, "wordle.txt"))

    from solvers.infogain_solver import InfoGainSolver
    solvers["Information Gain"] = lambda: InfoGainSolver(os.path.join(project_root, "wordle.txt"))

    try:
        from solvers.tabular_q_solver import TabularQSolver
        q_path = os.path.join(project_root, "models", "q_table.pkl")
        if os.path.exists(q_path):
            from engine.word_lists import build_curated_lists
            s1, s2, s3, _ = build_curated_lists(os.path.join(project_root, "wordle.txt"), verbose=False)
            solvers["Tabular Q-Learning"] = lambda: TabularQSolver(
                q_table_path=q_path,
                word_list_path=os.path.join(project_root, "wordle.txt"),
                curated_words=s1
            )
    except:
        pass

    try:
        from solvers.rollout_solver import RolloutSolver
        cache_path = os.path.join(project_root, "models", "rollout_cache.pkl")
        if os.path.exists(cache_path):
            solvers["Rollout"] = lambda: RolloutSolver(
                word_list_path=os.path.join(project_root, "wordle.txt"),
                top_k=10,
                cache_path=cache_path
            )
    except:
        pass

    try:
        import torch
        from solvers.dqn_solver import DQNSolver
        model_path = os.path.join(project_root, "models", "dqn_model.pt")
        if os.path.exists(model_path):
            solvers["DQN"] = lambda: DQNSolver(
                model_path=model_path,
                word_list_path=os.path.join(project_root, "wordle.txt")
            )
        model_v2 = os.path.join(project_root, "models", "dqn_v2_model.pt")
        if os.path.exists(model_v2):
            solvers["DQN v2 (Shaped)"] = lambda: DQNSolver(
                model_path=model_v2,
                word_list_path=os.path.join(project_root, "wordle.txt")
            )
    except:
        pass

    return solvers

solvers = load_solvers()

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;900&display=swap');

.grid-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    margin: 1em 0;
}

.grid-row { display: flex; gap: 6px; }

.grid-tile {
    width: 58px;
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    color: white;
    border-radius: 4px;
    text-transform: uppercase;
}

.grid-tile-empty { background-color: #121213; border: 2px solid #3a3a3c; }
.grid-tile-gray { background-color: #3a3a3c; border: 2px solid #3a3a3c; }
.grid-tile-yellow { background-color: #b59f3b; border: 2px solid #b59f3b; }
.grid-tile-green { background-color: #538d4e; border: 2px solid #538d4e; }
</style>
""", unsafe_allow_html=True)


def render_game(guesses, feedbacks):
    """Render played game as Wordle grid."""
    html = '<div class="grid-container">'
    for r in range(6):
        html += '<div class="grid-row">'
        for c in range(5):
            if r < len(guesses):
                letter = guesses[r][c]
                fb = feedbacks[r][c]
                if fb == 2:
                    css = "grid-tile grid-tile-green"
                elif fb == 1:
                    css = "grid-tile grid-tile-yellow"
                else:
                    css = "grid-tile grid-tile-gray"
                html += f'<div class="{css}">{letter.upper()}</div>'
            else:
                html += '<div class="grid-tile grid-tile-empty"> </div>'
        html += '</div>'
    html += '</div>'
    return html


def play_game(solver, target):
    """Play a game and return guesses + feedbacks."""
    game = WordleGame(target=target, word_list=WORDS)
    solver.reset()
    guesses = []
    feedbacks = []

    while not game.is_over():
        guess = solver.get_guess()
        feedback = game.make_guess(guess)
        solver.update(guess, feedback)
        guesses.append(guess)
        feedbacks.append(feedback)
        if game.is_solved():
            break

    solved = game.is_solved()
    return guesses, feedbacks, solved


# --- Page Layout ---
st.markdown("## ▶️ Autoplay")

solver_name = st.selectbox("Choose Solver", list(solvers.keys()))

col_word, col_rand = st.columns([3, 1])
with col_word:
    target_word = st.text_input(
        "Target word",
        max_chars=5,
        placeholder="Type a 5-letter word..."
    ).strip().lower()
with col_rand:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎲 Random", use_container_width=True):
        target_word = random.choice(WORDS)
        st.session_state["autoplay_target"] = target_word

# Use random word if button was pressed
if "autoplay_target" in st.session_state and not target_word:
    target_word = st.session_state["autoplay_target"]

if target_word:
    if target_word not in WORD_SET:
        st.warning(f"'{target_word.upper()}' is not in the word list")
    else:
        st.markdown(f"**Target: {target_word.upper()}**")

        solver = solvers[solver_name]()
        guesses, feedbacks, solved = play_game(solver, target_word)

        st.markdown(render_game(guesses, feedbacks), unsafe_allow_html=True)

        if solved:
            st.success(f"🎉 Solved in {len(guesses)} guesses!")
        else:
            st.error(f"❌ Failed! Used all 6 guesses.")

        # Show guess details
        with st.expander("Guess details"):
            for i, (g, fb) in enumerate(zip(guesses, feedbacks)):
                tiles = " ".join(["🟩" if f == 2 else "🟨" if f == 1 else "⬛" for f in fb])
                st.text(f"Turn {i+1}: {g.upper()}  {tiles}")
