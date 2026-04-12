"""
Solver Assistant — interactive Wordle helper.
Type a word or generate one, click tiles to set colors, get next guess.
"""

import streamlit as st
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from engine.wordle_env import load_word_list

st.set_page_config(page_title="Solver Assistant", page_icon="🎯", layout="centered")

# --- Load word list ---
@st.cache_resource
def get_words():
    return load_word_list(os.path.join(project_root, "wordle.txt"))

WORDS = get_words()
WORD_SET = set(WORDS)

# --- Load solvers ---
@st.cache_resource
def load_solvers():
    solvers = {}
    from solvers.frequency_solver import FrequencySolver
    solvers["Frequency Heuristic"] = FrequencySolver(os.path.join(project_root, "wordle.txt"))

    from solvers.infogain_solver import InfoGainSolver
    solvers["Information Gain"] = InfoGainSolver(os.path.join(project_root, "wordle.txt"))

    try:
        from solvers.tabular_q_solver import TabularQSolver
        q_path = os.path.join(project_root, "models", "q_table.pkl")
        if os.path.exists(q_path):
            from engine.word_lists import build_curated_lists
            s1, s2, s3, _ = build_curated_lists(os.path.join(project_root, "wordle.txt"), verbose=False)
            solvers["Tabular Q-Learning"] = TabularQSolver(
                q_table_path=q_path,
                word_list_path=os.path.join(project_root, "wordle.txt"),
                curated_words=s1
            )
    except Exception as e:
        pass

    try:
        from solvers.rollout_solver import RolloutSolver
        cache_path = os.path.join(project_root, "models", "rollout_cache.pkl")
        if os.path.exists(cache_path):
            solvers["Rollout"] = RolloutSolver(
                word_list_path=os.path.join(project_root, "wordle.txt"),
                top_k=10,
                cache_path=cache_path
            )
    except Exception as e:
        pass

    try:
        import torch
        from solvers.dqn_solver import DQNSolver
        model_path = os.path.join(project_root, "models", "dqn_model.pt")
        if os.path.exists(model_path):
            solvers["DQN"] = DQNSolver(
                model_path=model_path,
                word_list_path=os.path.join(project_root, "wordle.txt")
            )
        model_v2 = os.path.join(project_root, "models", "dqn_v2_model.pt")
        if os.path.exists(model_v2):
            solvers["DQN v2 (Shaped)"] = DQNSolver(
                model_path=model_v2,
                word_list_path=os.path.join(project_root, "wordle.txt")
            )
    except Exception as e:
        pass

    return solvers

solvers = load_solvers()

# --- CSS ---
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

.grid-row {
    display: flex;
    gap: 6px;
}

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
    border: 2px solid #555;
}

.grid-tile-empty {
    background-color: #121213;
    border: 2px solid #3a3a3c;
}

.grid-tile-gray { background-color: #3a3a3c; border-color: #3a3a3c; }
.grid-tile-yellow { background-color: #b59f3b; border-color: #b59f3b; }
.grid-tile-green { background-color: #538d4e; border-color: #538d4e; }

.suggestion-box {
    text-align: center;
    padding: 1em;
    margin: 1em 0;
    background: #1a1a2e;
    border: 2px solid #538d4e;
    border-radius: 8px;
}

.suggestion-word {
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    font-size: 2.5rem;
    letter-spacing: 0.2em;
    color: #538d4e;
}

.suggestion-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 0.3em;
}

.status-text {
    text-align: center;
    font-family: 'Outfit', sans-serif;
    color: #888;
    font-size: 0.9rem;
    margin: 0.5em 0;
}
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "grid_letters" not in st.session_state:
    st.session_state.grid_letters = [["" for _ in range(5)] for _ in range(6)]
if "grid_colors" not in st.session_state:
    # 0 = gray, 1 = yellow, 2 = green
    st.session_state.grid_colors = [[0 for _ in range(5)] for _ in range(6)]
if "current_row" not in st.session_state:
    st.session_state.current_row = 0
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "solver_name" not in st.session_state:
    st.session_state.solver_name = list(solvers.keys())[0] if solvers else "Frequency Heuristic"


def reset_game():
    st.session_state.grid_letters = [["" for _ in range(5)] for _ in range(6)]
    st.session_state.grid_colors = [[0 for _ in range(5)] for _ in range(6)]
    st.session_state.current_row = 0
    st.session_state.game_over = False
    if "word_input" in st.session_state:
        st.session_state.word_input = ""


def get_suggestion():
    """Get the solver's suggested next guess based on game history."""
    solver_name = st.session_state.solver_name
    if solver_name not in solvers:
        return "CARES"

    solver = solvers[solver_name]
    solver.reset()

    # Feed all completed rows to the solver
    for row in range(st.session_state.current_row):
        letters = st.session_state.grid_letters[row]
        colors = st.session_state.grid_colors[row]
        word = "".join(letters).lower()
        if len(word) == 5:
            solver.update(word, colors)

    return solver.get_guess().upper()


def render_grid():
    """Render the Wordle grid as HTML."""
    html = '<div class="grid-container">'
    for r in range(6):
        html += '<div class="grid-row">'
        for c in range(5):
            letter = st.session_state.grid_letters[r][c]
            color = st.session_state.grid_colors[r][c]

            if letter == "":
                css_class = "grid-tile grid-tile-empty"
            elif color == 0:
                css_class = "grid-tile grid-tile-gray"
            elif color == 1:
                css_class = "grid-tile grid-tile-yellow"
            else:
                css_class = "grid-tile grid-tile-green"

            html += f'<div class="{css_class}">{letter.upper()}</div>'
        html += '</div>'
    html += '</div>'
    return html


# --- Page Layout ---
st.markdown("## 🎯 Solver Assistant")

# Solver selector
solver_name = st.selectbox(
    "Choose Solver",
    list(solvers.keys()),
    key="solver_name"
)

# Suggestion
if not st.session_state.game_over:
    suggestion = get_suggestion()
    st.markdown(f"""
    <div class="suggestion-box">
        <div class="suggestion-label">SUGGESTED GUESS</div>
        <div class="suggestion-word">{suggestion}</div>
    </div>
    """, unsafe_allow_html=True)

# Render grid
st.markdown(render_grid(), unsafe_allow_html=True)

# Status
row = st.session_state.current_row
if st.session_state.game_over:
    # Check if solved (all green on last completed row)
    last_row = st.session_state.current_row - 1
    if last_row >= 0 and all(c == 2 for c in st.session_state.grid_colors[last_row]):
        st.success(f"🎉 Solved in {st.session_state.current_row} guesses!")
    else:
        st.error("Game over — 6 guesses used")
elif row > 0:
    st.markdown(f'<div class="status-text">Row {row + 1} of 6 — set colors for row {row} below, then enter next word</div>', unsafe_allow_html=True)

# Color toggles for current completed row
if row > 0 and not st.session_state.game_over:
    active_row = row - 1
    letters = st.session_state.grid_letters[active_row]

    if any(l != "" for l in letters):
        st.markdown(f"**Set colors for: {''.join(l.upper() for l in letters)}**")
        cols = st.columns(5)
        for c in range(5):
            with cols[c]:
                color_names = ["⬛ Gray", "🟨 Yellow", "🟩 Green"]
                current = st.session_state.grid_colors[active_row][c]
                new_color = st.selectbox(
                    letters[c].upper(),
                    [0, 1, 2],
                    index=current,
                    format_func=lambda x: color_names[x],
                    key=f"color_{active_row}_{c}"
                )
                st.session_state.grid_colors[active_row][c] = new_color

    # Check if all green → game won
    if all(c == 2 for c in st.session_state.grid_colors[active_row]):
        st.session_state.game_over = True
        st.rerun()

# Word input and buttons
if not st.session_state.game_over and row < 6:
    col_input, col_gen = st.columns([3, 1])

    with col_input:
        word = st.text_input(
            "Enter a word (5 letters)",
            max_chars=5,
            key="word_input",
            placeholder="Type a 5-letter word..."
        ).strip().lower()

    with col_gen:
        st.markdown("<br>", unsafe_allow_html=True)
        generate = st.button("📝 Generate", use_container_width=True)

    if generate and not word:
        word = suggestion.lower()

    if word and len(word) == 5:
        if word in WORD_SET or generate:
            # Place word in current row
            for c in range(5):
                st.session_state.grid_letters[row][c] = word[c]
                st.session_state.grid_colors[row][c] = 0  # gray default
            st.session_state.current_row = row + 1

            if row + 1 >= 6:
                st.session_state.game_over = True

            st.rerun()
        elif len(word) == 5:
            st.warning(f"'{word.upper()}' is not in the word list")

# Reset button
st.markdown("---")
if st.button("🔄 New Game", use_container_width=True):
    reset_game()
    st.rerun()
