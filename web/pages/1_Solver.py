"""
Solver Assistant — interactive Wordle helper.
Click tiles to cycle colors, type or generate guesses.
"""

import streamlit as st
import sys
import os

# Add project root to path (pages/ → web/ → project root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    wl = os.path.join(project_root, "wordle.txt")

    try:
        from solvers.frequency_solver import FrequencySolver
        solvers["Frequency Heuristic"] = FrequencySolver(wl)
    except:
        pass

    try:
        from solvers.infogain_solver import InfoGainSolver
        solvers["Information Gain"] = InfoGainSolver(wl)
    except:
        pass

    try:
        from solvers.tabular_q_solver import TabularQSolver
        q_path = os.path.join(project_root, "models", "q_table.pkl")
        if os.path.exists(q_path):
            solvers["Tabular Q-Learning"] = TabularQSolver(
                q_table_path=q_path,
                word_list_path=wl
            )
    except:
        pass

    try:
        from solvers.rollout_solver import RolloutSolver
        cache_path = os.path.join(project_root, "models", "rollout_cache.pkl")
        if os.path.exists(cache_path):
            solvers["Rollout"] = RolloutSolver(
                word_list_path=wl,
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
            solvers["DQN v1"] = DQNSolver(model_path=model_path, word_list_path=wl)
        model_v2 = os.path.join(project_root, "models", "dqn_v2_model.pt")
        if os.path.exists(model_v2):
            solvers["DQN v2 (Shaped)"] = DQNSolver(model_path=model_v2, word_list_path=wl)
    except:
        pass

    return solvers

solvers = load_solvers()

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;900&display=swap');

.suggestion-box {
    text-align: center;
    padding: 0.8em;
    margin: 0.5em 0;
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


def reset_game():
    st.session_state.grid_letters = [["" for _ in range(5)] for _ in range(6)]
    st.session_state.grid_colors = [[0 for _ in range(5)] for _ in range(6)]
    st.session_state.current_row = 0
    st.session_state.game_over = False


def get_suggestion():
    """Get solver's suggested guess based on completed rows."""
    solver_name = st.session_state.get("solver_select", list(solvers.keys())[0])
    if solver_name not in solvers:
        return "CARES"

    solver = solvers[solver_name]
    solver.reset()

    for row in range(st.session_state.current_row):
        letters = st.session_state.grid_letters[row]
        colors = st.session_state.grid_colors[row]
        word = "".join(letters).lower()
        if len(word) == 5 and all(l != "" for l in letters):
            solver.update(word, colors)

    return solver.get_guess().upper()


def place_word(word):
    """Place a word in the current empty row."""
    row = st.session_state.current_row
    if row >= 6 or st.session_state.game_over:
        return False
    word = word.lower().strip()
    if len(word) != 5:
        return False
    for c in range(5):
        st.session_state.grid_letters[row][c] = word[c]
        st.session_state.grid_colors[row][c] = 0
    st.session_state.current_row = row + 1
    if row + 1 >= 6:
        st.session_state.game_over = True
    return True


# --- Page Layout ---
st.markdown("## 🎯 Solver Assistant")

# Solver selector
solver_name = st.selectbox(
    "Choose Solver",
    list(solvers.keys()),
    key="solver_select"
)

# Suggestion box
if not st.session_state.game_over:
    suggestion = get_suggestion()
    st.markdown(f"""
    <div class="suggestion-box">
        <div class="suggestion-label">SUGGESTED GUESS</div>
        <div class="suggestion-word">{suggestion}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    last_row = st.session_state.current_row - 1
    if last_row >= 0 and all(c == 2 for c in st.session_state.grid_colors[last_row]):
        st.success(f"🎉 Solved in {st.session_state.current_row} guesses!")
    else:
        st.warning("Game over — 6 guesses used.")

# --- Wordle Grid ---
# Each filled tile is a button showing LETTER + color indicator
# Clicking cycles: gray → yellow → green → gray
# Empty tiles show as disabled gray squares

color_indicator = {0: "⬜", 1: "🟨", 2: "🟩"}

for r in range(6):
    cols = st.columns(5, gap="small")
    for c in range(5):
        letter = st.session_state.grid_letters[r][c]
        color = st.session_state.grid_colors[r][c]

        with cols[c]:
            if letter:
                # Filled tile — show letter with color indicator, clickable
                indicator = color_indicator[color]
                btn_label = f"{indicator} {letter.upper()}"

                if st.button(
                    btn_label,
                    key=f"tile_{r}_{c}",
                    use_container_width=True,
                ):
                    # Cycle color on click
                    if r < st.session_state.current_row:
                        st.session_state.grid_colors[r][c] = (color + 1) % 3
                        # Check win condition
                        if all(cc == 2 for cc in st.session_state.grid_colors[r]):
                            st.session_state.game_over = True
                        st.rerun()
            else:
                # Empty tile
                st.button(
                    "·",
                    key=f"tile_{r}_{c}",
                    use_container_width=True,
                    disabled=True
                )

# --- Input and Buttons ---
st.markdown("---")

if not st.session_state.game_over and st.session_state.current_row < 6:
    word_input = st.text_input(
        "Type a 5-letter word and press Enter",
        max_chars=5,
        key="word_input",
        placeholder="Type here or click Generate Guess..."
    ).strip().lower()

    col_gen, col_reset = st.columns(2)

    with col_gen:
        if st.button("📝 Generate Guess", use_container_width=True):
            place_word(suggestion.lower())
            st.rerun()

    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            reset_game()
            st.rerun()

    # Handle typed word on enter
    if word_input and len(word_input) == 5:
        if word_input in WORD_SET:
            place_word(word_input)
            st.rerun()
        else:
            st.error(f"'{word_input.upper()}' is not in the word list")
else:
    if st.button("🔄 Reset", use_container_width=True):
        reset_game()
        st.rerun()

# --- Instructions ---
with st.expander("How to use"):
    st.markdown("""
    1. **Choose a solver** from the dropdown
    2. **See the suggested guess** in the green box
    3. **Type a word** and press Enter, or click **Generate Guess** to use the suggestion
    4. **Click each tile** to cycle its color: ⬜ gray → 🟨 yellow → 🟩 green
    5. The suggestion updates based on your feedback
    6. When all tiles in a row are green, you win!

    *Use alongside the real Wordle — enter the same word, copy the colors, get the next suggestion.*
    """)
