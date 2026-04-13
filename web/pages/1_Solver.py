"""
Solver Assistant — interactive Wordle helper.
Uses embedded HTML/JS for the grid, Streamlit for solver logic.
"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import json

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
    except: pass
    try:
        from solvers.infogain_solver import InfoGainSolver
        solvers["Information Gain"] = InfoGainSolver(wl)
    except: pass
    try:
        from solvers.tabular_q_solver import TabularQSolver
        q_path = os.path.join(project_root, "models", "q_table.pkl")
        if os.path.exists(q_path):
            solvers["Tabular Q-Learning"] = TabularQSolver(
                q_table_path=q_path, word_list_path=wl)
    except: pass
    try:
        from solvers.rollout_solver import RolloutSolver
        cache_path = os.path.join(project_root, "models", "rollout_cache.pkl")
        if os.path.exists(cache_path):
            solvers["Rollout"] = RolloutSolver(
                word_list_path=wl, top_k=10, cache_path=cache_path)
    except: pass
    try:
        import torch
        from solvers.dqn_solver import DQNSolver
        for name, fname in [("DQN v1", "dqn_model.pt"), ("DQN v2 (Shaped)", "dqn_v2_model.pt")]:
            path = os.path.join(project_root, "models", fname)
            if os.path.exists(path):
                solvers[name] = DQNSolver(model_path=path, word_list_path=wl)
    except: pass
    return solvers

solvers = load_solvers()

# --- Session State ---
if "grid_letters" not in st.session_state:
    st.session_state.grid_letters = [["" for _ in range(5)] for _ in range(6)]
if "grid_colors" not in st.session_state:
    st.session_state.grid_colors = [[0 for _ in range(5)] for _ in range(6)]
if "current_row" not in st.session_state:
    st.session_state.current_row = 0
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0


def reset_game():
    st.session_state.grid_letters = [["" for _ in range(5)] for _ in range(6)]
    st.session_state.grid_colors = [[0 for _ in range(5)] for _ in range(6)]
    st.session_state.current_row = 0
    st.session_state.game_over = False
    st.session_state.input_counter += 1


def get_suggestion():
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
    st.session_state.input_counter += 1
    if row + 1 >= 6:
        st.session_state.game_over = True
    return True


def build_grid_html():
    """Build a self-contained HTML/CSS Wordle grid."""
    colors_map = {0: "#3a3a3c", 1: "#b59f3b", 2: "#538d4e"}
    border_map = {0: "#3a3a3c", 1: "#b59f3b", 2: "#538d4e"}

    tiles_html = ""
    for r in range(6):
        tiles_html += '<div style="display:flex;gap:6px;margin-bottom:6px;justify-content:center;">'
        for c in range(5):
            letter = st.session_state.grid_letters[r][c]
            color = st.session_state.grid_colors[r][c]
            if letter:
                bg = colors_map[color]
                border = border_map[color]
                tiles_html += f'''<div style="
                    width:58px;height:58px;
                    background:{bg};border:2px solid {border};border-radius:4px;
                    display:flex;align-items:center;justify-content:center;
                    font-family:'Segoe UI',Arial,sans-serif;font-weight:700;font-size:1.8rem;
                    color:white;text-transform:uppercase;
                ">{letter.upper()}</div>'''
            elif r == st.session_state.current_row and not st.session_state.game_over:
                tiles_html += '''<div style="
                    width:58px;height:58px;
                    background:#272729;border:2px solid #565758;border-radius:4px;
                    display:flex;align-items:center;justify-content:center;
                "></div>'''
            else:
                tiles_html += '''<div style="
                    width:58px;height:58px;
                    background:#121213;border:2px solid #3a3a3c;border-radius:4px;
                    display:flex;align-items:center;justify-content:center;
                "></div>'''
        tiles_html += '</div>'

    return f'''<div style="display:flex;flex-direction:column;align-items:center;padding:10px 0;">
        {tiles_html}
    </div>'''


# ==================== PAGE LAYOUT ====================

st.markdown("## 🎯 Solver Assistant")
st.selectbox("Choose Solver", list(solvers.keys()), key="solver_select")

# Suggestion
if not st.session_state.game_over:
    suggestion = get_suggestion()
    st.markdown(f"""<div style="
        text-align:center;padding:12px;margin:8px 0;
        background:#1a1a2e;border:2px solid #538d4e;border-radius:8px;">
        <div style="font-size:0.85rem;color:#888;">SUGGESTED GUESS</div>
        <div style="font-family:'Segoe UI',Arial,sans-serif;font-weight:900;font-size:2.5rem;
            letter-spacing:0.2em;color:#538d4e;">{suggestion}</div>
    </div>""", unsafe_allow_html=True)
else:
    suggestion = ""
    last_row = st.session_state.current_row - 1
    if last_row >= 0 and all(c == 2 for c in st.session_state.grid_colors[last_row]):
        st.success(f"🎉 Solved in {st.session_state.current_row} guesses!")
    else:
        st.warning("Game over — 6 guesses used.")

# Grid display
st.markdown(build_grid_html(), unsafe_allow_html=True)

# Color controls for completed rows
for r in range(st.session_state.current_row):
    if all(c == 2 for c in st.session_state.grid_colors[r]):
        continue

    letters = st.session_state.grid_letters[r]
    word_str = "".join(l.upper() for l in letters)
    st.caption(f"Set colors for **{word_str}** (Row {r+1}):")

    cols = st.columns(5)
    color_options = ["⬛ Gray", "🟨 Yellow", "🟩 Green"]
    for c in range(5):
        with cols[c]:
            current_color = st.session_state.grid_colors[r][c]
            new_color = st.selectbox(
                f"{letters[c].upper()}",
                options=[0, 1, 2],
                index=current_color,
                format_func=lambda x: color_options[x],
                key=f"color_{r}_{c}_{st.session_state.input_counter}",
                label_visibility="collapsed"
            )
            if new_color != st.session_state.grid_colors[r][c]:
                st.session_state.grid_colors[r][c] = new_color
                if all(cc == 2 for cc in st.session_state.grid_colors[r]):
                    st.session_state.game_over = True
                st.rerun()

# Word input
st.markdown("---")

if not st.session_state.game_over and st.session_state.current_row < 6:
    with st.form(key=f"word_form_{st.session_state.input_counter}", clear_on_submit=True):
        word_input = st.text_input(
            "Type a 5-letter word",
            max_chars=5,
            placeholder="Type here and press Enter..."
        ).strip().lower()
        submitted = st.form_submit_button("Enter Word")
        if submitted and word_input:
            if len(word_input) != 5:
                st.error("Word must be 5 letters")
            elif word_input not in WORD_SET:
                st.error(f"'{word_input.upper()}' is not in the word list")
            else:
                place_word(word_input)
                st.rerun()

    col_gen, col_reset = st.columns(2)
    with col_gen:
        if st.button("📝 Generate Guess", use_container_width=True):
            if suggestion:
                place_word(suggestion.lower())
                st.rerun()
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            reset_game()
            st.rerun()
else:
    if st.button("🔄 Reset", use_container_width=True):
        reset_game()
        st.rerun()

with st.expander("How to use"):
    st.markdown("""
    1. **Choose a solver** from the dropdown
    2. **See the suggested guess** in the green box
    3. **Type a word** and press Enter, or click **Generate Guess**
    4. **Set tile colors** using the dropdowns below the grid
    5. Suggestion updates based on your feedback
    6. Repeat until solved!
    """)
