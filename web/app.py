"""
Wordle ML Project — Web Interface
Home page with navigation to Solver Assistant, Autoplay, and About.
"""

import streamlit as st

st.set_page_config(
    page_title="Wordle ML Project",
    page_icon="🟩",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;900&display=swap');

.main-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    font-size: 3.5rem;
    text-align: center;
    letter-spacing: 0.15em;
    margin-bottom: 0.2em;
}

.tile-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-bottom: 8px;
}

.tile {
    width: 56px;
    height: 56px;
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

.tile-green { background-color: #538d4e; }
.tile-yellow { background-color: #b59f3b; }
.tile-gray { background-color: #3a3a3c; }

.subtitle {
    font-family: 'Outfit', sans-serif;
    font-weight: 400;
    font-size: 1.1rem;
    text-align: center;
    color: #888;
    margin-bottom: 2em;
    line-height: 1.6;
}

.nav-button {
    display: block;
    width: 100%;
    padding: 1.2em 1.5em;
    margin-bottom: 12px;
    background: #1a1a2e;
    border: 2px solid #333;
    border-radius: 8px;
    color: white;
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: left;
    cursor: pointer;
    transition: all 0.2s ease;
}

.nav-button:hover {
    border-color: #538d4e;
    background: #1e1e3a;
}

.nav-desc {
    font-weight: 400;
    font-size: 0.85rem;
    color: #888;
    margin-top: 4px;
}

.divider {
    border: none;
    border-top: 1px solid #333;
    margin: 2em 0;
}
</style>
""", unsafe_allow_html=True)

# Title with Wordle-style tiles
st.markdown("""
<div class="tile-row">
    <div class="tile tile-green">W</div>
    <div class="tile tile-yellow">O</div>
    <div class="tile tile-green">R</div>
    <div class="tile tile-gray">D</div>
    <div class="tile tile-green">L</div>
    <div class="tile tile-yellow">E</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">ML PROJECT</div>', unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
    A comparative study of heuristic and machine learning approaches to solving Wordle.<br>
    Six solvers — from simple letter frequency to deep reinforcement learning — tested
    against the theoretical optimal.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Navigation buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎯 Solver Assistant", use_container_width=True):
        st.switch_page("pages/1_Solver.py")
    st.caption("Get help solving today's Wordle. Enter your guesses, mark the colors, get the next best word.")

with col2:
    if st.button("▶️ Autoplay", use_container_width=True):
        st.switch_page("pages/2_Autoplay.py")
    st.caption("Watch any solver play Wordle. Enter a target word and see the strategy unfold.")

with col3:
    if st.button("📊 About", use_container_width=True):
        st.switch_page("pages/3_About.py")
    st.caption("Project writeup, methodology, performance results, and comparison charts.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Quick stats
st.markdown("""
<div style="text-align: center; color: #888; font-family: 'Outfit', sans-serif; font-size: 0.9rem;">
    <strong>6 Solvers</strong> · <strong>2,983 Words</strong> · <strong>Optimal: 3.421 avg guesses</strong>
</div>
""", unsafe_allow_html=True)
