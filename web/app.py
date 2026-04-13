"""
Wordle ML Project — Dash Web Interface
"""

import sys
import os
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dash import Dash, html, dcc, callback, Input, Output, State, ctx, ALL, no_update
from engine.wordle_env import WordleGame, load_word_list, get_feedback, filter_words

# ---- Load resources ----
WORDS = load_word_list(os.path.join(PROJECT_ROOT, "wordle.txt"))
WORD_SET = set(WORDS)

SOLVERS = {}

def load_solvers():
    wl = os.path.join(PROJECT_ROOT, "wordle.txt")
    try:
        from solvers.frequency_solver import FrequencySolver
        SOLVERS["Frequency Heuristic"] = FrequencySolver(wl)
    except: pass
    try:
        from solvers.infogain_solver import InfoGainSolver
        SOLVERS["Information Gain"] = InfoGainSolver(wl)
    except: pass
    try:
        from solvers.tabular_q_solver import TabularQSolver
        q_path = os.path.join(PROJECT_ROOT, "models", "q_table.pkl")
        if os.path.exists(q_path):
            SOLVERS["Tabular Q-Learning"] = TabularQSolver(
                q_table_path=q_path, word_list_path=wl)
    except: pass
    try:
        from solvers.rollout_solver import RolloutSolver
        cache_path = os.path.join(PROJECT_ROOT, "models", "rollout_cache.pkl")
        if os.path.exists(cache_path):
            SOLVERS["Rollout"] = RolloutSolver(
                word_list_path=wl, top_k=10, cache_path=cache_path)
    except: pass
    try:
        import torch
        from solvers.dqn_solver import DQNSolver
        for name, fname in [("DQN v1", "dqn_model.pt"), ("DQN v2 (Shaped)", "dqn_v2_model.pt")]:
            path = os.path.join(PROJECT_ROOT, "models", fname)
            if os.path.exists(path):
                SOLVERS[name] = DQNSolver(model_path=path, word_list_path=wl)
    except: pass

load_solvers()
SOLVER_NAMES = list(SOLVERS.keys()) or ["Frequency Heuristic"]

# ---- Helper functions ----

def get_suggestion(game_state, solver_name):
    if solver_name not in SOLVERS:
        return "CARES"
    solver = SOLVERS[solver_name]
    solver.reset()
    letters = game_state["grid_letters"]
    colors = game_state["grid_colors"]
    current_row = game_state["current_row"]
    for r in range(current_row):
        word = "".join(letters[r]).lower()
        if len(word) == 5 and all(l != "" for l in letters[r]):
            solver.update(word, colors[r])
    return solver.get_guess().upper()


def make_empty_state():
    return {
        "grid_letters": [["", "", "", "", ""] for _ in range(6)],
        "grid_colors": [[0, 0, 0, 0, 0] for _ in range(6)],
        "current_row": 0,
        "game_over": False,
    }


def play_full_game(solver_name, target):
    if solver_name not in SOLVERS:
        return [], [], False
    solver = SOLVERS[solver_name]
    solver.reset()
    game = WordleGame(target=target, word_list=WORDS)
    guesses, feedbacks = [], []
    while not game.is_over():
        guess = solver.get_guess()
        feedback = game.make_guess(guess)
        solver.update(guess, feedback)
        guesses.append(guess)
        feedbacks.append(feedback)
        if game.is_solved():
            break
    return guesses, feedbacks, game.is_solved()


# ---- Build tile grid HTML ----

COLOR_BG = {0: "#3a3a3c", 1: "#b59f3b", 2: "#538d4e"}

def render_grid_html(letters, colors, current_row, game_over, clickable=True):
    """Build Wordle grid. If clickable, tiles in completed rows are buttons."""
    rows_html = []
    for r in range(6):
        tiles = []
        for c in range(5):
            letter = letters[r][c] if letters[r][c] else ""
            color = colors[r][c]
            if letter:
                bg = COLOR_BG[color]
                tiles.append(f'<div class="tile tile-{"gray" if color == 0 else "yellow" if color == 1 else "green"}">{letter.upper()}</div>')
            elif r == current_row and not game_over:
                tiles.append('<div class="tile tile-active"></div>')
            else:
                tiles.append('<div class="tile tile-empty"></div>')
        rows_html.append(f'<div class="wordle-row">{"".join(tiles)}</div>')
    return f'<div class="wordle-grid">{"".join(rows_html)}</div>'


# ---- App ----

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # For gunicorn

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content", className="app-container"),
], style={"background": "#121213", "minHeight": "100vh"})


# ---- Page Layouts ----

def home_layout():
    return html.Div([
        html.Div([
            html.Div("W", className="home-tile", style={"background": "#538d4e"}),
            html.Div("O", className="home-tile", style={"background": "#b59f3b"}),
            html.Div("R", className="home-tile", style={"background": "#538d4e"}),
            html.Div("D", className="home-tile", style={"background": "#3a3a3c"}),
            html.Div("L", className="home-tile", style={"background": "#538d4e"}),
            html.Div("E", className="home-tile", style={"background": "#b59f3b"}),
        ], className="home-tiles"),
        html.Div("ML PROJECT", className="page-title"),
        html.Div(
            "A comparative study of heuristic and machine learning approaches to solving Wordle. "
            "Six solvers — from simple letter frequency to deep reinforcement learning — "
            "tested against the theoretical optimal.",
            className="subtitle"
        ),
        html.Hr(className="divider"),
        html.Div([
            dcc.Link(html.Div([
                html.Div("🎯 Solver Assistant"),
                html.Div("Get help solving Wordle", className="nav-desc"),
            ]), href="/solver", className="nav-btn"),
            dcc.Link(html.Div([
                html.Div("▶️ Autoplay"),
                html.Div("Watch a solver play", className="nav-desc"),
            ]), href="/autoplay", className="nav-btn"),
            dcc.Link(html.Div([
                html.Div("📊 About"),
                html.Div("Results & methodology", className="nav-desc"),
            ]), href="/about", className="nav-btn"),
        ], className="nav-buttons"),
        html.Hr(className="divider"),
        html.Div(
            f"{len(SOLVERS)} Solvers · {len(WORDS):,} Words · Optimal: 3.421 avg guesses",
            className="footer-stats"
        ),
    ])


def solver_layout():
    return html.Div([
        html.Div("🎯 Solver Assistant", className="page-title", style={"fontSize": "1.8rem"}),

        # Solver dropdown
        html.Div([
            dcc.Dropdown(
                id="solver-dropdown",
                options=[{"label": s, "value": s} for s in SOLVER_NAMES],
                value=SOLVER_NAMES[0],
                clearable=False,
                style={"background": "#1a1a2e", "color": "#121213"},
            ),
        ], style={"marginBottom": "10px"}),

        # Suggestion box
        html.Div(id="suggestion-box"),

        # Game state store
        dcc.Store(id="game-state", data=make_empty_state()),

        # Grid — rendered as HTML, tiles are clickable via JS
        html.Div(id="grid-display"),

        # Color controls for active row
        html.Div(id="color-controls"),

        # Word input
        html.Div([
            dcc.Input(
                id="word-input",
                type="text",
                maxLength=5,
                placeholder="Type a 5-letter word...",
                className="word-input",
                debounce=True,
                n_submit=0,
            ),
        ], style={"margin": "12px 0"}),

        # Buttons
        html.Div([
            html.Button("📝 Generate Guess", id="btn-generate", className="btn btn-generate"),
            html.Button("🔄 Reset", id="btn-reset", className="btn btn-reset"),
        ], className="action-buttons"),

        # Status message
        html.Div(id="status-message"),

        # Back link
        html.Div([
            dcc.Link("← Back to Home", href="/", style={"color": "#888", "fontSize": "0.85rem"}),
        ], style={"textAlign": "center", "marginTop": "2em"}),
    ])


def autoplay_layout():
    return html.Div([
        html.Div("▶️ Autoplay", className="page-title", style={"fontSize": "1.8rem"}),

        html.Div([
            dcc.Dropdown(
                id="auto-solver-dropdown",
                options=[{"label": s, "value": s} for s in SOLVER_NAMES],
                value=SOLVER_NAMES[0],
                clearable=False,
                style={"background": "#1a1a2e", "color": "#121213"},
            ),
        ], style={"marginBottom": "10px"}),

        html.Div([
            dcc.Input(
                id="target-input",
                type="text",
                maxLength=5,
                placeholder="Type target word...",
                className="word-input",
                style={"flex": "3"},
            ),
            html.Button("🎲 Random", id="btn-random", className="btn btn-generate",
                        style={"flex": "1", "marginLeft": "10px"}),
        ], style={"display": "flex", "margin": "12px 0"}),

        html.Div([
            html.Button("▶️ Solve!", id="btn-solve", className="btn btn-generate",
                        style={"width": "100%"}),
        ], style={"margin": "12px 0"}),

        html.Div(id="autoplay-result"),

        html.Div([
            dcc.Link("← Back to Home", href="/", style={"color": "#888", "fontSize": "0.85rem"}),
        ], style={"textAlign": "center", "marginTop": "2em"}),
    ])


def about_layout():
    return html.Div([
        html.Div("📊 About", className="page-title", style={"fontSize": "1.8rem"}),
        html.Div([
            html.H3("Overview"),
            html.P(
                "A comparative study of heuristic and machine learning approaches to solving Wordle, "
                "developed as a graduate ML course project. We implement and evaluate six solvers "
                "ranging from simple letter frequency heuristics to deep reinforcement learning, "
                "benchmarked against the known optimal solution."
            ),
            html.H3("Results"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Solver"), html.Th("Type"), html.Th("Win Rate"), html.Th("Avg Guesses"),
                ])),
                html.Tbody([
                    html.Tr([html.Td("Frequency Heuristic"), html.Td("Heuristic"), html.Td("99.8%"), html.Td("3.707")]),
                    html.Tr([html.Td("Information Gain"), html.Td("Heuristic"), html.Td("100.0%"), html.Td("3.728")]),
                    html.Tr([html.Td("Tabular Q-Learning"), html.Td("RL (Learned)"), html.Td("98.4%"), html.Td("3.812")]),
                    html.Tr([html.Td("Rollout (POMDP)"), html.Td("RL (Planning)"), html.Td("100.0%"), html.Td("3.568")]),
                    html.Tr([html.Td("DQN v1 (Pure)"), html.Td("Deep RL"), html.Td("64.4%"), html.Td("4.536")]),
                    html.Tr([html.Td("DQN v2 (Shaped)"), html.Td("Deep RL"), html.Td("TBD"), html.Td("TBD")]),
                    html.Tr([
                        html.Td("Optimal", style={"fontWeight": "bold"}),
                        html.Td("Exact DP", style={"fontWeight": "bold"}),
                        html.Td("100%", style={"fontWeight": "bold"}),
                        html.Td("3.421", style={"fontWeight": "bold"}),
                    ]),
                ]),
            ]),
            html.H3("Key Findings"),
            html.P(
                "Understanding problem structure matters more than model complexity. "
                "Wordle's effective state space collapses to just 431 reachable game states "
                "under a good policy — trivially solved by dynamic programming but intractable "
                "for deep RL's exploration-based learning. A 543,000-parameter neural network (DQN) "
                "was our worst performer, while a 431-entry lookup table (rollout cache) achieved "
                "near-optimal results."
            ),
            html.H3("References"),
            html.Ol([
                html.Li("Bhambri, S. et al. (2022). RL Methods for Wordle. arXiv:2211.10298."),
                html.Li("Liu, C.-L. (2022). Using Wordle for Learning to Design and Compare Strategies. IEEE CoG."),
                html.Li("Bertsimas, D. & Paskov, A. (2024). An Exact and Interpretable Solution to Wordle. Operations Research."),
                html.Li("Anderson, B.J. & Meyer, J.G. (2022). Finding the optimal human strategy for Wordle. arXiv:2202.00557."),
                html.Li("Ho, A. (2022). Wordle Solving with Deep Reinforcement Learning."),
            ]),
            html.H3("Links"),
            html.P([
                html.A("GitHub Repository", href="https://github.com/jhffmn82/wordle-ML-project", target="_blank"),
                " · ",
                html.A("Evaluation Notebook", href="https://github.com/jhffmn82/wordle-ML-project/blob/main/notebooks/evaluation.ipynb", target="_blank"),
            ]),
        ], className="about-content"),
        html.Div([
            dcc.Link("← Back to Home", href="/", style={"color": "#888", "fontSize": "0.85rem"}),
        ], style={"textAlign": "center", "marginTop": "2em"}),
    ])


# ---- Routing ----

@callback(Output("page-content", "children"), Input("url", "pathname"))
def route(pathname):
    if pathname == "/solver":
        return solver_layout()
    elif pathname == "/autoplay":
        return autoplay_layout()
    elif pathname == "/about":
        return about_layout()
    return home_layout()


# ---- Solver Callbacks ----

@callback(
    Output("game-state", "data"),
    Input("word-input", "n_submit"),
    Input("btn-generate", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    Input({"type": "color-btn", "row": ALL, "col": ALL}, "n_clicks"),
    State("word-input", "value"),
    State("game-state", "data"),
    State("solver-dropdown", "value"),
    prevent_initial_call=True,
)
def update_game_state(n_submit, n_gen, n_reset, color_clicks, word_value, state, solver_name):
    triggered = ctx.triggered_id

    # Reset
    if triggered == "btn-reset":
        return make_empty_state()

    # Generate guess
    if triggered == "btn-generate":
        if state["game_over"] or state["current_row"] >= 6:
            return no_update
        suggestion = get_suggestion(state, solver_name)
        row = state["current_row"]
        for c in range(5):
            state["grid_letters"][row][c] = suggestion[c].lower()
            state["grid_colors"][row][c] = 0
        state["current_row"] = row + 1
        if row + 1 >= 6:
            state["game_over"] = True
        return state

    # Word submit
    if triggered == "word-input":
        word = (word_value or "").strip().lower()
        if len(word) != 5 or word not in WORD_SET:
            return no_update
        if state["game_over"] or state["current_row"] >= 6:
            return no_update
        row = state["current_row"]
        for c in range(5):
            state["grid_letters"][row][c] = word[c]
            state["grid_colors"][row][c] = 0
        state["current_row"] = row + 1
        if row + 1 >= 6:
            state["game_over"] = True
        return state

    # Color toggle
    if isinstance(triggered, dict) and triggered.get("type") == "color-btn":
        r = triggered["row"]
        c = triggered["col"]
        current = state["grid_colors"][r][c]
        state["grid_colors"][r][c] = (current + 1) % 3
        # Check win
        if all(cc == 2 for cc in state["grid_colors"][r]):
            state["game_over"] = True
        return state

    return no_update


@callback(
    Output("grid-display", "children"),
    Output("color-controls", "children"),
    Output("suggestion-box", "children"),
    Output("status-message", "children"),
    Output("word-input", "value"),
    Input("game-state", "data"),
    State("solver-dropdown", "value"),
)
def render_solver_page(state, solver_name):
    letters = state["grid_letters"]
    colors = state["grid_colors"]
    current_row = state["current_row"]
    game_over = state["game_over"]

    # Build grid using Dash components
    grid_rows = []
    for r in range(6):
        row_tiles = []
        for c in range(5):
            letter = letters[r][c] if letters[r][c] else ""
            color = colors[r][c]
            if letter:
                css_class = f"tile tile-{'gray' if color == 0 else 'yellow' if color == 1 else 'green'}"
                row_tiles.append(html.Div(letter.upper(), className=css_class))
            elif r == current_row and not game_over:
                row_tiles.append(html.Div("", className="tile tile-active"))
            else:
                row_tiles.append(html.Div("", className="tile tile-empty"))
        grid_rows.append(html.Div(row_tiles, className="wordle-row"))
    grid = html.Div(grid_rows, className="wordle-grid")

    # Color controls — buttons for each completed row
    color_controls = []
    for r in range(current_row):
        if all(c == 2 for c in colors[r]):
            continue
        word = "".join(l.upper() for l in letters[r])
        row_buttons = []
        for c in range(5):
            color = colors[r][c]
            color_emoji = ["⬜", "🟨", "🟩"]
            label = f"{color_emoji[color]} {letters[r][c].upper()}"
            bg = COLOR_BG[color]
            row_buttons.append(
                html.Button(
                    label,
                    id={"type": "color-btn", "row": r, "col": c},
                    style={
                        "background": bg, "color": "white", "border": f"2px solid {bg}",
                        "borderRadius": "4px", "padding": "8px 12px", "margin": "2px",
                        "fontFamily": "'Outfit', sans-serif", "fontWeight": "700",
                        "fontSize": "1.1rem", "cursor": "pointer", "flex": "1",
                    },
                )
            )
        color_controls.append(html.Div([
            html.Div(f"Set colors for {word}:", style={"color": "#888", "fontSize": "0.85rem", "textAlign": "center", "marginBottom": "4px"}),
            html.Div(row_buttons, style={"display": "flex", "gap": "4px", "justifyContent": "center"}),
        ], style={"marginBottom": "8px"}))

    # Suggestion
    if not game_over:
        suggestion = get_suggestion(state, solver_name)
        suggestion_box = html.Div([
            html.Div("SUGGESTED GUESS", className="suggestion-label"),
            html.Div(suggestion, className="suggestion-word"),
        ], className="suggestion-box")
    else:
        last_row = current_row - 1
        if last_row >= 0 and all(c == 2 for c in colors[last_row]):
            suggestion_box = html.Div(f"🎉 Solved in {current_row} guesses!", className="status-win")
        else:
            suggestion_box = html.Div("Game over — 6 guesses used", className="status-fail")

    # Status
    status = ""
    if not game_over and current_row > 0:
        remaining_count = "..."
        status = html.Div(f"Row {current_row + 1} of 6", className="status-info")

    # Clear input
    return grid, color_controls, suggestion_box, status, ""


# ---- Autoplay Callbacks ----

@callback(
    Output("target-input", "value"),
    Input("btn-random", "n_clicks"),
    prevent_initial_call=True,
)
def random_word(n):
    import random
    return random.choice(WORDS)


@callback(
    Output("autoplay-result", "children"),
    Input("btn-solve", "n_clicks"),
    State("target-input", "value"),
    State("auto-solver-dropdown", "value"),
    prevent_initial_call=True,
)
def run_autoplay(n_clicks, target, solver_name):
    if not target or len(target.strip()) != 5:
        return no_update
    target = target.strip().lower()
    if target not in WORD_SET:
        return html.Div(f"'{target.upper()}' is not in the word list", className="status-fail")

    guesses, feedbacks, solved = play_full_game(solver_name, target)

    # Build result grid
    letters = [list(g) for g in guesses] + [[""] * 5] * (6 - len(guesses))
    colors = [list(fb) for fb in feedbacks] + [[0] * 5] * (6 - len(feedbacks))

    # Build result grid using Dash components
    grid_rows = []
    for r in range(6):
        row_tiles = []
        for c in range(5):
            letter = letters[r][c] if letters[r][c] else ""
            color = colors[r][c]
            if letter:
                css_class = f"tile tile-{'gray' if color == 0 else 'yellow' if color == 1 else 'green'}"
                row_tiles.append(html.Div(letter.upper(), className=css_class))
            else:
                row_tiles.append(html.Div("", className="tile tile-empty"))
        grid_rows.append(html.Div(row_tiles, className="wordle-row"))

    result = [
        html.Div(f"Target: {target.upper()}", style={"textAlign": "center", "fontWeight": "700", "fontSize": "1.2rem", "margin": "10px 0"}),
        html.Div(grid_rows, className="wordle-grid"),
    ]

    if solved:
        result.append(html.Div(f"🎉 Solved in {len(guesses)} guesses!", className="status-win"))
    else:
        result.append(html.Div(f"❌ Failed after 6 guesses", className="status-fail"))

    # Details
    details = []
    for i, (g, fb) in enumerate(zip(guesses, feedbacks)):
        tiles = " ".join(["🟩" if f == 2 else "🟨" if f == 1 else "⬛" for f in fb])
        details.append(html.Div(f"Turn {i+1}: {g.upper()}  {tiles}", style={"fontFamily": "monospace", "margin": "2px 0"}))

    result.append(html.Details([
        html.Summary("Guess details", style={"cursor": "pointer", "color": "#888", "margin": "10px 0"}),
        html.Div(details, style={"padding": "10px", "background": "#1a1a2e", "borderRadius": "6px"}),
    ]))

    return html.Div(result)


# ---- Run ----

if __name__ == "__main__":
    app.run(debug=True, port=8050)
