"""
Wordle ML Project — Flask web server for Hugging Face Spaces.

Routes:
    GET  /                  → SPA (index.html)
    GET  /api/solvers       → list available solvers
    POST /api/suggest       → get solver's next guess
    POST /api/autoplay      → run full game
    GET  /api/words/random  → random target word
    GET  /api/words/validate/<word> → check if word is in list
"""

import os
import random
from flask import Flask, render_template, jsonify, request

from solver_adapter import (
    list_solvers,
    suggest_guess,
    autoplay,
    WORD_LIST,
)

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/solvers")
def api_solvers():
    return jsonify(list_solvers())


@app.route("/api/suggest", methods=["POST"])
def api_suggest():
    data = request.get_json(force=True)
    solver_id = data.get("solver", "frequency")
    history = data.get("history", [])
    return jsonify(suggest_guess(solver_id, history))


@app.route("/api/autoplay", methods=["POST"])
def api_autoplay():
    data = request.get_json(force=True)
    solver_id = data.get("solver", "frequency")
    target = data.get("target", "")
    return jsonify(autoplay(solver_id, target))


@app.route("/api/words/random")
def api_random_word():
    if WORD_LIST:
        return jsonify({"word": random.choice(WORD_LIST).upper()})
    return jsonify({"error": "Word list not loaded"}), 500


@app.route("/api/words/validate/<word>")
def api_validate_word(word):
    valid = word.lower().strip() in WORD_LIST
    return jsonify({"word": word.upper(), "valid": valid})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
