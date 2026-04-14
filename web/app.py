"""
Wordle ML Project — Flask web server for Hugging Face Spaces.
"""

import os
import random
from flask import Flask, render_template, jsonify, request

from solver_adapter import (
    list_solvers,
    suggest_guess,
    autoplay,
    _ensure_word_list,
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
    words = _ensure_word_list()
    if words:
        return jsonify({"word": random.choice(words).upper()})
    return jsonify({"error": "Word list not loaded"}), 500


@app.route("/api/words/validate/<word>")
def api_validate_word(word):
    _ensure_word_list()
    valid = word.lower().strip() in WORD_LIST
    return jsonify({"word": word.upper(), "valid": valid})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
