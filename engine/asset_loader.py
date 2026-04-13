import os
import pickle
from typing import Any


def load_pickle_asset(path: str) -> Any:
    """
    Load a pickle asset from disk.

    Args:
        path: Path to .pkl file

    Returns:
        Unpickled object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Asset not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def load_curated_sets(path: str) -> dict:
    """
    Load curated opening sets from a single pickle file.

    Expected format:
        {
            "set1": [...],
            "set2": [...],
            "set3": [...]
        }
    """
    data = load_pickle_asset(path)

    if not isinstance(data, dict):
        raise ValueError(f"Curated sets file must contain a dict, got {type(data)}")

    for key in ("set1", "set2"):
        if key not in data:
            raise ValueError(f"Missing required curated set: {key}")

    return data
