"""Read the project settings from config/config.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml

# The project root folder. This file is in <root>/src/sms_fraud/config.py,
# so we go up 2 folders. Why? Then paths work no matter which folder
# you start Python from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# PATTERN: Singleton — the YAML file is read only once; every later call returns the same dict object, so all modules share one config.
# lru_cache remembers the result of the first call. Later calls do not
# open the file again; they get the saved result.
@lru_cache(maxsize=1)
def get_config() -> dict:
    """Load config.yaml once and return it as a dictionary."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        # safe_load only reads simple data (text, numbers, lists).
        # It cannot run code, so it is safer than yaml.load.
        return yaml.safe_load(f)


def project_path(relative_path: str) -> Path:
    """Turn a path from the config (like "data/raw") into a full path."""
    return PROJECT_ROOT / relative_path
