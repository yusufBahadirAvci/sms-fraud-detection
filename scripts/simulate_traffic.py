"""Send realistic traffic to the running API, so the prediction log has data.

1. 100 English messages (from the English test set)
2. 100 Turkish messages (from the Turkish dataset)

Run it with (the API must be running, e.g. "docker compose up -d"):
    python scripts/simulate_traffic.py

After this, logs/predictions.jsonl has: lines 1-100 English, lines 101-200 Turkish.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

N_MESSAGES = 100
# Send messages in small groups, like many users would, not all at once.
BATCH_SIZE = 10
# Fixed seed: the same messages are chosen every time (reproducible).
RANDOM_STATE = 42


def sample_messages(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, encoding="utf-8", keep_default_na=False)
    return df["message"].sample(n=N_MESSAGES, random_state=RANDOM_STATE).tolist()


def send(messages: list[str], name: str) -> None:
    """Send the messages in batches and print a short summary."""
    flagged = 0
    for start in range(0, len(messages), BATCH_SIZE):
        batch = messages[start:start + BATCH_SIZE]
        response = requests.post(f"{API_URL}/predict/batch", json={"messages": batch}, timeout=30)
        response.raise_for_status()
        flagged += sum(r["is_fraud"] for r in response.json()["results"])
    print(f"Sent {len(messages)} {name} messages -> {flagged} predicted as fraud")


def main() -> None:
    # Check first that the API is up, and give a clear message if not.
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.exceptions.ConnectionError:
        sys.exit(f"API not reachable at {API_URL}. Start it with: docker compose up -d")
    if not health.get("model_loaded"):
        sys.exit("API is running, but the model is not loaded.")

    send(sample_messages(PROJECT_ROOT / "data/processed/test.csv"), "English")
    send(sample_messages(PROJECT_ROOT / "data/processed/turkish.csv"), "Turkish")


if __name__ == "__main__":
    main()
