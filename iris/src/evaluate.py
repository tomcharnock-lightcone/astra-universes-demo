"""Stub standing in for evaluating a classifier's macro-F1 on held-out iris data.

The score is made up but repeatable - it depends only on what the model was
trained with - and the record keeps the model it read.
"""

import argparse
import hashlib
import json
from pathlib import Path


def score(model: dict[str, str]) -> float:
    """A repeatable stand-in for macro-F1, between 0.85 and 0.95."""
    digest = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).digest()
    return round(0.85 + digest[0] / 255 * 0.1, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = json.loads(Path(args.model).read_text())
    record = {
        "stub": "evaluate",
        "metric": "f1_macro",
        "model": args.model,
        "trained_with": model,
        "value": score(model),
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
