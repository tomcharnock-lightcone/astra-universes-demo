"""Stub standing in for training a classifier on iris.

In place of a pickled model it writes a JSON record of what it was given, so
the files on disk show which data and decisions each universe's model was
trained with.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scaling", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-size", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    record = {
        "stub": "train",
        "data": args.data,
        "scaling": args.scaling,
        "model": args.model,
        "test_size": args.test_size,
        "seed": args.seed,
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
