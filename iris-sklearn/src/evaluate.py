"""Score a trained classifier's macro-F1 on its held-out iris test set."""

import argparse
import json
import pickle
from pathlib import Path

from sklearn.metrics import f1_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with Path(args.model).open("rb") as model_file:
        model = pickle.load(model_file)
    x_test, y_test = model["test"]
    predicted = model["pipeline"].predict(x_test)
    record = {
        "metric": "f1_macro",
        "model": args.model,
        "trained_with": model["trained_with"],
        "value": round(float(f1_score(y_test, predicted, average="macro")), 4),
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
