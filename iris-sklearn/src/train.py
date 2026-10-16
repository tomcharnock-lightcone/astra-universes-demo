"""Train a classifier on iris with scikit-learn.

The decisions arrive as option ids and are mapped to what they mean here.
The pickle holds the fitted pipeline, the held-out test set it's scored on
and the options it was trained with.
"""

import argparse
import importlib
import pickle
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

TEST_SIZES = {"small": 0.2, "medium": 0.3}
SEEDS = {"seed_42": 42, "seed_123": 123}


def load(source: str) -> Any:
    """Features and labels from a loader named like ``sklearn.datasets.load_iris``."""
    module, _, name = source.rpartition(".")
    loader = getattr(importlib.import_module(module), name)
    return loader(return_X_y=True)


def classifier(model: str, seed: int) -> Any:
    if model == "random_forest":
        return RandomForestClassifier(random_state=seed)
    if model == "svm":
        return SVC(random_state=seed)
    if model == "logistic":
        return LogisticRegression(max_iter=1000, random_state=seed)
    raise SystemExit(f"unknown model option '{model}'")


def scaler(scaling: str) -> list[Any]:
    if scaling == "standard":
        return [StandardScaler()]
    if scaling == "minmax":
        return [MinMaxScaler()]
    if scaling == "none":
        return []
    raise SystemExit(f"unknown scaling option '{scaling}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scaling", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-size", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seed = SEEDS[args.seed]
    features, labels = load(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZES[args.test_size], random_state=seed, stratify=labels
    )
    pipeline = make_pipeline(*scaler(args.scaling), classifier(args.model, seed))
    pipeline.fit(x_train, y_train)

    trained_with = {
        "data": args.data,
        "scaling": args.scaling,
        "model": args.model,
        "test_size": args.test_size,
        "seed": args.seed,
    }
    model = {"pipeline": pipeline, "test": (x_test, y_test), "trained_with": trained_with}
    with Path(args.out).open("wb") as out:
        pickle.dump(model, out)


if __name__ == "__main__":
    main()
