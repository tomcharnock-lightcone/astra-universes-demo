"""RFC-0003's extended iris example.

``f1_score`` is built once per universe and ``model_comparison`` once from
``results/rf_standard/`` and ``results/svm_standard/``. Each test runs the
runner on a fresh copy of the project. Run with ``pytest`` once
``requirements.txt`` is installed.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).parent


def copy_project(name: str, tmp_path: Path) -> Path:
    shutil.copytree(DEMO / name, tmp_path / name, ignore=shutil.ignore_patterns("results"))
    return tmp_path / name


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return copy_project("iris", tmp_path)


@pytest.fixture
def sklearn_project(tmp_path: Path) -> Path:
    pytest.importorskip("sklearn")
    return copy_project("iris-sklearn", tmp_path)


def run(project: Path, *targets: str) -> str:
    result = subprocess.run(
        [sys.executable, str(DEMO / "run.py"), str(project), *targets],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def built(project: Path) -> list[str]:
    return sorted(
        str(p.relative_to(project)) for p in (project / "results").rglob("*") if p.is_file()
    )


def test_f1_is_built_per_universe_and_the_comparison_once(project: Path):
    run(project)
    assert built(project) == [
        "results/model_comparison.json",
        "results/rf_standard/f1_score.json",
        "results/rf_standard/trained_output.pkl",
        "results/svm_standard/f1_score.json",
        "results/svm_standard/trained_output.pkl",
    ]
    comparison = json.loads((project / "results/model_comparison.json").read_text())
    assert comparison["a"]["path"] == "results/rf_standard/f1_score.json"
    assert comparison["b"]["path"] == "results/svm_standard/f1_score.json"
    assert (comparison["a"]["model"], comparison["b"]["model"]) == ("random_forest", "svm")


def test_a_diff_universe_inherits_its_base(project: Path):
    run(project, "trained_output#svm_standard")
    model = json.loads((project / "results/svm_standard/trained_output.pkl").read_text())
    assert model == {
        "stub": "train",
        "data": "sklearn.datasets.load_iris",
        "scaling": "standard",
        "model": "svm",
        "test_size": "small",
        "seed": "seed_42",
    }


def test_only_what_is_asked_for_is_built(project: Path):
    run(project, "f1_score#svm_standard")
    assert built(project) == [
        "results/svm_standard/f1_score.json",
        "results/svm_standard/trained_output.pkl",
    ]


def test_results_on_disk_are_skipped(project: Path):
    run(project)
    again = run(project)
    assert "build" not in again
    assert again.count("skip") == 5


def test_the_scikit_learn_version_builds_the_same_results(sklearn_project: Path):
    run(sklearn_project)
    assert built(sklearn_project) == [
        "results/model_comparison.json",
        "results/rf_standard/f1_score.json",
        "results/rf_standard/trained_output.pkl",
        "results/svm_standard/f1_score.json",
        "results/svm_standard/trained_output.pkl",
    ]
    comparison = json.loads((sklearn_project / "results/model_comparison.json").read_text())
    assert comparison["a"]["path"] == "results/rf_standard/f1_score.json"
    assert comparison["b"]["path"] == "results/svm_standard/f1_score.json"
    assert 0 < comparison["a"]["value"] <= 1
    assert 0 < comparison["b"]["value"] <= 1
