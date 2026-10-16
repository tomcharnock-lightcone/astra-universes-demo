"""Run an ASTRA analysis's recipes - the prototype runner for the universes RFC.

Usage:
    python run.py PROJECT [TARGET ...] [--dry-run] [--force]

A TARGET is an output id (``f1_score``) or an output in one universe
(``f1_score#svm_standard`` - the same pin syntax as in ``astra.yaml``).
Without targets everything is built. Each target is built with everything it
reads, in dependency order, and a result already on disk is skipped unless
``--force`` is given.

The analysis is validated with astra-tools first and errors stop the run.
The work is planned with ``astra.resolve.plan`` and each command is filled
in with ``astra.resolve.render_command``. Recipes run from the project
folder with this interpreter's folder first on their ``PATH`` - so
``python`` in a recipe is the Python the runner runs under.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from astra.helpers import load_yaml, resolve_analysis_tree
from astra.resolve import PlannedOutput, plan, render_command, split_pin
from astra.validation.schema import validate_analysis_data
from astra.validation.semantic import collect_semantic_warnings, validate_analysis


def load(project: Path) -> dict[str, Any]:
    """Loads and validates the project's ``astra.yaml``.

    Errors stop the run and warnings are printed. Returns the analysis with
    its external sub-analyses resolved, ready to plan.
    """
    data = load_yaml(project / "astra.yaml")
    errors = [*validate_analysis_data(data), *map(str, validate_analysis(data, project))]
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
    resolved = resolve_analysis_tree(data, project)
    for warning in collect_semantic_warnings(resolved):
        print(f"warning: {warning}", file=sys.stderr)
    return resolved


def label(entry: PlannedOutput) -> str:
    """``id#universe`` for an output built per universe and ``id`` for one built once."""
    return f"{entry.output.id}#{entry.universe}" if entry.universe else entry.output.id


def select(entries: list[PlannedOutput], targets: list[str]) -> list[PlannedOutput]:
    """The targets and everything they read, in dependency order.

    ``id`` matches the output in every universe it's planned in and
    ``id#universe`` matches it in that one universe. An entry depends on the
    entry whose result path it reads.
    """
    if targets:
        wanted = []
        for target in targets:
            artifact, universe = split_pin(target)
            found = [
                e
                for e in entries
                if e.output.id == artifact and (universe is None or e.universe == universe)
            ]
            if not found:
                raise SystemExit(f"error: nothing is planned for '{target}'")
            wanted.extend(found)
    else:
        wanted = list(entries)

    by_path = {entry.path: entry for entry in entries}
    order: list[PlannedOutput] = []
    seen: set[str] = set()

    def visit(entry: PlannedOutput) -> None:
        if entry.path in seen:
            return
        seen.add(entry.path)
        for path in entry.reads.values():
            if path in by_path:
                visit(by_path[path])
        order.append(entry)

    for entry in wanted:
        visit(entry)
    return order


def build(project: Path, entry: PlannedOutput, *, force: bool, dry_run: bool) -> None:
    """Runs one entry's recipe unless its result is already on disk."""
    if (project / entry.path).exists() and not force:
        print(f"skip   {label(entry)} ({entry.path} exists)")
        return
    if entry.output.command is None:
        print(f"skip   {label(entry)} (no recipe)")
        return
    command = render_command(
        entry.output.command,
        inputs=entry.reads,
        decisions=entry.output.decisions,
        output=entry.path,
    )
    print(f"build  {label(entry)}: {command}")
    if dry_run:
        return
    (project / entry.path).parent.mkdir(parents=True, exist_ok=True)
    path = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    subprocess.run(command, shell=True, cwd=project, check=True, env={**os.environ, "PATH": path})
    if not (project / entry.path).exists():
        raise SystemExit(f"error: {label(entry)}'s recipe didn't write {entry.path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("project", type=Path, help="folder holding astra.yaml")
    parser.add_argument("targets", nargs="*", help="output ids, optionally pinned (id#universe)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    parser.add_argument("--force", action="store_true", help="rebuild results already on disk")
    args = parser.parse_args()

    entries = plan(load(args.project))
    for entry in select(entries, args.targets):
        build(args.project, entry, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
