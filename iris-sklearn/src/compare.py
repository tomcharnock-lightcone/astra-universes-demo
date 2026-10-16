"""Compare macro-F1 under two specifications.

It records which two results it read and what each contained so the file
shows the comparison really drew on the two different universes.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    record = {
        "a": {"path": args.a, "model": a["trained_with"]["model"], "value": a["value"]},
        "b": {"path": args.b, "model": b["trained_with"]["model"], "value": b["value"]},
        "difference": round(a["value"] - b["value"], 4),
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
