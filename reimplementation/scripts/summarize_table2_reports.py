from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPORT_RE = re.compile(r"^(?P<method>oracle|wan0s|wanft)_(?P<task>pusht|wall)_t(?P<horizon>\d+)")


def _parse_since(value: str | None) -> float | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            pass
    raise ValueError(f"Unsupported --since format: {value!r}")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[skip] {path}: {exc}")
        return None


def _text_for_filter(path: Path, data: dict[str, Any]) -> str:
    return f"{path}\n{json.dumps(data, sort_keys=True, ensure_ascii=False)}"


def _result_counts(data: dict[str, Any]) -> tuple[int, int]:
    results = data.get("results")
    if isinstance(results, list):
        return sum(1 for row in results if bool(row.get("success"))), len(results)
    if "num_success" in data and "num_results" in data:
        return int(data["num_success"]), int(data["num_results"])
    if "num_successes" in data and "num_results" in data:
        return int(data["num_successes"]), int(data["num_results"])
    return 0, 0


def _iter_reports(args: argparse.Namespace) -> list[tuple[Path, dict[str, Any], re.Match[str]]]:
    report_dir = Path(args.report_dir)
    since_ts = _parse_since(args.since)
    rows = []
    for path in sorted(report_dir.glob("*.json")):
        if since_ts is not None and path.stat().st_mtime < since_ts:
            continue
        match = REPORT_RE.match(path.name)
        if not match:
            continue
        data = _load_json(path)
        if data is None:
            continue
        text = _text_for_filter(path, data)
        if args.require_token and not all(token in text for token in args.require_token):
            continue
        if args.any_token and not any(token in text for token in args.any_token):
            continue
        rows.append((path, data, match))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Table 2-style JSON reports.")
    parser.add_argument("--report-dir", default="/gpfs/home2/scur0196/dl2runtime/reports")
    parser.add_argument("--since", default=None, help="Only include files modified after this local time.")
    parser.add_argument(
        "--require-token",
        action="append",
        default=[],
        help="Require this token in either the file path or JSON payload. Can be repeated.",
    )
    parser.add_argument(
        "--any-token",
        action="append",
        default=[],
        help="Require at least one of these tokens in either the file path or JSON payload.",
    )
    parser.add_argument("--details", action="store_true", help="Print per-file rows before aggregates.")
    args = parser.parse_args()

    files = _iter_reports(args)
    groups: dict[tuple[str, str, int], list[tuple[Path, int, int]]] = defaultdict(list)
    for path, data, match in files:
        success, total = _result_counts(data)
        key = (match.group("method"), match.group("task"), int(match.group("horizon")))
        groups[key].append((path, success, total))

    if args.details:
        print("| file | success | total | rate |")
        print("| --- | ---: | ---: | ---: |")
        for key in sorted(groups):
            for path, success, total in groups[key]:
                rate = success / total if total else 0.0
                print(f"| {path.name} | {success} | {total} | {rate:.3f} |")
        print()

    print("| method | task | T | files | success | total | rate |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for method, task, horizon in sorted(groups, key=lambda item: (item[0], item[1], item[2])):
        entries = groups[(method, task, horizon)]
        success = sum(item[1] for item in entries)
        total = sum(item[2] for item in entries)
        rate = success / total if total else 0.0
        print(f"| {method} | {task} | {horizon} | {len(entries)} | {success} | {total} | {rate:.3f} |")


if __name__ == "__main__":
    main()
