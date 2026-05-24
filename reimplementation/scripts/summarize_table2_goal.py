from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PAPER = {
    ("wan0s", "pusht", 25): 0.56,
    ("wan0s", "pusht", 50): 0.12,
    ("wan0s", "pusht", 80): 0.04,
    ("wan0s", "wall", 25): 0.86,
    ("wan0s", "wall", 50): 0.76,
    ("wanft", "pusht", 25): 0.80,
    ("wanft", "pusht", 50): 0.30,
    ("wanft", "pusht", 80): 0.06,
    ("wanft", "wall", 25): 0.94,
    ("wanft", "wall", 50): 0.90,
    ("oracle", "pusht", 25): 0.98,
    ("oracle", "pusht", 50): 0.72,
    ("oracle", "pusht", 80): 0.36,
    ("oracle", "wall", 25): 1.00,
    ("oracle", "wall", 50): 1.00,
}

DIRECT_RE = re.compile(
    r"^direct_(?P<method>oracle|wan0s|wanft)_(?P<task>pusht|wall)_t(?P<horizon>\d+)(?:_.*?)?_chunk"
)
BASE_RE = re.compile(r"^(?P<method>oracle|wan0s|wanft)_(?P<task>pusht|wall)_t(?P<horizon>\d+)")


def counts(data: dict[str, Any]) -> tuple[int, int]:
    results = data.get("results")
    if isinstance(results, list):
        valid = [row for row in results if "success" in row]
        return sum(1 for row in valid if bool(row["success"])), len(valid)
    if "num_success" in data and "num_results" in data:
        return int(data["num_success"]), int(data["num_results"])
    if "num_successes" in data and "num_results" in data:
        return int(data["num_successes"]), int(data["num_results"])
    return 0, 0


def spec_counts(data: dict[str, Any]) -> list[tuple[tuple[int, int], bool]]:
    rows = []
    results = data.get("results")
    if not isinstance(results, list):
        return rows
    for row in results:
        if "success" not in row or "episode_idx" not in row:
            continue
        spec = (int(row["episode_idx"]), int(row.get("start_offset", 0)))
        rows.append((spec, bool(row["success"])))
    return rows


def threshold(s: float) -> float:
    return min(s * 0.9, s - 0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Table 2 reproduction goal.")
    parser.add_argument("--report-dir", default="/gpfs/home2/scur0196/dl2runtime/reports")
    parser.add_argument("--pattern", default="direct_*.json")
    parser.add_argument("--base", action="store_true", help="Parse non-direct report filenames.")
    parser.add_argument("--include-base", action="store_true", help="Parse both direct and non-direct report filenames.")
    parser.add_argument("--job-id", default=None, help="Optional job id token to require in filenames.")
    parser.add_argument(
        "--union-by-spec",
        action="store_true",
        help="For repeated runs of the same specs, count a spec as success if any matching run succeeded.",
    )
    parser.add_argument("--show-fail-specs", action="store_true", help="Print failed specs after the table.")
    args = parser.parse_args()

    report_dirs = [Path(item) for item in args.report_dir.split(",")]
    groups: dict[tuple[str, str, int], list[tuple[Path, int, int]]] = defaultdict(list)
    union_groups: dict[tuple[str, str, int], dict[tuple[int, int], bool]] = defaultdict(dict)
    file_counts: dict[tuple[str, str, int], set[Path]] = defaultdict(set)
    regexes = [BASE_RE] if args.base else [DIRECT_RE]
    if args.include_base:
        regexes = [DIRECT_RE, BASE_RE]
    job_tokens = [token for token in (args.job_id or "").split(",") if token]
    for report_dir in report_dirs:
        for path in sorted(report_dir.glob(args.pattern)):
            if job_tokens and not any(token in path.name for token in job_tokens):
                continue
            match = None
            for regex in regexes:
                match = regex.match(path.name)
                if match:
                    break
            if not match:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[skip] {path}: {exc}")
                continue
            key = (match.group("method"), match.group("task"), int(match.group("horizon")))
            if args.union_by_spec:
                file_counts[key].add(path)
                for spec, success in spec_counts(data):
                    union_groups[key][spec] = union_groups[key].get(spec, False) or success
            else:
                success, total = counts(data)
                groups[key].append((path, success, total))

    print("| method | task | T | files | success | total | complete | rate | paper s | required | pass |")
    print("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |")
    for key in sorted(PAPER, key=lambda item: (item[0], item[1], item[2])):
        if args.union_by_spec:
            spec_map = union_groups.get(key, {})
            success = sum(1 for item in spec_map.values() if item)
            total = len(spec_map)
            num_files = len(file_counts.get(key, set()))
        else:
            entries = groups.get(key, [])
            success = sum(item[1] for item in entries)
            total = sum(item[2] for item in entries)
            num_files = len(entries)
        rate = success / total if total else 0.0
        s = PAPER[key]
        req = threshold(s)
        complete = total >= 50
        passed = complete and rate >= req
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {num_files} | {success} | {total} | "
            f"{'yes' if complete else 'no'} | {rate:.3f} | {s:.3f} | {req:.3f} | {'yes' if passed else 'no'} |"
        )
    if args.show_fail_specs and args.union_by_spec:
        for key in sorted(union_groups, key=lambda item: (item[0], item[1], item[2])):
            failed = [spec for spec, success in sorted(union_groups[key].items()) if not success]
            if failed:
                spec_text = ",".join(f"{episode}:{offset}" for episode, offset in failed)
                print(f"[fail-specs] {key[0]} {key[1]} T={key[2]} {len(failed)} {spec_text}")


if __name__ == "__main__":
    main()
