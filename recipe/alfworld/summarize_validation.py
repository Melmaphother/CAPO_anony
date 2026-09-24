#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

TASK_ORDER = [
    "pick_and_place",
    "look_at_obj_in_light",
    "pick_clean_then_place",
    "pick_heat_then_place",
    "pick_cool_then_place",
    "pick_two_obj_and_place",
]


def _iter_jsonl_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.jsonl"))


def _load_entries(input_path: Path) -> list[dict[str, Any]]:
    files = _iter_jsonl_files(input_path)
    entries: list[dict[str, Any]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
    return entries


def _filter_entries(entries: list[dict[str, Any]], step: str) -> list[dict[str, Any]]:
    if step != "latest":
        target_step = int(step)
        return [entry for entry in entries if int(entry.get("step", -1)) == target_step]

    if not entries:
        return entries
    max_step = max(int(entry.get("step", -1)) for entry in entries)
    return [entry for entry in entries if int(entry.get("step", -1)) == max_step]


def _group_by_task(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce a flattened step dump to one terminal result per trajectory.

    Validation JSONL deliberately mirrors rollout JSONL and consequently has one
    row for every agent/environment step.  Averaging ``score`` over those rows
    would count a successful 15-step episode as ``1 / 15`` rather than one
    success.  Select the last dumped row for each trajectory instead; its
    ``success`` field is the terminal ALFWorld outcome used by online validation.
    """
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for input_order, entry in enumerate(entries):
        split = entry.get("split")
        trajectory_uid = entry.get("trajectory_uids")
        if not split or not trajectory_uid:
            continue
        key = (str(split), str(trajectory_uid))
        position = (
            int(entry.get("num_steps", 0)),
            int(entry.get("step_indices", -1)),
            input_order,
        )
        previous = grouped.get(key)
        if previous is None or position >= previous["position"]:
            grouped[key] = {
                "position": position,
                "split": str(split),
                "task_id": str(entry.get("task_id", "")),
                "task_family": entry.get("task_family"),
                "score": float(bool(entry.get("success", False))),
            }

    grouped_rows = []
    for row in grouped.values():
        grouped_rows.append(
            {
                "split": row["split"],
                "task_id": row["task_id"],
                "task_family": row["task_family"],
                "score": row["score"],
            }
        )
    return grouped_rows


def _compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    summary: dict[str, Any] = {}
    for split_name, split_rows in sorted(by_split.items()):
        per_task: dict[str, Any] = {}
        for task_family in TASK_ORDER:
            task_rows = [row for row in split_rows if row.get("task_family") == task_family]
            if task_rows:
                per_task[task_family] = {
                    "score": mean(float(row["score"]) for row in task_rows),
                    "num_tasks": len(task_rows),
                }

        summary[split_name] = {
            "overall": mean(float(row["score"]) for row in split_rows) if split_rows else 0.0,
            "num_tasks": len(split_rows),
            "tasks": per_task,
        }

    valid_rows = [row for row in rows if row["split"] in {"valid_seen", "valid_unseen"}]
    merged_tasks: dict[str, Any] = {}
    for task_family in TASK_ORDER:
        task_rows = [row for row in valid_rows if row.get("task_family") == task_family]
        if task_rows:
            merged_tasks[task_family] = {
                "score": mean(float(row["score"]) for row in task_rows),
                "num_tasks": len(task_rows),
            }

    summary["all_valid"] = {
        "overall": mean(float(row["score"]) for row in valid_rows) if valid_rows else 0.0,
        "num_tasks": len(valid_rows),
        "tasks": merged_tasks,
    }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    for split_name in ("valid_seen", "valid_unseen", "all_valid"):
        if split_name not in summary:
            continue
        split_summary = summary[split_name]
        print(f"\n[{split_name}] overall={split_summary['overall']:.4f} num_tasks={split_summary['num_tasks']}")
        for task_family in TASK_ORDER:
            task_stats = split_summary["tasks"].get(task_family)
            if not task_stats:
                continue
            print(
                f"  {task_family:<28} score={task_stats['score']:.4f} "
                f"num_tasks={task_stats['num_tasks']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ALFWorld validation dumps.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Validation dump directory or a single jsonl file.",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="latest",
        help="Step to summarize. Use 'latest' or a concrete integer step.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to write summary JSON.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    entries = _load_entries(input_path)
    filtered = _filter_entries(entries, args.step)
    grouped = _group_by_task(filtered)
    summary = _compute_summary(grouped)
    _print_summary(summary)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nWrote summary -> {output_path}")


if __name__ == "__main__":
    main()
