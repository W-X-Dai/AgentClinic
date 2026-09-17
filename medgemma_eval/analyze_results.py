"""Aggregate MedGemma case results by NEJM subtype and task type."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def wilson_interval(
    correct: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_exact_p(helpful: int, harmful: int) -> float:
    """Return the two-sided exact McNemar p-value for paired flips."""
    discordant = helpful + harmful
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value) for value in range(min(helpful, harmful) + 1)
    )
    return min(1.0, 2 * tail / (2**discordant))


def load_latest(path: Path) -> list[Record]:
    latest: dict[tuple[Any, ...], Record] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            key = (
                row.get("case_id"),
                row.get("mode"),
                row.get("model"),
                row.get("prompt_version"),
                row.get("seed"),
            )
            latest[key] = row
    return list(latest.values())


def validate_comparable_config(records: list[Record]) -> None:
    """Reject silent aggregation across incompatible benchmark configurations."""
    configurations = {
        (
            row.get("prompt_version"),
            row.get("seed"),
            row.get("dataset_sha256"),
            row.get("temperature"),
            row.get("num_predict"),
        )
        for row in records
    }
    if len(configurations) != 1:
        raise ValueError(
            "results contain multiple prompt/seed/dataset/generation configurations; "
            "analyze each configuration in a separate JSONL file"
        )


def _metric(
    model: str, mode: str, scope_kind: str, scope_value: str, rows: Iterable[Record]
) -> Record:
    values = list(rows)
    total = len(values)
    correct = sum(row.get("correct") is True for row in values)
    scorable = sum(row.get("status") == "ok" for row in values)
    low, high = wilson_interval(correct, total)
    latencies = [
        float(row["latency_seconds"])
        for row in values
        if row.get("latency_seconds") is not None
    ]
    return {
        "model": model,
        "mode": mode,
        "scope_kind": scope_kind,
        "scope_value": scope_value,
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "ci95_low": low,
        "ci95_high": high,
        "scorable": scorable,
        "scorable_rate": scorable / total if total else None,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
    }


def aggregate(records: list[Record]) -> list[Record]:
    grouped: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for row in records:
        grouped[(str(row.get("model")), str(row.get("mode")))].append(row)
    metrics: list[Record] = []
    for (model, mode), rows in sorted(grouped.items()):
        metrics.append(_metric(model, mode, "overall", "all", rows))
        subtype_rows: dict[str, list[Record]] = defaultdict(list)
        task_rows: dict[str, list[Record]] = defaultdict(list)
        for row in rows:
            for subtype in row.get("subtypes") or []:
                subtype_rows[str(subtype)].append(row)
            task_rows[str(row.get("task_type", "unknown"))].append(row)
        metrics.extend(
            _metric(model, mode, "subtype", subtype, values)
            for subtype, values in sorted(subtype_rows.items())
        )
        metrics.extend(
            _metric(model, mode, "task_type", task_type, values)
            for task_type, values in sorted(task_rows.items())
        )
    return metrics


def _pair_metric(
    model: str, scope_kind: str, scope_value: str, pairs: list[tuple[Record, Record]]
) -> Record:
    total = len(pairs)
    text_correct = sum(left.get("correct") is True for left, _ in pairs)
    image_correct = sum(right.get("correct") is True for _, right in pairs)
    helpful = sum(
        left.get("correct") is not True and right.get("correct") is True
        for left, right in pairs
    )
    harmful = sum(
        left.get("correct") is True and right.get("correct") is not True
        for left, right in pairs
    )
    return {
        "model": model,
        "scope_kind": scope_kind,
        "scope_value": scope_value,
        "n_pairs": total,
        "text_only_accuracy": text_correct / total if total else None,
        "multimodal_accuracy": image_correct / total if total else None,
        "vision_delta": (image_correct - text_correct) / total if total else None,
        "image_helpful": helpful,
        "image_harmful": harmful,
        "mcnemar_exact_p": mcnemar_exact_p(helpful, harmful),
        "stable_correct": sum(
            left.get("correct") is True and right.get("correct") is True
            for left, right in pairs
        ),
        "stable_wrong": sum(
            left.get("correct") is not True and right.get("correct") is not True
            for left, right in pairs
        ),
    }


def paired(records: list[Record]) -> list[Record]:
    by_key: dict[tuple[str, str, str, Any, Any], dict[str, Record]] = defaultdict(dict)
    for row in records:
        key = (
            str(row.get("model")),
            str(row.get("case_id")),
            str(row.get("prompt_version")),
            row.get("seed"),
            row.get("dataset_sha256"),
        )
        by_key[key][str(row.get("mode"))] = row
    by_model: dict[str, list[tuple[Record, Record]]] = defaultdict(list)
    for key, modes in by_key.items():
        if "text_only" in modes and "multimodal" in modes:
            by_model[key[0]].append((modes["text_only"], modes["multimodal"]))

    output: list[Record] = []
    for model, pairs in sorted(by_model.items()):
        output.append(_pair_metric(model, "overall", "all", pairs))
        subtypes = sorted(
            {str(tag) for left, _ in pairs for tag in left.get("subtypes") or []}
        )
        tasks = sorted({str(left.get("task_type", "unknown")) for left, _ in pairs})
        for subtype in subtypes:
            subset = [
                (left, right)
                for left, right in pairs
                if subtype in (left.get("subtypes") or [])
            ]
            output.append(_pair_metric(model, "subtype", subtype, subset))
        for task in tasks:
            subset = [
                (left, right)
                for left, right in pairs
                if str(left.get("task_type", "unknown")) == task
            ]
            output.append(_pair_metric(model, "task_type", task, subset))
    return output


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def markdown_report(metrics: list[Record], pairs: list[Record], min_rank_n: int) -> str:
    lines = [
        "# MedGemma AgentClinic-NEJM evaluation",
        "",
        "> Research benchmark only. These results are not clinical validation and must not guide patient care.",
        "",
        "Failures and unparseable answers remain in the primary accuracy denominator. Subtypes are multi-label,",
        "so subtype sample counts must not be summed. Public NEJM cases may also overlap model pretraining data.",
        "",
    ]
    models = sorted({str(row["model"]) for row in metrics})
    for model in models:
        lines.extend(
            [
                f"## {model}",
                "",
                "### Overall",
                "",
                "| Mode | Correct / n | Accuracy | 95% CI | Scorable |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in metrics:
            if row["model"] == model and row["scope_kind"] == "overall":
                lines.append(
                    f"| {row['mode']} | {row['correct']} / {row['n']} | {_percent(row['accuracy'])} | "
                    f"{_percent(row['ci95_low'])}–{_percent(row['ci95_high'])} | {_percent(row['scorable_rate'])} |"
                )
        overall_pair = next(
            (
                row
                for row in pairs
                if row["model"] == model and row["scope_kind"] == "overall"
            ),
            None,
        )
        if overall_pair:
            lines.extend(
                [
                    "",
                    (
                        "Paired image effect: "
                        f"{_percent(overall_pair['vision_delta'])} "
                        f"({overall_pair['image_helpful']} helpful, "
                        f"{overall_pair['image_harmful']} harmful; exact McNemar "
                        f"p={overall_pair['mcnemar_exact_p']:.3f})."
                    ),
                ]
            )
        subtype_pairs = [
            row
            for row in pairs
            if row["model"] == model and row["scope_kind"] == "subtype"
        ]
        if subtype_pairs:
            lines.extend(
                [
                    "",
                    "### Paired subtype comparison",
                    "",
                    "| Subtype | n | Text | Text + image | Δ image | Helpful / harmful | Evidence |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for row in sorted(
                subtype_pairs, key=lambda item: (-item["n_pairs"], item["scope_value"])
            ):
                evidence = (
                    "rankable" if row["n_pairs"] >= min_rank_n else "descriptive only"
                )
                lines.append(
                    f"| {row['scope_value']} | {row['n_pairs']} | {_percent(row['text_only_accuracy'])} | "
                    f"{_percent(row['multimodal_accuracy'])} | {_percent(row['vision_delta'])} | "
                    f"{row['image_helpful']} / {row['image_harmful']} | {evidence} |"
                )
            rankable = [row for row in subtype_pairs if row["n_pairs"] >= min_rank_n]
            if rankable:
                strongest = sorted(
                    rankable,
                    key=lambda item: (-item["multimodal_accuracy"], -item["n_pairs"]),
                )[:3]
                weakest = sorted(
                    rankable,
                    key=lambda item: (item["multimodal_accuracy"], -item["n_pairs"]),
                )[:3]
                lines.extend(
                    [
                        "",
                        f"With the configured minimum n={min_rank_n}, the highest multimodal accuracies are: "
                        + ", ".join(
                            f"{row['scope_value']} ({_percent(row['multimodal_accuracy'])}, n={row['n_pairs']})"
                            for row in strongest
                        )
                        + ".",
                        "",
                        "The lowest multimodal accuracies are: "
                        + ", ".join(
                            f"{row['scope_value']} ({_percent(row['multimodal_accuracy'])}, n={row['n_pairs']})"
                            for row in weakest
                        )
                        + ". Rankings are descriptive, not proof of clinically meaningful differences.",
                    ]
                )
        lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[Record]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-rank-n", type=int, default=10)
    parser.add_argument("--task-types", default="", help="comma-separated task filter")
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.results.parent / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_latest(args.results)
    task_types = {
        value.strip() for value in args.task_types.split(",") if value.strip()
    }
    if task_types:
        records = [row for row in records if row.get("task_type") in task_types]
    if not records:
        raise SystemExit("no result records found")
    validate_comparable_config(records)
    metrics = aggregate(records)
    pair_metrics = paired(records)
    summary = {
        "source": str(args.results),
        "records": len(records),
        "metrics": metrics,
        "paired": pair_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "metrics.csv", metrics)
    _write_csv(output_dir / "paired.csv", pair_metrics)
    report = markdown_report(metrics, pair_metrics, args.min_rank_n)
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSummary files written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
