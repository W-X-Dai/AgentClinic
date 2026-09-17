"""Analyze image-focused ablations and mismatched-image controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from medgemma_eval.analyze_results import (
        aggregate,
        load_latest,
        mcnemar_exact_p,
        validate_comparable_config,
    )
else:
    from .analyze_results import (
        aggregate,
        load_latest,
        mcnemar_exact_p,
        validate_comparable_config,
    )


Record = dict[str, Any]
VISUAL_MODES = (
    "context_only",
    "context_image",
    "context_mismatched",
    "image_only",
)


def binomial_upper_p(correct: int, total: int, chance: float = 0.2) -> float:
    """Exact one-sided probability of at least `correct` under chance."""
    if total <= 0:
        return 1.0
    return min(
        1.0,
        sum(
            math.comb(total, value) * chance**value * (1 - chance) ** (total - value)
            for value in range(correct, total + 1)
        ),
    )


def _comparison(
    records: list[Record],
    *,
    name: str,
    baseline_mode: str,
    target_mode: str,
) -> list[Record]:
    by_case: dict[tuple[Any, ...], dict[str, Record]] = defaultdict(dict)
    for row in records:
        key = (
            row.get("model"),
            row.get("case_id"),
            row.get("prompt_version"),
            row.get("seed"),
            row.get("dataset_sha256"),
        )
        by_case[key][str(row.get("mode"))] = row
    pairs_by_model: dict[str, list[tuple[Record, Record]]] = defaultdict(list)
    for key, modes in by_case.items():
        if baseline_mode in modes and target_mode in modes:
            pairs_by_model[str(key[0])].append(
                (modes[baseline_mode], modes[target_mode])
            )

    output: list[Record] = []
    for model, pairs in sorted(pairs_by_model.items()):
        scopes: list[tuple[str, str, list[tuple[Record, Record]]]] = [
            ("overall", "all", pairs)
        ]
        subtypes = sorted(
            {
                str(tag)
                for baseline, _ in pairs
                for tag in baseline.get("subtypes") or []
            }
        )
        tasks = sorted(
            {str(baseline.get("task_type", "unknown")) for baseline, _ in pairs}
        )
        scopes.extend(
            (
                "subtype",
                subtype,
                [pair for pair in pairs if subtype in (pair[0].get("subtypes") or [])],
            )
            for subtype in subtypes
        )
        scopes.extend(
            (
                "task_type",
                task,
                [
                    pair
                    for pair in pairs
                    if str(pair[0].get("task_type", "unknown")) == task
                ],
            )
            for task in tasks
        )
        for scope_kind, scope_value, scoped_pairs in scopes:
            total = len(scoped_pairs)
            baseline_correct = sum(
                baseline.get("correct") is True for baseline, _ in scoped_pairs
            )
            target_correct = sum(
                target.get("correct") is True for _, target in scoped_pairs
            )
            helpful = sum(
                baseline.get("correct") is not True and target.get("correct") is True
                for baseline, target in scoped_pairs
            )
            harmful = sum(
                baseline.get("correct") is True and target.get("correct") is not True
                for baseline, target in scoped_pairs
            )
            changed = sum(
                baseline.get("predicted_index") != target.get("predicted_index")
                for baseline, target in scoped_pairs
            )
            output.append(
                {
                    "comparison": name,
                    "model": model,
                    "scope_kind": scope_kind,
                    "scope_value": scope_value,
                    "baseline_mode": baseline_mode,
                    "target_mode": target_mode,
                    "n_pairs": total,
                    "baseline_accuracy": baseline_correct / total if total else None,
                    "target_accuracy": target_correct / total if total else None,
                    "delta": (target_correct - baseline_correct) / total
                    if total
                    else None,
                    "helpful": helpful,
                    "harmful": harmful,
                    "answer_changed": changed,
                    "answer_change_rate": changed / total if total else None,
                    "mcnemar_exact_p": mcnemar_exact_p(helpful, harmful),
                }
            )
    return output


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _write_csv(path: Path, rows: list[Record]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric_index(metrics: list[Record]) -> dict[tuple[str, str, str, str], Record]:
    return {
        (
            str(row["model"]),
            str(row["scope_kind"]),
            str(row["scope_value"]),
            str(row["mode"]),
        ): row
        for row in metrics
    }


def markdown_report(
    metrics: list[Record], comparisons: list[Record], min_rank_n: int
) -> str:
    lines = [
        "# Generalist medical VLM image-reading ablation",
        "",
        "> Research benchmark only. This is not clinical validation or patient-care evidence.",
        "",
        (
            "`context_image` uses a patient-actor proxy context that omits the original "
            "formal image report, but may retain patient-visible findings; it is not "
            "clinician-redacted. `context_mismatched` substitutes a deterministic wrong "
            "image from the closest available subtype. `image_only` retains only the "
            "task question, answer choices, and image."
        ),
        "",
    ]
    index = _metric_index(metrics)
    models = sorted({str(row["model"]) for row in metrics})
    for model in models:
        lines.extend(
            [
                f"## {model}",
                "",
                "| Condition | Correct / n | Accuracy | 95% CI | Scorable |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for mode in VISUAL_MODES:
            row = index.get((model, "overall", "all", mode))
            if row:
                lines.append(
                    f"| {mode} | {row['correct']} / {row['n']} | "
                    f"{_percent(row['accuracy'])} | {_percent(row['ci95_low'])}–"
                    f"{_percent(row['ci95_high'])} | {_percent(row['scorable_rate'])} |"
                )
        image_only = index.get((model, "overall", "all", "image_only"))
        if image_only:
            chance_p = binomial_upper_p(image_only["correct"], image_only["n"])
            lines.extend(
                [
                    "",
                    f"Image-only versus 20% five-choice chance: exact one-sided p={chance_p:.3g}.",
                ]
            )
        for comparison_name in ("matched_image_gain", "matched_vs_mismatched"):
            row = next(
                (
                    value
                    for value in comparisons
                    if value["model"] == model
                    and value["scope_kind"] == "overall"
                    and value["comparison"] == comparison_name
                ),
                None,
            )
            if row:
                lines.append(
                    f"- {comparison_name}: Δ {_percent(row['delta'])}; "
                    f"{row['helpful']} helpful / {row['harmful']} harmful; "
                    f"answer changed {_percent(row['answer_change_rate'])}; "
                    f"McNemar p={row['mcnemar_exact_p']:.3f}."
                )

        subtype_gain = {
            str(row["scope_value"]): row
            for row in comparisons
            if row["model"] == model
            and row["scope_kind"] == "subtype"
            and row["comparison"] == "matched_image_gain"
        }
        subtype_grounding = {
            str(row["scope_value"]): row
            for row in comparisons
            if row["model"] == model
            and row["scope_kind"] == "subtype"
            and row["comparison"] == "matched_vs_mismatched"
        }
        lines.extend(
            [
                "",
                "### Subtype comparison",
                "",
                (
                    "| Subtype | n | Context | Matched image | Wrong image | Image only | "
                    "True-image Δ | Matched-over-wrong Δ | Evidence |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        subtypes = sorted(
            subtype_gain,
            key=lambda subtype: (-subtype_gain[subtype]["n_pairs"], subtype),
        )
        for subtype in subtypes:
            gain = subtype_gain[subtype]
            grounding = subtype_grounding.get(subtype)
            n = gain["n_pairs"]
            context = index.get((model, "subtype", subtype, "context_only"))
            matched = index.get((model, "subtype", subtype, "context_image"))
            mismatch = index.get((model, "subtype", subtype, "context_mismatched"))
            image_only_row = index.get((model, "subtype", subtype, "image_only"))
            evidence = "preliminary" if n >= min_rank_n else "descriptive only"
            lines.append(
                f"| {subtype} | {n} | {_percent(context['accuracy'] if context else None)} | "
                f"{_percent(matched['accuracy'] if matched else None)} | "
                f"{_percent(mismatch['accuracy'] if mismatch else None)} | "
                f"{_percent(image_only_row['accuracy'] if image_only_row else None)} | "
                f"{_percent(gain['delta'])} | "
                f"{_percent(grounding['delta'] if grounding else None)} | {evidence} |"
            )
        lines.extend(
            [
                "",
                (
                    "A useful image reader should improve over context-only, outperform "
                    "mismatched images, and exceed chance in image-only testing. Any "
                    "single criterion is insufficient."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-rank-n", type=int, default=10)
    parser.add_argument("--task-types", default="")
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.results.parent / "visual-summary"
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
    present_modes = {str(row.get("mode")) for row in records}
    missing = set(VISUAL_MODES) - present_modes
    if missing:
        raise SystemExit(f"visual analysis requires modes: {sorted(missing)}")

    metrics = aggregate(records)
    comparisons = [
        *_comparison(
            records,
            name="matched_image_gain",
            baseline_mode="context_only",
            target_mode="context_image",
        ),
        *_comparison(
            records,
            name="matched_vs_mismatched",
            baseline_mode="context_mismatched",
            target_mode="context_image",
        ),
    ]
    summary = {
        "source": str(args.results),
        "records": len(records),
        "metrics": metrics,
        "comparisons": comparisons,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "metrics.csv", metrics)
    _write_csv(output_dir / "comparisons.csv", comparisons)
    report = markdown_report(metrics, comparisons, args.min_rank_n)
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSummary files written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
