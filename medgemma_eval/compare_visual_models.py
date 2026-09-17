"""Compare two models on identical AgentClinic visual-ablation records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from medgemma_eval.analyze_results import load_latest, mcnemar_exact_p
    from medgemma_eval.analyze_visual_results import VISUAL_MODES, _comparison
else:
    from .analyze_results import load_latest, mcnemar_exact_p
    from .analyze_visual_results import VISUAL_MODES, _comparison


Record = dict[str, Any]


def _model_name(records: list[Record], label: str) -> str:
    models = {str(row.get("model")) for row in records}
    if len(models) != 1:
        raise ValueError(
            f"{label} results must contain exactly one model: {sorted(models)}"
        )
    return next(iter(models))


def _comparable_key(row: Record) -> tuple[Any, ...]:
    return (
        row.get("case_id"),
        row.get("mode"),
        row.get("prompt_version"),
        row.get("seed"),
        row.get("dataset_sha256"),
    )


def pair_records(
    baseline: list[Record], candidate: list[Record]
) -> list[tuple[Record, Record]]:
    baseline_by_key = {_comparable_key(row): row for row in baseline}
    candidate_by_key = {_comparable_key(row): row for row in candidate}
    if baseline_by_key.keys() != candidate_by_key.keys():
        missing_candidate = sorted(baseline_by_key.keys() - candidate_by_key.keys())
        missing_baseline = sorted(candidate_by_key.keys() - baseline_by_key.keys())
        raise ValueError(
            "model result keys differ: "
            f"candidate missing {len(missing_candidate)}, "
            f"baseline missing {len(missing_baseline)}"
        )
    pairs = []
    for key in sorted(
        baseline_by_key, key=lambda value: tuple(str(item) for item in value)
    ):
        left = baseline_by_key[key]
        right = candidate_by_key[key]
        for field in (
            "displayed_correct_index",
            "displayed_options",
            "option_order_original_indices",
            "prompt_sha256",
            "image_case_id",
            "image_sha256",
        ):
            if left.get(field) != right.get(field):
                raise ValueError(f"paired records differ in {field}: {key}")
        pairs.append((left, right))
    return pairs


def _metric(
    pairs: list[tuple[Record, Record]],
    *,
    mode: str,
    scope_kind: str,
    scope_value: str,
) -> Record:
    scoped = [pair for pair in pairs if pair[0].get("mode") == mode]
    if scope_kind == "subtype":
        scoped = [
            pair for pair in scoped if scope_value in (pair[0].get("subtypes") or [])
        ]
    total = len(scoped)
    baseline_correct = sum(left.get("correct") is True for left, _ in scoped)
    candidate_correct = sum(right.get("correct") is True for _, right in scoped)
    candidate_helpful = sum(
        left.get("correct") is not True and right.get("correct") is True
        for left, right in scoped
    )
    candidate_harmful = sum(
        left.get("correct") is True and right.get("correct") is not True
        for left, right in scoped
    )
    return {
        "mode": mode,
        "scope_kind": scope_kind,
        "scope_value": scope_value,
        "n_pairs": total,
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "baseline_accuracy": baseline_correct / total if total else None,
        "candidate_accuracy": candidate_correct / total if total else None,
        "candidate_delta": (candidate_correct - baseline_correct) / total
        if total
        else None,
        "candidate_helpful": candidate_helpful,
        "candidate_harmful": candidate_harmful,
        "mcnemar_exact_p": mcnemar_exact_p(candidate_helpful, candidate_harmful),
    }


def model_metrics(pairs: list[tuple[Record, Record]]) -> list[Record]:
    subtypes = sorted(
        {str(tag) for left, _ in pairs for tag in left.get("subtypes") or []}
    )
    rows = [
        _metric(pairs, mode=mode, scope_kind="overall", scope_value="all")
        for mode in VISUAL_MODES
    ]
    rows.extend(
        _metric(pairs, mode=mode, scope_kind="subtype", scope_value=subtype)
        for subtype in subtypes
        for mode in VISUAL_MODES
    )
    return rows


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _pvalue(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _overall_grounding(records: list[Record]) -> dict[str, Record]:
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
    return {
        str(row["comparison"]): row
        for row in comparisons
        if row["scope_kind"] == "overall"
    }


def markdown_report(
    *,
    baseline_name: str,
    candidate_name: str,
    baseline: list[Record],
    candidate: list[Record],
    metrics: list[Record],
    min_rank_n: int,
) -> str:
    lines = [
        "# Generalist medical VLM visual comparison",
        "",
        "> Research benchmark only. This is not clinical validation or patient-care evidence.",
        "",
        f"Baseline: `{baseline_name}`. Candidate: `{candidate_name}`.",
        "Failures and unparseable outputs remain in the denominator.",
        "",
        "## Paired overall accuracy",
        "",
        (
            f"| Condition | n | {baseline_name} | {candidate_name} | Candidate Δ | "
            "Helpful / harmful | McNemar p |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    overall = [row for row in metrics if row["scope_kind"] == "overall"]
    for row in overall:
        lines.append(
            f"| {row['mode']} | {row['n_pairs']} | "
            f"{_percent(row['baseline_accuracy'])} | "
            f"{_percent(row['candidate_accuracy'])} | "
            f"{_percent(row['candidate_delta'])} | "
            f"{row['candidate_helpful']} / {row['candidate_harmful']} | "
            f"{_pvalue(row['mcnemar_exact_p'])} |"
        )

    lines.extend(
        [
            "",
            "## Within-model image grounding",
            "",
            (
                "| Model | Matched vs no image | Helpful / harmful | "
                "Matched vs wrong image | Helpful / harmful |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, records in ((baseline_name, baseline), (candidate_name, candidate)):
        grounding = _overall_grounding(records)
        gain = grounding["matched_image_gain"]
        mismatch = grounding["matched_vs_mismatched"]
        lines.append(
            f"| {name} | {_percent(gain['delta'])} | "
            f"{gain['helpful']} / {gain['harmful']} | "
            f"{_percent(mismatch['delta'])} | "
            f"{mismatch['helpful']} / {mismatch['harmful']} |"
        )

    context_rows = [
        row
        for row in metrics
        if row["scope_kind"] == "subtype"
        and row["mode"] == "context_image"
        and row["n_pairs"] >= min_rank_n
    ]
    lines.extend(
        [
            "",
            f"## Context + matched image by subtype (n ≥ {min_rank_n})",
            "",
            f"| Subtype | n | {baseline_name} | {candidate_name} | Candidate Δ |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        context_rows, key=lambda item: (-item["n_pairs"], item["scope_value"])
    ):
        lines.append(
            f"| {row['scope_value']} | {row['n_pairs']} | "
            f"{_percent(row['baseline_accuracy'])} | "
            f"{_percent(row['candidate_accuracy'])} | "
            f"{_percent(row['candidate_delta'])} |"
        )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[Record]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-types", default="")
    parser.add_argument("--min-rank-n", type=int, default=10)
    args = parser.parse_args(argv)

    baseline = load_latest(args.baseline)
    candidate = load_latest(args.candidate)
    task_types = {
        value.strip() for value in args.task_types.split(",") if value.strip()
    }
    if task_types:
        baseline = [row for row in baseline if row.get("task_type") in task_types]
        candidate = [row for row in candidate if row.get("task_type") in task_types]
    baseline_name = _model_name(baseline, "baseline")
    candidate_name = _model_name(candidate, "candidate")
    pairs = pair_records(baseline, candidate)
    metrics = model_metrics(pairs)
    report = markdown_report(
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        baseline=baseline,
        candidate=candidate,
        metrics=metrics,
        min_rank_n=args.min_rank_n,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "paired-model-comparison.csv", metrics)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "baseline": str(args.baseline),
                "candidate": str(args.candidate),
                "task_types": sorted(task_types),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nComparison files written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
