"""Summarize a completed vLLM model-matrix run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from medgemma_eval.analyze_results import aggregate, load_latest
    from medgemma_eval.analyze_visual_results import _comparison
else:
    from .analyze_results import aggregate, load_latest
    from .analyze_visual_results import _comparison


Record = dict[str, Any]
VISUAL_MODES = (
    "context_only",
    "context_image",
    "context_mismatched",
    "image_only",
)


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _metric_by_mode(records: list[Record]) -> dict[str, Record]:
    return {
        str(row["mode"]): row
        for row in aggregate(records)
        if row["scope_kind"] == "overall"
    }


def _comparison_row(
    records: list[Record], name: str, baseline: str, target: str
) -> Record | None:
    rows = _comparison(
        records,
        name=name,
        baseline_mode=baseline,
        target_mode=target,
    )
    return next((row for row in rows if row["scope_kind"] == "overall"), None)


def summarize(matrix: dict[str, Any], task_types: set[str]) -> list[Record]:
    rows: list[Record] = []
    for item in matrix.get("models") or []:
        result_path = item.get("result")
        summary: Record = {
            "model": item.get("name"),
            "vision": item.get("vision"),
            "run_status": item.get("status"),
            "vllm_version": item.get("vllm_version"),
            "result": result_path,
        }
        if item.get("status") != "complete" or not result_path:
            rows.append(summary)
            continue
        records = load_latest(Path(str(result_path)))
        if task_types:
            records = [row for row in records if row.get("task_type") in task_types]
        metrics = _metric_by_mode(records)
        for mode in VISUAL_MODES:
            metric = metrics.get(mode)
            summary[f"{mode}_n"] = metric.get("n") if metric else None
            summary[f"{mode}_accuracy"] = metric.get("accuracy") if metric else None
            summary[f"{mode}_scorable_rate"] = (
                metric.get("scorable_rate") if metric else None
            )
            summary[f"{mode}_mean_latency_seconds"] = (
                metric.get("mean_latency_seconds") if metric else None
            )
        present_modes = set(metrics)
        if set(VISUAL_MODES).issubset(present_modes):
            gain = _comparison_row(
                records, "matched_image_gain", "context_only", "context_image"
            )
            grounding = _comparison_row(
                records,
                "matched_vs_mismatched",
                "context_mismatched",
                "context_image",
            )
            if gain:
                summary.update(
                    image_gain=gain["delta"],
                    image_gain_helpful=gain["helpful"],
                    image_gain_harmful=gain["harmful"],
                    image_gain_p=gain["mcnemar_exact_p"],
                )
            if grounding:
                summary.update(
                    matched_over_wrong=grounding["delta"],
                    matched_over_wrong_helpful=grounding["helpful"],
                    matched_over_wrong_harmful=grounding["harmful"],
                    matched_over_wrong_p=grounding["mcnemar_exact_p"],
                )
        rows.append(summary)
    return rows


def markdown_report(matrix: dict[str, Any], rows: list[Record]) -> str:
    lines = [
        "# vLLM AgentClinic-NEJM model matrix",
        "",
        "> Research benchmark only. This is not clinical validation or patient-care evidence.",
        "",
        f"Run: `{matrix.get('run_id', 'unknown')}`.",
        "Text-only models are not assigned image-mode failures and are excluded from image rankings.",
        "",
        (
            "| Model | Vision | Status | Context | Matched image | Wrong image | "
            "Image + task | Image gain | Matched over wrong |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {'yes' if row.get('vision') else 'no'} | "
            f"{row.get('run_status')} | "
            f"{_percent(row.get('context_only_accuracy'))} | "
            f"{_percent(row.get('context_image_accuracy'))} | "
            f"{_percent(row.get('context_mismatched_accuracy'))} | "
            f"{_percent(row.get('image_only_accuracy'))} | "
            f"{_percent(row.get('image_gain'))} | "
            f"{_percent(row.get('matched_over_wrong'))} |"
        )
    eligible = [
        row
        for row in rows
        if row.get("run_status") == "complete"
        and row.get("vision")
        and row.get("context_image_accuracy") is not None
    ]
    if eligible:
        lines.extend(
            [
                "",
                "## Matched-image ranking",
                "",
            ]
        )
        for index, row in enumerate(
            sorted(
                eligible,
                key=lambda item: item["context_image_accuracy"],
                reverse=True,
            ),
            1,
        ):
            lines.append(
                f"{index}. {row['model']}: "
                f"{_percent(row['context_image_accuracy'])}; "
                f"image gain {_percent(row.get('image_gain'))}; "
                f"matched over wrong {_percent(row.get('matched_over_wrong'))}."
            )
    lines.extend(
        [
            "",
            (
                "A high matched-image score is insufficient by itself. Prefer models that "
                "also improve over context-only and outperform a subtype-near wrong image. "
                "Public NEJM cases may overlap pretraining data."
            ),
        ]
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
    parser.add_argument("matrix_record", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task-types", default="")
    args = parser.parse_args(argv)
    matrix = json.loads(args.matrix_record.read_text(encoding="utf-8"))
    task_types = {
        value.strip() for value in args.task_types.split(",") if value.strip()
    }
    rows = summarize(matrix, task_types)
    output_dir = args.output_dir or args.matrix_record.parent / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "models.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {"matrix_record": str(args.matrix_record), "models": rows},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = markdown_report(matrix, rows)
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
