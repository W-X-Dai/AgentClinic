"""Run paired AgentClinic-NEJM evaluations against a local Ollama model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from medgemma_eval import PROMPT_VERSION
    from medgemma_eval.dataset import (
        Case,
        case_metadata,
        ensure_image,
        file_sha256,
        load_cases,
        mismatched_image_case,
        permuted_options,
        select_cases,
    )
    from medgemma_eval.ollama_client import OllamaClient, parse_choice
else:
    from . import PROMPT_VERSION
    from .dataset import (
        Case,
        case_metadata,
        ensure_image,
        file_sha256,
        load_cases,
        mismatched_image_case,
        permuted_options,
        select_cases,
    )
    from .ollama_client import OllamaClient, parse_choice


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "agentclinic_nejm_extended.jsonl"
DEFAULT_CACHE = Path(__file__).resolve().parent / "cache"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results" / "cases.jsonl"
MODES = (
    "text_only",
    "multimodal",
    "context_only",
    "context_image",
    "context_mismatched",
    "image_only",
)
IMAGE_MODES = {"multimodal", "context_image", "context_mismatched", "image_only"}
SYSTEM_PROMPT = (
    "You are answering a retrospective medical benchmark question for research only. "
    "This is not patient care. Select exactly one best answer from A through E. "
    'Return only a JSON object such as {"choice":"A"}, following the supplied schema.'
)


def _csv_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def _index_set(value: str) -> set[int]:
    result: set[int] = set()
    for part in _csv_set(value):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def _prompt(case: Case, options: tuple[str, ...], mode: str) -> str:
    choices = "\n".join(f"{chr(65 + i)}. {answer}" for i, answer in enumerate(options))
    if mode == "image_only":
        context = (
            "Use the attached clinical image as the only case evidence.\n\n"
            f"Question:\n{case.question_target}"
        )
    elif mode in {"context_only", "context_image", "context_mismatched"}:
        context = (
            "Clinical history without test or image interpretation:\n"
            f"{case.clinical_context}\n\nQuestion:\n{case.question_target}"
        )
        if mode != "context_only":
            context += "\n\nAlso inspect the attached clinical image."
    else:
        context = f"Clinical question:\n{case.question}"
        if mode == "multimodal":
            context += "\n\nAlso inspect the attached clinical image."
    return f"{context}\n\nAnswer choices:\n{choices}\n\nSelect the single best answer."


def _model_metadata(info: dict[str, Any], digest: str | None) -> dict[str, Any]:
    details = info.get("details") if isinstance(info.get("details"), dict) else {}
    return {
        "model_digest": info.get("digest") or digest,
        "model_family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "capabilities": info.get("capabilities"),
    }


def _completed_keys(path: Path, retry_errors: bool) -> set[tuple[Any, ...]]:
    completed: set[tuple[Any, ...]] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if retry_errors and row.get("status") != "ok":
                continue
            completed.add(
                (
                    row.get("case_id"),
                    row.get("mode"),
                    row.get("model"),
                    row.get("prompt_version"),
                    row.get("seed"),
                )
            )
    return completed


def _write_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default="medgemma:27b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--modes", default="text_only,multimodal")
    parser.add_argument("--subtypes", default="", help="comma-separated subtype filter")
    parser.add_argument("--task-types", default="", help="comma-separated task filter")
    parser.add_argument("--indices", default="", help="e.g. 0,3,10-14")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--num-predict", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--download-timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    unknown_modes = set(modes) - set(MODES)
    if unknown_modes:
        raise SystemExit(f"unsupported modes: {sorted(unknown_modes)}")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")

    dataset_sha = file_sha256(args.dataset)
    all_cases = load_cases(args.dataset)
    selected = select_cases(
        all_cases,
        _csv_set(args.subtypes),
        _csv_set(args.task_types),
        _index_set(args.indices),
        args.limit,
    )
    if not selected:
        raise SystemExit("filters selected zero cases")

    client = OllamaClient(args.ollama_url, args.timeout)
    ollama_version = client.version()
    model_info = client.model_info(args.model)
    model_digest = client.model_digest(args.model)
    capabilities = set(model_info.get("capabilities") or [])
    if any(mode in IMAGE_MODES for mode in modes) and "vision" not in capabilities:
        raise SystemExit(f"{args.model} does not report the vision capability")
    metadata = _model_metadata(model_info, model_digest)
    print(
        f"Preflight OK: {len(all_cases)} cases, {len(selected)} selected, "
        f"model={args.model}, Ollama={ollama_version}, modes={','.join(modes)}"
    )
    if args.preflight_only:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        set() if args.no_resume else _completed_keys(args.output, args.retry_errors)
    )
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_common = {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "ollama_version": ollama_version,
        "prompt_version": PROMPT_VERSION,
        "dataset_sha256": dataset_sha,
        "seed": args.seed,
        "temperature": 0,
        "num_predict": args.num_predict,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        **metadata,
    }
    total = len(selected) * len(modes)
    finished = 0
    with args.output.open("a", encoding="utf-8") as output:
        for case in selected:
            options, displayed_correct, option_order = permuted_options(case, args.seed)
            for mode in modes:
                key = (case.case_id, mode, args.model, PROMPT_VERSION, args.seed)
                if key in completed:
                    finished += 1
                    print(f"[{finished}/{total}] skip {case.case_id} {mode}")
                    continue
                prompt = _prompt(case, options, mode)
                record: dict[str, Any] = {
                    **run_common,
                    **case_metadata(case),
                    "mode": mode,
                    "displayed_options": list(options),
                    "option_order_original_indices": list(option_order),
                    "displayed_correct_index": displayed_correct,
                    "displayed_correct_label": chr(65 + displayed_correct),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "status": "error",
                    "attempts": 0,
                }
                image = None
                try:
                    if mode in IMAGE_MODES:
                        image_case = (
                            mismatched_image_case(all_cases, case, args.seed)
                            if mode == "context_mismatched"
                            else case
                        )
                        image, image_sha, _ = ensure_image(
                            image_case,
                            args.cache_dir,
                            timeout=args.download_timeout,
                        )
                        record.update(
                            {
                                "image_sha256": image_sha,
                                "image_case_id": image_case.case_id,
                                "image_subtypes": list(image_case.subtypes),
                                "image_matched": image_case.case_id == case.case_id,
                            }
                        )
                    last_error: Exception | None = None
                    for attempt in range(1, args.retries + 1):
                        record["attempts"] = attempt
                        try:
                            response = client.chat(
                                model=args.model,
                                system_prompt=SYSTEM_PROMPT,
                                user_prompt=prompt,
                                image=image,
                                seed=args.seed,
                                num_predict=args.num_predict,
                                keep_alive=args.keep_alive,
                            )
                            predicted, parse_strategy = parse_choice(
                                response.content, options
                            )
                            record.update(
                                {
                                    "status": "ok"
                                    if predicted is not None
                                    else "parse_error",
                                    "raw_response": response.content,
                                    "predicted_index": predicted,
                                    "predicted_label": chr(65 + predicted)
                                    if predicted is not None
                                    else None,
                                    "parse_strategy": parse_strategy,
                                    "correct": predicted == displayed_correct
                                    if predicted is not None
                                    else False,
                                    "latency_seconds": response.latency_seconds,
                                    "total_duration_ns": response.total_duration_ns,
                                    "load_duration_ns": response.load_duration_ns,
                                    "prompt_eval_count": response.prompt_eval_count,
                                    "eval_count": response.eval_count,
                                }
                            )
                            last_error = None
                            break
                        except RuntimeError as error:
                            last_error = error
                            if attempt < args.retries:
                                time.sleep(2 ** (attempt - 1))
                    if last_error is not None:
                        raise last_error
                except Exception as error:  # noqa: BLE001 - failures stay in denominator
                    record.update(
                        {
                            "status": "error",
                            "error_type": type(error).__name__,
                            "error": str(error)[:2000],
                            "correct": False,
                        }
                    )
                _write_record(output, record)
                finished += 1
                print(
                    f"[{finished}/{total}] {case.case_id} {mode}: "
                    f"{record['status']} correct={record.get('correct', False)}"
                )
    print(f"Results appended to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
