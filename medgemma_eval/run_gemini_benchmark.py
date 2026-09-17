"""Run AgentClinic-NEJM image ablations with the backend Gemini API setup."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
        case_metadata,
        ensure_image,
        file_sha256,
        load_cases,
        mismatched_image_case,
        permuted_options,
        select_cases,
    )
    from medgemma_eval.gemini_client import GeminiClient, load_env_file
    from medgemma_eval.ollama_client import parse_choice
    from medgemma_eval.run_benchmark import (
        IMAGE_MODES,
        MODES,
        SYSTEM_PROMPT,
        _completed_keys,
        _csv_set,
        _index_set,
        _prompt,
        _write_record,
    )
else:
    from . import PROMPT_VERSION
    from .dataset import (
        case_metadata,
        ensure_image,
        file_sha256,
        load_cases,
        mismatched_image_case,
        permuted_options,
        select_cases,
    )
    from .gemini_client import GeminiClient, load_env_file
    from .ollama_client import parse_choice
    from .run_benchmark import (
        IMAGE_MODES,
        MODES,
        SYSTEM_PROMPT,
        _completed_keys,
        _csv_set,
        _index_set,
        _prompt,
        _write_record,
    )


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "agentclinic_nejm_extended.jsonl"
DEFAULT_CACHE = Path(__file__).resolve().parent / "cache"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results" / "gemini-cases.jsonl"
DEFAULT_ENV_FILE = REPOSITORY_ROOT / "backend" / ".env"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--model")
    parser.add_argument(
        "--modes",
        default="context_only,context_image,context_mismatched,image_only",
    )
    parser.add_argument("--subtypes", default="")
    parser.add_argument("--task-types", default="")
    parser.add_argument("--indices", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--num-predict",
        type=int,
        default=1024,
        help="output budget including Gemini reasoning tokens",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--download-timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _safe_error(error: Exception, api_key: str) -> str:
    message = str(error)
    if api_key:
        message = message.replace(api_key, "<redacted>")
    return message[:2000]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    unknown_modes = set(modes) - set(MODES)
    if unknown_modes:
        raise SystemExit(f"unsupported modes: {sorted(unknown_modes)}")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")

    file_env = load_env_file(args.env_file)
    api_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or file_env.get("GEMINI_API_KEY", "").strip()
    )
    model = (
        args.model
        or os.environ.get("GEMINI_MODEL", "").strip()
        or file_env.get("GEMINI_MODEL", "").strip()
        or "gemini-2.5-flash"
    )
    client = GeminiClient(api_key, args.timeout)
    model_info = client.model_info(model)

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
    print(
        f"Preflight OK: {len(all_cases)} cases, {len(selected)} selected, "
        f"model={model}, modes={','.join(modes)}"
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
        "provider": "gemini",
        "model": model,
        "model_info": model_info,
        "gemini_sdk_version": importlib.metadata.version("google-genai"),
        "prompt_version": PROMPT_VERSION,
        "dataset_sha256": dataset_sha,
        "seed": args.seed,
        "temperature": None,
        "num_predict": args.num_predict,
        "host": socket.gethostname(),
        "platform": platform.platform(),
    }
    total = len(selected) * len(modes)
    finished = 0
    with args.output.open("a", encoding="utf-8") as output:
        for case in selected:
            options, displayed_correct, option_order = permuted_options(case, args.seed)
            for mode in modes:
                key = (case.case_id, mode, model, PROMPT_VERSION, args.seed)
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
                            response = client.generate_choice(
                                model=model,
                                system_prompt=SYSTEM_PROMPT,
                                user_prompt=prompt,
                                image=image,
                                seed=args.seed,
                                max_output_tokens=args.num_predict,
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
                                    "finish_reason": response.finish_reason,
                                    "prompt_token_count": response.prompt_token_count,
                                    "candidates_token_count": response.candidates_token_count,
                                    "total_token_count": response.total_token_count,
                                    "response_model_version": response.model_version,
                                    "response_id": response.response_id,
                                }
                            )
                            last_error = None
                            break
                        except Exception as error:  # noqa: BLE001 - bounded API retry
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
                            "error": _safe_error(error, api_key),
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
