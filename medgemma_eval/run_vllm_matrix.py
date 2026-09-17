"""Sequentially serve and benchmark a local model matrix with vLLM."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from medgemma_eval.vllm_client import VLLMClient
else:
    from .vllm_client import VLLMClient


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "vllm_models.example.json"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "vllm-matrix"
VISUAL_MODES = "context_only,context_image,context_mismatched,image_only"
TEXT_MODES = "context_only"


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("manifest must contain a non-empty models list")
    names = [str(item.get("name", "")) for item in models if isinstance(item, dict)]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("every model needs a unique non-empty name")
    return data


def inspect_model(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        return {"config_found": False}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = data.get("quantization_config")
    return {
        "config_found": True,
        "architectures": data.get("architectures"),
        "model_type": data.get("model_type"),
        "dtype": data.get("torch_dtype") or data.get("dtype"),
        "quantization_method": quantization.get("quant_method")
        if isinstance(quantization, dict)
        else None,
        "has_vision_config": isinstance(data.get("vision_config"), dict),
    }


def detect_vision(path: Path, inspection: dict[str, Any]) -> bool:
    if inspection.get("has_vision_config"):
        return True
    names = " ".join(
        [str(inspection.get("model_type", ""))]
        + [str(value) for value in inspection.get("architectures") or []]
    ).lower()
    markers = ("vision", "visual", "vl", "omni", "gemma3", "gemma4", "medgemma")
    if any(marker in names for marker in markers):
        return True
    for filename in ("preprocessor_config.json", "processor_config.json"):
        candidate = path / filename
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8", errors="replace").lower()
            if "image_processor" in text or "vision" in text:
                return True
    return False


def resolve_vision(model: dict[str, Any], inspection: dict[str, Any]) -> bool:
    configured = model.get("vision")
    if isinstance(configured, bool):
        return configured
    if configured == "auto":
        return detect_vision(Path(str(model["path"])), inspection)
    raise ValueError(f"{model['name']}: vision must be true, false, or 'auto'")


def build_serve_command(
    manifest: dict[str, Any], model: dict[str, Any], vision: bool | None = None
) -> list[str]:
    server = manifest.get("server") or {}
    command = list(server.get("command") or ["vllm", "serve"])
    command.extend(
        [
            str(model["path"]),
            "--served-model-name",
            str(model["name"]),
            "--host",
            str(server.get("host", "127.0.0.1")),
            "--port",
            str(server.get("port", 8000)),
            "--tensor-parallel-size",
            str(server.get("tensor_parallel_size", 1)),
            "--gpu-memory-utilization",
            str(server.get("gpu_memory_utilization", 0.9)),
            "--max-model-len",
            str(model.get("max_model_len", server.get("max_model_len", 8192))),
            "--generation-config",
            "vllm",
        ]
    )
    command.extend(str(value) for value in model.get("serve_args") or [])
    if vision and not any(
        value.startswith("--limit-mm-per-prompt") for value in command
    ):
        command.extend(["--limit-mm-per-prompt.image", "1"])
    return command


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _capture_command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)[:1000]}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip()[:4000],
        "stderr": result.stderr.strip()[:2000],
    }


def capture_environment(server_command: list[str]) -> dict[str, Any]:
    executable = server_command[0] if server_command else "vllm"
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "vllm_executable": shutil.which(executable),
        "vllm_cli_version": _capture_command([executable, "--version"]),
        "nvidia_smi": _capture_command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }


def wait_until_ready(
    client: VLLMClient, process: subprocess.Popen[Any], timeout: float
) -> tuple[list[str], str | None]:
    deadline = time.monotonic() + timeout
    last_error = "server did not become ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited during startup with code {process.returncode}"
            )
        try:
            if client.health():
                return client.models(), client.version()
        except (RuntimeError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(5)
    raise RuntimeError(last_error)


def stop_process_group(process: subprocess.Popen[Any], timeout: float = 60) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _benchmark_command(
    *,
    model_name: str,
    base_url: str,
    vision: bool,
    response_format: str,
    output: Path,
    smoke: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_vllm_benchmark.py"),
        "--base-url",
        base_url,
        "--model",
        model_name,
        "--modes",
        VISUAL_MODES if vision else TEXT_MODES,
        "--response-format",
        response_format,
        "--output",
        str(output),
    ]
    if vision:
        command.append("--vision-capable")
    if smoke:
        command.extend(["--indices", "30"])
    return command


def _smoke_ok(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return len(rows) == expected and all(row.get("status") == "ok" for row in rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--only", default="", help="comma-separated model names")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop after the first failed model instead of continuing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    selected_names = {value.strip() for value in args.only.split(",") if value.strip()}
    models = [
        model
        for model in manifest["models"]
        if model.get("enabled", True)
        and (not selected_names or model["name"] in selected_names)
    ]
    if not models:
        raise SystemExit("no enabled models selected")
    if selected_names - {str(model["name"]) for model in models}:
        missing = selected_names - {str(model["name"]) for model in models}
        raise SystemExit(f"unknown or disabled models: {sorted(missing)}")

    server = manifest.get("server") or {}
    host = str(server.get("host", "127.0.0.1"))
    port = int(server.get("port", 8000))
    base_url = f"http://{host}:{port}"
    api_key = os.environ.get("VLLM_API_KEY", "").strip()
    startup_timeout = float(server.get("startup_timeout_seconds", 1200))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.results_root / run_id
    if run_dir.exists() and not args.dry_run:
        raise SystemExit(f"run directory already exists: {run_dir}")
    if not args.dry_run:
        run_dir.mkdir(parents=True)

    matrix: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "environment": capture_environment(list(server.get("command") or ["vllm"])),
        "models": [],
    }
    overall_success = True
    for model in models:
        model_path = Path(str(model["path"]))
        inspection = inspect_model(model_path)
        vision = resolve_vision(model, inspection)
        command = build_serve_command(manifest, model, vision)
        item: dict[str, Any] = {
            "name": model["name"],
            "path": str(model_path),
            "vision": vision,
            "inspection": inspection,
            "serve_command": command,
            "response_format": model.get("response_format", "json_schema"),
            "status": "planned" if args.dry_run else "pending",
        }
        matrix["models"].append(item)
        if args.dry_run:
            print(json.dumps(item, ensure_ascii=False))
            continue
        model_dir = run_dir / str(model["name"])
        model_dir.mkdir()
        if not model_path.is_dir():
            item.update(status="invalid_model_path", error="model directory not found")
            overall_success = False
            _write_json(run_dir / "matrix-run.json", matrix)
            if args.stop_on_error:
                break
            continue
        executable = command[0]
        if shutil.which(executable) is None:
            item.update(status="missing_vllm_command", error=f"not found: {executable}")
            overall_success = False
            _write_json(run_dir / "matrix-run.json", matrix)
            if args.stop_on_error:
                break
            continue

        log_path = model_dir / "vllm-server.log"
        process: subprocess.Popen[Any] | None = None
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                item["server_pid"] = process.pid
                client = VLLMClient(base_url, 10, api_key)
                served_models, version = wait_until_ready(
                    client, process, startup_timeout
                )
                item.update(
                    status="server_ready",
                    served_models=served_models,
                    vllm_version=version,
                )
                _write_json(run_dir / "matrix-run.json", matrix)

                response_format = str(item["response_format"])
                if not args.skip_smoke:
                    smoke_path = model_dir / "smoke.jsonl"
                    smoke_command = _benchmark_command(
                        model_name=str(model["name"]),
                        base_url=base_url,
                        vision=vision,
                        response_format=response_format,
                        output=smoke_path,
                        smoke=True,
                    )
                    smoke_result = subprocess.run(smoke_command, check=False)
                    expected = 4 if vision else 1
                    if smoke_result.returncode != 0 or not _smoke_ok(
                        smoke_path, expected
                    ):
                        raise RuntimeError("benchmark smoke test failed")

                result_path = model_dir / "cases.jsonl"
                benchmark_command = _benchmark_command(
                    model_name=str(model["name"]),
                    base_url=base_url,
                    vision=vision,
                    response_format=response_format,
                    output=result_path,
                    smoke=False,
                )
                benchmark_result = subprocess.run(benchmark_command, check=False)
                if benchmark_result.returncode != 0:
                    raise RuntimeError(
                        f"benchmark exited with code {benchmark_result.returncode}"
                    )
                item.update(status="complete", result=str(result_path.resolve()))
        except Exception as error:  # noqa: BLE001 - persist matrix failure and continue
            item.update(status="failed", error=str(error)[:2000])
            overall_success = False
        finally:
            if process is not None:
                stop_process_group(process)
            item["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(run_dir / "matrix-run.json", matrix)
        if item["status"] != "complete" and args.stop_on_error:
            break

    if args.dry_run:
        return 0
    print(f"Matrix record: {run_dir / 'matrix-run.json'}")
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
