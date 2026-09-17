"""Download and checksum AgentClinic-NEJM images into the local evaluation cache."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from medgemma_eval.dataset import ensure_image, load_cases, select_cases
else:
    from .dataset import ensure_image, load_cases, select_cases


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent


def _csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "agentclinic_nejm_extended.jsonl"
    )
    parser.add_argument("--cache-dir", type=Path, default=EVAL_DIR / "cache")
    parser.add_argument(
        "--manifest", type=Path, default=EVAL_DIR / "results" / "image_manifest.jsonl"
    )
    parser.add_argument("--subtypes", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    cases = select_cases(
        load_cases(args.dataset), _csv(args.subtypes), set(), set(), args.limit
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with args.manifest.open("w", encoding="utf-8") as manifest:
        for position, case in enumerate(cases, 1):
            record = {
                "case_index": case.index,
                "case_id": case.case_id,
                "image_url": case.image_url,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            try:
                path, checksum, downloaded = ensure_image(
                    case,
                    args.cache_dir,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                record.update(
                    {
                        "status": "ok",
                        "path": str(path),
                        "sha256": checksum,
                        "bytes": path.stat().st_size,
                        "downloaded": downloaded,
                    }
                )
            except RuntimeError as error:
                failures += 1
                record.update({"status": "error", "error": str(error)})
            manifest.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            print(f"[{position}/{len(cases)}] {case.case_id}: {record['status']}")
    print(f"Manifest written to {args.manifest}; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
