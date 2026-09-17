"""Dataset loading, validation, and image caching for the NEJM benchmark."""

from __future__ import annotations

import hashlib
import json
import random
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    index: int
    case_id: str
    image_url: str
    question: str
    question_target: str
    clinical_context: str
    options: tuple[str, ...]
    correct_index: int
    subtypes: tuple[str, ...]
    task_type: str

    @property
    def correct_label(self) -> str:
        return chr(ord("A") + self.correct_index)

    @property
    def correct_answer(self) -> str:
        return self.options[self.correct_index]


def _case_id(image_url: str, index: int) -> str:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(image_url).query)
    candidate = query.get("id", [""])[0]
    if re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
        return candidate
    return f"case-{index:03d}-{hashlib.sha256(image_url.encode()).hexdigest()[:10]}"


def classify_task(question: str) -> str:
    """Classify the requested answer, without using the gold answer."""
    text = final_question(question).lower().removesuffix("?").strip()

    if (
        re.search(r"\btest is most likely to reveal the diagnosis\b", text)
        or re.search(
            r"\bdiagnosis(?: in this case)? is .*associated with which .*syndromes?\b",
            text,
        )
        or re.search(r"\bassociated with increased mortality in this diagnosis\b", text)
    ):
        return "test_or_association"
    if "diagnos" in text:
        return "diagnosis"
    if re.search(r"\b(treatment|treated|therapy|management)\b", text):
        return "treatment"
    if re.search(
        r"\b(cause|etiology|mechanism|pathogen|organism|vector|toxicity|"
        r"contributory factor|activit(?:y|ies) likely preceded)\b",
        text,
    ):
        return "etiology"
    if re.search(r"\b(name of this|what is seen|exam finding|these crystals)\b", text):
        return "finding"
    if "site of the culprit lesion" in text:
        return "localization"
    return "other"


def final_question(question: str) -> str:
    """Extract the final requested task while retaining its question mark."""
    normalized = " ".join(question.lower().split())
    without_final_mark = normalized.removesuffix("?")
    boundary = max(
        without_final_mark.rfind("."),
        without_final_mark.rfind("!"),
        without_final_mark.rfind("?"),
    )
    target = " ".join(question.split())
    if boundary >= 0:
        # The normalized and original strings can differ in spacing, so extract the
        # final sentence from the whitespace-normalized original independently.
        original_without_mark = target.removesuffix("?")
        original_boundary = max(
            original_without_mark.rfind("."),
            original_without_mark.rfind("!"),
            original_without_mark.rfind("?"),
        )
        target = original_without_mark[original_boundary + 1 :].strip()
    return f"{target.removesuffix('?')}?"


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            answers = row.get("answers")
            if not isinstance(answers, list) or len(answers) != 5:
                raise ValueError(f"row {index}: expected exactly five answers")
            correct = [
                i for i, answer in enumerate(answers) if answer.get("correct") is True
            ]
            if len(correct) != 1:
                raise ValueError(f"row {index}: expected exactly one correct answer")
            subtypes = row.get("type")
            if not isinstance(subtypes, list) or not subtypes:
                raise ValueError(f"row {index}: extended NEJM row has no subtype")
            image_url = str(row["image_url"])
            identifier = _case_id(image_url, index)
            if identifier in seen_ids:
                raise ValueError(f"row {index}: duplicate case id {identifier}")
            seen_ids.add(identifier)
            question = str(row["question"])
            clinical_context = str(row.get("patient_info", "")).strip()
            if not clinical_context:
                raise ValueError(f"row {index}: patient_info is required")
            cases.append(
                Case(
                    index=index,
                    case_id=identifier,
                    image_url=image_url,
                    question=question,
                    question_target=final_question(question),
                    clinical_context=clinical_context,
                    options=tuple(str(answer["text"]) for answer in answers),
                    correct_index=correct[0],
                    subtypes=tuple(str(value) for value in subtypes),
                    task_type=classify_task(question),
                )
            )
    return cases


def select_cases(
    cases: Iterable[Case],
    subtypes: set[str],
    task_types: set[str],
    indices: set[int],
    limit: int | None,
) -> list[Case]:
    selected = [
        case
        for case in cases
        if (not subtypes or subtypes.intersection(case.subtypes))
        and (not task_types or case.task_type in task_types)
        and (not indices or case.index in indices)
    ]
    return selected[:limit] if limit is not None else selected


def image_path(cache_dir: Path, case: Case) -> Path:
    return cache_dir / f"{case.case_id}.jpg"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_image(data: bytes, content_type: str) -> bool:
    signatures = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a")
    return content_type.startswith("image/") and data.startswith(signatures)


def ensure_image(
    case: Case,
    cache_dir: Path,
    *,
    timeout: float = 30.0,
    retries: int = 3,
    max_bytes: int = 25 * 1024 * 1024,
) -> tuple[Path, str, bool]:
    """Return path, sha256, and whether a new download was performed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = image_path(cache_dir, case)
    if target.is_file() and target.stat().st_size > 0:
        return target, file_sha256(target), False

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                case.image_url,
                headers={"User-Agent": "AgentClinic-MedGemma-Eval/0.1"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get_content_type()
                data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError(f"image exceeds {max_bytes} bytes")
            if not _looks_like_image(data, content_type):
                raise ValueError(f"unexpected response type {content_type!r}")
            with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as temp:
                temp.write(data)
                temp_path = Path(temp.name)
            temp_path.replace(target)
            return target, hashlib.sha256(data).hexdigest(), True
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {case.case_id}: {last_error}")


def case_metadata(case: Case) -> dict[str, Any]:
    return {
        "case_index": case.index,
        "case_id": case.case_id,
        "subtypes": list(case.subtypes),
        "task_type": case.task_type,
        "question_target": case.question_target,
        "correct_index": case.correct_index,
        "correct_label": case.correct_label,
        "correct_answer": case.correct_answer,
    }


def permuted_options(
    case: Case, seed: int
) -> tuple[tuple[str, ...], int, tuple[int, ...]]:
    """Return a stable per-case permutation shared by every evaluation mode."""
    material = f"{seed}:{case.case_id}:option-order".encode()
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    order = list(range(len(case.options)))
    random.Random(derived_seed).shuffle(order)
    options = tuple(case.options[index] for index in order)
    displayed_correct = order.index(case.correct_index)
    return options, displayed_correct, tuple(order)


def mismatched_image_case(cases: Iterable[Case], case: Case, seed: int) -> Case:
    """Select a deterministic wrong image, preferring maximum subtype overlap."""
    candidates = [candidate for candidate in cases if candidate.case_id != case.case_id]
    if not candidates:
        raise ValueError("at least two cases are required for mismatched-image control")
    overlap = {
        candidate.case_id: len(set(candidate.subtypes).intersection(case.subtypes))
        for candidate in candidates
    }
    best_overlap = max(overlap.values())
    if best_overlap > 0:
        candidates = [
            candidate
            for candidate in candidates
            if overlap[candidate.case_id] == best_overlap
        ]
    material = f"{seed}:{case.case_id}:mismatched-image".encode()
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return candidates[derived_seed % len(candidates)]
