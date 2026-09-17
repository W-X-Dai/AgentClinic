# MedGemma 27B × AgentClinic-NEJM subtype evaluation

This directory is an isolated, reproducible harness for measuring the local Ollama
model `medgemma:27b` on the 120-case `agentclinic_nejm_extended.jsonl` set. It does
not modify or depend on the legacy `agentclinic.py` runner. A separate Gemini adapter
uses the backend's existing configuration for a paired generalist-model comparison.

This is a retrospective research benchmark, not clinical validation. Do not use its
outputs to diagnose or treat patients. The NEJM cases are public and may overlap the
model's pretraining data.

## What it measures

The legacy benchmark comparison runs each case under paired conditions with the same
prompt, option permutation, seed, and generation settings:

- `text_only`: original question and five answer choices.
- `multimodal`: the same question and choices plus the NEJM image.
- `image_only` (optional): image and choices without the case narrative.

For image-reading screening, use the visual ablation modes:

- `context_only`: `patient_info` proxy context plus the final task and choices.
- `context_image`: identical proxy context plus the matched image.
- `context_mismatched`: identical prompt plus a deterministic wrong image, chosen to
  maximize overlap with the case's subtype tags.
- `image_only`: final task, choices, and matched image without clinical history.

`patient_info` removes most formal radiology/pathology interpretations but is not a
clinician-redacted nonvisual history. It can retain patient-visible skin or eye
findings and high-level mentions of tests. These modes are therefore a model-screening
proxy, not a definitive pure-vision or clinical-validation dataset.

The main score is exact A–E multiple-choice accuracy; no second LLM judges the
answers. Answer order is deterministically shuffled per case to reduce position bias.
Transport failures and unparseable answers count as incorrect in the primary
denominator. The analyzer reports overall and per-subtype accuracy, Wilson 95%
confidence intervals, scorable rate, and paired image-helpful/image-harmful flips.

The dataset's `type` field is multi-label. A case may count under more than one of
the 13 image/test subtypes, but counts only once in the overall score. Only these six
subtypes have at least 10 cases and are suitable for even preliminary ranking:
`physical`, `CT scan`, `dermatology`, `hist/path`, `xray`, and `ophthalmology`.
Results for smaller groups are descriptive only.

The source data also mixes question intents. Exactly 90 cases directly ask for a
diagnosis; the other 30 ask for an etiology, treatment, finding, association, test,
or anatomical localization. Every result is tagged with `task_type`, so the full
clinical-question benchmark and direct-diagnosis subset can be reported separately.

## Requirements and preflight

- Python 3.10 or later (standard library only)
- Ollama running at `http://127.0.0.1:11434`
- Local `medgemma:27b` model with the `vision` capability
- Network access to `csvc.nejm.org` once, to populate the ignored image cache

From `AgentClinic/`:

```bash
python3 medgemma_eval/run_benchmark.py --preflight-only
python3 -m unittest discover -s medgemma_eval/tests -v
```

`medgemma:latest` is not an alias for the requested 27B model on the current host;
always specify `medgemma:27b` explicitly. The model is about 17 GB, so a 16 GB GPU
will use CPU/RAM offload even when otherwise empty. Inspect resource use before a
long run:

```bash
ollama ps
nvidia-smi
```

Do not automatically kill unrelated GPU processes. If a known service must be
paused, stop it deliberately and restore it after the evaluation.

## Download and smoke test

Download all images and write a checksum manifest:

```bash
python3 medgemma_eval/download_images.py
```

The following seven cases collectively cover all 13 subtypes. This is a plumbing
smoke test only; it is far too small for model-quality conclusions:

```bash
python3 medgemma_eval/run_benchmark.py \
  --indices 21,30,40,45,53,100,105 \
  --modes text_only,multimodal \
  --output medgemma_eval/results/smoke.jsonl

python3 medgemma_eval/analyze_results.py \
  medgemma_eval/results/smoke.jsonl \
  --output-dir medgemma_eval/results/smoke-summary
```

The runner appends one JSON object per case-condition and flushes it immediately.
Rerunning the same command resumes from completed keys. Add `--retry-errors` to
retry failed records, or `--no-resume` to deliberately append a fresh attempt.

## Full benchmark

Run the paired 120-case benchmark:

```bash
python3 medgemma_eval/run_benchmark.py \
  --model medgemma:27b \
  --modes text_only,multimodal \
  --output medgemma_eval/results/medgemma-27b-all.jsonl

python3 medgemma_eval/analyze_results.py \
  medgemma_eval/results/medgemma-27b-all.jsonl \
  --output-dir medgemma_eval/results/medgemma-27b-all-summary
```

Run the four-arm image-reading ablation and its analyzer:

```bash
python3 medgemma_eval/run_benchmark.py \
  --modes context_only,context_image,context_mismatched,image_only \
  --output medgemma_eval/results/medgemma-27b-visual.jsonl

python3 medgemma_eval/analyze_visual_results.py \
  medgemma_eval/results/medgemma-27b-visual.jsonl \
  --output-dir medgemma_eval/results/medgemma-27b-visual-summary
```

Run the identical visual ablation through the backend-configured Gemini API. This
reads `GEMINI_API_KEY` and `GEMINI_MODEL` from `backend/.env` without copying the key
into AgentClinic or result records. Use the backend interpreter because it owns the
`google-genai` dependency:

```bash
../backend/venv/bin/python \
  medgemma_eval/run_gemini_benchmark.py \
  --output medgemma_eval/results/gemini-visual.jsonl

../backend/venv/bin/python \
  medgemma_eval/compare_visual_models.py \
  medgemma_eval/results/medgemma-27b-visual.jsonl \
  medgemma_eval/results/gemini-visual.jsonl \
  --output-dir medgemma_eval/results/medgemma-vs-gemini
```

Gemini 3.5 and newer may consume reasoning tokens before returning the short JSON
answer, so its runner defaults to a 1024-token total output budget. This avoids
counting truncated JSON as model quality while retaining failures in the denominator.

Run only the 90 direct-diagnosis questions:

```bash
python3 medgemma_eval/run_benchmark.py \
  --task-types diagnosis \
  --modes text_only,multimodal \
  --output medgemma_eval/results/medgemma-27b-diagnosis.jsonl
```

If the full run already exists, produce the direct-diagnosis report without rerunning
the model:

```bash
python3 medgemma_eval/analyze_results.py \
  medgemma_eval/results/medgemma-27b-all.jsonl \
  --task-types diagnosis \
  --output-dir medgemma_eval/results/medgemma-27b-diagnosis-summary
```

Useful filters include `--subtypes 'CT scan,hist/path'`, `--indices 0-14`, and
`--limit 3`. Use a distinct output file when changing the model, prompt code, seed,
or study cohort. The strict response schema requests only `{"choice":"A"}` to
prevent long rationale generation from turning a valid choice into a truncated
response. The JSONL captures the model digest and quantization, Ollama
version, dataset and prompt hashes, generation settings, image checksum, latency,
raw response, parsing method, and correctness.

Generated images and results remain under this directory's ignored `cache/` and
`results/` folders and should not be committed. The manifest retains source URLs and
checksums because remote images can change or disappear.

## Interpreting strengths and weaknesses

Use `report.md` and `paired.csv` together:

- Multimodal accuracy answers “how often was this subtype correct with the image?”
- `vision_delta` compares text+image against the paired text-only baseline.
- `image_helpful` and `image_harmful` show individual 0→1 and 1→0 flips.
- High accuracy with near-zero image delta may reflect a text-solvable question, not
  image understanding; many questions explicitly describe the image in words.
- Small-n subtype rankings are unstable. The report labels groups below the default
  `--min-rank-n 10` as descriptive only.
- Correct multiple-choice answers do not prove visual grounding or clinical safety.

The current harness deliberately avoids open-ended semantic grading because reliable
equivalence would require a frozen, clinically reviewed alias set or human review.
That should be a separate secondary experiment rather than an LLM judge mixed into
the primary score.

## vLLM and specialist-model decisions

vLLM is a serving engine, not an image-reading model. Moving identical MedGemma
weights from Ollama to vLLM can change throughput, memory use, preprocessing, and
generation defaults, but does not itself add medical visual expertise. Verify
processor and answer parity before comparing runtimes.

Use this decision sequence:

1. Establish context-only, matched-image, mismatched-image, image-only, and
   clinician-verified oracle-findings arms on an image-essential holdout.
2. Run MedGemma alone and each specialist without answer-choice access.
3. Feed the specialist's blinded structured findings to the same MedGemma reasoner.
4. Introduce the specialist only if the hybrid has a predeclared, paired improvement
   with no unacceptable increase in harmful flips or critical misses.

Current candidates worth a separate, license-reviewed PoC are MedGemma 1.5 4B and
MedSigLIP for broad screening; MAIRA-2 or CheXagent for chest radiographs; RadFM for
native CT/MRI research; and SlideChat or TITAN when native whole-slide pathology data
is available. Do not compare a WSI or volume model using only a downscaled NEJM panel.

The completed Gemini comparison shows that a stronger generalist can outperform the
local MedGemma baseline and can demonstrate positive matched-image grounding on this
benchmark. It does not establish performance on unseen hospital images: the cases are
public, the task exposes answer choices, and the unusually high image-plus-task score
is compatible with pretraining contamination or case memorization. Add a task-and-
choices-without-image control and a private image-essential holdout before deciding
that a generalist model can replace modality specialists.

## RTX Pro 6000 vLLM matrix

The checked-in [`vllm_models.example.json`](vllm_models.example.json) describes the
11 checkpoint directories supplied on the target host. The matrix runner starts one
server at a time on `127.0.0.1`, waits for `/health` and `/v1/models`, runs one smoke
case, executes the full cohort, and terminates only the child process group that it
started. A failed model is recorded and the default behavior continues to the next
checkpoint; add `--stop-on-error` when debugging a single model. The matrix record
also captures the GPU/driver, Python/platform, vLLM CLI/API versions, inspected local
checkpoint metadata, and exact launch command.

The capability split is intentional:

- Image arms: GLM-4.5V, both MedGemma checkpoints, both Nemotron Omni checkpoints,
  and Qwen2.5-VL.
- Context-only control: DeepSeek-R1-Distill-Qwen, GPT-OSS 20B/120B, and
  Qwen2.5-32B-Instruct.
- `gemma-4-31b`: configured as `auto`; the target host's `config.json` and processor
  metadata determine whether it receives image arms. Review the dry-run output.

Do not count an unsupported image request as a text model's diagnostic error. Text-
only checkpoints are excluded from matched-image rankings by design.

On the target host, from `AgentClinic/`:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
vllm --version

python3 medgemma_eval/download_images.py
python3 medgemma_eval/run_vllm_matrix.py --dry-run
```

Start with one small VLM and inspect its smoke/result/log files:

```bash
python3 medgemma_eval/run_vllm_matrix.py \
  --only medgemma-4b-it \
  --run-id 20260917-medgemma-4b
```

Then run the full matrix. Use a new immutable run ID rather than overwriting the
first run:

```bash
python3 medgemma_eval/run_vllm_matrix.py \
  --run-id 20260917-rtx-pro-6000
```

Summarize all completed models, then optionally produce a diagnosis-only report:

```bash
python3 medgemma_eval/analyze_vllm_matrix.py \
  medgemma_eval/results/vllm-matrix/20260917-rtx-pro-6000/matrix-run.json

python3 medgemma_eval/analyze_vllm_matrix.py \
  medgemma_eval/results/vllm-matrix/20260917-rtx-pro-6000/matrix-run.json \
  --task-types diagnosis \
  --output-dir medgemma_eval/results/vllm-matrix/20260917-rtx-pro-6000/diagnosis-summary
```

For an already running vLLM endpoint, bypass orchestration and call the runner
directly:

```bash
python3 medgemma_eval/run_vllm_benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --model medgemma-27b-it \
  --vision-capable \
  --modes context_only,context_image,context_mismatched,image_only \
  --output medgemma_eval/results/medgemma-27b-vllm.jsonl
```

Operational notes:

- The manifest uses `--generation-config vllm` so checkpoint-level generation
  defaults do not silently change the comparison.
- Checkpoint quantization is read from local configuration; the runner does not infer
  AWQ or NVFP4 merely from a directory name.
- NVIDIA's Nemotron Omni model card currently calls for vLLM 0.20.0,
  `--trust-remote-code`, `--reasoning-parser nemotron_v3`, and `--moe-backend triton`
  on RTX Pro. The NVFP4 entry additionally uses an FP8 KV cache. Keep separate vLLM
  environments if other checkpoints require a conflicting version.
- Qwen2.5-VL AWQ uses `--dtype half` following its official model card.
- The server remains loopback-only. Do not expose it with `0.0.0.0`; vLLM documents
  that its API key does not protect every endpoint.
- Images are sent as base64 data URLs. The harness does not enable vLLM local-file
  access or `--allowed-local-media-path`.
- Structured-output incompatibility fails the smoke test instead of silently changing
  scoring. A checkpoint can explicitly set `"response_format": "json_object"` or
  `"none"` in a copied manifest, and the resulting difference is recorded.

Official references: [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html),
[multimodal example](https://docs.vllm.ai/en/latest/examples/generate/multimodal/),
[serve CLI](https://docs.vllm.ai/en/latest/cli/serve/), and
[supported models](https://docs.vllm.ai/en/latest/models/supported_models/).
