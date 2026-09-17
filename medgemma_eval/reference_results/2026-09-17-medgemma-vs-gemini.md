# Gemini 3.6 Flash versus MedGemma 27B visual ablation — 2026-09-17

This is a retrospective model-screening result, not clinical validation and not
evidence for autonomous patient diagnosis.

## Question

Does the backend-configured generalist Gemini model outperform the local
`medgemma:27b` model when both receive the same pre-examination proxy context and
NEJM image? Does the stronger model specifically benefit from the correct image?

## Matched design and provenance

- Dataset: all 120 public AgentClinic extended NEJM cases; 90 directly ask for a
  diagnosis. Public cases may overlap either model's pretraining data.
- Conditions: proxy context only, context plus matched image, context plus a
  deterministic subtype-near wrong image, and task/choices plus matched image without
  clinical history.
- Pairing: identical case, prompt version `nejm-visual-ablation-v3`, seed `20260916`,
  option permutation, image checksum, and scoring rule.
- Local baseline: Ollama `medgemma:27b`, digest
  `58238ae38f99827496301de22016b2f94157552971f5a38db226f13802c2437e`, Q4_K_M.
- Candidate: Gemini Developer API `gemini-3.6-flash`, with the API response also
  identifying `gemini-3.6-flash`; `google-genai` SDK 2.16.0.
- Gemini used the backend's existing key and model configuration. The key was neither
  copied nor written to results. The 1024-token output budget includes model reasoning
  tokens; only a structured A–E choice was scored.
- All 480 Gemini responses and all 480 MedGemma responses were valid and scorable.
  Failures would have remained in the primary denominator.

## All 120 clinical questions

| Condition | MedGemma 27B | Gemini 3.6 Flash | Gemini Δ | Gemini helpful / harmful | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proxy context only | 65.0% | 88.3% | +23.3 pp | 32 / 4 | <0.001 |
| Context + matched image | 62.5% | 96.7% | +34.2 pp | 41 / 0 | <0.001 |
| Context + mismatched image | 66.7% | 90.0% | +23.3 pp | 30 / 2 | <0.001 |
| Task/choices + matched image | 25.8% | 90.0% | +64.2 pp | 79 / 2 | <0.001 |

Gemini was decisively more accurate than MedGemma in every condition. With the
clinically relevant proxy-context-plus-image condition, Gemini answered 116/120
(96.7%, Wilson 95% CI 91.7%–98.7%) versus MedGemma's 75/120 (62.5%, 53.6%–70.6%).

## Direct-diagnosis subset

| Condition | MedGemma 27B | Gemini 3.6 Flash | Gemini Δ | Gemini helpful / harmful | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proxy context only | 70.0% | 87.8% | +17.8 pp | 20 / 4 | 0.002 |
| Context + matched image | 66.7% | 97.8% | +31.1 pp | 28 / 0 | <0.001 |
| Context + mismatched image | 67.8% | 90.0% | +22.2 pp | 21 / 1 | <0.001 |
| Task/choices + matched image | 23.3% | 90.0% | +66.7 pp | 61 / 1 | <0.001 |

For the 90 questions that directly ask for a diagnosis, Gemini answered 88/90 with
the matched image and proxy context. This benchmark estimate is 97.8% (95% CI
92.3%–99.4%), not a real-world diagnostic probability.

## Does Gemini specifically use the correct image?

| Model | Matched image vs no image | Helpful / harmful | Matched vs wrong image | Helpful / harmful |
| --- | ---: | ---: | ---: | ---: |
| MedGemma 27B | −2.5 pp (p=0.453) | 2 / 5 | −4.2 pp (p=0.227) | 3 / 8 |
| Gemini 3.6 Flash | +8.3 pp (p=0.013) | 12 / 2 | +6.7 pp (p=0.039) | 10 / 2 |

Unlike MedGemma, Gemini showed statistically detectable positive matched-image
grounding in both controls. This is evidence that Gemini used image-specific
information on this benchmark, rather than only being a stronger text reasoner.

The evidence is not clean enough for a clinical claim. Gemini also scored 90.0% when
given only the final task, answer choices, and image. This unusually high result on a
public, dated NEJM set is compatible with several explanations: strong image and OCR
ability, answer-option cues, retrieval-like memory of cases seen during pretraining,
or a mixture of all three. A missing task-and-choices-without-image arm should be
added to quantify how much of this score requires the image at all.

## Adequately sized subtype screen

| Subtype | n | MedGemma matched | Gemini matched | Gemini Δ | Gemini matched over wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| physical | 51 | 62.7% | 94.1% | +31.4 pp | 0.0 pp |
| CT scan | 19 | 42.1% | 94.7% | +52.6 pp | +21.1 pp |
| dermatology | 16 | 81.2% | 100.0% | +18.8 pp | 0.0 pp |
| hist/path | 13 | 69.2% | 100.0% | +30.8 pp | +15.4 pp |
| xray | 12 | 66.7% | 100.0% | +33.3 pp | +8.3 pp |
| ophthalmology | 11 | 81.8% | 100.0% | +18.2 pp | 0.0 pp |

The CT and hist/path screens show the clearest matched-over-wrong improvements, but
the samples are small and the images are publication panels rather than native DICOM
volumes or whole-slide images. These results cannot select a production model for
those modalities.

## Runtime observation

Mean Gemini latency was approximately 6.2 seconds for proxy context plus matched
image, compared with 3.1 seconds for the local MedGemma run. This is not a controlled
serving benchmark: Gemini used a remote API and internal reasoning tokens, whereas
MedGemma used local quantized weights with CPU/GPU offload.

## Decision

On this public benchmark, Gemini 3.6 Flash is substantially better than the tested
MedGemma 27B quantization and shows positive image-specific grounding. The result
supports using Gemini as the stronger generalist comparator in the next Agentic
Clinic experiment.

It does not prove that a generalist model can safely replace modality specialists.
Before that decision, repeat the comparison on unseen, institution-approved,
image-essential cases with native inputs, a task-only no-image control, blinded
specialist findings, clinician-reviewed critical-miss labels, and an external-site
holdout. Real patient images must not be sent to an external API without an approved
privacy, security, contractual, and clinical-governance boundary.
