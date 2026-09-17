# MedGemma 27B image-reading ablation — 2026-09-16

This is a model-screening research result, not clinical validation or evidence for
autonomous diagnosis.

## Research question

Does `medgemma:27b` use the AgentClinic-NEJM images specifically enough to justify a
single-model architecture, or should the next prototype compare specialist medical
image models?

The four paired conditions were:

- `context_only`: patient-actor proxy context, final task, and five choices.
- `context_image`: the same prompt plus the matched NEJM image.
- `context_mismatched`: the same prompt plus a deterministic wrong image selected to
  maximize available subtype overlap.
- `image_only`: final task, choices, and matched image without clinical history.

The proxy context excludes the original question's formal imaging/pathology report,
but it is not clinician-redacted and can retain patient-visible findings. NEJM images
are composite publication screenshots rather than native DICOM volumes or WSI data.

## Complete 120-case result

All 480 responses were valid and scorable. Every case had all four conditions, and
no mismatched-image condition reused its own image.

| Condition | Correct / n | Accuracy | Wilson 95% CI |
| --- | ---: | ---: | ---: |
| Proxy context only | 78 / 120 | 65.0% | 56.1%–72.9% |
| Context + matched image | 75 / 120 | 62.5% | 53.6%–70.6% |
| Context + mismatched image | 80 / 120 | 66.7% | 57.8%–74.5% |
| Image only | 31 / 120 | 25.8% | 18.8%–34.3% |

- Matched image versus no image: −2.5 points; 2 helpful and 5 harmful flips;
  answer-change rate 9.2%; exact McNemar p=0.453.
- Matched versus mismatched image: −4.2 points; 3 helpful and 8 harmful flips;
  answer-change rate 12.5%; exact McNemar p=0.227.
- Image-only accuracy was not significantly above five-choice chance of 20%
  (one-sided exact binomial p=0.072).

## Direct-diagnosis subset

| Condition | Correct / n | Accuracy | Wilson 95% CI |
| --- | ---: | ---: | ---: |
| Proxy context only | 63 / 90 | 70.0% | 59.9%–78.5% |
| Context + matched image | 60 / 90 | 66.7% | 56.4%–75.5% |
| Context + mismatched image | 61 / 90 | 67.8% | 57.6%–76.5% |
| Image only | 21 / 90 | 23.3% | 15.8%–33.1% |

Matched images were −3.3 points versus no image and −1.1 points versus mismatched
images. Image-only performance was consistent with chance (p=0.25).

## Subtype screening

| Subtype | n | Context | Matched | Wrong | Image only | Matched gain | Matched over wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| physical | 51 | 68.6% | 62.7% | 66.7% | 25.5% | −5.9 pp | −3.9 pp |
| CT scan | 19 | 42.1% | 42.1% | 52.6% | 31.6% | 0.0 pp | −10.5 pp |
| dermatology | 16 | 93.8% | 81.2% | 81.2% | 18.8% | −12.5 pp | 0.0 pp |
| hist/path | 13 | 76.9% | 69.2% | 69.2% | 30.8% | −7.7 pp | 0.0 pp |
| xray | 12 | 58.3% | 66.7% | 66.7% | 41.7% | +8.3 pp | 0.0 pp |
| ophthalmology | 11 | 81.8% | 81.8% | 90.9% | 27.3% | 0.0 pp | −9.1 pp |

X-ray was the only adequately sized group with a positive matched-image gain and the
highest image-only accuracy, so it is the best first specialist-model PoC. However,
matched and mismatched images had identical X-ray accuracy, so this run did not
demonstrate image specificity. Other subtype differences are negative or null and
have wide uncertainty.

## Decision

This evaluation rejects the claim that the current MedGemma 27B + NEJM screenshot
pipeline has already demonstrated sufficient image-reading ability. Correct images
did not improve accuracy, did not outperform closely matched wrong images, and
image-only performance was statistically compatible with chance.

The result supports a specialist-model proof of concept, not immediate production
integration. vLLM cannot replace that comparison because vLLM is an inference engine,
not a visual model. The correct architecture experiment is:

1. MedGemma alone.
2. Specialist alone, blinded to answer choices.
3. Specialist structured findings passed to the same MedGemma reasoner.
4. Clinician-verified oracle visual findings passed to MedGemma, to measure the
   maximum value the images could add.

Adopt a specialist only if the hybrid shows a predeclared paired improvement on a
clinician-redacted, image-essential private holdout. A reasonable research gate is an
overall gain of at least 5 percentage points with a paired 95% interval above zero,
or at least 10 points in a priority modality replicated on an independent cohort,
with no increase in harmful flips or critical visual misses. These are study-design
gates, not clinical acceptance thresholds.

## Candidate sequence

- Broad updated baseline: MedGemma 1.5 4B, then MedSigLIP for classification/retrieval.
- Chest X-ray: MAIRA-2 and CheXagent.
- CT/MRI: MedGemma 1.5 4B on native volumes; RadFM as a research comparator.
- Whole-slide pathology: SlideChat or TITAN only after obtaining native WSI data.

The local host currently has no vLLM installation or specialist VLM weights. Its
16 GB GPU already requires CPU offload for the 17 GB Q4 MedGemma 27B model, so vLLM
deployment of 27B vision weights requires a separate memory and parity study.

Official model and runtime references:

- [MedGemma model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
  and [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedSigLIP 448](https://huggingface.co/google/medsiglip-448)
- [MAIRA-2](https://huggingface.co/microsoft/maira-2) and
  [CheXagent](https://github.com/Stanford-AIMI/CheXagent)
- [RadFM](https://github.com/chaoyi-wu/RadFM),
  [SlideChat](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_SlideChat_A_Large_Vision-Language_Assistant_for_Whole-Slide_Pathology_Image_Understanding_CVPR_2025_paper.pdf),
  and [TITAN](https://huggingface.co/MahmoodLab/TITAN)
- [vLLM supported-model documentation](https://docs.vllm.ai/en/latest/models/supported_models/)

## Limitations

- Proxy context was not manually redacted by clinicians.
- Images are public, possibly present in model pretraining data.
- Many are multi-panel publication composites; subtype tags are overlapping and not
  primary-modality annotations.
- The experiment measures five-choice case reasoning, not visual finding sensitivity,
  localization, laterality, report quality, calibration, or clinical safety.
- Most subtype sample sizes are insufficient for model procurement or deployment
  decisions.
