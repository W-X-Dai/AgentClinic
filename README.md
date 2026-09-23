# AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments

<p align="center">
  <img src="media/mainfigure.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

## Release
- [09/13/2024] 🍓 We release new results and support for o1!
- [08/17/2024] 🎆 Major updates 🎇
  - 🏥 A new suite of cases (**AgentClinic-MIMIC-IV**), based on real clinical cases from MIMIC-IV (requires approval from https://physionet.org/content/mimiciv/2.2/)! 
  - More AgentClinic-MedQA cases [107] → [215] 
  - More AgentClinic-NEJM cases [15] → [120] 
  - 💼 Tutorials on building your own AgentClinic cases!
  - Support for three new models--☀️ Anthropic's Claude 3.5 Sonnet, 📗 OpenAI's GPT 4o-mini, and 🦙 Llama 3 70B

- [06/28/2024] 🩻 We added support for vision models and the NEJM case questions
- [05/18/2024] 🤗 We added support for HuggingFace models!
- [05/17/2024] We release new results and support for GPT-4o!
- [05/13/2024] 🔥 We release **AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environment**. We propose a multimodal benchmark based on language agents which simulate the clinical environment.  Checkout the [paper](media/AgentClinicPaper.pdf) and the [website](https://agentclinic.github.io/) for this code.

## Local research extension version

### v0.4.1 (2026-09-17)

- Added an A100 80GB-specific vLLM manifest. It attempts all 11 supplied checkpoints,
  omits the RTX Pro-only Nemotron Triton workaround, and applies conservative
  single-sequence/eager settings to memory-tight models. The Blackwell-oriented
  Nemotron NVFP4 entry is retained as an explicit hardware-compatibility probe; a
  startup failure is recorded and does not abort the remaining matrix.

### v0.4.0 (2026-09-17)

- Added a portable vLLM matrix harness for the RTX Pro 6000 host. It can inspect
  local checkpoint metadata, launch one model at a time on loopback, run a smoke
  test, execute the benchmark, persist failures, stop only the server process it
  started, and continue to the next model.
- Added OpenAI-compatible text/image requests using base64 data URLs and strict JSON
  schema, plus matrix summaries that keep text-only models out of image denominators.
- Matrix provenance includes GPU/driver, Python/platform, vLLM CLI/API versions,
  inspected checkpoint metadata, and each exact launch command.
- Added a reviewed example manifest for the 11 checkpoints under `/mnt/models`,
  including special RTX Pro flags for Nemotron Omni and automatic local-config
  inspection for the ambiguous `gemma-4-31b` directory.
- This version adds reproducible tooling only; no result is claimed until it is run
  on the target host. See the
  [vLLM instructions](medgemma_eval/README.md#vllm-model-matrix-rtx-pro-6000--a100).

### v0.3.0 (2026-09-17)

- Added a Gemini Developer API adapter that reuses the backend's configured
  `GEMINI_API_KEY` and `GEMINI_MODEL` without copying or recording the credential.
- Completed the same 120-case, four-arm visual ablation with
  `gemini-3.6-flash`: 480/480 responses were scorable. Context plus the matched
  image reached 96.7%, compared with 62.5% for local `medgemma:27b`.
- Added strict case-level model comparison. Gemini improved matched-image accuracy
  by 34.2 points over MedGemma (41 helpful, 0 harmful paired flips; p<0.001), and
  also showed positive within-model image grounding versus no image (+8.3 points,
  p=0.013) and a mismatched image (+6.7 points, p=0.039).
- The result remains a public-case research screen. Gemini's 90.0% image-plus-task
  score creates a substantial pretraining-contamination or case-memorization risk;
  private image-essential evaluation is required before architecture or clinical
  conclusions. See the
  [paired comparison report](medgemma_eval/reference_results/2026-09-17-medgemma-vs-gemini.md).

### v0.2.0 (2026-09-16)

- Added an image-focused four-arm ablation: proxy context only, matched image,
  subtype-matched wrong image, and image only. This avoids treating the original
  question's embedded formal report as visual evidence.
- Completed all 480 responses for 120 cases. Matched images scored 62.5%, compared
  with 65.0% without an image and 66.7% with a mismatched image; image-only accuracy
  was 25.8% and was not significantly above 20% five-choice chance (`p=0.072`).
- Added paired grounding diagnostics, wrong-image controls, subtype reports, and a
  documented decision protocol for comparing specialist image models with a fixed
  MedGemma reasoner. See the
  [visual-ablation report](medgemma_eval/reference_results/2026-09-16-medgemma-27b-visual-ablation.md).
- The proxy context is not clinician-redacted, and publication images are not native
  DICOM volumes or whole-slide images. Results support a specialist-model proof of
  concept, not clinical deployment.

### v0.1.0 (2026-09-16)

- Added an isolated [`medgemma_eval/`](medgemma_eval/README.md) harness for paired
  text-only and text-plus-image evaluation of the local Ollama `medgemma:27b` model
  on all 120 extended NEJM cases.
- Added deterministic A–E scoring, per-subtype and per-task aggregation, Wilson 95%
  intervals, image helpful/harmful analysis, resumable JSONL records, image checksum
  caching, and an all-subtype smoke cohort.
- Recorded a complete 120-case reference run: 72.5% text-only and 70.8% multimodal
  accuracy overall; see the
  [reproducible result and limitations](medgemma_eval/reference_results/2026-09-16-medgemma-27b.md).
- This is a research benchmark only. It does not establish clinical safety or
  authorize diagnostic use; small subtypes remain descriptive and require cautious
  interpretation.


## Contents
- [Install](#install)
- [Evaluation](#evaluation)
- [Code Examples](#code-examples)



## Install

1. This library has few dependencies, so you can simply install the requirements.txt!
```bash
pip install -r requirements.txt
```

## Evaluation

All of the models from the paper are available (GPT-4/4o/3.5, Mixtral-8x7B, Llama-70B-chat). You can try them for any of the agents, make sure you have either an OpenAI or Replicate key ready for evaluation! HuggingFace wrappers are also implemented if you don't want to use API keys.

Just change modify the following parameters in the CLI

```
parser.add_argument('--openai_api_key', type=str, required=True, help='OpenAI API Key')
parser.add_argument('--replicate_api_key', type=str, required=False, help='Replicate API Key')
parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
parser.add_argument('--doctor_bias', type=str, help='Doctor bias type', default='None', choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None', choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
parser.add_argument('--doctor_llm', type=str, default='gpt4', choices=['gpt4', 'gpt3.5', 'llama-2-70b-chat', 'mixtral-8x7b', 'gpt4o'])
parser.add_argument('--patient_llm', type=str, default='gpt4', choices=['gpt4', 'gpt3.5', 'mixtral-8x7b', 'gpt4o'])
parser.add_argument('--measurement_llm', type=str, default='gpt4', choices=['gpt4'])
parser.add_argument('--moderator_llm', type=str, default='gpt4', choices=['gpt4'])
parser.add_argument('--num_scenarios', type=int, default=1, required=False, help='Number of scenarios to simulate')
parser.add_argument('--agent_dataset', type=str, default='MedQA')
parser.add_argument('--doctor_image_request', type=bool, default=False)
parser.add_argument('--total_inferences', type=int, default=20, required=False, help='Number of inferences between patient and doctor')
```


## Code Examples

🎆 And then run it!

```
python3 agentclinic.py --openai_api_key "YOUR_OPENAIAPI_KEY" --inf_type "llm"
```

🤗 You can also try ANY custom HuggingFace language model very simply! All you have to do is pass "HF_{hf_path}" where hf_path is the HuggingFace path (e.g. mistralai/Mixtral-8x7B-v0.1). Here is how you can run Mixtral-8x7B locally with AgentClinic for both doctor and patient agents. 🤗


🔥 Here is an example with gpt-4o!

```
python3 agentclinic.py --openai_api_key "YOUR_OPENAIAPI_KEY" --doctor_llm gpt4o --patient_llm gpt4o --inf_type llm
```

⚖️ Here is an example with doctor and patient bias with gpt-3.5!

```
python3 agentclinic.py --openai_api_key "YOUR_OPENAIAPI_KEY" --doctor_llm gpt3.5 --patient_llm gpt4 --patient_bias self_diagnosis --doctor_bias recency --inf_type llm
```


🩻 Here is an example with gpt-4o on the NEJM reports

```
python3 agentclinic.py --openai_api_key "YOUR_OPENAIAPI_KEY" --doctor_llm gpt4o --patient_llm gpt4o --inf_type llm --agent_dataset NEJM --doctor_image_request True
```

⚠️ Can be quite slow ⚠️

```
python3 agentclinic.py --inf_type "llm" --inf_type "llm" --patient_llm "HF_mistralai/Mixtral-8x7B-v0.1"  --moderator_llm "HF_mistralai/Mixtral-8x7B-v0.1"  --doctor_llm "HF_mistralai/Mixtral-8x7B-v0.1"  --measurement_llm "HF_mistralai/Mixtral-8x7B-v0.1"
```

- The extended MedQA and NEJM datasets are available through setting the keyword `agent_dataset=NEJM_Ext` and `agent_dataset=MedQA_Ext`

BIBTEX Citation
```
@misc{schmidgall2024agentclinic,
      title={AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments}, 
      author={Samuel Schmidgall and Rojin Ziaei and Carl Harris and Eduardo Reis and Jeffrey Jopling and Michael Moor},
      year={2024},
      eprint={2405.07960},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}
```
