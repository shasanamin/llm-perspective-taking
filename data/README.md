# Data Layout

## Toxicity Detection

- `toxicity_detection/original/comments.csv` -- 120 online comments with
  comment IDs and text.
- `toxicity_detection/human_annotations/` -- Human annotation data:
  - `direct_annotations.csv` -- Per-item aggregated subgroup ground truth
    (fraction who rated toxic, by gender).
  - `pilot_annotations.csv` -- Individual direct annotations from
    crowdworkers, used for direct-baseline bootstrapping.
  - `formal_annotations.csv` -- Human perspective-taking (PT) study file.
  - `in_group/` -- In-group PT annotations by gender (female, male,
    non-binary).
  - `out_group/` -- Out-group PT annotations (female, male).

## DICES-350

- `dices/original/dices350_raw.csv` -- Raw per-rater annotations (~140K rows)
  from Aroyo et al. (2024).
- `dices/original/dices_comments.csv` -- Conversation text items (1,486 items).

## LLM Annotations

Pre-generated LLM perspective-taking outputs (JSONL). Each record includes
the model name, target group, item ID, parsed percentage estimate, decoding
settings, and optional reasoning trace.

- `llm_annotations/toxicity_detection/`
  - `main/` -- Primary experiments: 3 genders x ~20 models.
  - `prompt/` -- Prompt-mode ablation (question / definition / levels / examples).
  - `temperature/` -- Temperature ablation (T=0.3 / 0.6 / 0.9 / 1.2 / 1.5).
  - `reasoning_followup/` -- Qwen3 reasoning-mode experiments.
  - `generated/` -- Additional runs (HuggingFace models, GPT-5.4 reasoning tiers, non-binary panel).
- `llm_annotations/dices/`
  - `main/` -- DICES-350 perspective-taking outputs: 16 models x 350 items x 10 generations.
