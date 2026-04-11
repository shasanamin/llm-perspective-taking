# From Fallback to Frontline: LLMs as Annotators of Human Perspectives

Code and data for the paper:

> **From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?**
> Hasan Amin, Harry Yizhou Tian, Xiaoni Duan, Chien-Ju Ho, Rajiv Khanna, Ming Yin

We formalize **perspective-taking**—estimating how a demographic subgroup would collectively judge subjective content—as a statistical estimation problem. Through a bias–variance–correlation decomposition and experiments across various LLM configurations, datasets, and subgroups, we show that a single LLM estimate often rivals multiple direct human labels and identify clear regimes where LLMs excel or fall short.

## Repository Structure

```
├── src/                        # Core library
│   ├── datasets.py             #   Dataset loaders (toxicity + DICES)
│   ├── prompts.py              #   Prompt templates
│   ├── generation.py           #   LLM annotation generation engine
│   ├── config.py               #   YAML config loader with env-var expansion
│   ├── paths.py                #   Project path constants
│   ├── providers/              #   LLM backend adapters
│   │   ├── openai_compatible.py      (GenAI Studio / Ollama)
│   │   ├── openai_responses.py       (OpenAI Responses API)
│   │   └── huggingface_local.py      (local HF models)
│   ├── analysis/               #   Statistical analysis
│   │   ├── bootstrap.py              Bootstrap engine (MSE / Bias / Variance)
│   │   ├── toxicity.py               Toxicity-specific helpers
│   │   ├── dices.py                  DICES-specific helpers
│   │   └── differential.py           Differential perspective-taking
│   ├── plotting/               #   Figure generation
│   │   ├── style.py                  Shared theme, colors, model metadata
│   │   ├── toxicity.py               Toxicity figure functions
│   │   └── dices.py                  DICES figure functions
│   └── utils/                  #   Utilities (JSONL I/O, text parsing, model names)
├── scripts/                    # CLI entry points
│   ├── run_toxicity.py         #   Generate toxicity LLM annotations
│   ├── run_dices.py            #   Generate DICES LLM annotations
│   ├── analyze_*.py            #   Compute bootstrap / DPT / reasoning analysis
│   ├── plot_*.py               #   Produce paper figures
│   └── reproduce_paper_figures.py  # One command to reproduce all figures
├── configs/
│   ├── generation/             #   Example experiment configs
│   └── runtime/                #   Provider configs (API keys via env vars)
├── data/
│   ├── toxicity_detection/     #   comments + human annotations
│   ├── dices/                  #   DICES-350 raw annotations
│   └── llm_annotations/        #   ~200 pre-generated LLM output files
├── results/                    #   Regenerated figures and processed CSVs (gitignored)
└── requirements.txt
```

## Quick Start

### 1. Set up environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce all paper figures

All pre-generated LLM annotations are bundled in `data/llm_annotations/`.
No API access or GPUs are needed.

```bash
python scripts/reproduce_paper_figures.py
```

This runs the full analysis-and-plotting pipeline and writes PDF figures to `results/figures/`. Use `--force` to recompute cached bootstrap CSVs.

### 3. Run individual scripts

```bash
# Main paper figures (toxicity)
python scripts/plot_toxicity_paper.py

# Ablation figures (prompt modes, temperature, model mixing)
python scripts/plot_toxicity_ablations.py

# DICES specificity and prevalence figures
python scripts/plot_dices_paper.py

# Differential perspective-taking analysis + figures
python scripts/analyze_differential_pt.py
python scripts/plot_differential_pt.py

# Pretrained vs post-trained comparison (requires plot_hf_females.py first)
python scripts/plot_hf_females.py
python scripts/plot_pt_vs_it_paper.py

# Reasoning paradox analysis
python scripts/analyze_reasoning_paradox.py
```

## Generating New LLM Annotations

To collect new annotations you need access to an LLM provider. Three backends are supported:

**OpenAI-compatible endpoint** (GenAI Studio, Ollama, etc.):
```bash
cp configs/runtime/genai_studio.example.yaml configs/runtime/local_genai_studio.yaml
# Edit local_genai_studio.yaml with your API URL and key, then:
python scripts/run_toxicity.py \
  --config configs/generation/toxicity_api.yaml \
  --runtime-config configs/runtime/local_genai_studio.yaml \
  --model qwen3:8b
```

**OpenAI Responses API**:
```bash
export OPENAI_API_KEY=<your-key>
python scripts/run_toxicity.py \
  --config configs/generation/toxicity_api.yaml \
  --runtime-config configs/runtime/openai.example.yaml \
  --model gpt-5.4-nano --reasoning-effort none
```

**Local HuggingFace models** (requires GPU, `transformers`, `accelerate`, `torch`):
```bash
cp configs/runtime/huggingface_cluster.example.yaml configs/runtime/local_hf.yaml
# Edit local_hf.yaml with your cache directory and token, then:
export HF_HOME=/path/to/cache/huggingface
export HF_TOKEN=<your-token>   # for gated models (Llama 3.1, Gemma 3)
python scripts/run_toxicity.py \
  --config configs/generation/toxicity_hf.yaml \
  --runtime-config configs/runtime/local_hf.yaml \
  --model qwen3.5:0.8b
```

## Data

### Datasets

- **Toxicity Detection** — 120 online comments with annotations from 50+ U.S.-based crowdworkers per gender subgroup (female, male, non-binary). Based on [Duan et al. (2025)](https://aclanthology.org/2025.naacl-long.119.pdf), extended with new non-binary annotations collected on Prolific.

- **DICES-350** — 350 conversational safety examples from [Aroyo et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/a74b697bce4cac6c91896372abaa8863-Paper-Datasets_and_Benchmarks.pdf) with 100+ raters per item across gender, race, age, and education demographics.

### LLM Annotation Format

Each JSONL record includes:

| Field | Description |
|-------|-------------|
| `model_name` | Canonical model identifier |
| `target_group` | Target demographic group |
| `comment_id` | Source item identifier |
| `generation_index` | Index within the repeated-generation run |
| `parsed_percentage` | Extracted numeric estimate (0–100) |
| `reasoning_trace` | Model reasoning chain (when available) |
| `experiment_group` | Groups related runs for analysis |
| `temperature`, `top_p` | Decoding parameters used |

See [`data/README.md`](data/README.md) for the full data layout.

## Citation

```bibtex
@inproceedings{amin2025fallback,
  title={From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?},
  author={Amin, Hasan and Tian, Harry Yizhou and Duan, Xiaoni and Ho, Chien-Ju and Khanna, Rajiv and Yin, Ming},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}
```
