# STCIR — Semi-Synthetic Test Collection for IR

**STCIR** is an end-to-end, language-adaptive framework for building semi-synthetic IR test collections.
It supports both standard benchmark evaluation (Mr. TyDi, mMARCO, MS MARCO) and custom domain corpora
(e.g., Algerian legal text), with full checkpointing, pre-built run reuse, and LLM-assisted annotation.

---

## Key Features

| Feature | Details |
|---|---|
| **Languages** | Arabic · English (registry-driven, extensible) |
| **Retrieval Stage 1** | BM25 (rank-bm25) + 6 bi-encoders (SentenceTransformer + FAISS) → RRF fusion → top-1000 pool |
| **Retrieval Stage 2** | 5–6 cross-encoders → RRF fusion → top-10 candidates |
| **Pre-built runs** | Download Stage-1 and Stage-2 runs from HuggingFace — skip GPU encoding entirely |
| **Annotation** | Human (Flask UI) or automatic (Gemma 3 4B LLM) |
| **Evaluation** | MRR@k · nDCG@k · MAP · Recall@k · Hit@k · P@k · Kendall's τ · Spearman's ρ |
| **Checkpointing** | Every phase writes a marker file; interrupted runs resume from the last completed phase |
| **Batch mode** | Run all three benchmarks sequentially with one command (`batch_run.ipynb`) |
| **Domain mode** | Chunk a raw corpus → sample passages → create topics → full pipeline |

---

## Repository Structure

```
STCIR/
├── main.ipynb              # Interactive single-dataset pipeline (10 cells)
├── batch_run.ipynb         # Unattended batch runner (all datasets)
├── configs/
│   ├── mrtydi_arabic.yaml
│   ├── mmarco_arabic.yaml
│   ├── algerian_legal.yaml
│   ├── mrtydi_english.yaml
│   └── msmarco_english.yaml
├── stcir/
│   ├── config.py           # STCIRConfig (dataclass, YAML-serialisable)
│   ├── registry.py         # Model registry, dataset maps, prebuilt folder map
│   ├── pipeline/
│   │   └── runner.py       # PipelineRunner — programmatic phase executor
│   ├── corpus/             # Corpus loader + token-aware sliding-window chunker
│   ├── topics/             # Topic loader (HuggingFace primary / ir_datasets fallback)
│   ├── indexing/           # BM25Index · BiEncoder · FaissIndex (Flat/IVFFlat)
│   ├── retrieval/          # BM25Retriever · DenseRetriever · RRF · Stage1Pool
│   ├── reranking/          # CrossEncoderReranker · RRF Stage-2
│   ├── annotation/         # GemmaAnnotator · GemmaTopicGenerator
│   ├── evaluation/         # IR metrics · system-level correlation · scatter plots
│   └── ui/                 # Flask relevance UI · Flask topic-creation UI
└── archive_v1/             # Original Kaggle notebooks (preserved)
```

---

## Pipeline Overview

```
Corpus ──► [BM25 + 6 Bi-encoders] ──► RRF Stage-1 ──► Pool (top-1000)
                                                              │
                                              [5–6 Cross-encoders]
                                                              │
                                              RRF Stage-2 ──► Top-10
                                                              │
                                         [Human UI / Gemma4 LLM Annotation]
                                                              │
                                              IR Metrics + Correlation
```

When **pre-built runs** are available, Stage-1 and/or Stage-2 are downloaded directly from
HuggingFace — no bi-encoder encoding or cross-encoder reranking needed.

---

## Pre-built Runs

Pre-built runs are hosted at **[hatemestinbejaia/STCIR_Synthetic-Test-Collection-IR](https://huggingface.co/datasets/hatemestinbejaia/STCIR_Synthetic-Test-Collection-IR)**.

| Dataset | Stage | HF Folder | Models included |
|---|---|---|---|
| Mr. TyDi Arabic | Stage-1 | `FirstStage_mrTydi/` | AraElectra KD/NoKD · AraDPR KD/NoKD · mMiniLM KD/NoKD |
| Mr. TyDi Arabic | Stage-2 | `MrTydi_second_stage/` | AraElectra · AraBERT · mMiniLM · CAMeL · mBERT cross-encoders |
| mMARCO Arabic | Stage-1 | `FirstStage_mmarco/` | AraElectra KD/NoKD · AraDPR KD/NoKD · mMiniLM KD/NoKD |

Set `stage1_source: prebuilt_mrtydi` (or `prebuilt_mmarco`) and `stage2_source: prebuilt_mrtydi`
in your config to use them. The batch runner detects availability automatically.

---

## Supported Datasets

| Key | Language | Corpus loader | Topics / Qrels |
|---|---|---|---|
| `mrtydi_arabic` | Arabic | HF `hatemestinbejaia/mr-tydi-ar-dev` → ir_datasets fallback | Mr. TyDi test split |
| `mmarco_arabic` | Arabic | HF `hatemestinbejaia/mmarco-arabic-dev` → ir_datasets fallback | mMARCO dev/small |
| `algerian_legal` | Arabic | Local JSONL corpus | Domain topics (human / LLM) |
| `mrtydi_english` | English | ir_datasets `mr-tydi/en` | Mr. TyDi test split |
| `msmarco` | English | HF `hatemestinbejaia/mmarco-english-dev` → ir_datasets fallback | MS MARCO dev/small |

---

## LLM Annotation with Gemma 3 4B

When `annotation_mode: llm`, the framework uses **Google Gemma 3 4B Instruct**
(`google/gemma-3-4b-it`) to judge the relevance of each (query, passage) pair in the top-10 list.

- **Arabic prompt**: instructs the model to respond `نعم` (relevant) or `لا` (not relevant)
- **English prompt**: instructs the model to respond `Yes` or `No`
- Batched inference via HuggingFace `transformers` (configurable `llm_batch_size`)
- LLM annotation is the only supported mode in `batch_run.ipynb` (human UI cannot run unattended)

To switch to human annotation, set `annotation_mode: human` in your config and run `main.ipynb` —
a Flask UI will launch for relevance judging.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/hatemamine/STCIR-Semi-Synthetic-Test-Collection-for-Algerian-Legal-IR.git
cd STCIR-Semi-Synthetic-Test-Collection-for-Algerian-Legal-IR
pip install -e .
```

### 2. Single dataset (interactive)

Open `main.ipynb`, pick a language/dataset in Cell 0, and run all cells.
Pre-built runs are downloaded automatically when configured.

### 3. Batch run (all benchmarks, unattended)

```bash
jupyter nbconvert --to notebook --execute batch_run.ipynb --output batch_run_output.ipynb
```

Or open `batch_run.ipynb` and run all cells. Datasets run sequentially;
each resumes from its last checkpoint if interrupted.

### 4. Load from YAML config

```python
from stcir import STCIRConfig
from stcir.pipeline.runner import PipelineRunner

config = STCIRConfig.from_yaml("configs/mrtydi_arabic.yaml")
runner = PipelineRunner(config, auto_use_prebuilt=True)
result = runner.run_all()   # returns dict with qrels, results DataFrame, ...
```

---

## Models

### Arabic bi-encoders (Stage-1)

| Model | Training | HF repo |
|---|---|---|
| AraElectra bi-encoder | KD | `hatemestinbejaia/mmarco-Arabic-AraElectra-bi-encoder-KD-v1` |
| AraElectra bi-encoder | No KD | `hatemestinbejaia/mmarco-Arabic-AraElectra-bi-encoder-NoKD-v1` |
| AraDPR bi-encoder | KD | `hatemestinbejaia/mmarco-Arabic-AraDPR-bi-encoder-KD-v1` |
| AraDPR bi-encoder | No KD | `hatemestinbejaia/mmarco-Arabic-AraDPR-bi-encoder-NoKD-v1` |
| mMiniLM bi-encoder | KD | `hatemestinbejaia/mmarco-Arabic-mMiniLML-bi-encoder-KD-v1` |
| mMiniLM bi-encoder | No KD | `hatemestinbejaia/mmarco-Arabic-mMiniLML-bi-encoder-NoKD-v1` |

### Arabic cross-encoders (Stage-2)

| Model | HF repo |
|---|---|
| AraElectra cross-encoder | `hatemestinbejaia/mmarco-Arabic-AraElectra-cross-encoder-v1` |
| AraBERT cross-encoder | `hatemestinbejaia/mmarco-Arabic-AraBERT-cross-encoder-v1` |
| mMiniLM cross-encoder | `hatemestinbejaia/mmarco-Arabic-mMiniLML-cross-encoder-v1` |
| CAMeL cross-encoder | `hatemestinbejaia/mmarco-Arabic-CAMeL-cross-encoder-v1` |
| mBERT cross-encoder | `hatemestinbejaia/mmarco-Arabic-mBERT-cross-encoder-v1` |

---

## Evaluation Metrics

- **Ranking metrics**: MRR@10, nDCG@10, MAP, Recall@10, Hit@10, P@10
- **System-level correlation** (synthetic vs. human qrels):
  - Global Kendall's τ
  - Global Spearman's ρ
  - Per-metric scatter plots

---

## Citation

If you use STCIR in your research, please cite:

```bibtex
@misc{stcir2024,
  title  = {STCIR: Semi-Synthetic Test Collection for Algerian Legal Information Retrieval},
  author = {Hatem Amine},
  year   = {2024},
  url    = {https://github.com/hatemamine/STCIR-Semi-Synthetic-Test-Collection-for-Algerian-Legal-IR}
}
```
