# STCALIR-Semi-Synthetic-Test-Collection-for-Algerian-Legal-Information-Retrieval
STCALIR: Semi-Synthetic Test Collection for Algerian Legal Information Retrieval
# STCALIR: Semi-Synthetic Test Collection for Algerian Legal Information Retrieval

## Overview
This repository provides the resources and code supporting the STCALIR framework, a **semi-synthetic test collection pipeline** designed for Algerian Arabic legal information retrieval. STCALIR combines human topic generation, multi-system bi-encoder retrieval, RRF pooling, and cross-encoder reranking to produce high-quality semi-synthetic relevance judgments while minimizing manual annotation effort.

The framework is particularly useful for **low-resource legal domains**, where fully human-annotated corpora are scarce.

---

## Data
The repository includes:

- **Semi-synthetic dataset**: a collection of Algerian legal passages with generated relevance labels.  
  Available at [Hugging Face](https://huggingface.co/hatemestinbejaia).


---

## Code
The STCALIR pipeline code is available at [GitHub](https://github.com/hatemamine/STCALIR-Semi-Synthetic-Test-Collection-for-Algerian-Legal-IR) and includes:

- Multi-system retrieval scripts  
- RRF pooling implementation  
- Cross-encoder reranking  
- Evaluation scripts for computing Hit@10, Kendall’s τ, and Spearman’s ρ  

---

## Installation
Clone the repository:

```bash
git clone https://github.com/hatemamine/STCALIR-Semi-Synthetic-Test-Collection-for-Algerian-Legal-IR.git
cd STCALIR-Semi-Synthetic-Test-Collection-for-Algerian-Legal-IR
