```
# ATOC-LLM: Automatic Task Ontology Construction with Large Language Models
## Running Instructions
---
### 1. Project Overview
This repository implements the full technical framework proposed in the paper *ATOC-LLM: An LLM-Driven Automatic Task Ontology Construction Method*, covering six functional modules:
1.  Automated domain text collection and corpus preprocessing (three-agent pipeline: retrieval agent / filtering agent / preprocessing agent)
2.  Hierarchical task ontology feature extraction: bottom-level concrete features (actor, time, environment, goal) and top-level abstract features (activity, information requirement)
3.  Core task relation modeling: four relation types (decomposition, division of labor, collaboration, information exchange)
4.  Hierarchical ontology construction via Dynamic Weight Hierarchical Clustering (DWHC, Algorithm 1)
5.  OWL ontology generation (OWL 2 DL profile, with the property set: `hasSubTask` / `isSubTaskOf` / `performedBy` / `collaboratesWith` / `requiresInformation`, plus disjointness axioms between sibling atomic tasks and a minimum cardinality 1 axiom on the `performedBy` property)
6.  Dynamic ontology update mechanism: triggered when the maximum cluster similarity falls below the threshold δ=0.55.

**Fixed embedding backbone**: BGE-M3 (1024-dimensional dense vectors, locally self-hosted), held constant across all backbones to eliminate embedding-space confounding in cross-backbone comparison.
**Generation backbones**: DeepSeek-V3 (deepseek-chat, snapshot 2025-07-15) / Doubao-pro-32k (snapshot 2025-07-20) / ERNIE-4.0-Turbo (snapshot 2025-08-01). The `<backbone>` placeholder in all prompt templates is replaced at runtime with the selected model identifier.
**Independent blinded evaluation judges**: GPT-4o (gpt-4o-2024-08-06) / Qwen-2.5-72B-Instruct / Llama-3.1-70B-Instruct, all run at temperature 0.0 for scoring consistency.

**Dataset note**: Domain corpora are automatically collected by the multi-agent module and are not redistributed with the code. The `data/` directory only contains two minimal sample documents (including the Myanmar earthquake rescue running example) for end-to-end pipeline validation.

---
### 2. Environment Requirements
Python 3.9 or higher. Install dependencies via:
```bash
pip install -r requirements.txt
```

For Pellet reasoner-based consistency checking, Java 8+ is required. The `owlready2` library will automatically download the Pellet JAR on first run; if Java is unavailable, consistency checks are gracefully skipped and all other pipeline steps run normally.

---

### 3. API Key Configuration

Configure model credentials as environment variables (set only the keys for backbones you intend to run):

```
# Windows
set DEEPSEEK_API_KEY=sk-xxxx
set ARK_API_KEY=xxxx          # Doubao / Volcano Engine Ark
set QIANFAN_API_KEY=xxxx      # ERNIE / Baidu Qianfan
set OPENAI_API_KEY=sk-xxxx    # GPT-4o / GPT-3.5
set DASHSCOPE_API_KEY=sk-xxxx # Qwen
set LLAMA_API_KEY=xxxx        # Llama-3.1-70B (any OpenAI-compatible endpoint; custom base URL can be set in config/settings.py)

# Linux/macOS
export DEEPSEEK_API_KEY=sk-xxxx
export ARK_API_KEY=xxxx
export QIANFAN_API_KEY=xxxx
export OPENAI_API_KEY=sk-xxxx
export DASHSCOPE_API_KEY=sk-xxxx
export LLAMA_API_KEY=xxxx
```

---

### 4. Quick Start

1. Run the full pipeline on the sample corpus (DeepSeek backbone):```
python main.py build --corpus data/sample_documents.json
```
2. Switch between other backbones:```
python main.py build --corpus data/sample_documents.json --backbone doubao
python main.py build --corpus data/sample_documents.json --backbone ernie
```
3. Outputs are written to `output/<backbone>/`:
   - `task_ontology.ttl`: Generated OWL-Turtle ontology file
   - `build_report.json`: Structured build statistics (corpus size, extracted elements, hierarchy structure, relations, runtime, token consumption)
   - `construction_report.txt`: Human-readable build summary.

---

### 5. Input Data Format

The input corpus is a JSON file, where each document entry contains the following fields:

| Field | Description |
| --- | --- |
| `doc_id` | Unique document identifier |
| `title` | Document title |
| `url` | Source URL (used for authority scoring, per Appendix A of the paper) |
| `source_type` | Source category: `academic` / `official` / `news` |
| `content` | Raw document text |
| `elements` / `relations` | Optional fields for self-built gold standard evaluation. |

---

### 6. Available Commands

#### Full Ontology Build

```
python main.py build --corpus <corpus.json> [--backbone deepseek]
```

Ablation switches (matching the ablation experiments in Section 5.2.6):

- `--no-sfe`: ATOC-w/o-SFE — flat single-pass extraction replaces hierarchical feature extraction
- `--no-dhwc`: ATOC-w/o-DWHC — equal-weight Ward agglomerative clustering replaces DWHC
- `--no-context`: ATOC-w/o-CTX — α_ctx=0, cosine similarity only
- `--no-filtering`: ATOC-w/o-FLT — disable the corpus filtering score (Eq.1)
- `--no-agents`: ATOC-w/o-AGT — single-pass keyword crawler replaces the three-agent collection pipeline
- `--no-update`: ATOC-w/o-UPD — disable the dynamic update module.

#### Incremental Update Experiment (Section 5.2.6, Table 12)

```
python experiments/run_incremental.py --corpus <base_corpus.json> \
    --stream <stream_documents.json> [--backbone deepseek]
```

Compares three strategies (incremental update / full reconstruction / no update), and outputs sensitivity analysis across δ thresholds (0.45 / 0.50 / 0.55 / 0.65).

#### Ablation Experiment (Table 13)

```
python experiments/run_ablation.py --corpus <corpus.json> \
    [--gold-standard <gold.json>] [--quality]
```

#### Baseline Comparison (Tables 7 / 8 / 9)

```
python experiments/run_baselines.py --corpus <corpus.json> \
    --gold-standard <gold.json>
```

Includes supervised baselines (BiLSTM-CRF, BERT-CRF; BIO labeling, AdamW optimizer, early stopping) and prompt-based LLM baselines (GPT-3.5 zero-shot / GPT-3.5 Few-shot+CoT / GPT-4o / Qwen / Llama fair-prompt).

#### Cross-Domain Experiment (Section 5.3)

```
python experiments/run_cross_domain.py --domain intelligent_healthcare \
    --corpus <healthcare_corpus.json> [--gold-standard <gold.json>]
```

Supported domains: `emergency_rescue` / `intelligent_healthcare` / `intelligent_manufacturing`.

#### Quality Evaluation (Section 5.2.2, three-judge blinded scoring, Table 5 weights, ICC(C,1))

```
python main.py evaluate --backbone deepseek \
    [--gold-standard <gold.json>] [--ontology output/deepseek/task_ontology.ttl]
```

---

### 7. Repository Structure

```
ATOC-LLM/
├── main.py                 CLI entrypoint (build / update / evaluate)
├── pipeline.py             End-to-end pipeline orchestrator (ATOCPipeline)
├── requirements.txt        Python dependencies
├── config/
│   ├── settings.py         All hyperparameters (matching Appendix E):
│   │                       w_rel/w_com/w_aut=0.4/0.3/0.3, w_term=0.5,
│   │                       eta=0.65, theta=0.70, theta_split=0.60,
│   │                       delta=0.55, gamma_merge=0.55, alpha_ctx=0.6,
│   │                       L=4, K1-K4=127/32/8/1
│   └── prompts.py          All prompt templates (including context relevance scoring prompt from Appendix C)
├── corpus/                 Module 1: Multi-agent corpus collection
│   ├── agents.py           Retrieval / filtering / preprocessing agents and message bus
│   ├── collector.py        Query expansion, multi-source retrieval, deduplication, statistics
│   ├── filtering.py        Eq.1-3 filtering scoring (Relevance / Completeness / Authority) and keyword crawler baseline
│   ├── preprocessing.py    Denoising, date/institution name normalization, task snippet extraction
│   └── authority.py       Source authority scoring table (Appendix A)
├── extraction/             Module 2: Hierarchical feature extraction
│   ├── bottom_level.py     Two-stage bottom-level feature extraction (extraction + refinement, with few-shot examples and supporting sentence citation)
│   ├── top_level.py        Bottom-up abstraction of top-level features (activity / information requirement)
│   ├── flat.py            w/o-SFE ablation: single-pass flat extraction
│   ├── grounding.py       Source-grounding verification: elements without supporting sentences are marked unverified and removed
│   └── cross_level.py      Cross-level mapping (bottom-level features ↔ goal ↔ top-level features)
├── relationship/           Module 3: Core task relation modeling
│   ├── decomposition.py     Task decomposition (chain-of-thought reasoning)
│   ├── division.py         Task division of labor
│   ├── collaboration.py     Task collaboration
│   ├── information_interaction.py  Information exchange
│   └── pipeline.py         Relation modeling orchestrator
├── clustering/             Module 4: Hierarchy construction
│   ├── embeddings.py       BGE-M3 embedding wrapper (with local cache)
│   ├── context.py           Eq.6 combined similarity: 0.6*C_ctx + 0.4*S'_cos, S'_cos=(cos+1)/2 normalised to [0,1]
│   ├── assignment.py      Section 4.1.1 bottom-level clustering: θ=0.70 assignment, θ_split=0.60 median-based split, S_cos>=0.9 merge threshold, explicit unassigned feature set
│   ├── dhwc.py             Algorithm 1: Dynamic Weight Hierarchical Clustering (layer-wise linear weight adjustment, γ_merge=0.55, target cluster counts K_l, normalised cosine similarity)
│   └── ward_baseline.py    w/o-DWHC ablation: equal-weight (0.25 per dimension) Ward clustering
├── ontology/               Module 5: OWL ontology generation
│   ├── mapping.py          Mapping rules: decomposition → hasSubTask (inverse isSubTaskOf), division → performedBy, collaboration → collaboratesWith, information exchange → requiresInformation; sibling disjointness axioms; performedBy minCardinality=1; rdfs:subClassOf reserved for genuine is-a concept hierarchies
│   ├── serialization.py    Turtle serialization and build report generation
│   └── consistency.py      Pellet consistency checking and entailment evaluation (50 curated test questions)
├── update/                 Module 6: Dynamic update
│   ├── monitor.py          Trigger: Sim_max = max_C (cos(V,u_C)+1)/2 < δ
│   └── incremental.py      Three update strategies: incremental / full reconstruction / no update
├── evaluation/
│   ├── metrics.py         P/R/F1 (span matching, reported separately for bottom/top level features), mean±std over repeated runs
│   ├── icc.py              ICC(C,1) (Shrout & Fleiss 1979) with bootstrap 95% CI
│   ├── quality.py         Three-judge blinded scoring pipeline: 5-dimension scoring, average judge weights, weighted comprehensive score Q, per-dimension ICC
│   ├── agreement.py       Element-level consistency (equivalent span matching, used for Table 12)
│   └── runtime.py          Runtime and token consumption statistics
├── models/
│   ├── bilstm_crf.py      BiLSTM + linear-chain CRF baseline (GloVe-100d embeddings, 128-d BiLSTM encoder, Viterbi decoding)
│   ├── bert_crf.py         BERT + CRF baseline
│   └── train.py           Supervised baseline training (70/15/15 train/val/test split, BIO labeling, AdamW, early stopping)
├── experiments/
│   ├── run_ablation.py     Unified experiment for all six ablation variants
│   ├── run_incremental.py Incremental vs full reconstruction vs no update + δ sensitivity analysis
│   ├── run_baselines.py    All comparison baselines
│   └── run_cross_domain.py Cross-domain feasibility experiment
└── data/
    └── sample_documents.json  Minimal sample corpus (including running example)
```

---

### 8. Notes for Paper Reproduction

1. The F1 scores and quality scores reported in the paper are based on the authors' self-collected corpus (412 emergency rescue documents, 31% off-topic filtering rate), which is not redistributed with this repository. Results obtained on custom corpora will differ from the reported numbers.
2. All LLM experiments are recommended to be repeated 5 times (with different random seeds, and temperature set to 0.1/0.3/0.4 per pipeline stage), reporting mean±std (implemented via the `repeated_runs` function in `evaluation/metrics.py`).
3. Gold standard format matches the `elements`/`relations` fields in `data/sample_documents.json`, following the annotation protocol in Section 5.1.1 (3 independent expert annotators, pairwise macro-F1 reported for inter-annotator agreement).
4. Source-grounding verification is enabled by default: elements without supporting sentences are marked unverified and excluded from the final ontology, matching the source-grounding mechanism described in Section 5.2.5.

---

### 9. Frequently Asked Questions

**Q: Pellet reasoner throws errors?**
A: Confirm Java is installed (`java -version`). Consistency checks are automatically skipped if Java is unavailable, with no impact on other pipeline steps.

**Q: A backbone's API is not compatible?**
A: All providers follow the OpenAI-compatible `/chat/completions` protocol. If using a custom gateway, update the corresponding `base_url` in `config/settings.py`.

**Q: Embedding generation is very slow on first run?**
A: BGE-M3 downloads weights (~2GB) from HuggingFace on first use; weights are cached locally afterward. For offline environments, pre-download weights and set the `HF_HOME` environment variable.

---

### 10. Citation

If you use this code in your work, please cite:

> 
> Z. Lu, J. Yang, C. Zhang, J. Wan, Y. Li. *ATOC-LLM: An LLM-Driven Automatic Task Ontology Construction Method*. Electronics (under review), 2026.
> 
> 
> ```
> 
> ```
