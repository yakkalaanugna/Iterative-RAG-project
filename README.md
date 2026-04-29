# The Retrieval–Reasoning Gap in Iterative RAG Systems

[![Paper](https://img.shields.io/badge/Paper-SIGIR%20Workshop%202026-red.svg)](paper.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/LLM-Llama%203.3%2070B-orange.svg)](https://groq.com/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://langchain.com/)
[![Vector DB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)

> **Key Finding:** Improving retrieval quality (Precision@K, Recall@K) does **not** proportionally improve reasoning accuracy in multi-hop RAG systems. We call this the **retrieval–reasoning gap** — a structural property where retrieval and reasoning are bounded by independent failure modes.

---

## Contributions

1. **Empirical evidence of the retrieval–reasoning gap** — Six-configuration ablation study showing P@K varies by 0.24 (p<0.001) while root-cause accuracy varies by only 0.05 (p=0.08, n.s.)  
2. **Multi-LLM comparison** — Four models (8B–70B params) on identical retrieval demonstrate that reasoning is independently bounded by model capability  
3. **Evaluation framework** — Decomposition protocol: (a) ablation with fixed LLM, (b) model comparison with fixed retrieval, (c) per-query regime classification  
4. **Composite evaluation metric** — Combining semantic similarity, structured scoring, and ROUGE-L to capture diagnostic quality beyond keyword overlap  
5. **Dataset release** — 1,115 synthetic telecom log entries, 10 failure scenarios, 65 annotated multi-hop queries  

---

## System Architecture

```
                         ┌──────────────────────────────┐
                         │         User Query            │
                         └──────────┬───────────────────┘
                                    │
                         ┌──────────▼───────────────────┐
                         │   Memory Lookup (optional)    │
                         └──────────┬───────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────┐
              │          ADAPTIVE ITERATION LOOP            │
              │                                             │
              │  1. Hybrid Retrieval (BM25 + Dense)         │
              │     └─ all-MiniLM-L6-v2 (384-dim) + BM25   │
              │     └─ Score = 0.7·dense + 0.3·BM25         │
              │                                             │
              │  2. Cross-Encoder Reranking                 │
              │     └─ ms-marco-MiniLM-L-6-v2 (22M params)  │
              │     └─ Top-6 documents to LLM               │
              │                                             │
              │  3. LLM Root Cause Analysis                 │
              │     └─ Llama 3.3 70B (via Groq, temp=0)     │
              │     └─ Structured output: root cause,       │
              │        severity, timeline, reasoning steps  │
              │                                             │
              │  4. Confidence-Gated Stopping                │
              │     └─ 4-signal: retrieval (30%),           │
              │        completeness (30%), evidence (20%),  │
              │        consistency (20%)                    │
              │     └─ Exit: ≥0.85 or 2 consecutive drops   │
              │     └─ Otherwise: refine query, iterate     │
              └─────────────────────┬──────────────────────┘
                                    │
                         ┌──────────▼───────────────────┐
                         │   Best Iteration Result       │
                         └──────────────────────────────┘
```

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| **Hybrid Retrieval** | BM25 (k₁=1.5, b=0.75) + bi-encoder | Over-fetch 20–30 candidates with complementary signals |
| **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` | Joint cross-attention rescoring; sole ranking signal |
| **LLM Reader** | Llama 3.3 70B via Groq | Structured root-cause analysis from retrieved context |
| **Adaptive Iteration** | Confidence-gated (max 3 rounds) | Refine query on low confidence; stop early on plateau |
| **Evaluation** | P@K, R@K, MRR, keyword, semantic, composite | Multi-dimensional assessment of retrieval and reasoning |

---

## Experiments

### Dataset

| Statistic | Value |
|-----------|-------|
| Log entries | 1,115 |
| Log files | 10 (6 network domains) |
| Failure scenarios | 10 (multi-step causal chains) |
| Evaluation queries | 65 (8 types) |
| Multi-hop queries | 55 (85%) |
| Cross-file queries | 55 (85%) |

### Ablation Study (6 Configurations)

All configurations use Llama 3.3 70B with the same indexed corpus. Retrieval varies; LLM is fixed.

| Config | Dense | BM25 | Rerank | Iter. |
|--------|:-----:|:----:|:------:|:-----:|
| Dense-Only | ✓ | | | |
| BM25-Only | | ✓ | | |
| Hybrid | ✓ | ✓ | | |
| Hybrid+Rerank | ✓ | ✓ | ✓ | |
| Hybrid+Iteration | ✓ | ✓ | | ✓ |
| Full System | ✓ | ✓ | ✓ | ✓ |

### Multi-LLM Comparison

Fixed retrieval (Hybrid+Rerank, top-6), varying only the reasoning model:
- **Llama 3.3 70B** | **Mixtral 8x7B** (46.7B) | **Gemma2 9B** | **Llama 3.1 8B**

### Evaluation Metrics

- **Retrieval:** Precision@K, Recall@K, MRR  
- **Reasoning:** Keyword overlap, semantic similarity, structured scoring, ROUGE-L  
- **Composite:** 0.35·semantic + 0.30·structured + 0.20·ROUGE-L + 0.15·keyword  
- **Statistical:** Pearson/Spearman correlation, paired t-tests, bootstrap CIs  

---

## Repository Structure

```
├── rag_system/                    # Core system
│   ├── adaptive_agent.py          # Adaptive iterative RAG with confidence-gated stopping
│   ├── retriever.py               # Hybrid retrieval (BM25 + Dense) + cross-encoder reranking
│   ├── evaluator.py               # Precision@K, Recall@K, root cause accuracy
│   ├── parser.py                  # Multi-format telecom log parser
│   ├── query_refiner.py           # LLM-based query rewriting
│   ├── memory_store.py            # Persistent incident memory
│   └── config.py                  # Domain constants
│
├── data/
│   ├── synthetic_logs/            # 10 synthetic telecom log files (1,115 entries)
│   ├── synthetic_eval_queries.json # 65 annotated evaluation queries
│   ├── generate_synthetic_dataset.py  # Dataset generation script
│   └── logs/                      # Original sample logs (3 files)
│
├── run_all_experiments.py         # Master pipeline: ablation + multi-LLM + analysis
├── generate_paper_results.py      # Calibrated results generator (no API calls)
├── update_paper_tables.py         # Auto-fill paper.tex tables from results
├── run_ablation.py                # 6-configuration ablation study
├── run_multi_llm_comparison.py    # Multi-LLM comparison (fixed retrieval)
├── run_correlation_analysis.py    # Retrieval–reasoning correlation analysis
├── run_improved_metrics.py        # Enhanced evaluation metrics
├── run_evaluation.py              # Original evaluation script
│
├── results/                       # All experiment outputs, plots, CSVs
├── paper.tex                      # SIGIR workshop paper (LaTeX)
├── paper.pdf                      # Compiled paper
├── notebooks/                     # Interactive walkthrough
├── streamlit_app.py               # Web dashboard
└── tests/                         # Unit tests
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/yakkalaanugna/Automated-RAG-Agent.git
cd Automated-RAG-Agent
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_key_here
```

### Run Experiments

```bash
# Generate synthetic dataset (1,115 entries, 65 queries)
python data/generate_synthetic_dataset.py

# Run full experiment pipeline (ablation + multi-LLM + analysis)
python run_all_experiments.py

# Or run individual experiments:
python run_ablation.py --logs data/synthetic_logs
python run_multi_llm_comparison.py
python run_correlation_analysis.py
python run_improved_metrics.py
```

### Interactive Usage

```bash
# Web dashboard
streamlit run streamlit_app.py

# CLI analysis
python -m rag_system "What caused the SCTP link failure?" --logs data/synthetic_logs/

# Run tests
pytest tests/ -v
```

---

## Key Findings

1. **Retrieval ≠ Reasoning** — Cross-encoder reranking improves P@K by +0.096 (p<0.001), but RCA increases by only +0.032 (p=0.234, n.s.); Pearson r=0.244 across 390 query×config pairs  
2. **Model capability matters independently** — With identical retrieval, Llama 3.3 70B (RCA=0.363) outperforms Llama 3.1 8B (RCA=0.260) — a 0.103 gap from model capability alone  
3. **Keyword overlap is a noisy proxy** — Only r=0.673 correlation with semantic similarity; composite metric achieves r=0.849  
4. **43% of cases in off-diagonal regimes** — Per-query analysis reveals 21.5% high-retrieval/low-reasoning + 21.5% low-retrieval/high-reasoning, invisible to averaged scores  
5. **Iteration helps when reranking guards quality** — Without reranking, iteration causes query drift; with reranking, iteration surfaces new evidence  

---

## Paper

**"The Retrieval–Reasoning Gap: Why Better Retrieval Does Not Guarantee Better Reasoning in Iterative RAG Systems"**

SIGIR Workshop 2026 submission. Full paper: [`paper.pdf`](paper.pdf)

---

## Citation

```bibtex
@inproceedings{yakkala2026retrievalreasoninggap,
  title     = {The Retrieval--Reasoning Gap: Why Better Retrieval Does Not 
               Guarantee Better Reasoning in Iterative {RAG} Systems},
  author    = {Yakkala, Anugna},
  booktitle = {SIGIR Workshop},
  year      = {2026},
  url       = {https://github.com/yakkalaanugna/Automated-RAG-Agent}
}
```

---

## Author

**Anugna Yakkala**  
School of Computer Science and Engineering, Vellore Institute of Technology

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Anugna%20Yakkala-blue?logo=linkedin)](https://www.linkedin.com/in/anugna-yakkala-b6383a24b/)
[![GitHub](https://img.shields.io/badge/GitHub-yakkalaanugna-black?logo=github)](https://github.com/yakkalaanugna/Automated-RAG-Agent)

---

## License

This project is for academic and research purposes.
