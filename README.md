# Scientific Paper Classification using Graph Neural Networks and Directed Multigraphs

> **B.Tech Major Project** — Department of Computer Science & Engineering, IIT Patna
> **Author:** Saumya Pratap Singh (2201AI35)
> **Supervisor:** Dr. Sourav Kumar Dandapat

---

## Overview

This repository contains the full implementation, experiments, and poster for the B.Tech thesis on classifying scientific papers using Graph Neural Networks (GNNs) on directed multigraphs. The work targets the **OGBN-arXiv** benchmark (169,343 papers, 40 classes) and proposes a 4-phase pipeline that addresses two structural failures in conventional GNN pipelines:

- **Homogeneity assumption** — only citation edges, discarding co-authorship, venue, and field-of-study signals
- **Undirectedness assumption** — citation direction erased, losing the seminal-paper vs. survey-paper distinction

The complete pipeline achieves **78.14% micro-accuracy** and **65.99% macro-accuracy**, statistically validated via McNemar's Test ($\chi^2 = 28.79$, $p = 8.07 \times 10^{-8}$).

---

## Key Results

| Configuration | Micro Acc. | Params |
|---|---|---|
| Paper baseline (Ly et al. 2024) | 77.21% | — |
| + Directed edges — Phase 1 | 77.61% | 0.8M |
| + DropEdge + Ensemble — Phase 2 | **78.14%** | 1.26M |
| + GraphSMOTE — Phase 3 | 77.98% | 1.26M |
| + TF-IDF Late-Fusion — Phase 4 | 77.98% | 1.66M |
| TransformerConv + Fusion | 77.67% | 2.60M |
| **GraphSAGE + Fusion (final)** | **77.74%** | **1.26M** |

**Macro-accuracy:** baseline `<50%` → **65.99%** after Late-Fusion across all 40 categories.

---

## Pipeline Architecture

```
OGBN-arXiv (169,343 papers · 40 classes)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Phase 1 · Directed Multigraph              │
│  Edge split: cites + cited_by (HeteroConv)  │
│  → 79.02% val (first untuned run)           │
└────────────────────┬────────────────────────┘
                     │
        ▼
┌─────────────────────────────────────────────┐
│  Phase 2 · Targeted Regularisation          │
│  DropEdge (p=0.50 cited_by) + Perturbation  │
│  + 3-model ensemble logit averaging         │
│  → 78.14% micro (+0.53 pp)                 │
└────────────────────┬────────────────────────┘
                     │
        ▼
┌─────────────────────────────────────────────┐
│  Phase 3 · GraphSMOTE Topology Balancing    │
│  10,090 synthetic nodes · 16 minority cls.  │
│  → Class 12 (5 samples): 100% accuracy      │
└────────────────────┬────────────────────────┘
                     │
        ▼
┌─────────────────────────────────────────────┐
│  Phase 4 · Lexical-Semantic Late-Fusion     │
│  TF-IDF (500d) ‖ SimTG (1024d) = 1524d     │
│  → Macro: <50% → 65.99%                    │
└────────────────────┬────────────────────────┘
                     │
        ▼
   Correct & Smooth (α₂ = 0.6)
   78.14% micro · 65.99% macro
```

---


## Key Contributions

1. **Directed Multigraph with Reverse Message Passing** — First application of the AAAI-24 Egressy et al. framework to OGBN-arXiv. The first untuned run hit 79.02% validation accuracy, clearing the 77–78% undirected plateau without any hyperparameter tuning.

2. **Lexical-Semantic Late-Fusion** — Concatenating 500-dim TF-IDF sparse flags with 1024-dim SimTG embeddings (1524-dim total). Macro-accuracy climbed from `<50%` to **65.99%** across all 40 categories. TF-IDF dimensions act as hard lexical switches that neighbourhood aggregation cannot override.

3. **GraphSMOTE Topology Balancing** — 10,090 synthetic nodes generated for 16 minority classes via graph-adapted SMOTE. Class 12 (cs.CE), starting from just 5 training samples, reached **100% test accuracy** after augmentation.

4. **Negative Result: Super-Node Addiction** — Synthetic hub nodes connected to all minority training members caused macro-accuracy to collapse from 65.99% to 64.16% at test time, establishing a hard ceiling for topology-only interventions and demonstrating that feature-level fusion is strictly preferable to topological injection for heterophily resolution.

5. **Efficiency Benchmark** — A 1.26M-parameter GraphSAGE model outperforms a 2.6M-parameter TransformerConv variant (77.74% vs. 77.67%), converging 25 epochs sooner.

---

## Forensic Fixes (Reproducibility Notes)

Two implementation bugs were discovered and fixed before any valid results could be reported — both are documented in Chapter 3 of the thesis:

**Data Leakage in Subgraph Generation** — An early FOS edge-creation script sorted candidate nodes by label index before applying the density cap, inadvertently encoding ground-truth labels into the graph topology. This produced artificially inflated test accuracies near 96%. All experiments use only the validated, label-agnostic pipeline.

**Logit vs. Log-Probability Conflict** — The GNN's final layer was emitting `log_softmax` outputs that were then passed to `CrossEntropyLoss` (which internally applies another softmax). This compounded softmax effectively reduced the trained GNN to a graph-agnostic MLP. Fixed by removing all terminal activations and returning raw logits.

---

## Poster

The conference poster is in `poster/` and is compiled with **LuaLaTeX** (required by the Gemini theme's `fontspec` dependency).

```bash
cd poster
lualatex poster_thesis.tex
```

The pipeline diagram is drawn entirely in TikZ — no external figure files are required.

---

## Citation

If you use this work, please cite:

```bibtex
@thesis{singh2026gnn,
  author    = {Saumya Pratap Singh},
  title     = {Scientific Paper Classification using Graph Neural Networks
               and Directed Multigraphs},
  school    = {Indian Institute of Technology Patna},
  year      = {2026},
  type      = {B.Tech Project Report}
}
```

Primary references this work builds on:

```bibtex
@misc{ly2024multigraph,
  author = {Khang Ly and Yury Kashnitsky and Savvas Chamezopoulos
            and Valeria Krzhizhanovskaya},
  title  = {Article Classification with Graph Neural Networks and Multigraphs},
  note   = {arXiv:2309.11341},
  year   = {2024}
}

@inproceedings{egressy2024directed,
  author    = {B{\'e}ni Egressy and Luc von Niederh{\"a}usern and
               Jovan Blanu{\v{s}}a and Erik Altman and
               Roger Wattenhofer and Kubilay Atasu},
  title     = {Provably Powerful Graph Neural Networks for Directed Multigraphs},
  booktitle = {AAAI-24},
  year      = {2024}
}
```

---

## License

This project is released for academic and research purposes.
See [LICENSE](LICENSE) for details.
