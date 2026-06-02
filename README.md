# Scientific Paper Classification using Graph Neural Networks and Directed Multigraphs

> **B.Tech Major Project** — Department of Computer Science & Engineering, IIT Patna
> **Author:** Saumya Pratap Singh (2201AI35)
> **Supervisor:** Dr. Sourav Kumar Dandapat

---

## Abstract

This work investigates GNN expressivity limitations in large-scale scientific paper classification on **OGBN-arXiv** (169,343 papers, 40 classes). Standard GNN pipelines rest on two flawed structural assumptions — graph homogeneity and directional symmetry — that together cause **Minority Class Collapse**: sparse, cross-disciplinary classes drift toward dominant-class centroids. A 4-phase pipeline is proposed that removes both assumptions, achieving **78.14% micro-accuracy** and **65.99% macro-accuracy**, statistically validated via McNemar's Test ($χ² = 28.79$, $p = 8.07 × 10⁻⁸$).

---

## Poster

The poster (`poster/poster_thesis.tex`) is compiled with **LuaLaTeX** — required by the Gemini theme's `fontspec` dependency. Do **not** use `pdflatex`.

```bash
cd poster
lualatex poster_thesis.tex
```

### Poster Files

```
poster/
├── poster_thesis.tex           # Main poster source
├── beamerthemegemini.sty       # Gemini beamer theme
├── beamercolorthememsu.sty     # MSU colour theme
├── poster.bib                  # Bibliography
└── logos/
    └── iitpatna.png            # Institute logo
```

> The pipeline diagram is drawn entirely in TikZ — no external figure files are required.

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

**Macro-accuracy:** `<50%` → **65.99%** across all 40 categories after Late-Fusion.

---

## Key Contributions

1. **Directed Multigraph (Reverse MP)** — First AAAI-24 Egressy et al. application to OGBN-arXiv. Untuned Run 1 hit 79.02% validation, clearing the 77–78% undirected plateau immediately.

2. **Lexical-Semantic Late-Fusion** — TF-IDF (500d) concatenated with SimTG (1024d) embeddings yields a 1524-dim feature set. Macro-accuracy climbed from `<50%` to **65.99%** across all 40 categories.

3. **GraphSMOTE Topology Balancing** — 10,090 synthetic nodes for 16 minority classes. Class 12 (cs.CE, just 5 training samples) reached **100% test accuracy** after augmentation.

4. **Negative Result — Super-Node Addiction** — Synthetic hub nodes collapsed macro-accuracy from 65.99% to 64.16% at test time, establishing that feature-level fusion is strictly preferable to topological injection for heterophily resolution.

5. **Efficiency Benchmark** — 1.26M-parameter GraphSAGE outperforms 2.6M-parameter TransformerConv in accuracy and convergence speed (55 vs. 80 epochs).

---

## Citation

```bibtex
@thesis{singh2026gnn,
  author = {Saumya Pratap Singh},
  title  = {Scientific Paper Classification using Graph Neural Networks
            and Directed Multigraphs},
  school = {Indian Institute of Technology Patna},
  year   = {2026},
  type   = {B.Tech Project Report}
}
```

---

## References

\[1\] Ly et al., *Article Classification with GNNs and Multigraphs*, arXiv:2309.11341, 2024.
\[2\] Egressy et al., *Provably Powerful GNNs for Directed Multigraphs*, AAAI-24, 2024.
\[3\] Hamilton et al., *GraphSAGE*, NeurIPS 2017.
\[4\] Zhao et al., *GraphSMOTE*, WSDM 2021.
\[5\] Duan et al., *SimTeG*, 2023.
\[6\] Huang et al., *Correct & Smooth*, ICLR 2021.
