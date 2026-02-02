# 📝 Paper Writing Guide

This document explains how to continue working on the SC'26 Cascade paper.

---

## Table of Contents

1. [Paper Overview](#paper-overview)
2. [File Structure](#file-structure)
3. [Building the Paper](#building-the-paper)
4. [Section Guide](#section-guide)
5. [Updating Results](#updating-results)
6. [Figure Generation](#figure-generation)
7. [Submission Checklist](#submission-checklist)

---

## Paper Overview

**Title**: Cascade: HPC-Scale KV Cache Storage for LLM Inference

**Target Venue**: SC'26 (Supercomputing Conference)

**Deadline**: TBD (typically April/May for November conference)

**Page Limit**: 10 pages + references

### Key Contributions

1. **Content-addressed deduplication** for KV cache blocks (98% storage reduction)
2. **4-tier hierarchical storage** (GPU → SHM → Remote DRAM → Lustre)
3. **Semantic eviction policy** preserving shared prefixes
4. **HPC-native multi-node scaling** up to 256 nodes

---

## File Structure

```
paper/
├── main.tex                    # Main document (includes all sections)
├── 1. Introduction.tex         # Motivation, contributions
├── 2. Background.tex           # LLM inference, KV cache basics
├── 3. Design.tex               # Cascade architecture
├── 4. Evaluation.tex           # Experiments, results  ← MAIN RESULTS
├── 5. Related Works.tex        # Comparison to prior work
├── 6. Discussion and Limitation.tex
├── 7. Conclusion.tex
├── references.bib              # Bibliography
├── main.pdf                    # Compiled output
├── generate_figures.py         # Script to generate TikZ figures
└── Figures/                    # All figures
    ├── architecture.pdf        # System architecture diagram
    ├── throughput_comparison.pdf
    └── ...
```

---

## Building the Paper

### On Perlmutter (Manual)

```bash
cd paper/

# Full build (including bibliography)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# View PDF
evince main.pdf  # Or download to local machine
```

### Using Makefile (if available)

```bash
cd paper/
make        # Build PDF
make clean  # Remove aux files
```

### Local Build (Overleaf Alternative)

Download the `paper/` folder and build locally with TeX Live:

```bash
# macOS
brew install mactex

# Ubuntu
apt install texlive-full

# Build
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Section Guide

### 1. Introduction (`1. Introduction.tex`)

**Purpose**: Hook reader, state problem, list contributions

**Key elements**:
- LLM inference cost problem (80% of time in memory)
- Why existing solutions fail at HPC scale
- Cascade's approach (1 paragraph)
- Contributions list (numbered)
- Results preview (headline numbers)

**Headlines to maintain**:
- "98% storage reduction"
- "1.93× higher throughput than LMCache"
- "100% vs 12% cache hit rate"

---

### 2. Background (`2. Background.tex`)

**Purpose**: Educate reader on LLM inference and KV caching

**Key concepts to explain**:
- Transformer attention mechanism
- KV cache purpose (avoid recomputation)
- Memory hierarchy on HPC systems
- Perlmutter hardware specs

**Figure needed**: Memory access pattern diagram

---

### 3. Design (`3. Design.tex`)

**Purpose**: Technical architecture of Cascade

**Subsections**:
1. Content-Addressed Block IDs (SHA-256 hashing)
2. 4-Tier Storage Hierarchy
3. Semantic Eviction Policy
4. MPI-based Communication
5. Aggregated Lustre I/O

**Figures needed**:
- Architecture diagram (4 tiers)
- Eviction flow chart
- Block format diagram

**Code listings**:
```latex
\begin{lstlisting}[language=Python, caption=Block ID computation]
def compute_block_id(key: bytes, value: bytes) -> str:
    h = hashlib.sha256()
    h.update(key)
    h.update(value)
    return h.hexdigest()[:32]
\end{lstlisting}
```

---

### 4. Evaluation (`4. Evaluation.tex`) ⭐ MOST IMPORTANT

**Purpose**: Prove Cascade works via experiments

**Structure**:
1. **Experimental Setup** (hardware, model, workload)
2. **Overall Comparison** (6-system table)
3. **Deduplication Effectiveness** (bar chart)
4. **Throughput Analysis** (read/write breakdown)
5. **Tier Distribution** (pie chart)
6. **Baseline Failures** (vLLM data loss)
7. **Scalability** (1-32 nodes)
8. **Microbenchmarks** (latency breakdown)

**Key tables**:
- Table 1: 6-system comparison (main results)
- Table 2: Tier statistics
- Table 3: Scaling results
- Table 4: Latency breakdown

**Key figures**:
- Figure 1: Storage efficiency (blocks stored vs lost)
- Figure 2: Throughput (horizontal bar chart)
- Figure 3: Tier distribution (pie chart)
- Figure 4: Cascade vs vLLM eviction (stacked bar)
- Figure 5: Scaling curve (line chart)

---

### 5. Related Works (`5. Related Works.tex`)

**Categories to cover**:
- GPU memory management (vLLM, PagedAttention)
- KV cache systems (LMCache, SGLang)
- HPC I/O systems (HDF5, PDC, DAOS)
- Deduplication (storage systems)
- In-memory caches (Redis, Memcached)

**Format for each**:
> **System X** [cite] does Y. However, it lacks Z, which Cascade addresses through W.

---

### 6. Discussion (`6. Discussion and Limitation.tex`)

**Topics**:
- Hash computation overhead (42ms per block)
- GPU memory pressure trade-offs
- Single-node vs multi-node dedup
- Lustre metadata scalability
- Future: NVMe tier integration

---

### 7. Conclusion (`7. Conclusion.tex`)

**Structure**:
1. Restate problem (1 sentence)
2. Restate solution (2 sentences)
3. Key results (bullet points)
4. Future work (1 sentence)

---

## Updating Results

### After Running Benchmarks

1. **Get Result JSON**:
```bash
cat benchmark/results/full_6sys_<jobid>.json
```

2. **Update Table 1** in `4. Evaluation.tex`:
```latex
\begin{tabular}{l|rr|r|l}
\textbf{System} & \textbf{Write} & \textbf{Read} & \textbf{Dedup} & \textbf{Observation} \\
\midrule
\Cascade & 1.70 & 10.27 & 49× & 98\% dedup \\
LMCache & 0.88 & 6.55 & 1× & Per-file overhead \\
...
\end{tabular}
```

3. **Update TikZ charts**:
```latex
\addplot coordinates {(Cascade, 1.70) (LMCache, 0.88) ...};
```

4. **Update headline numbers** in Introduction:
```latex
achieves \textbf{1.93$\times$} higher throughput
```

### Consistency Check

Run this to ensure all numbers match:
```bash
grep -n "1.70\|10.27\|98\%\|1.93" paper/*.tex
```

All instances should reference the same experiment.

---

## Figure Generation

### TikZ Charts (In-Document)

Charts are embedded in `4. Evaluation.tex` using TikZ/pgfplots:

```latex
\begin{tikzpicture}
\begin{axis}[ybar, ...]
\addplot coordinates {(Cascade, 60) (LMCache, 3000) ...};
\end{axis}
\end{tikzpicture}
```

### External PDF Figures

For complex diagrams, generate PDFs:

```bash
cd paper/
python generate_figures.py
# Creates: Figures/architecture.pdf, etc.
```

### Required Packages

Add to `main.tex` preamble:
```latex
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{patterns, shapes, positioning}
\usepackage{tcolorbox}
```

---

## Submission Checklist

### Content

- [ ] All sections complete
- [ ] All figures render correctly
- [ ] All tables have captions
- [ ] All citations resolve
- [ ] Page limit met (10 pages)
- [ ] Abstract < 250 words

### Technical

- [ ] All numbers from actual experiments
- [ ] Experiment configuration fully specified
- [ ] Hardware specs accurate (A100-40GB, etc.)
- [ ] Baselines use real code (not simulation)

### Formatting

- [ ] SC format template used
- [ ] Single-column PDF for review
- [ ] Author info for camera-ready only
- [ ] Supplementary materials prepared

### Final Steps

```bash
# Check page count
pdfinfo main.pdf | grep Pages

# Check for common errors
grep -i "todo\|fixme\|xxx" paper/*.tex

# Generate camera-ready
cp main.pdf Cascade_SC26_submission.pdf
```

---

## Quick Edit Reference

### Update a Number

```bash
# Find all occurrences
grep -rn "1.70" paper/

# Edit file
vim paper/4.\ Evaluation.tex
```

### Add a Citation

1. Add to `references.bib`:
```bibtex
@inproceedings{newpaper2025,
  author = {Author, A.},
  title = {Title},
  booktitle = {Conference},
  year = {2025}
}
```

2. Cite in text:
```latex
Recent work~\cite{newpaper2025} shows...
```

3. Rebuild:
```bash
bibtex main && pdflatex main.tex
```

### Add a Figure

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{Figures/myfig.pdf}
\caption{Description.}
\label{fig:myfig}
\end{figure}
```

Reference: `Figure~\ref{fig:myfig}`
