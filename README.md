# Emergent Chiral Asymmetry in 3+1D Causal Sets
### Numerical code and data accompanying the preprint:
**"Emergent Chiral Asymmetry in 3+1D Causal Sets from Dirac–Kähler Fermions with Parity-Biased Sprinklings"**

Author: Greg Bakker  
DOI: https://doi.org/10.5281/zenodo.17714411  
GitHub: https://github.com/604Bakker/3-1D-entropic-foam  

---

## Overview

This repository contains the exact code, data, figures, and manuscript sources used
to produce the results in the above paper. All numerical simulations were conducted
using Python 3.11.9 on consumer hardware. The results demonstrate an emergent 
topological chiral index in 3+1D causal sets under parity-biased Poisson sprinklings 
using a Dirac–Kähler fermion operator with a Wilson term.

This package enables **full reproducibility** of:

- The N=6000 sweep used for the phase diagram  
- The chiral index computation for each (r, ε) parameter point  
- All three main plots included in the paper  
- LaTeX compilation of the manuscript  

---

## Folder Contents

### **Simulation Code**
- `3d_sweep.py`  
  Main script to run the full parameter sweep, generating chiral index measurements.

### **Plotting Code**
- `create_figs.py`  
  Script that reads `live_sweep.csv` and generates:  
  - `fig1_phase_diagram.pdf`  
  - `fig2_slices.pdf`  
  - `fig3_distribution.pdf`

### **Data**
- `live_sweep.csv`  
  Contains the complete sweep results for:
    - r ∈ [0.1, 0.55]  
    - ε ∈ [-1.8, -0.3]  
    - 40 trials per grid point  
    - Column fields: r, epsilon, mean_index, std_index, trials_done  

### **Figures**
- `fig1_phase_diagram.pdf`  
- `fig2_slices.pdf`  
- `fig3_distribution.pdf`  

These were generated directly from `create_figs.py` and included unchanged in the paper.

### **Manuscript**
- `main.tex`  
- `references.bib`  
- `Emergent Chiral Asymmetry in 3 + 1D Causal Sets.pdf`  

The exact document submitted for journal review.

### **Environment Requirements**
- `requirements.txt`  
Lists the Python dependencies needed to reproduce all computations and figures.

---

## How to Reproduce the Sweep

The full sweep used in the paper is already included as `live_sweep.csv`.  
However, if you wish to re-run it:

```bash
python3 3d_sweep.py
```

Estimated runtime: several hours on a modern CPU (N=6000 per run).

---

## How to Reproduce All Figures

Run the plotting script:

```bash
python3 create_figs.py
```

This will regenerate:

- `fig1_phase_diagram.pdf`  
- `fig2_slices.pdf`  
- `fig3_distribution.pdf`  

These should match the versions included in the paper.

---

## Installation Instructions

Ensure you have Python 3.11.9 installed.  
Then install all required packages:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this code or data in your work, please cite:

```
Greg Bakker (2025). Emergent Chiral Asymmetry in 3+1D Causal Sets from Dirac–Kähler Fermions with Parity-Biased Sprinklings.
Zenodo. https://doi.org/10.5281/zenodo.17714411
```

---

## License

This project is released under an open license to ensure full transparency and reproducibility.
All scientific conclusions are the responsibility of the author.

---
