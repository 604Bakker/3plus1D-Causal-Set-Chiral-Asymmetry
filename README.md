# Emergent Chiral Asymmetry in 3+1D Causal Sets  
### Dirac–Kähler Fermions with Parity-Biased Poisson Sprinklings

This repository contains the full simulation code, data-generation tools, and analysis scripts used in the paper:

**“Emergent Chiral Asymmetry in 3+1D Causal Sets from Dirac–Kähler Fermions with Parity-Biased Sprinklings”  
by Greg Bakker (2025)**

The project provides numerical evidence for a sharp topological transition in random causal sets: once a small parity-violating sprinkling bias is introduced, the Dirac–Kähler spectrum develops a large, robust chiral index of ~40–50 zero modes of uniform handedness. This extends the previously discovered 2+1D “chiral plateau” into 3+1 dimensions.

---

## ✨ Key Features

- **3+1D Poisson sprinkling generator** with adjustable parity-violating bias  
- **Causal relation and link construction** with proper-time cutoff  
- **Dirac–Kähler operator** built directly from incidence matrices  
- **Minimal Wilson term** to suppress doublers  
- **Chern–Simons-like orientation term** for parity bias  
- **Eigenvalue computation** via sparse Hermitian solvers  
- **Chiral index measurement** from low-lying modes of `i γ₅ D`  
- **Phase diagram tools** to reproduce the “chiral plateau”  
- **100% reproducible**: all parameters, seeds, and code provided

---

## 🔬 Scientific Background

Causal-set theory models spacetime as a discrete partially ordered set, replacing the metric continuum with locally finite order structure. The Dirac–Kähler (DK) formulation represents fermions on this discrete geometry using chain complexes rather than local tetrads or spin structures.

In 2+1 dimensions, it was previously found that a slight parity-violating deformation of the sprinkling distribution produces a **topologically stable excess of chiral zero modes**. This repository extends that investigation to **3+1 dimensions**, where we observe:

- A **sharp critical line** around  
  - `r ≳ 0.11` (bias strength)  
  - `ε ≳ 0.35` (discreteness scale)

- A stable chiral index plateau of **≈ −45** for `N = 6000–8000`  

- Reversal of handedness when the sign of the parity bias is flipped  

These results indicate that **discrete spacetime microstructure alone** can support nontrivial chiral structure — without gauge fields, Higgs dynamics, or continuum limits.

## Installation & Running

1. Clone the repo: `git clone https://github.com/604Bakker/3plus1D-Causal-Set-Chiral-Asymmetry.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python final_3plus1_chiral_cliff.py`
4. To regenerate the plot: `python make_3plus1_cliff_plot.py` (requires live_sweep.csv from full sweep)


  - final_3plus1_chiral_cliff.py   # Reproduces the key result
  -full_phase_sweep_3plus1.py     # Generates the full phase diagram (live_sweep.csv)
  - make_3plus1_cliff_plot.py      # Creates the publication plot from CSV
  - live_sweep.csv                 # Raw data from sweep
  - 3plus1_cliff_plot.png          # The chiral cliff figure
  - README.md

📚 Citing This Work
If you use this repository or build upon it, please cite:

G. Bakker, “Emergent Chiral Asymmetry in 3+1D Causal Sets from Dirac–Kähler Fermions with Parity-Biased Sprinklings” (2025).
Zenodo DOI: [to be added]

🤝 Acknowledgments
Certain aspects of implementation, debugging, and text polishing benefited from interactive assistance with large language models (Grok by xAI and ChatGPT by OpenAI). All scientific ideas, physical conclusions, and numerical results are solely the responsibility of the author.

📝 License
Released under the MIT License.
You are free to use, modify, and build upon this work for research or education.

🌟 Contributions
Pull requests, reproducibility improvements, and extended experiments (e.g., with gauge fields, alternative DK discretizations, or different parity-bias models) are welcome.

Feel free to open an issue with questions or feature requests.

