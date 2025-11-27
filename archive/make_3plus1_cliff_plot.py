# make_3plus1_cliff_plot.py
# Creates the cliff plot from the live CSV

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the live CSV you already have
df = pd.read_csv("live_sweep.csv")

# Pivot to get one line per r value
pivot = df.pivot(index='epsilon', columns='r', values='mean_index')
errors = df.pivot(index='epsilon', columns='r', values='std_index')

plt.figure(figsize=(10, 6.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(pivot.columns)))

for i, r_val in enumerate(sorted(pivot.columns)):
    eps = pivot.index
    mean = pivot[r_val]
    std = errors[r_val]
    
    plt.errorbar(eps, mean, yerr=std, 
                 fmt='o-', capsize=4, capthick=1.5,
                 color=colors[i], label=f'r = {r_val:.3f}',
                 markersize=6, linewidth=2)

plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xscale('log')
plt.xlabel(r'Discreteness scale $\varepsilon$', fontsize=14)
plt.ylabel(r'Chiral index $\langle n_+ - n_- \rangle$', fontsize=14)
plt.title(r'3+1D Chiral Phase Transition in Causal Sets ($N \approx 6000$--$8000$)', 
          fontsize=15, pad=15)
plt.legend(title='Parity bias $r$', fontsize=11, title_fontsize=12)
plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Save the exact figure used in the paper
plt.savefig("3plus1_cliff_plot.png", dpi=300, bbox_inches='tight')
plt.savefig("3plus1_cliff_plot.pdf", bbox_inches='tight')
print("Plot saved as 3plus1_cliff_plot.png and .pdf")
plt.show()
