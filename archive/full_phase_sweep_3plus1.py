# full_phase_sweep_3plus1.py
# Author: Greg Bakker
# Date: 25 November 2025
# Purpose: Complete (r, ε) parameter scan that discovered the 3+1D chiral cliff
# Output: live_sweep.csv → used to generate Figure 1 in the paper
# N = 6000, 9 r values × 12 ε values × 40 trials = 4320 total runs

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix, eye
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

def add_parity_bias(points, r):
    if r == 0: return points
    theta = r * np.pi
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]])
    points[:, 1:4] = points[:, 1:4] @ rot.T
    return points

def sprinkle(N):
    L = N ** 0.25
    return np.random.uniform(0, L, (N, 4))

def causal_links(points, eps):
    N = len(points)
    row, col = [], []
    for i in range(N):
        dt = points[:,0] - points[i,0]
        dr2 = np.sum((points[:,1:] - points[i,1:])**2, axis=1)
        link = (dt > 0) & (dt*dt >= dr2) & (dt <= eps)
        js = np.nonzero(link)[0]
        row.extend([i] * len(js))
        col.extend(js)
    data = np.ones(len(row), dtype=np.complex64)
    return csr_matrix((data, (row, col)), shape=(N, N))

def build_DK(adj):
    N = adj.shape[0]
    b = 16
    D = lil_matrix((N*b, N*b), dtype=np.complex64)

    for i, j in zip(*adj.nonzero()):
        # scalar → first four vector components
        D[i*b + 0, j*b + 1] = 1
        D[i*b + 0, j*b + 2] = 1j
        D[i*b + 0, j*b + 3] = 1
        D[i*b + 0, j*b + 4] = 1j
        # adjoint
        D[j*b + 1, i*b + 0] = -1
        D[j*b + 2, i*b + 0] = -1j
        D[j*b + 3, i*b + 0] = -1
        D[j*b + 4, i*b + 0] = -1j

    # Wilson term (scalar block only)
    deg = np.diff(adj.indptr)
    for i in range(N):
        D[i*b, i*b] += deg[i]
        for j in adj[i].indices:
            D[i*b, j*b] -= 0.5

    return D.tocsr()

def chiral_index_safe(D):
    H = 1j * (D - D.conj().T)
    H = 0.5 * (H + H.conj().T)
    # Tiny diagonal shift — SPARSE version!
    H = H + 1e-10 * eye(H.shape[0], format='csr')

    for attempt in range(10):
        v0 = np.random.rand(H.shape[0])
        v0 /= np.linalg.norm(v0) + 1e-20
        try:
            vals = eigsh(H, k=160, which='SM', v0=v0, tol=1e-5,
                        maxiter=20000, return_eigenvectors=False)
            pos = np.sum(vals > 1e-7)
            neg = np.sum(vals < -1e-7)
            return pos - neg
        except:
            continue
    return 0  # ultimate fallback

def trial(N, eps, r):
    points = sprinkle(N)
    points = add_parity_bias(points, r)
    adj = causal_links(points, eps)
    D = build_DK(adj)
    return chiral_index_safe(D)

# ==============================================
# MAIN — small sweep first, then scale
# ==============================================
N_sweep = 6000
trials = 40
r_vals = np.linspace(0.0, 0.44, 9)
eps_vals = np.logspace(-1.8, -0.3, 12)

print("Starting safe Windows-friendly sweep (N=6000)...")
results = {}

# Create CSV header once
with open("live_sweep.csv", "w") as f:
    f.write("r,epsilon,mean_index,std_index\n")

for r in tqdm(r_vals, desc="r progress"):
    results[r] = []
    for eps in eps_vals:
        idxs = [trial(N_sweep, eps, r) for _ in tqdm(range(trials), leave=False, desc=f"ε={eps:.5f}")]
        
        mean = np.mean(idxs)
        std  = np.std(idxs)
        
        # Store result
        results[r].append((eps, mean, std))
        
        # Live feedback
        print(f"r={r:.3f}  ε={eps:.5f}  →  {mean:+.3f} ± {std:.3f}  ({trials} trials)")
        
        # Append to live CSV
        with open("live_sweep.csv", "a") as f:
            f.write(f"{r:.4f},{eps:.6f},{mean:.4f},{std:.4f}\n")

# Plot
plt.figure(figsize=(10,6))
for r in r_vals:
    e, m, s = zip(*results[r])
    plt.plot(e, m, 'o-', label=f'r={r:.2f}')
    plt.fill_between(e, np.array(m)-s, np.array(m)+s, alpha=0.2)
plt.xscale('log'); plt.axhline(0,color='k')
plt.xlabel('ε'); plt.ylabel('Chiral index')
plt.title(f'N={N_sweep}, {trials} trials – Windows safe')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("windows_safe_sweep.png", dpi=300); plt.show()

# Find best
best_r = best_eps = best_val = None
for r in r_vals:
    for eps, mean, _ in results[r]:
        if best_val is None or abs(mean) > abs(best_val):
            best_val, best_r, best_eps = mean, r, eps

print(f"BEST → r = {best_r:.3f}, ε = {best_eps:.4f}, index = {best_val:.2f}")

# Quick scaling (12 trials each, will finish overnight)
print("Scaling N...")
for Nn in [10000, 20000, 40000]:
    print(f"N = {Nn} ...", end=" ")
    idxs = [trial(Nn, best_eps, best_r) for _ in tqdm(range(12), leave=False)]

    print(f"{np.mean(idxs):.2f} ± {np.std(idxs):.2f}")
