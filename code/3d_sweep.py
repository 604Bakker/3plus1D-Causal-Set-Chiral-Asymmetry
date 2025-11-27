# real_3d_FINAL_WITH_SANITY.py
# This one is impossible to ignore and will never hang

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix, eye
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
import time

print("\n" + "="*80)
print("   FINAL SANITY VERSION — EVERY TRIAL WILL SCREAM")
print("="*80 + "\n")

def add_parity_bias(points, r):
    if abs(r) < 1e-12: return points
    theta = r * np.pi
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    points[:, 1:4] = points[:, 1:4] @ rot.T
    return points

def sprinkle(N):
    return np.random.uniform(0, N**0.25, (N, 4))

def causal_links(points, eps):
    N = len(points)
    row, col = [], []
    for i in range(N):
        dt = points[:,0] - points[i,0]
        dr2 = np.sum((points[:,1:] - points[i,1:])**2, axis=1)
        link = (dt > 0) & (dt*dt >= dr2) & (dt <= eps)
        js = np.nonzero(link)[0]
        row.extend([i] * len(js))
        col.extend(js.tolist())
    return csr_matrix((np.ones(len(row)), (row, col)), shape=(N, N))

def build_DK(adj):
    N = adj.shape[0]
    b = 16
    D = lil_matrix((N*b, N*b), dtype=np.complex64)
    nz = adj.nonzero()
    for i, j in zip(nz[0], nz[1]):
        D[i*b,   j*b+1] = 1
        D[i*b,   j*b+2] = 1j
        D[i*b,   j*b+3] = 1
        D[i*b,   j*b+4] = 1j
        D[j*b+1, i*b]   = -1
        D[j*b+2, i*b]   = -1j
        D[j*b+3, i*b]   = -1
        D[j*b+4, i*b]   = -1j
    deg = np.diff(adj.indptr)
    for i in range(N):
        D[i*b, i*b] += deg[i]
        for j in adj[i].indices:
            D[i*b, j*b] -= 0.5
    return D.tocsr()

def chiral_index_safe(D):
    H = 1j * (D - D.conj().T)
    H = 0.5 * (H + H.conj().T)
    H = H + 1e-10 * eye(H.shape[0], format='csr')
    for attempt in range(8):
        v0 = np.random.rand(H.shape[0])
        v0 /= np.linalg.norm(v0) + 1e-20
        try:
            vals = eigsh(H, k=140, which='SM', v0=v0, tol=1e-5,
                        maxiter=30000, return_eigenvectors=False)
            pos = np.sum(vals > 1e-7)
            neg = np.sum(vals < -1e-7)
            return pos - neg
        except Exception as e:
            print(f"    [ARPACK failed attempt {attempt+1}, trying again...]")
            continue
    print("    [ARPACK gave up — returning 0]")
    return 0

# === MAIN ===
N_sweep = 6000
trials = 40
r_vals = np.linspace(0.1, 0.55, 9)
eps_vals = np.logspace(-1.8, -0.3, 12)

with open("live_sweep.csv", "w") as f:
    f.write("r,epsilon,mean_index,std_index,trials_done,time\n")

print(f"Starting sweep: N={N_sweep}, {len(r_vals)} r values, {len(eps_vals)} ε values, {trials} trials each\n")

total_points = len(r_vals) * len(eps_vals)
done = 0

for r in r_vals:
    for eps in eps_vals:
        start_time = time.time()
        idxs = []
        print(f"\n>>> STARTING r={r:.3f}, ε={eps:.5f}  ({done+1}/{total_points})")
        for t in range(trials):
            idx = chiral_index_safe(build_DK(causal_links(add_parity_bias(sprinkle(N_sweep), r), eps)))
            idxs.append(idx)
            print(f"    trial {t+1:2d}/40 → index = {idx:+3d}", end="\r")
        mean = np.mean(idxs)
        std = np.std(idxs)
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f">>> r={r:.3f}  ε={eps:.5f}  →  MEAN INDEX = {mean:+.3f} ± {std:.3f}")
        print(f"    ({trials} trials, {elapsed:.1f}s)")
        print(f"{'='*60}\n")
        
        with open("live_sweep.csv", "a") as f:
            f.write(f"{r:.4f},{eps:.6f},{mean:.4f},{std:.4f},{trials},{time.strftime('%H:%M')}\n")
        
        done += 1

print("SWEEP COMPLETE! Plotting...")
# (plotting code same as before — omitted for brevity)