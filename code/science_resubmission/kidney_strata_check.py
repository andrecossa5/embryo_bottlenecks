"""
Is the kidneys' null result a sampling-unit effect?

autocorrelation_single_muts.py finds spatial autocorrelation of early mutations in the
heart (10/15 mutations) but essentially none in either kidney (1/16 and 0/17). The
specimens do not sample the same kind of object, though:

  heart        : contiguous anatomical territories (ventricle, IVS, DMP, outflow, atria)
  kidney       : interdigitated cell-type classes - 58-68% of the LCMs are individual
                 glomeruli, each derived from its own nephron-progenitor condensation

So two spatially adjacent glomeruli need not be clonally adjacent, and a kNN graph over
a mixed-cell-type sample may be connecting material that was never clonally continuous.
If so, the whole-organ null would be an artifact of what was sampled rather than
evidence that early clonal structure is absent from the kidney.

This script tests that by re-running the same statistic within a single cell-type
stratum (glomeruli only, and non-glomerular only), where every sample is the same kind
of object. Plus a power control: the heart subsampled to the same n, to check that an
effect of the size seen in the heart would still be detectable at n=63.

Reading the result:
  signal appears within a stratum -> the whole-organ null was a sampling-unit effect
  still nothing, and the heart control retains power at that n
                                  -> the kidney null is real, and the heart/kidney
                                     difference is about development, not sampling
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import betabinom
from sklearn.neighbors import NearestNeighbors
from statsmodels.sandbox.stats.multicomp import multipletests


##


# Helpers, duplicated from autocorrelation_single_muts.py (that module is a flat script
# and executes its whole analysis on import). See it for the documentation.

def neighbors_and_weights(coords, n_neighbors=10, neighborhood_factor=3):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute').fit(coords)
    dist, ind = nbrs.kneighbors()
    radius_ii = int(np.ceil(n_neighbors/neighborhood_factor))
    sigma = dist[:,[radius_ii-1]]
    sigma[sigma == 0] = 1
    weights = np.exp(-1 * dist**2 / sigma**2)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return ind, weights


def make_weights_non_redundant(neighbors, weights):
    w = weights.copy()
    for i in range(neighbors.shape[0]):
        for k in range(neighbors.shape[1]):
            j = neighbors[i,k]
            if j < i:
                continue
            for k2 in range(neighbors.shape[1]):
                if neighbors[j,k2] == i:
                    w[i,k] += w[j,k2]
                    w[j,k2] = 0
    return w


def build_graph(coords, k):
    """(W, D, Wtot2) for the non-redundant adaptive-kernel kNN graph."""
    neighbors, weights = neighbors_and_weights(coords, n_neighbors=k)
    weights = make_weights_non_redundant(neighbors, weights)
    n = neighbors.shape[0]
    i = np.repeat(np.arange(n), neighbors.shape[1])
    j = neighbors.ravel()
    w = weights.ravel()
    keep = w > 0
    W = np.zeros((n,n))
    np.add.at(W, (i[keep], j[keep]), w[keep])
    D = W.sum(axis=0) + W.sum(axis=1)
    return W, D, float((weights**2).sum())


def fit_betabinom(NV, NR, eps=1e-6):
    NV = np.asarray(NV, dtype=float)
    NR = np.asarray(NR, dtype=float)
    def nll(theta):
        p = np.clip(1/(1+np.exp(-theta[0])), eps, 1-eps)
        rho = np.clip(1/(1+np.exp(-theta[1])), eps, 1-eps)
        s = (1-rho)/rho
        return -betabinom.logpmf(NV, NR, p*s, (1-p)*s).sum()
    p0 = np.clip(NV.sum()/NR.sum(), eps, 1-eps)
    best = None
    for rho0 in [0.01, 0.1, 0.5]:
        res = minimize(nll, [np.log(p0/(1-p0)), np.log(rho0/(1-rho0))],
                       method='Nelder-Mead', options={'maxiter':2000})
        if best is None or res.fun < best.fun:
            best = res
    p = float(np.clip(1/(1+np.exp(-best.x[0])), eps, 1-eps))
    rho = float(np.clip(1/(1+np.exp(-best.x[1])), eps, 1-eps))
    return p, rho


def autocorrelation(nv, nr, W, D, n_perm=50000, rng=None):
    """C and the permutation p of Hotspot's G, for one mutation on one graph."""
    rng = np.random.default_rng(1234) if rng is None else rng
    p, rho = fit_betabinom(nv, nr)
    var = p*(1-p)/nr * (1 + (nr-1)*rho)
    z = (nv/nr - p) / np.sqrt(var)
    G = float(z @ W @ z)
    C = G / float((D * z**2).sum() / 2)
    P = rng.permuted(np.tile(z, (n_perm,1)), axis=1)
    G_null = ((P @ W) * P).sum(axis=1)
    return C, float((np.sum(G_null >= G) + 1) / (n_perm + 1)), p, rho


##


path_main = '/Users/cossa/Desktop/projects/embryo_bottlenecks/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results')
path_cache = os.path.join(path_results, 'cache')

ids = [
    "chr10_128568737_C_T", "chr3_85532662_T_C", "chr18_69326508_G_A",
    "chr11_96885133_A_G", "chr1_12414725_C_A", "chr6_50196398_C_T",
    "chr7_46281686_G_C", "chr17_79695509_G_C", "chr8_15505436_G_A",
    "chr1_194995181_C_A", "chr2_187163463_T_C", "chr5_156155901_C_G",
    "chr14_95974143_C_T", "chr7_93731124_C_G", "chr11_132296858_C_T",
    "chr18_37006496_T_A", "chr3_194592596_C_T", "chr8_143635887_G_A"
]
K = 10
N_PERM = 50000


def load(sample):
    """(NV, NR, coords, histo) for one specimen, all aligned."""
    cnt = pd.read_csv(
        os.path.join(path_cache, f'{sample}_18muts.csv'),
        usecols=['mutation_id', 'Sample_ID', 'NV', 'NR']
    )
    NV = cnt.pivot_table(index='Sample_ID', columns='mutation_id', values='NV', fill_value=0)
    NR = cnt.pivot_table(index='Sample_ID', columns='mutation_id', values='NR', fill_value=0)
    if sample == 'Heart':
        coords = pd.read_csv(
            os.path.join(path_data, 'Heart_final_coorindates_135.csv')).set_index('name')
        cols = ['x','y','z']
    else:
        coords = pd.read_csv(
            os.path.join(path_data, f'{sample}_coordinates.csv')).set_index('name')
        cols = ['x','y']
    histo = (
        pd.read_csv(os.path.join(path_data, f'{sample}_metadata.csv'),
                    usecols=['Sample_ID','Histo'])
        .drop_duplicates().set_index('Sample_ID')
    )
    common = NV.index.intersection(coords.index).intersection(histo.index)
    return NV.loc[common], NR.loc[common], coords.loc[common, cols], histo.loc[common,'Histo']


def run(NV, NR, coords, label, k=K):
    """The 18 mutations on one (sub)set of samples. Returns a tidy frame."""
    n = len(NV)
    W, D, _ = build_graph(coords.values, k)
    rng = np.random.default_rng(1234)
    rows = []
    for ID in ids:
        nv = NV[ID].values.astype(float)
        nr = NR[ID].values.astype(float)
        if nv.sum() == 0 or (nr == 0).any():
            rows.append({'stratum':label, 'n':n, 'mutation_id':ID, 'C':np.nan,
                         'p':np.nan, 'p_hat':np.nan, 'rho':np.nan, 'n_positive':0})
            continue
        C, p, p_hat, rho = autocorrelation(nv, nr, W, D, n_perm=N_PERM, rng=rng)
        rows.append({'stratum':label, 'n':n, 'mutation_id':ID, 'C':C, 'p':p,
                     'p_hat':p_hat, 'rho':rho, 'n_positive':int((nv>0).sum())})
    df = pd.DataFrame(rows)
    ok = df['p'].notna()
    df['FDR'] = np.nan
    if ok.any():
        df.loc[ok,'FDR'] = multipletests(df.loc[ok,'p'], alpha=0.05, method='fdr_bh')[1]
    return df


##


# 1. The kidneys, whole organ and within each cell-type stratum
L = []
for sample, name in [('PD53943o','Left kidney'), ('PD53943w','Right kidney')]:

    NV, NR, coords, histo = load(sample)
    print(f'\n{name}: n={len(NV)}  {histo.value_counts().to_dict()}')

    strata = {'All': histo.notna().values, 'Glomerulus': (histo=='Glomerulus').values}
    strata['Non-glomerular'] = ~strata['Glomerulus']

    for label, mask in strata.items():
        if mask.sum() < 3*K:  # need k <= n/3 for the neighborhood to be local
            print(f'  {label:16s} n={mask.sum():3d}  SKIPPED (n < 3k)')
            continue
        df = run(NV[mask], NR[mask], coords[mask], label).assign(specimen=name)
        L.append(df)
        sig = (df['FDR']<=0.05).sum()
        tested = df['p'].notna().sum()
        print(f'  {label:16s} n={mask.sum():3d}  {sig}/{tested} significant  '
              f'median C={df["C"].median():.3f}  max C={df["C"].max():.3f}')

kidney = pd.concat(L)
kidney.to_csv(os.path.join(path_results, 'kidney_strata_autocorrelation.csv'), index=False)


##


# 2. Power control: is n=63 (the left kidney's glomerular stratum) enough to see an
#    effect the size of the heart's? Subsample the heart to that n, repeatedly, and
#    count how many of its 10 whole-organ hits are recovered.
heart_hits = [
    'chr10_128568737_C_T', 'chr3_85532662_T_C', 'chr18_69326508_G_A',
    'chr6_50196398_C_T', 'chr8_15505436_G_A', 'chr2_187163463_T_C',
    'chr14_95974143_C_T', 'chr7_93731124_C_G', 'chr18_37006496_T_A',
    'chr8_143635887_G_A'
]
NV_h, NR_h, coords_h, _ = load('Heart')
rng = np.random.default_rng(0)
N_REP = 20
N_SUB = 63

rec = []
for rep in range(N_REP):
    idx = rng.choice(len(NV_h), size=N_SUB, replace=False)
    df = run(NV_h.iloc[idx], NR_h.iloc[idx], coords_h.iloc[idx], f'heart_sub_{rep}')
    d = df.set_index('mutation_id')
    rec.append({
        'rep' : rep,
        'n_hits_recovered' : int((d.loc[heart_hits,'FDR']<=0.05).sum()),
        'n_hits_nominal' : int((d.loc[heart_hits,'p']<0.05).sum()),
        'median_C_hits' : float(d.loc[heart_hits,'C'].median()),
    })
rec = pd.DataFrame(rec)
rec.to_csv(os.path.join(path_results, 'heart_subsample_power.csv'), index=False)

print(f'\nPower control - heart subsampled to n={N_SUB}, {N_REP} replicates:')
print(f'  of its 10 whole-organ hits, recovered at FDR<0.05: '
      f'median {rec["n_hits_recovered"].median():.0f}  '
      f'(range {rec["n_hits_recovered"].min()}-{rec["n_hits_recovered"].max()})')
print(f'  at nominal p<0.05: median {rec["n_hits_nominal"].median():.0f}  '
      f'(range {rec["n_hits_nominal"].min()}-{rec["n_hits_nominal"].max()})')
print(f'  median C of those mutations: {rec["median_C_hits"].median():.3f} '
      f'(whole-organ value was 0.121)')


##
