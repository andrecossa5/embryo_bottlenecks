"""
Spatial autocorrelation of single early mutations, across all specimens: left kidney,
right kidney, heart, liver and brain.

Same question as muts_in_space.py - is the AF of an individual first-divisions mutation
spatially structured within an organ? - but computed the way Hotspot does it
(DeTomaso & Yosef 2021, github.com/YosefLab/Hotspot), which differs from the Moran's I
of muts_in_space.py in the two places that matter here.

1) THE NULL IS A MODEL OF THE MEASUREMENT, NOT A PERMUTATION.
   Moran's I permutes the AFs across LCM samples, which treats every AF as if it were
   known exactly. It is not: an AF from an LCM covered at 12 reads is a far noisier
   number than one covered at 90, and the AFs here are computed from NV/NR with NR
   varying several-fold across samples. Hotspot's answer is to give every observation
   its own null mean and variance from a model of the counts (its danb / bernoulli
   models condition on each cell's UMI depth), and to standardize against that before
   measuring any spatial signal. The direct analogue for LCM data is a beta-binomial
   on NV given NR: sampling noise that scales as p(1-p)/NR, plus a per-mutation
   overdispersion for the real cell-to-cell spread. A low-coverage sample then simply
   carries less weight, instead of contributing a noisy AF at face value.

2) INFERENCE IS CLOSED-FORM.
   Once the values are model-standardized, E[G] = 0 and Var[G] = sum(w^2) exactly, so
   the z-score is analytic and no permutation is needed. This matters at these n: a
   permutation test cannot return a p below 1/(n_perm+1), and that floor is what a
   BH correction runs into first.
   The flip side is that turning that Z into a p-value assumes G is normally
   distributed, which is an asymptotic result: Hotspot is normally run on 10^4-10^5
   cells, against the 11-137 LCM samples here. It does not hold at this n. The null of
   G is right-skewed (the AFs are near-zero for most samples, so z is skewed and G is a
   sum of products of skewed terms), which makes its upper tail heavier than normal and
   the analytic p roughly 2-4x too small - see median_p_ratio in the organ summary.
   So every analytic p is checked against a permutation null of the same statistic
   (calibrate=True) and both are reported, but PVAL_PERM / FDR_PERM ARE THE ONES TO
   REPORT. Pval is kept only as the calibration reference.

What the statistic is, in one line: G = sum over neighboring pairs of w_ij * z_i * z_j,
i.e. how much model-standardized AF covaries between physically adjacent LCM samples.
Note that this is one number per mutation per organ - like the global Moran's I of
muts_in_space.py, and unlike a per-sample LISA. C is its effect size, scaled to [-1,1].
"""

import os
import numpy as np
import pandas as pd
import plotting_utils as plu
from scipy.optimize import minimize
from scipy.stats import betabinom, norm
from sklearn.neighbors import NearestNeighbors
from statsmodels.sandbox.stats.multicomp import multipletests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('macOSX')
plu.set_rcParams()


##


def neighbors_and_weights(coords, n_neighbors=10, neighborhood_factor=3):
    """
    kNN graph with Hotspot's adaptive gaussian kernel (hotspot/knn.py).

    The kernel width for sample i is its own distance to neighbor
    ceil(n_neighbors/neighborhood_factor), so the neighborhood adapts to how densely
    that part of the organ was sampled - the sparse-sampling problem that power_filter()
    had to handle by exclusion in AOC_brain_liver.py is here handled by the weights.
    Weights are row-normalized to sum to 1.

    Returns (neighbors, weights), both (n, n_neighbors); self is excluded.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute').fit(coords)
    dist, ind = nbrs.kneighbors()  # self excluded

    radius_ii = int(np.ceil(n_neighbors/neighborhood_factor))
    sigma = dist[:,[radius_ii-1]]
    sigma[sigma == 0] = 1
    weights = np.exp(-1 * dist**2 / sigma**2)
    wnorm = weights.sum(axis=1, keepdims=True)
    wnorm[wnorm == 0] = 1.0
    weights = weights / wnorm

    return ind, weights


##


def make_weights_non_redundant(neighbors, weights):
    """
    Port of hotspot/knn.py. If i and j are mutual neighbors the edge appears twice;
    fold w_ji into w_ij and zero the duplicate, so every undirected edge is counted once.
    """
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


def edge_arrays(neighbors, weights):
    """
    (i, j, w) of the non-zero edges, and the (n,n) dense weight matrix. The dense form
    is what makes the permutation calibration cheap: G = z @ W @ z.
    """
    n = neighbors.shape[0]
    i = np.repeat(np.arange(n), neighbors.shape[1])
    j = neighbors.ravel()
    w = weights.ravel()
    keep = w > 0
    i, j, w = i[keep], j[keep], w[keep]
    W = np.zeros((n,n))
    np.add.at(W, (i,j), w)

    return i, j, w, W


def compute_node_degree(i, j, w, n):
    """Port of hotspot/knn.py: D_i = sum of the weights of every edge touching i."""
    D = np.zeros(n)
    np.add.at(D, i, w)
    np.add.at(D, j, w)
    return D


##


def fit_betabinom(NV, NR, eps=1e-6):
    """
    Per-mutation beta-binomial fit of NV given NR, by maximum likelihood over
    (p, rho): p the organ-wide AF of the mutation, rho the overdispersion
    (rho -> 0 is the plain binomial, i.e. all spread is sequencing noise).

    This is the LCM analogue of Hotspot's danb_model: a mean shared across samples and
    a variance that is a function of each sample's own depth.

    Returns (p, rho, mu, var) with mu and var per sample, ON THE AF SCALE:
        mu_i  = p
        var_i = p(1-p)/NR_i * (1 + (NR_i-1)*rho)
    so a sample sequenced deeply gets a small null variance and a shallow one a large
    one, and standardizing by it is what downweights the noisy samples.
    """

    NV = np.asarray(NV, dtype=float)
    NR = np.asarray(NR, dtype=float)

    def nll(theta):
        p = 1/(1+np.exp(-theta[0]))
        rho = 1/(1+np.exp(-theta[1]))
        p = np.clip(p, eps, 1-eps)
        rho = np.clip(rho, eps, 1-eps)
        s = (1-rho)/rho
        a, b = p*s, (1-p)*s
        return -betabinom.logpmf(NV, NR, a, b).sum()

    p0 = np.clip(NV.sum()/NR.sum(), eps, 1-eps)
    best = None
    for rho0 in [0.01, 0.1, 0.5]:  # multi-start: the likelihood is flat in rho at low n
        theta0 = [np.log(p0/(1-p0)), np.log(rho0/(1-rho0))]
        try:
            res = minimize(nll, theta0, method='Nelder-Mead',
                           options={'maxiter':2000, 'xatol':1e-6, 'fatol':1e-6})
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res

    p = float(np.clip(1/(1+np.exp(-best.x[0])), eps, 1-eps))
    rho = float(np.clip(1/(1+np.exp(-best.x[1])), eps, 1-eps))

    mu = np.full(len(NR), p)
    var = p*(1-p)/NR * (1 + (NR-1)*rho)

    return p, rho, mu, var


##


def hotspot_autocorrelation(
    x, mu, var, neighbors, weights, W, D, observed=None, calibrate=True,
    n_perm=10000, random_state=1234
    ):
    """
    Hotspot's centered local autocorrelation (hotspot/local_stats.py, the centered=True
    path that compute_autocorrelations() actually uses).

    x, mu, var : observed values and their per-sample null mean / variance
    neighbors, weights, W, D : the non-redundant kNN graph, its dense form, node degrees
    observed   : optional boolean mask; samples with no coverage for this mutation get
                 z_i = 0, so they neither contribute to G nor to its normalization,
                 which is the right thing for a missing measurement (as opposed to a
                 measured zero). They stay in the graph, so their neighbors keep theirs.

    z_i = (x_i - mu_i)/sd_i, then
        G  = sum_ij w_ij z_i z_j     (over the undirected edges)
        Z  = G / sqrt(sum w_ij^2)    (since E[G]=0, Var[G]=sum w^2 for standardized z)
        C  = G / (sum_i D_i z_i^2/2) effect size in [-1,1], comparable across mutations
             and organs - C is the number to compare, Z only says how sure we are.

    With calibrate=True the same G is also referred to a permutation null (z shuffled
    across LCM samples, graph held fixed), returning Z_perm and p_perm. At n in the
    hundreds the analytic Var[G] is an asymptotic result; this is the check on it.
    """

    sd = np.sqrt(var)
    sd[sd == 0] = 1
    z = (x-mu) / sd
    if observed is not None:
        z = np.where(observed, z, 0.0)

    G = float(z @ W @ z)
    Wtot2 = float((weights**2).sum())
    Z = G / np.sqrt(Wtot2)
    G_max = float((D * z**2).sum() / 2)
    C = G / G_max if G_max > 0 else np.nan

    out = {
        'G' : G, 'Z' : Z, 'C' : C, 'Pval' : float(norm.sf(Z)),
        'Z_perm' : np.nan, 'Pval_perm' : np.nan, 'sd_G_perm' : np.nan,
        'sd_G_analytic' : np.sqrt(Wtot2)
    }

    if calibrate:
        rng = np.random.default_rng(random_state)
        Zp = rng.permuted(np.tile(z, (n_perm,1)), axis=1)
        G_null = ((Zp @ W) * Zp).sum(axis=1)
        out['sd_G_perm'] = float(G_null.std(ddof=1))
        out['Z_perm'] = float((G-G_null.mean()) / G_null.std(ddof=1))
        out['Pval_perm'] = float((np.sum(G_null >= G) + 1) / (n_perm + 1))

    return out


##


def load_counts(path_csv, mut_ids, path_cache=None, chunksize=5_000_000):
    """
    (NV, NR) matrices, Sample_ID x mutation_id, restricted to mut_ids. The mutation
    tables run up to 2.7 GB, so they are read in chunks and the subset cached to disk.
    """
    cols = ['mutation_id', 'Sample_ID', 'NV', 'NR', 'VAF']
    if path_cache is not None and os.path.exists(path_cache):
        df = pd.read_csv(path_cache, usecols=cols)
    else:
        L = []
        for chunk in pd.read_csv(path_csv, usecols=cols, chunksize=chunksize):
            L.append(chunk[chunk['mutation_id'].isin(mut_ids)])
        df = pd.concat(L)
        if path_cache is not None:
            df.to_csv(path_cache, index=False)

    NV = df.pivot_table(index='Sample_ID', columns='mutation_id', values='NV', fill_value=0)
    NR = df.pivot_table(index='Sample_ID', columns='mutation_id', values='NR', fill_value=0)
    NV = NV.reindex(columns=list(mut_ids), fill_value=0.0)
    NR = NR.reindex(columns=list(mut_ids), fill_value=0.0)

    return NV, NR


##


# Paths
path_main = '/Users/cossa/Desktop/projects/embryo_bottlenecks/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results')
path_figures = os.path.join(path_main, 'figures')
path_cache = os.path.join(path_results, 'cache')
os.makedirs(path_cache, exist_ok=True)


##


# Early cell division mutations (as in muts_in_space.py)
branches = [
    "A", "A", "B", "C", "C", "D", "E", "E", "F",
    "G", "H", "H", "I", "I", "J", "J", "J", "J"
]
ids = [
    "chr10_128568737_C_T",
    "chr3_85532662_T_C",
    "chr18_69326508_G_A",
    "chr11_96885133_A_G",
    "chr1_12414725_C_A",
    "chr6_50196398_C_T",
    "chr7_46281686_G_C",
    "chr17_79695509_G_C",
    "chr8_15505436_G_A",
    "chr1_194995181_C_A",
    "chr2_187163463_T_C",
    "chr5_156155901_C_G",
    "chr14_95974143_C_T",
    "chr7_93731124_C_G",
    "chr11_132296858_C_T",
    "chr18_37006496_T_A",
    "chr3_194592596_C_T",
    "chr8_143635887_G_A"
]
interesting_muts = pd.DataFrame({'mutation_id':ids, 'Branch':branches})


##


def get_specimen(organ):
    """
    (NV, NR, spatial_coords) for one specimen, all indexed the same way. Coordinates and
    counts live in different files per specimen: the kidneys have their own coordinate
    file, the heart the 3D one also used by spatial_association.py, brain and liver a
    single annotation file, with their counts in the embryo-wide mutation table.
    """

    if organ in ['PD53943o', 'PD53943w', 'Heart']:

        NV, NR = load_counts(
            os.path.join(path_data, f'{organ}_metadata.csv'), ids,
            path_cache=os.path.join(path_cache, f'{organ}_18muts.csv')
        )
        if organ == 'Heart':
            coords = (
                pd.read_csv(os.path.join(path_data, 'Heart_final_coorindates_135.csv'))
                .set_index('name')
            )
            coord_cols = ['x', 'y', 'z']  # Heart was sectioned in 3D
        else:
            coords = (
                pd.read_csv(os.path.join(path_data, f'{organ}_coordinates.csv'))
                .set_index('name')
            )
            coord_cols = ['x', 'y']

    elif organ in ['Brain', 'Liver']:

        NV, NR = load_counts(
            os.path.join(path_data, 'Final_Dataframe_heart_annotations_raw_trophoblasts.csv'),
            ids, path_cache=os.path.join(path_cache, 'brain_liver_18muts.csv')
        )
        coords = (
            pd.read_csv(os.path.join(path_data, 'df_brain_liver_annotations_unique_for_Andrea.csv'))
            .dropna(subset=['x', 'y'])
            .query('Bulk_phenotype == @organ')
            .set_index('Sample_ID')
        )
        coord_cols = ['x', 'y']

    else:
        raise ValueError(f'Unknown specimen {organ}')

    # Only samples with both coordinates and counts
    common = NV.index.intersection(coords.index)
    NV = NV.loc[common]
    NR = NR.loc[common]
    coords = coords.loc[common, coord_cols]
    assert np.all(NV.index == coords.index)

    return NV, NR, coords


##


SPECIMENS = {
    'PD53943o' : 'Left kidney',
    'PD53943w' : 'Right kidney',
    'Heart' : 'Heart',
    'Liver' : 'Liver',
    'Brain' : 'Brain',
}
# Hotspot's default is 30 neighbors on 10^4-10^5 cells; at n=11-137 that would be most
# of the organ, so the scan is over small k, capped at n/3.
K_VALUES = [5, 10, 15, 20]
K_MAIN = 10
N_PERM = 50000  # p_perm floor 2e-5; the top hits need tail resolution below 1e-4

L = []
data = {}
k_main = {}
fits = {}  # (organ, mutation) -> beta-binomial fit; the null does not depend on k

for organ, name in SPECIMENS.items():

    NV, NR, coords = get_specimen(organ)
    data[organ] = (NV, NR, coords)
    n = NV.shape[0]
    print(f'{name}: n={n} LCM samples, median NR={np.median(NR.values):.0f}')

    # Neighborhoods are capped at n/3, past which 'local' stops meaning anything. The
    # liver (n=11) has no k satisfying that and is run at the smallest k instead: it is
    # kept for completeness and should not be read as evidence either way.
    ks = [ k for k in K_VALUES if k <= n//3 ] or [min(K_VALUES)]
    k_main[organ] = min(K_MAIN, max(ks))
    if max(ks) > n//3:
        print(f'  WARNING: {name} has n={n}, too few LCM samples for a local '
              f'neighborhood; running at k={max(ks)} ({max(ks)/n:.0%} of the organ)')

    for k in ks:

        neighbors, weights = neighbors_and_weights(coords.values, n_neighbors=k)
        weights = make_weights_non_redundant(neighbors, weights)
        i_e, j_e, w_e, W = edge_arrays(neighbors, weights)
        D = compute_node_degree(i_e, j_e, w_e, n)

        for ID in ids:

            nv = NV[ID].values.astype(float)
            nr = NR[ID].values.astype(float)
            ok = nr > 0  # Covered here; NR==0 is a missing measurement, not an AF of 0
            vaf = np.divide(nv, nr, out=np.zeros_like(nv), where=ok)
            row = {
                'mutation_id' : ID, 'organ' : name, 'sample' : organ, 'k' : k, 'n' : n,
                'n_covered' : int(ok.sum()), 'n_positive' : int((nv>0).sum()),
                'median_NR' : float(np.median(nr[ok])) if ok.any() else np.nan,
                'mean_VAF' : float(vaf[ok].mean()) if ok.any() else np.nan
            }

            # Never detected in this organ: the beta-binomial degenerates at p=0 and
            # there is no variation to correlate. Reported, not tested.
            if nv.sum() == 0 or ok.sum() < 5:
                row.update({
                    'p_hat':np.nan, 'rho':np.nan, 'G':np.nan, 'Z':np.nan, 'C':np.nan,
                    'Pval':np.nan, 'Z_perm':np.nan, 'Pval_perm':np.nan,
                    'sd_G_perm':np.nan, 'sd_G_analytic':np.nan
                })
                L.append(row)
                continue

            if (organ, ID) not in fits:
                fits[(organ, ID)] = fit_betabinom(nv[ok], nr[ok])
            p_hat, rho, _, _ = fits[(organ, ID)]
            mu = np.full(n, p_hat)
            var = np.where(ok, p_hat*(1-p_hat)/np.where(ok, nr, 1) * (1+(nr-1)*rho), 1)

            res = hotspot_autocorrelation(
                vaf, mu, var, neighbors, weights, W, D, observed=ok,
                calibrate=True, n_perm=N_PERM
            )
            row.update({'p_hat':p_hat, 'rho':rho})
            row.update(res)
            L.append(row)

df = pd.DataFrame(L).merge(interesting_muts, on='mutation_id', how='left')

# BH within specimen and k: the 18 mutations are the family of tests
for col, out in [('Pval', 'FDR'), ('Pval_perm', 'FDR_perm')]:
    df[out] = np.nan
    for (s, k), x in df.groupby(['sample', 'k']):
        v = x[col].notna()
        if v.any():
            df.loc[x.index[v], out] = \
                multipletests(x.loc[v, col], alpha=0.05, method='fdr_bh')[1]

# Specimen-level readout, each at its reported k
k_main_by_name = { SPECIMENS[s]:k for s,k in k_main.items() }
df['k_main'] = df['organ'].map(k_main_by_name)
df.to_csv(os.path.join(path_results, 'hotspot_autocorrelation_single_muts.csv'), index=False)

main = df.query('k == k_main')

organ_summary = (
    main
    .groupby('organ')
    .apply(lambda x: pd.Series({
        'n' : x['n'].iloc[0],
        'k' : x['k'].iloc[0],
        'n_muts_tested' : x['Z'].notna().sum(),
        'n_sig_analytic' : (x['FDR']<=0.05).sum(),
        'n_sig_perm' : (x['FDR_perm']<=0.05).sum(),
        'median_C' : x['C'].median(),
        'max_C' : x['C'].max(),
        # Calibration of the closed-form inference against the permutation null.
        # sd_ratio compares only the SPREAD of the two nulls and is not by itself
        # enough to judge the p-values: the null of G is right-skewed, so the normal
        # tail can be wrong even where the two sds agree. p_ratio = Pval/Pval_perm is
        # the diagnostic that matters, and it is < 1 here, i.e. the analytic p is
        # ANTIconservative. Report Pval_perm / FDR_perm.
        'median_sd_ratio' : (x['sd_G_perm']/x['sd_G_analytic']).median(),
        'median_p_ratio' : (x['Pval']/x['Pval_perm']).median(),
        'min_p_ratio' : (x['Pval']/x['Pval_perm']).min(),
    }), include_groups=False)
)
organ_summary.to_csv(os.path.join(path_results, 'hotspot_autocorrelation_organ_summary.csv'))
print(organ_summary)

print('\nTop autocorrelated mutations (at each specimen\'s reported k):')
print(
    main.sort_values('C', ascending=False)
    [['organ','mutation_id','Branch','n_positive','p_hat','rho','C','Z','Pval','FDR',
      'Z_perm','Pval_perm','FDR_perm']]
    .head(15).to_string(index=False)
)


##


# Fig: calibration of the analytic z-score against the permutation null. Points on the
# diagonal are mutations where Hotspot's closed-form inference holds at this n.
fig, ax = plt.subplots(figsize=(3.2,3.2), constrained_layout=True)
x = main.dropna(subset=['Z','Z_perm'])
colors = plu.create_palette(x, 'organ', palette='tab10')
for organ, d in x.groupby('organ'):
    ax.scatter(d['Z'], d['Z_perm'], s=12, color=colors[organ], label=organ, alpha=.8)
lims = [min(x['Z'].min(), x['Z_perm'].min())-.5, max(x['Z'].max(), x['Z_perm'].max())+.5]
ax.plot(lims, lims, 'k--', linewidth=.5)
ax.legend(frameon=False, fontsize=6)
plu.format_ax(ax=ax, xlabel='Z (analytic)', ylabel='Z (permutation)',
              title='Calibration of the closed-form null', title_size=8)
fig.savefig(os.path.join(path_figures, 'hotspot_Z_calibration.pdf'))
plt.close(fig)


##


# Fig: C per mutation and specimen, at each specimen's reported k
piv = main.pivot_table(index='mutation_id', columns='organ', values='C')
piv = piv.loc[[ i for i in ids if i in piv.index ]]
sig = main.pivot_table(index='mutation_id', columns='organ', values='FDR_perm').reindex(piv.index)

fig, ax = plt.subplots(figsize=(4.5,5), constrained_layout=True)
ax.set_facecolor('#e6e6e6')  # grey = not detected in that organ, distinct from C=0
im = ax.imshow(np.ma.masked_invalid(piv.values), cmap='RdBu_r',
               vmin=-np.nanmax(np.abs(piv.values)),
               vmax=np.nanmax(np.abs(piv.values)), aspect='auto')
for r in range(piv.shape[0]):
    for c in range(piv.shape[1]):
        if sig.values[r,c] <= 0.05:
            ax.text(c, r, '*', ha='center', va='center', fontsize=8)
ax.set_xticks(range(piv.shape[1]))
ax.set_xticklabels(piv.columns, rotation=90, fontsize=6)
ax.set_yticks(range(piv.shape[0]))
ax.set_yticklabels(
    [ f'{i} ({interesting_muts.set_index("mutation_id").loc[i,"Branch"]})' for i in piv.index ],
    fontsize=6
)
fig.colorbar(im, ax=ax, shrink=.4, label='C (autocorrelation)')
ax.set_title('Spatial autocorrelation of early mutations\n(* FDR<0.05, permutation)', fontsize=8)
fig.savefig(os.path.join(path_figures, 'hotspot_autocorrelation_C.pdf'))
plt.close(fig)


##


# Fig: sensitivity to k
fig, axs = plt.subplots(1, len(SPECIMENS), figsize=(11,2.4), constrained_layout=True)

for ax, (organ, name) in zip(axs, SPECIMENS.items()):
    d = df.query('organ == @name')
    for ID, x in d.groupby('mutation_id'):
        ax.plot(x['k'], x['C'], marker='o', markersize=2, linewidth=.5, color='grey')
    med = d.groupby('k')['C'].median()
    ax.plot(med.index, med.values, marker='o', markersize=3, color='r')
    ax.axhline(0, color='k', linestyle='--', linewidth=.5)
    plu.format_ax(ax=ax, title=f'{name} (n={int(d["n"].iloc[0])})', xlabel='k',
                  ylabel='C', title_size=8)

fig.savefig(os.path.join(path_figures, 'hotspot_autocorrelation_k_sensitivity.pdf'))
plt.close(fig)


##


# Fig: AF maps, with the autocorrelation result on each panel - the muts_in_space.py
# figure, recomputed. One figure per specimen.
for organ, name in SPECIMENS.items():

    NV, NR, coords = data[organ]
    k = k_main[organ]
    d = df.query('sample == @organ and k == @k').set_index('mutation_id')

    fig = plt.figure(figsize=(8,5))
    for i, ID in enumerate(ids):

        ax = fig.add_subplot(3, 6, i+1)
        nv = NV[ID].values.astype(float)
        nr = NR[ID].values.astype(float)
        df_plot = coords[['x','y']].copy()
        df_plot['mut'] = np.divide(nv, nr, out=np.zeros_like(nv), where=nr>0)
        plu.scatter(
            df_plot, 'x', 'y', by='mut', continuous_cmap='Blues', ax=ax,
            size=5, kwargs={'edgecolor':'k', 'linewidth':.01}
        )
        C = d.loc[ID,'C']
        q = d.loc[ID,'FDR_perm']
        if np.isnan(C):
            lab = 'n.d.'  # not detected in this organ
        else:
            lab = f'C={C:.2f}' + ('*' if q <= 0.05 else '')
        plu.format_ax(ax=ax, title=f'{ID}\n{d.loc[ID,"Branch"]} ({lab})', title_size=6)
        ax.axis('off')

    fig.suptitle(f'{name} - AF, C = Hotspot autocorrelation (k={k}, * FDR<0.05)')
    fig.subplots_adjust(top=.85, bottom=.1, left=.1, right=.9, wspace=.8, hspace=.5)
    fig.savefig(os.path.join(path_figures, f'{organ}_hotspot_autocorrelation.pdf'))
    plt.close(fig)


##
