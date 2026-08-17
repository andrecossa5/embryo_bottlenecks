"""
Spatial autocorrelation of mutation allele fractions, across all five specimens.

Question: if an early cell division seeded a spatially coherent part of an organ, the AF
of the mutations it carries should be more similar between physically adjacent LCM
samples than between distant ones.

This single script replaces the three that came before it - autocorrelation_single_muts.py
(targeted, 18 early mutations), autocorrelation_all_muts.py (genome-wide scan, kidneys
stratified) and kidney_strata_check.py (strata + power control). They shared a statistic
but each re-assembled its own inputs, with its own pixel handling and its own liver/brain
source. Everything now reads the two tables built by build_metadata_table.py, so a
coordinate fix or a new sample set propagates to every analysis at once.

WHAT IT RUNS

  1. TARGETED   the 18 first-division mutations, every one permuted, BH over the 18
                within (specimen, graph). The confirmatory frame for the pre-specified
                early-mutation question. Report FDR_perm.
  2. SCAN       every mutation with at least MIN_POSITIVE carriers, kidneys split by
                histology, two-stage inference (analytic screen, then permutations for
                the candidates), BH within specimen x stratum. A discovery scan, reported
                at FDR 0.10.
  3. POWER      the heart subsampled to a smaller n, repeatedly, to ask whether an organ
                sampled like the kidneys could have seen a heart-sized effect at all.
  4. EARLY vs REST  are the early mutations more autocorrelated than the others? Rank
                test (threshold-free, preferred) plus Fisher on the thresholded calls.

THE STATISTIC (unchanged - a verified port of Hotspot, DeTomaso & Yosef 2021)

  z = (AF - mu)/sigma          beta-binomial standardization, per sample:
                               mu_i = p, var_i = p(1-p)/NR_i * (1 + (NR_i-1)*rho)
  G = sum_ij w_ij z_i z_j      covariance over the spatial graph
  C = G / (sum_i D_i z_i^2/2)  effect size, ~[-1,1], descriptive only
  Z = G / sqrt(sum w^2)        exact under the model, since E[G]=0 and Var[G]=sum(w^2)

The beta-binomial null is the reason for using Hotspot rather than permuting AF directly:
NR runs 8-63 reads (median 27-44 by organ), and a 12-read AF should not weigh as much as
a 90-read one. Fitted rho is small but non-zero, so var_i is about 1.5x the binomial value
and the depth term still dominates.

INFERENCE. Report FDR_perm, never FDR. The analytic p assumes G is normal; it is not
(z is right-skewed, so G's null is skewed), which makes norm.sf understate the far tail by
2-6x - exactly where calls are made. The analytic p is kept only as the calibration
reference, and in the scan as a screen: it is anticonservative in the tail, so screening
on it cannot discard a mutation the permutation test would have called.

TWO GRAPHS, AND WHY BOTH

  'knn'    Hotspot's adaptive-bandwidth kNN: sigma_i is sample i's own distance to its
           ceil(k/3)-th neighbour. This is the graph the earlier results were computed on,
           kept so the numbers remain comparable to them.
  'sigma'  a Gaussian kernel of FIXED bandwidth in microns over all samples. Slower to
           saturate but it means the same thing in every organ.

The distinction matters and is not cosmetic. An adaptive bandwidth renormalises every
sample to look equally well sampled: measured on this data it reports an effective
neighbourhood of 4-7 samples in EVERY organ, brain and liver included, at a kernel radius
of ~1 mm - it manufactures a neighbourhood where locality does not physically exist. That
is also why C could not previously be compared across specimens (the old notes' §5.2c:
"C is not invariant to sampling density... a fair cross-organ comparison would need
neighbourhoods matched in microns rather than in k, which is blocked by the heart's
coordinate-unit ambiguity"). Both halves of that blocker are now gone - coordinates are in
microns in metadata_table.csv, and the fixed-sigma graph is the matched-in-microns
comparison - so the fixed-sigma results are the ones to use across organs. `ess` (Kish
effective sample size) is reported for every organ and graph so the reader can see what
"local" actually bought.

WHAT CHANGED IN THE INPUTS SINCE THE EARLIER RUNS

  Heart      coordinates were being used raw, mixing a 16x-downsampled x/y pixel grid with
             a z already in microns. Correctly scaled (x,y * 7.36 um; z = section * 16 um)
             about a third of every heart neighbourhood changes membership. The heart
             carries the headline result, so every heart number here supersedes the old one.
  Brain      72 -> 86 samples (FINAL_df_brain_liver_annotations added coordinates for 14).
  Liver      11 -> 26 samples (LongData added 15 with counts). The old runs flagged the
             liver as "not evidence either way, do not report"; at n=26 it is a thin but
             real test, and its coverage is the highest of any organ (median DP 44).
  Kidneys    unchanged (109 and 137).

A CAUTION THAT SURVIVES. The brain's 86 samples sit in 29 spatial islands, 56% of them in
islands of fewer than 5, with the nearest sample from another island 3.7 mm away. The
permutation null conditions on the actual graph, so this cannot manufacture a false
positive - but "local" in the brain means something quite different from "local" in the
heart. Read the brain's ess column before reading its p-values.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import betabinom, norm, mannwhitneyu, fisher_exact
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from statsmodels.sandbox.stats.multicomp import multipletests


##


def neighbors_and_weights(coords, n_neighbors=10, neighborhood_factor=3):
    """
    kNN graph with Hotspot's adaptive gaussian kernel (hotspot/knn.py), verbatim.

    The kernel width for sample i is its own distance to neighbor
    ceil(n_neighbors/neighborhood_factor), so the neighborhood adapts to how densely that
    part of the organ was sampled. See the module docstring on why that adaptivity also
    hides sparsity, and why the fixed-sigma graph exists alongside it.

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


def make_weights_non_redundant(neighbors, weights):
    """
    Port of hotspot/knn.py. If i and j are mutual neighbors the edge appears twice; fold
    w_ji into w_ij and zero the duplicate, so every undirected edge is counted once.
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


##


def kish_ess(W_rownorm):
    """
    Per-sample effective neighbourhood size, (sum w)^2 / sum w^2 on the row-normalised
    weights. How many samples the local average actually averages over: ESS ~ 1 means the
    statistic is essentially "my single nearest sample". Reported, never used to filter.
    """
    s1 = W_rownorm.sum(axis=1)
    s2 = (W_rownorm**2).sum(axis=1)
    return np.divide(s1**2, s2, out=np.zeros_like(s1), where=s2 > 0)


def build_knn_graph(coords, k):
    """
    (W, D, Wtot2, ess) for the adaptive-bandwidth kNN graph. W is the non-redundant
    (upper-folded) weight matrix, D the node degrees, Wtot2 = sum w^2 = Var[G].
    """
    neighbors, weights = neighbors_and_weights(coords, n_neighbors=k)
    rownorm = np.zeros((len(coords), len(coords)))
    np.add.at(rownorm, (np.repeat(np.arange(len(coords)), k), neighbors.ravel()),
              weights.ravel())
    weights_nr = make_weights_non_redundant(neighbors, weights)

    n = neighbors.shape[0]
    i = np.repeat(np.arange(n), neighbors.shape[1])
    j = neighbors.ravel()
    w = weights_nr.ravel()
    keep = w > 0
    W = np.zeros((n,n))
    np.add.at(W, (i[keep], j[keep]), w[keep])

    return W, W.sum(axis=0) + W.sum(axis=1), float((weights_nr**2).sum()), kish_ess(rownorm)


def build_sigma_graph(coords_um, sigma):
    """
    (W, D, Wtot2, ess) for a Gaussian kernel of FIXED bandwidth `sigma` microns over all
    samples: w_ij = exp(-d_ij^2 / 2 sigma^2), row-normalised, then folded to one weight
    per undirected edge exactly as make_weights_non_redundant does for the kNN graph.

    Unlike the kNN graph this has no k and no per-sample rescaling, so a neighbourhood
    means the same physical thing in every organ - which is what makes C comparable across
    specimens. A sample with nothing nearby simply ends up with a low ess.
    """
    d = pairwise_distances(coords_um)
    Wf = np.exp(-d**2 / (2*sigma**2))
    np.fill_diagonal(Wf, 0.0)
    rownorm = Wf / np.where(Wf.sum(axis=1, keepdims=True) > 0, Wf.sum(axis=1, keepdims=True), 1.0)

    # Fold w_ji into w_ij, keeping the upper triangle only: one weight per undirected edge
    W = np.triu(rownorm + rownorm.T, k=1)

    return W, W.sum(axis=0) + W.sum(axis=1), float((W**2).sum()), kish_ess(rownorm)


##


def fit_betabinom(NV, NR, eps=1e-6):
    """
    Per-mutation beta-binomial fit of NV given NR by maximum likelihood over (p, rho):
    p the organ-wide AF, rho the overdispersion (rho -> 0 is the plain binomial, i.e. all
    spread is sequencing noise). The LCM analogue of Hotspot's danb_model.

    Returns (p, rho); the caller builds mu_i = p and
    var_i = p(1-p)/NR_i * (1 + (NR_i-1)*rho), so a deeply sequenced sample gets a small
    null variance and standardizing by it is what downweights the noisy ones.
    """
    NV = np.asarray(NV, dtype=float)
    NR = np.asarray(NR, dtype=float)

    def nll(theta):
        p = np.clip(1/(1+np.exp(-theta[0])), eps, 1-eps)
        rho = np.clip(1/(1+np.exp(-theta[1])), eps, 1-eps)
        s = (1-rho)/rho
        return -betabinom.logpmf(NV, NR, p*s, (1-p)*s).sum()

    p0 = np.clip(NV.sum()/NR.sum(), eps, 1-eps)
    best = None
    for rho0 in [0.01, 0.1, 0.5]:  # multi-start: the likelihood is flat in rho at low n
        try:
            res = minimize(nll, [np.log(p0/(1-p0)), np.log(rho0/(1-rho0))],
                           method='Nelder-Mead',
                           options={'maxiter':2000, 'xatol':1e-6, 'fatol':1e-6})
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res

    return (float(np.clip(1/(1+np.exp(-best.x[0])), eps, 1-eps)),
            float(np.clip(1/(1+np.exp(-best.x[1])), eps, 1-eps)))


def standardize(nv, nr, min_covered=5):
    """
    Model-standardized z for one mutation, plus its fitted (p, rho).

    NR == 0 is a missing measurement, not an AF of 0: those samples get z = 0, so they
    contribute neither to G nor to its normalization, while staying in the graph so their
    neighbours keep theirs. Returns None if there is nothing to fit.
    """
    ok = nr > 0
    if ok.sum() < min_covered or nv[ok].sum() == 0:
        return None, None, None, ok

    p, rho = fit_betabinom(nv[ok], nr[ok])
    vaf = np.divide(nv, nr, out=np.zeros_like(nv, dtype=float), where=ok)
    var = np.where(ok, p*(1-p)/np.where(ok, nr, 1) * (1 + (np.where(ok, nr, 1)-1)*rho), 1.0)
    z = np.where(ok, (vaf-p)/np.sqrt(var), 0.0)

    return z, p, rho, ok


##


def hotspot_stats(z, W, D, Wtot2):
    """G, its effect size C and the analytic one-sided p, for one standardized vector."""
    G = float(z @ W @ z)
    G_max = float((D * z**2).sum() / 2)
    Z = G / np.sqrt(Wtot2)
    return {'G':G, 'C':(G/G_max if G_max > 0 else np.nan), 'Z':Z, 'Pval':float(norm.sf(Z))}


def permutation_p(z, W, G, n_perm, rng):
    """
    Permutation null of the same G: reshuffle z across samples with the graph held fixed.
    Returns (p, Z_perm, sd). Floor 1/(n_perm+1).
    """
    P = rng.permuted(np.tile(z, (n_perm,1)), axis=1)
    G_null = ((P @ W) * P).sum(axis=1)
    sd = G_null.std(ddof=1)
    return ((np.sum(G_null >= G) + 1) / (n_perm + 1),
            float((G-G_null.mean())/sd) if sd > 0 else np.nan, float(sd))


##


# Paths
path_main = '/Users/cossa/Desktop/projects/embryo_bottlenecks/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'autocorrelation')
os.makedirs(path_results, exist_ok=True)

# Graphs. 'knn' keeps continuity with the earlier runs; 'sigma' (microns) is the one to
# use across organs. SIGMA_MAIN / K_MAIN are the reported settings.
K_VALUES = [5, 10, 15, 20]     # capped at n/3 per organ: past that 'local' stops meaning anything
K_MAIN = 10
SIGMAS = [200, 400, 800]       # microns
SIGMA_MAIN = 400

MIN_POSITIVE = 5      # carriers needed before a mutation is testable at all
MIN_N = 15            # samples needed in a stratum
N_PERM = 50_000       # p floor 2e-5; the top hits need tail resolution below 1e-4
SCREEN = 0.05         # analytic p below which the scan runs permutations
FDR_TARGETED = 0.05   # confirmatory
FDR_SCAN = 0.10       # discovery; a reporting cut only, FDR_perm is in the CSVs
N_SUB, N_REP = 63, 20  # power control: heart subsampled to this n, this many times

ORGANS = ['Heart', 'Kidney_left', 'Kidney_right', 'Brain', 'Liver']

# The 18 first-division mutations (branches A-J), as in muts_in_space.py
EARLY = pd.DataFrame({
    'mutation_id' : [
        "chr10_128568737_C_T", "chr3_85532662_T_C", "chr18_69326508_G_A",
        "chr11_96885133_A_G", "chr1_12414725_C_A", "chr6_50196398_C_T",
        "chr7_46281686_G_C", "chr17_79695509_G_C", "chr8_15505436_G_A",
        "chr1_194995181_C_A", "chr2_187163463_T_C", "chr5_156155901_C_G",
        "chr14_95974143_C_T", "chr7_93731124_C_G", "chr11_132296858_C_T",
        "chr18_37006496_T_A", "chr3_194592596_C_T", "chr8_143635887_G_A"],
    'Branch_early' : ["A","A","B","C","C","D","E","E","F","G","H","H","I","I","J","J","J","J"],
})


##


def load_organ(organ, df, ann):
    """
    (NV, NR, coords_um, strata) for one organ, all indexed identically.

    Coordinates arrive in microns and isotropic, so the graph builders need no conversion.
    strata maps a label to a boolean mask: 'All' always, plus histology for the kidneys,
    whose LCMs are interdigitated cell types rather than contiguous territories (two
    adjacent glomeruli each derive from their own nephron-progenitor condensation, so a
    kNN graph over them need not connect clonally related material).
    """
    sub = df.query('organ == @organ')
    NV = sub.pivot_table(index='sample', columns='MUT', values='AD', fill_value=0)
    NR = sub.pivot_table(index='sample', columns='MUT', values='DP', fill_value=0)
    a = ann.loc[NV.index]
    assert np.all(NV.index == NR.index) and np.all(NV.index == a.index)

    strata = {'All' : np.ones(len(NV), dtype=bool)}
    if organ in ['Kidney_left', 'Kidney_right']:
        for h, cnt in a['histo'].value_counts().items():
            if cnt >= MIN_N:
                strata[h] = (a['histo'] == h).values

    return NV, NR, a[['x','y','z']], strata


def knn_k_for(n):
    """
    The k this organ is reported at: K_MAIN, unless n/3 does not allow it. Organs smaller
    than 3*K_MAIN (the liver, n=26) are reported at the largest k they can support, which
    is why this has to be one function rather than a constant - hardcoding K_MAIN drops
    those organs from the summary and from the scan entirely.
    """
    ks = [k for k in K_VALUES if k <= n//3] or [min(K_VALUES)]
    return min(K_MAIN, max(ks))


def graphs_for(coords_um, n, which=('knn','sigma')):
    """
    The graphs to run for an organ, as {(kind, param) : (W, D, Wtot2, ess)}.
    kNN is capped at n/3; a fixed sigma has no such cap, it just yields a small ess where
    the organ is sparsely sampled.
    """
    out = {}
    if 'knn' in which:
        ks = [k for k in K_VALUES if k <= n//3] or [min(K_VALUES)]
        for k in ks:
            out[('knn', k)] = build_knn_graph(coords_um.values, k)
    if 'sigma' in which:
        for s in SIGMAS:
            out[('sigma', s)] = build_sigma_graph(coords_um.values, s)
    return out


##


def run_targeted(organ, NV, NR, coords_um, rng):
    """The 18 early mutations on every graph, all permuted."""
    n = len(NV)
    rows = []
    for (kind, param), (W, D, Wtot2, ess) in graphs_for(coords_um, n).items():
        for ID in EARLY['mutation_id']:
            if ID not in NV.columns:
                continue
            nv = NV[ID].values.astype(float)
            nr = NR[ID].values.astype(float)
            row = {'mutation_id':ID, 'organ':organ, 'graph':kind, 'param':param, 'n':n,
                   'median_ess':float(np.median(ess)), 'frac_ess_lt3':float(np.mean(ess<3)),
                   'n_covered':int((nr>0).sum()), 'n_positive':int((nv>0).sum()),
                   'median_NR':float(np.median(nr[nr>0])) if (nr>0).any() else np.nan}
            z, p, rho, ok = standardize(nv, nr)
            if z is None:  # never detected here: nothing to correlate. Reported, not tested
                rows.append(row)
                continue
            stats = hotspot_stats(z, W, D, Wtot2)
            pp, zp, sd = permutation_p(z, W, stats['G'], N_PERM, rng)
            row.update({'p_hat':p, 'rho':rho, **stats,
                        'Pval_perm':pp, 'Z_perm':zp, 'sd_G_perm':sd,
                        'sd_G_analytic':np.sqrt(Wtot2)})
            rows.append(row)
    return pd.DataFrame(rows)


def run_scan(NV, NR, coords_um, W, D, Wtot2, ess, rng):
    """
    Every testable mutation on one graph. Two-stage: the analytic p for all of them in one
    matrix product, permutations only for the candidates (analytic p < SCREEN).

    Safe because the analytic p is anticonservative in the tail, so screening on it cannot
    discard a mutation the permutation test would have called; and unscreened mutations sit
    at p >= 0.05, far above any BH threshold reached here.

    NOTE the resulting Pval_perm column is a MIXTURE - only rows with permuted == True hold
    an actual permutation p, the rest hold the analytic value. Do not describe it as
    "permutation p-values" without that qualification.
    """
    n = len(NV)
    nv_all = NV.values.astype(float)
    nr_all = NR.values.astype(float)
    testable = ((nv_all > 0).sum(axis=0) >= MIN_POSITIVE) & ((nr_all > 0).sum(axis=0) >= 5)
    muts = NV.columns[testable]

    Z = np.zeros((len(muts), n))
    p_hat, rho = np.zeros(len(muts)), np.zeros(len(muts))
    for m, ID in enumerate(muts):
        z, p, r, ok = standardize(nv_all[:, NV.columns.get_loc(ID)],
                                  nr_all[:, NR.columns.get_loc(ID)])
        Z[m], p_hat[m], rho[m] = z, p, r

    G = ((Z @ W) * Z).sum(axis=1)
    C = G / ((Z**2 @ D) / 2)
    Za = G / np.sqrt(Wtot2)
    Pa = norm.sf(Za)

    Pp = Pa.copy()
    cand = np.where(Pa < SCREEN)[0]
    for m in cand:
        Pp[m] = permutation_p(Z[m], W, G[m], N_PERM, rng)[0]

    return pd.DataFrame({
        'mutation_id':muts, 'n':n, 'median_ess':float(np.median(ess)),
        'n_positive':(nv_all[:,testable] > 0).sum(axis=0),
        'median_NR':np.median(np.where(nr_all[:,testable] > 0, nr_all[:,testable], np.nan), axis=0),
        'p_hat':p_hat, 'rho':rho, 'G':G, 'C':C, 'Z':Za, 'Pval':Pa, 'Pval_perm':Pp,
        'permuted':np.isin(np.arange(len(muts)), cand),
    })


##


# ---- inputs ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(path_data, 'metadata_table.csv'))
ann = pd.read_csv(os.path.join(path_data, 'sample_annotations.csv')).set_index('sample')

branch_df = pd.read_csv(
    os.path.join(path_data, 'Filteredmutations_14061_Sample_subset_snv_assigned_to_branches.txt'),
    sep='\t')
branch_df['mutation_id'] = (branch_df['Chr'] + '_' + branch_df['Pos'].astype(str) + '_' +
                            branch_df['Ref'] + '_' + branch_df['Alt'])
branches = branch_df.set_index('mutation_id')['Branch']

organ_data = {}
for organ in ORGANS:
    NV, NR, coords_um, strata = load_organ(organ, df, ann)
    organ_data[organ] = (NV, NR, coords_um, strata)
    print(f'{organ:13s} n={len(NV):3d} samples, {NV.shape[1]} mutations, '
          f'median DP={np.median(NR.values[NR.values>0]):.0f}')


##


# ---- 1. targeted: the 18 early mutations ----------------------------------------------
print('\n=== TARGETED: 18 first-division mutations')
L = []
for organ in ORGANS:
    NV, NR, coords_um, _ = organ_data[organ]
    L.append(run_targeted(organ, NV, NR, coords_um, np.random.default_rng(1234)))
targeted = pd.concat(L, ignore_index=True).merge(EARLY, on='mutation_id', how='left')

# BH within specimen x graph: the 18 mutations are the family
for col, out in [('Pval','FDR'), ('Pval_perm','FDR_perm')]:
    targeted[out] = np.nan
    for _, x in targeted.groupby(['organ','graph','param']):
        v = x[col].notna()
        if v.any():
            targeted.loc[x.index[v], out] = multipletests(
                x.loc[v, col], alpha=FDR_TARGETED, method='fdr_bh')[1]
targeted.to_csv(os.path.join(path_results, 'targeted_early_muts.csv'), index=False)

targeted['is_main'] = [
    (g == 'sigma' and p == SIGMA_MAIN) or (g == 'knn' and p == knn_k_for(n))
    for g, p, n in zip(targeted['graph'], targeted['param'], targeted['n'])
]
main = targeted.query('is_main')
targeted_summary = (
    main.groupby(['organ','graph','param'])
    .apply(lambda x: pd.Series({
        'n' : x['n'].iloc[0],
        'median_ess' : x['median_ess'].iloc[0],
        'frac_ess_lt3' : x['frac_ess_lt3'].iloc[0],
        'n_tested' : int(x['C'].notna().sum()),
        'n_sig_perm' : int((x['FDR_perm'] <= FDR_TARGETED).sum()),
        'n_sig_analytic' : int((x['FDR'] <= FDR_TARGETED).sum()),
        'median_C' : x['C'].median(),
        'max_C' : x['C'].max(),
        'median_p_ratio' : (x['Pval']/x['Pval_perm']).median(),
        'min_p_ratio' : (x['Pval']/x['Pval_perm']).min(),
    }), include_groups=False)
    .reset_index()
)
targeted_summary.to_csv(os.path.join(path_results, 'targeted_organ_summary.csv'), index=False)
pd.set_option('display.width', 250)
print(targeted_summary.round(3).to_string(index=False))


##


# ---- 2. scan: every mutation, kidneys stratified --------------------------------------
print('\n=== SCAN: all mutations with >= %d carriers' % MIN_POSITIVE)
L = []
for organ in ORGANS:
    NV, NR, coords_um, strata = organ_data[organ]
    for label, mask in strata.items():
        n = int(mask.sum())
        if n < MIN_N:
            continue
        for (kind, param), G in graphs_for(coords_um[mask], n,
                                           which=('knn','sigma')).items():
            if (kind == 'knn' and param != knn_k_for(n)) or \
               (kind == 'sigma' and param != SIGMA_MAIN):
                continue  # the scan runs one setting per graph kind
            W, D, Wtot2, ess = G
            out = run_scan(NV[mask], NR[mask], coords_um[mask], W, D, Wtot2, ess,
                           np.random.default_rng(1234))
            out = out.assign(organ=organ, stratum=label, graph=kind, param=param)
            out['FDR_perm'] = multipletests(out['Pval_perm'], alpha=FDR_SCAN,
                                            method='fdr_bh')[1]
            L.append(out)
            print(f'  {organ:13s} {label:18s} {kind}={param:<4} n={n:3d}  '
                  f'{len(out):4d} testable  '
                  f'{int((out["FDR_perm"]<=FDR_SCAN).sum()):3d} sig  '
                  f'median C={out["C"].median():+.3f}  max C={out["C"].max():.3f}')

scan = pd.concat(L, ignore_index=True)
scan['Branch'] = scan['mutation_id'].map(branches)
scan['is_early'] = scan['mutation_id'].isin(EARLY['mutation_id'])
scan.to_csv(os.path.join(path_results, 'scan_all_muts.csv'), index=False)

scan_summary = (
    scan.groupby(['organ','stratum','graph','param'])
    .apply(lambda x: pd.Series({
        'n' : x['n'].iloc[0], 'median_ess' : x['median_ess'].iloc[0],
        'n_tested' : len(x), 'n_sig' : int((x['FDR_perm'] <= FDR_SCAN).sum()),
        'frac_sig' : float((x['FDR_perm'] <= FDR_SCAN).mean()),
        'median_C' : x['C'].median(), 'max_C' : x['C'].max(),
        'n_branches_sig' : x.loc[x['FDR_perm'] <= FDR_SCAN, 'Branch'].nunique(),
    }), include_groups=False)
    .reset_index()
)
scan_summary.to_csv(os.path.join(path_results, 'scan_summary.csv'), index=False)
print('\n' + scan_summary.round(3).to_string(index=False))


##


# ---- 3. early vs the rest -------------------------------------------------------------
# The rank test is the one to quote: it uses no threshold, so it does not inherit the
# instability of a Fisher table built on FDR calls near the line.
rows = []
for (organ, stratum, graph, param), x in scan.groupby(['organ','stratum','graph','param']):
    early, rest = x.query('is_early'), x.query('~is_early')
    if len(early) < 3 or len(rest) < 3:
        continue
    u, p_rank = mannwhitneyu(early['C'], rest['C'], alternative='greater')
    tab = [[int((early['FDR_perm'] <= FDR_SCAN).sum()), int((early['FDR_perm'] > FDR_SCAN).sum())],
           [int((rest['FDR_perm'] <= FDR_SCAN).sum()), int((rest['FDR_perm'] > FDR_SCAN).sum())]]
    odds, p_fisher = fisher_exact(tab, alternative='greater')
    rows.append({'organ':organ, 'stratum':stratum, 'graph':graph, 'param':param,
                 'n_early':len(early), 'n_rest':len(rest),
                 'median_C_early':early['C'].median(), 'median_C_rest':rest['C'].median(),
                 'p_rank':p_rank, 'OR_fisher':odds, 'p_fisher':p_fisher,
                 'n_sig_early':tab[0][0], 'n_sig_rest':tab[1][0]})
early_vs_all = pd.DataFrame(rows)
early_vs_all.to_csv(os.path.join(path_results, 'early_vs_all.csv'), index=False)
print('\n=== EARLY vs REST (rank test preferred)')
print(early_vs_all.query('stratum == "All"').round(4).to_string(index=False))


##


# ---- 4. power control -----------------------------------------------------------------
# Could an organ sampled like the kidneys have seen a heart-sized effect at all? Subsample
# the heart to N_SUB and count how many of its whole-organ hits come back.
heart_hits = (
    main.query('organ == "Heart" and graph == "knn" and FDR_perm <= @FDR_TARGETED')
    ['mutation_id'].tolist()
)
NV_h, NR_h, coords_h, _ = organ_data['Heart']
rng = np.random.default_rng(0)
rec = []
for rep in range(N_REP):
    idx = rng.choice(len(NV_h), size=N_SUB, replace=False)
    W, D, Wtot2, ess = build_knn_graph(coords_h.iloc[idx].values, knn_k_for(N_SUB))
    r = []
    for ID in heart_hits:
        z, p, rho, ok = standardize(NV_h[ID].values[idx].astype(float),
                                    NR_h[ID].values[idx].astype(float))
        if z is None:
            continue
        s = hotspot_stats(z, W, D, Wtot2)
        pp = permutation_p(z, W, s['G'], 10_000, rng)[0]
        r.append({'mutation_id':ID, 'C':s['C'], 'Pval_perm':pp})
    r = pd.DataFrame(r)
    r['FDR_perm'] = multipletests(r['Pval_perm'], alpha=FDR_TARGETED, method='fdr_bh')[1]
    rec.append({'rep':rep, 'n_hits':len(heart_hits),
                'n_recovered_FDR':int((r['FDR_perm'] <= FDR_TARGETED).sum()),
                'n_recovered_nominal':int((r['Pval_perm'] < 0.05).sum()),
                'median_C':float(r['C'].median()), 'median_ess':float(np.median(ess))})
rec = pd.DataFrame(rec)
rec.to_csv(os.path.join(path_results, 'heart_subsample_power.csv'), index=False)

C_full = main.query('organ == "Heart" and graph == "knn" and mutation_id in @heart_hits')['C'].median()
print(f'\n=== POWER CONTROL: heart subsampled to n={N_SUB}, {N_REP} replicates, '
      f'kNN k={knn_k_for(N_SUB)}')
print(f'  of its {len(heart_hits)} whole-organ hits, recovered at FDR<={FDR_TARGETED}: '
      f'median {rec["n_recovered_FDR"].median():.0f} '
      f'(range {rec["n_recovered_FDR"].min()}-{rec["n_recovered_FDR"].max()})')
print(f'  at nominal p<0.05: median {rec["n_recovered_nominal"].median():.0f}')
print(f'  median C of those mutations: {rec["median_C"].median():.3f} '
      f'(whole organ: {C_full:.3f})')
print(f'\nwrote {path_results}')
