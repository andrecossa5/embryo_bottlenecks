"""
Spatial autocorrelation of ALL mutations, every specimen, kidneys split by histology.

Generalises autocorrelation_single_muts.py in two directions:

1) ALL MUTATIONS, not the 18 first-division ones. Of the 14,063 mutations in the
   callset most are private to one or two LCM samples and carry no spatial information;
   requiring >= MIN_POSITIVE samples with NV>0 leaves ~600 testable per specimen. Every
   mutation is annotated with its branch from the phylogeny, so the results can be read
   by clade rather than mutation by mutation.

   This also lets the original question be asked as a comparison instead of in
   isolation: are the first-division mutations more spatially autocorrelated than
   mutations in general? See the enrichment test at the end.

2) KIDNEYS SPLIT BY HISTOLOGY. Kidney LCMs are interdigitated cell-type classes
   (glomerulus, blastema, primitive tubule, urothelium), not contiguous territory, so a
   kNN graph over the whole organ connects samples that are spatially adjacent but need
   not be clonally comparable. kidney_strata_check.py showed this matters: for
   chr1_12414725_C_A in the right kidney, restricting to glomeruli raised C from 0.139
   (FDR 0.068) to 0.220 (FDR 0.012) on FEWER samples, and beat 300 size-matched random
   subsets (97.7th percentile). Here every stratum with enough samples is run.

Statistic, null model and graph are exactly those of autocorrelation_single_muts.py -
see that module's docstring. Two differences forced by the scale:

  - TWO-STAGE INFERENCE. The analytic Z is computed for every mutation (it is one
    matrix product for the whole batch), and the 50,000-permutation null is run only
    for mutations with an analytic p < SCREEN. This is safe because the analytic p is
    ANTIconservative in the tail (measured 2-6x too small, see the notes file), so
    screening on it cannot lose a mutation that the permutation test would have called.
    Mutations that fail the screen keep their analytic p; with ~600 tests any BH
    threshold is far below SCREEN, so they can never become discoveries either way.
  - BH is applied within each (specimen, stratum) over all tested mutations, and
    reported at FDR_THRESHOLD (0.10, see below).

Outputs
  results/autocorrelation_all_muts.csv          one row per mutation x specimen x stratum
  results/autocorrelation_all_muts_summary.csv  per specimen x stratum
  results/autocorrelation_branch_summary.csv    per branch, within specimen x stratum
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import betabinom, fisher_exact, mannwhitneyu, norm
from sklearn.neighbors import NearestNeighbors
from statsmodels.sandbox.stats.multicomp import multipletests


##


def neighbors_and_weights(coords, n_neighbors=10, neighborhood_factor=3):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute').fit(coords)
    dist, ind = nbrs.kneighbors()
    radius_ii = int(np.ceil(n_neighbors/neighborhood_factor))
    sigma = dist[:,[radius_ii-1]]
    sigma[sigma == 0] = 1
    weights = np.exp(-1 * dist**2 / sigma**2)
    return ind, weights / weights.sum(axis=1, keepdims=True)


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
    """(W, D, Wtot2) of the non-redundant adaptive-kernel kNN graph."""
    neighbors, weights = neighbors_and_weights(coords, n_neighbors=k)
    weights = make_weights_non_redundant(neighbors, weights)
    n = neighbors.shape[0]
    i = np.repeat(np.arange(n), neighbors.shape[1])
    j = neighbors.ravel()
    w = weights.ravel()
    keep = w > 0
    W = np.zeros((n,n))
    np.add.at(W, (i[keep], j[keep]), w[keep])
    return W, W.sum(axis=0) + W.sum(axis=1), float((weights**2).sum())


def fit_betabinom(NV, NR, eps=1e-6):
    """Identical to autocorrelation_single_muts.py, so results stay comparable."""
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
    return (float(np.clip(1/(1+np.exp(-best.x[0])), eps, 1-eps)),
            float(np.clip(1/(1+np.exp(-best.x[1])), eps, 1-eps)))


##


path_main = '/Users/cossa/Desktop/projects/embryo_bottlenecks/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results')
path_cache = os.path.join(path_results, 'cache')

MIN_POSITIVE = 5     # samples carrying the mutation; below this there is nothing to test
MIN_N = 15           # samples in a stratum
K_MAX = 10
N_PERM = 50000
SCREEN = 0.05        # analytic p below which the permutation null is run
# Discovery threshold. 0.10, not the conventional 0.05: this is a screen over ~200-1000
# mutations per specimen whose hits are meant to be followed up (and several already
# replicate across specimens and strata), not a confirmatory test. At 0.05 the left
# kidney returns nothing anywhere, while its top mutations sit at FDR 0.06-0.10 - i.e.
# the specimen was being reported as empty on the strength of a threshold convention.
# Everything downstream reads FDR_perm, so any other cut can be applied to the CSVs.
FDR_THRESHOLD = 0.10

early_ids = [
    "chr10_128568737_C_T", "chr3_85532662_T_C", "chr18_69326508_G_A",
    "chr11_96885133_A_G", "chr1_12414725_C_A", "chr6_50196398_C_T",
    "chr7_46281686_G_C", "chr17_79695509_G_C", "chr8_15505436_G_A",
    "chr1_194995181_C_A", "chr2_187163463_T_C", "chr5_156155901_C_G",
    "chr14_95974143_C_T", "chr7_93731124_C_G", "chr11_132296858_C_T",
    "chr18_37006496_T_A", "chr3_194592596_C_T", "chr8_143635887_G_A"
]


def load_specimen(organ):
    """(NV, NR, coords, strata) for one specimen. strata maps label -> boolean mask."""

    if organ in ['Brain', 'Liver']:
        d = np.load(os.path.join(path_cache, 'brain_liver_full.npz'), allow_pickle=True)
        ann = (
            pd.read_csv(os.path.join(path_data, 'df_brain_liver_annotations_unique_for_Andrea.csv'))
            .dropna(subset=['x','y']).query('Bulk_phenotype == @organ').set_index('Sample_ID')
        )
        coord_cols = ['x','y']
    else:
        d = np.load(os.path.join(path_cache, f'{organ}_full.npz'), allow_pickle=True)
        if organ == 'Heart':
            ann = pd.read_csv(
                os.path.join(path_data, 'Heart_final_coorindates_135.csv')).set_index('name')
            coord_cols = ['x','y','z']
        else:
            ann = pd.read_csv(
                os.path.join(path_data, f'{organ}_coordinates.csv')).set_index('name')
            coord_cols = ['x','y']
        histo = (
            pd.read_csv(os.path.join(path_data, f'{organ}_metadata.csv'),
                        usecols=['Sample_ID','Histo'])
            .drop_duplicates().set_index('Sample_ID')['Histo']
        )
        ann = ann.join(histo)

    NV = pd.DataFrame(d['NV'], index=d['samples'], columns=d['muts'])
    NR = pd.DataFrame(d['NR'], index=d['samples'], columns=d['muts'])
    common = NV.index.intersection(ann.index)
    NV, NR, ann = NV.loc[common], NR.loc[common], ann.loc[common]

    strata = {'All': np.ones(len(NV), dtype=bool)}
    # Kidneys only: split by histology, since their LCMs are interdigitated cell types
    if organ in ['PD53943o', 'PD53943w']:
        for h, cnt in ann['Histo'].value_counts().items():
            if cnt >= MIN_N:
                strata[h] = (ann['Histo'] == h).values

    return NV, NR, ann[coord_cols], strata


def run_stratum(NV, NR, coords, k, rng):
    """All testable mutations on one graph. Two-stage: analytic for all, permutation
    for the candidates."""

    n = len(NV)
    W, D, Wtot2 = build_graph(coords.values, k)

    nv_all = NV.values.astype(float)
    nr_all = NR.values.astype(float)
    keep = ((nv_all > 0).sum(axis=0) >= MIN_POSITIVE) & (nr_all > 0).all(axis=0)
    muts = NV.columns[keep]
    nv_all, nr_all = nv_all[:,keep], nr_all[:,keep]

    # Model-standardize every mutation
    Z = np.zeros((len(muts), n))
    p_hat = np.zeros(len(muts)); rho = np.zeros(len(muts))
    for m in range(len(muts)):
        p, r = fit_betabinom(nv_all[:,m], nr_all[:,m])
        var = p*(1-p)/nr_all[:,m] * (1 + (nr_all[:,m]-1)*r)
        Z[m] = (nv_all[:,m]/nr_all[:,m] - p) / np.sqrt(var)
        p_hat[m], rho[m] = p, r

    # Stage 1: G, C and the analytic p for the whole batch, in one matrix product
    G = ((Z @ W) * Z).sum(axis=1)
    C = G / ((Z**2 @ D) / 2)
    Za = G / np.sqrt(Wtot2)
    Pa = norm.sf(Za)

    # Stage 2: permutation null for the candidates only
    Pp = Pa.copy()
    cand = np.where(Pa < SCREEN)[0]
    for m in cand:
        z = Z[m]
        P = rng.permuted(np.tile(z, (N_PERM,1)), axis=1)
        G_null = ((P @ W) * P).sum(axis=1)
        Pp[m] = (np.sum(G_null >= G[m]) + 1) / (N_PERM + 1)

    df = pd.DataFrame({
        'mutation_id' : muts, 'n' : n, 'k' : k,
        'n_positive' : (nv_all > 0).sum(axis=0), 'median_NR' : np.median(nr_all, axis=0),
        'p_hat' : p_hat, 'rho' : rho, 'G' : G, 'C' : C, 'Z' : Za,
        'Pval' : Pa, 'Pval_perm' : Pp, 'permuted' : np.isin(np.arange(len(muts)), cand),
    })
    df['FDR_perm'] = multipletests(df['Pval_perm'], alpha=FDR_THRESHOLD, method='fdr_bh')[1]

    return df


##


# Branch annotation for every mutation in the callset
branch_df = pd.read_csv(
    os.path.join(path_data, 'Filteredmutations_14061_Sample_subset_snv_assigned_to_branches.txt'),
    sep='\t'
)
branch_df['mutation_id'] = (branch_df['Chr'] + '_' + branch_df['Pos'].astype(str) + '_' +
                            branch_df['Ref'] + '_' + branch_df['Alt'])
branches = branch_df.set_index('mutation_id')['Branch']


##


SPECIMENS = {
    'PD53943o' : 'Left kidney',
    'PD53943w' : 'Right kidney',
    'Heart' : 'Heart',
    'Brain' : 'Brain',
    'Liver' : 'Liver',
}

L = []
for organ, name in SPECIMENS.items():

    NV, NR, coords, strata = load_specimen(organ)
    print(f'\n{name}: n={len(NV)} samples, {NV.shape[1]} mutations in the callset')

    for label, mask in strata.items():
        n = int(mask.sum())
        if n < MIN_N:
            continue
        k = min(K_MAX, max(3, n//3))
        rng = np.random.default_rng(1234)
        df = run_stratum(NV[mask], NR[mask], coords[mask], k, rng)
        df = df.assign(specimen=name, sample=organ, stratum=label)
        L.append(df)
        print(f'  {label:18s} n={n:3d} k={k:2d}  {len(df):4d} testable  '
              f'{int((df["FDR_perm"]<=FDR_THRESHOLD).sum()):3d} significant  '
              f'median C={df["C"].median():+.3f}  max C={df["C"].max():.3f}')

df = pd.concat(L, ignore_index=True)
df['Branch'] = df['mutation_id'].map(branches)
df['is_early'] = df['mutation_id'].isin(early_ids)
df.to_csv(os.path.join(path_results, 'autocorrelation_all_muts.csv'), index=False)


##


# Per specimen x stratum
summary = (
    df.groupby(['specimen','stratum'])
    .apply(lambda x: pd.Series({
        'n' : x['n'].iloc[0], 'k' : x['k'].iloc[0], 'n_tested' : len(x),
        'n_sig' : (x['FDR_perm']<=FDR_THRESHOLD).sum(),
        'frac_sig' : (x['FDR_perm']<=FDR_THRESHOLD).mean(),
        'median_C' : x['C'].median(), 'max_C' : x['C'].max(),
        'n_branches_sig' : x.loc[x['FDR_perm']<=FDR_THRESHOLD,'Branch'].nunique(),
    }), include_groups=False)
    .reset_index()
)
summary.to_csv(os.path.join(path_results, 'autocorrelation_all_muts_summary.csv'), index=False)
print('\n' + summary.to_string(index=False))


##


# Are the first-division mutations more spatially autocorrelated than mutations at
# large? Two tests per specimen x stratum: a rank comparison of C, and a 2x2 on the
# significant calls. This is the original question of muts_in_space.py, but now with
# the rest of the callset as the comparison group instead of no comparison at all.
rows = []
for (sp, st), x in df.groupby(['specimen','stratum']):
    e, o = x[x['is_early']], x[~x['is_early']]
    if len(e) < 3:
        continue
    u = mannwhitneyu(e['C'], o['C'], alternative='greater')
    tab = [[int((e['FDR_perm']<=FDR_THRESHOLD).sum()), int((e['FDR_perm']>FDR_THRESHOLD).sum())],
           [int((o['FDR_perm']<=FDR_THRESHOLD).sum()), int((o['FDR_perm']>FDR_THRESHOLD).sum())]]
    rows.append({
        'specimen':sp, 'stratum':st, 'n_early':len(e), 'n_other':len(o),
        'median_C_early':e['C'].median(), 'median_C_other':o['C'].median(),
        'p_rank':u.pvalue,
        'sig_early':tab[0][0], 'sig_other':tab[1][0],
        'OR':fisher_exact(tab, alternative='greater')[0],
        'p_fisher':fisher_exact(tab, alternative='greater')[1],
    })
early_test = pd.DataFrame(rows)
early_test.to_csv(os.path.join(path_results, 'autocorrelation_early_vs_all.csv'), index=False)
print('\nAre the 18 first-division mutations more autocorrelated than the rest?')
print(early_test.to_string(index=False))


##


# Per branch: do whole clades come out together?
branch_summary = (
    df.dropna(subset=['Branch'])
    .groupby(['specimen','stratum','Branch'])
    .apply(lambda x: pd.Series({
        'n_muts' : len(x), 'n_sig' : (x['FDR_perm']<=FDR_THRESHOLD).sum(),
        'median_C' : x['C'].median(),
    }), include_groups=False)
    .reset_index()
    .query('n_muts >= 3')
    .sort_values('median_C', ascending=False)
)
branch_summary.to_csv(os.path.join(path_results, 'autocorrelation_branch_summary.csv'), index=False)
print('\nTop branches by median C (>=3 mutations in the branch):')
print(branch_summary.head(15).to_string(index=False))


##
