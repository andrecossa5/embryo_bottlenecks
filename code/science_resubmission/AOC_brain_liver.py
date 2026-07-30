"""
AOC on brain and liver samples.
"""

import os
import numpy as np
import pandas as pd
import mito as mt
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr, norm
from sklearn.metrics import pairwise_distances
from statsmodels.sandbox.stats.multicomp import multipletests


##


def pairwise_soft_cosine(X, W, rounding_decimals=7):
    """
    X: (n_obs, n_feat)
    W: (n_feat, n_feat) binary (or real) weights / mask for feature-pair contributions
    returns: (n_obs, n_obs) weighted cosine distance matrix
    """
    G = X @ W @ X.T
    norms = np.sqrt(np.diag(G))
    S = G / (norms[:,np.newaxis] * norms[np.newaxis,:])
    np.fill_diagonal(S, 1)
    np.clip(S, 0, 1, out=S)  # Numerical stability --> errors 
    np.round(S, rounding_decimals, out=S)  # Numerical stability --> simmetry

    return 1 - S


##


def rescale_distances(D):
    """
    Rescale (row-wise) pairwise distances to [0,1].
    """
    min_dist = D[~np.eye(D.shape[0], dtype=bool)].min()
    max_dist = D[~np.eye(D.shape[0], dtype=bool)].max()
    D = (D-min_dist)/(max_dist-min_dist)
    np.fill_diagonal(D, 0)
    return D


def spatial_AOC(
    D_gen, D_phys, D_gen_raw=None, D_phys_um=None, k=10, n_trials=1000,
    alpha=0.05, random_state=1234
    ):
    """
    mt.ut.AOC re-implemented so that, in the same pass over samples (and re-using the
    very same physical-space kNN graph), we also get an effect-size version of the
    same comparison: how genetically close a sample is to its physical neighbors,
    relative to equally-sized random sets of its physical NON-neighbors.

    Neighbors are always defined in PHYSICAL space (kNN on D_phys); every distance
    that gets averaged is a GENETIC distance. AOC scores that comparison on the rank
    scale, the statistics below score it on the raw genetic-distance scale.

    D_gen     : (n,n) rescaled genetic distances (the 'CAS' matrix of mt.ut.AOC).
    D_phys    : (n,n) rescaled physical distances, used to define the kNN graph.
    D_gen_raw : (n,n) *unrescaled* genetic distances (soft cosine). Used for the
                neighborhood statistics below. This matters for gen_ratio:
                rescale_distances() subtracts the global minimum, which shifts the
                scale and therefore distorts any ratio of distances. Defaults to D_gen.
    D_phys_um : (n,n) *unscaled* physical distances in microns, used only to report
                the physical scale of each neighborhood. Defaults to D_phys.
    Both nulls draw k random samples other than i itself. Note that this differs from
    mt.ut.AOC, whose null draws from all n samples and can therefore pick i against
    itself: a self-distance of 0 pulls the null means down, so AOC comes out slightly
    conservative. The effect is small (well within Monte Carlo noise at n_trials=1000)
    but there is no reason to keep it.

    Returns a (n, 9) DataFrame, one row per sample:

    AOC, p     : as in mt.ut.AOC (Weng et al., 2024). Rank-based.
                 The rank of the observed mean genetic distance to the k neighbors,
                 within row i, minus the same rank for random k-sets, over n.
    d_gen_nn   : mean genetic distance between the sample and its k physical NNs.
    d_gen_non  : mean genetic distance between the sample and all its non-neighbors.
                 This is exactly the expectation of the random-trial mean, so the
                 trials are not needed to estimate the null centre - only its spread.
    gen_ratio  : d_gen_nn / d_gen_non. Scale-free effect size, and the metric to use
                 as the primary readout: < 1 means the physical neighborhood is
                 genetically tighter than the rest of the organ, 1 means no local
                 genetic structure. Comparable across k, samples and organs.
    gen_diff   : d_gen_non - d_gen_nn. Same contrast on the additive scale.
    gen_z      : (d_gen_nn - mean(random)) / sd(random) over n_trials random k-subsets
                 of the non-neighbors. Same contrast, standardized by how variable a
                 random k-subset mean is for that sample. Negative = neighbors are
                 genetically closer. Use alongside gen_ratio: it is the more sensitive
                 of the two, but its magnitude also depends on the spread of the
                 sample's genetic distances, so it is less comparable across samples.
    gen_p      : one-sided permutation p, (1 + #{random <= observed}) / (n_trials + 1).
                 Unlike the equivalent test in physical space, this one is NOT
                 degenerate: the neighbors are picked by physical proximity, so their
                 mean genetic distance is in no way the minimum over k-subsets.
    d_kNN_um   : mean PHYSICAL distance (um) to the k NNs, i.e. the physical scale
                 the neighborhood statistics refer to.
    d_nonNN_um : mean PHYSICAL distance (um) to the non-neighbors.
    sep_ratio  : d_kNN_um / d_nonNN_um. How spatially distinct the neighborhood is:
                 --> 0 the k NNs are a tight local patch, --> 1 they sit at
                 essentially the same distance as everything else, i.e. sampling was
                 too sparse there for 'neighborhood' to mean anything. Note this can
                 only ever be an effect size: any rank-based separation test in
                 physical space is degenerate, the k NNs being the k closest samples
                 by construction (every sample would score a perfect separation).
    null_sd    : sd of the permutation null of d_gen_nn. Sets the resolution of the
                 test for that sample.
    mde_ratio  : Minimum Detectable Effect, on the gen_ratio scale: the largest (i.e.
                 least extreme) gen_ratio that would still reach p <= alpha for this
                 sample, = quantile(null, alpha) / d_gen_non. A sample with
                 mde_ratio 0.98 can resolve a 2% tightening; one with 0.60 needs 40%.
                 Feed this to power_filter() to decide who enters the final report.
    """

    n = D_gen.shape[0]
    D_gen_raw = D_gen if D_gen_raw is None else D_gen_raw
    D_phys_um = D_phys if D_phys_um is None else D_phys_um
    assert k <= n-1-k, f'k={k} too large: fewer than k non-neighbors left out of n={n}'

    idx, _, _ = mt.pp.kNN_graph(D=D_phys, k=k, from_distances=True)  # self excluded
    rng = np.random.default_rng(random_state)

    keys = [
        'AOC', 'p', 'd_gen_nn', 'd_gen_non', 'gen_ratio', 'gen_diff', 'gen_z', 
        'gen_p', 'null_sd', 'mde_ratio', 'd_kNN_um', 'd_nonNN_um', 'sep_ratio'
    ]
    res = { key : np.zeros(n) for key in keys }

    for i in range(n):

        nn = idx[i,:]
        others = np.delete(np.arange(n), i)
        non_nn = np.setdiff1d(others, nn)

        # AOC, rank-based (mt.ut.AOC): null drawn from all other samples
        sel_aoc = rng.permuted(np.tile(others, (n_trials,1)), axis=1)[:,:k]
        obs_gen = D_gen[i,nn].mean()
        obs_rank = np.sum(D_gen[i,:] < obs_gen) + 1
        rand_gen = D_gen[i,sel_aoc].mean(axis=1)
        random_ranks = (D_gen[i,:][np.newaxis,:] < rand_gen[:,np.newaxis]).sum(axis=1) + 1
        res['AOC'][i] = np.mean((random_ranks-obs_rank) / n)
        res['p'][i] = np.sum(random_ranks < obs_rank) / n_trials

        # Genetic compactness of the physical neighborhood: null drawn from non-neighbors
        sel = rng.permuted(np.tile(non_nn, (n_trials,1)), axis=1)[:,:k]
        d_nn = D_gen_raw[i,nn].mean()
        d_non = D_gen_raw[i,non_nn].mean()
        rand = D_gen_raw[i,sel].mean(axis=1)
        res['d_gen_nn'][i] = d_nn
        res['d_gen_non'][i] = d_non
        res['gen_ratio'][i] = d_nn / d_non
        res['gen_diff'][i] = d_non - d_nn
        res['gen_z'][i] = (d_nn-rand.mean()) / rand.std(ddof=1)
        res['gen_p'][i] = (np.sum(rand <= d_nn) + 1) / (n_trials + 1)

        # Resolution of the test for this sample: null spread, and the resulting
        # smallest effect it could have called significant at alpha
        res['null_sd'][i] = rand.std(ddof=1)
        res['mde_ratio'][i] = np.quantile(rand, alpha) / d_non

        # Physical separation of the neighborhood, i.e. was sampling dense enough here
        d_nn_um = D_phys_um[i,nn].mean()
        d_non_um = D_phys_um[i,non_nn].mean()
        res['d_kNN_um'][i] = d_nn_um
        res['d_nonNN_um'][i] = d_non_um
        res['sep_ratio'][i] = d_nn_um / d_non_um

    return pd.DataFrame(res)


##


def power_filter(df, target_ratio=0.8, min_power=0.8, max_sep_ratio=0.5):
    """
    Decide, sample by sample, whether the AOC / gen_ratio test had enough resolution
    to be worth reporting - before averaging anything across samples.

    The concern this addresses: where sampling is sparse, a sample's k physical
    neighbors sit at nearly the same distance as its non-neighbors (sep_ratio --> 1),
    so 'local' is not really being contrasted with 'distant'. Such a sample carries
    no information either way, and letting it into a median or a mean only dilutes
    the estimate and inflates the non-significant fraction.

    Rather than thresholding sep_ratio directly (which would need an arbitrary cut),
    this is done as an actual power calculation, using the permutation null already
    computed per sample by spatial_AOC(). A sample is retained if it could have
    detected a *typical* effect - by default the organ-wide median gen_ratio - with
    probability >= min_power:

        power_i = Phi( (mde_ratio_i - target_ratio) * d_gen_non_i / null_sd_i )

    Phi being the normal CDF, mde_ratio_i * d_gen_non_i the critical value of the
    one-sided permutation test at alpha. Note power_i = alpha exactly when
    target_ratio = 1 (no effect), as it must be.

    df           : per-sample output of spatial_AOC(), for a single k.
    target_ratio : the effect the test is required to be able to see, PRE-SPECIFIED.
                   0.8 = 'neighbors 20% genetically closer than non-neighbors'.
                   Do not set this to the observed median: the filter would become
                   self-referential and, on data where the typical effect is near the
                   detection limit, would drop every sample by construction.
    min_power    : retention threshold on that power. 0.8 by convention.
    max_sep_ratio: second, purely spatial gate, on the statistic that motivated the
                   filter in the first place. A sample is dropped if its k NNs sit at
                   more than this fraction of the mean non-neighbor distance, i.e. if
                   sampling around it was too sparse for its 'neighborhood' to be
                   spatially distinct at all. Set to 1 to disable.

    Returns df with added columns: target_ratio, power, fail_power, fail_sep, keep.
    """

    df = df.copy()
    delta = (df['mde_ratio']-target_ratio) * df['d_gen_non']
    df['target_ratio'] = target_ratio
    df['power'] = norm.cdf(delta / df['null_sd'])
    df['fail_power'] = df['power'] < min_power
    df['fail_sep'] = df['sep_ratio'] > max_sep_ratio
    df['keep'] = ~df['fail_power'] & ~df['fail_sep']

    return df


##


# Paths
path_main = '/Users/cossa/Desktop/projects/embryo_bottlenecks/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results')


##


# Read mutations table and coordinates
mut_table = pd.read_csv(os.path.join(path_data, 'Final_Dataframe_heart_annotations_raw_trophoblasts.csv'))
coords = pd.read_csv(os.path.join(path_data, 'df_brain_liver_annotations_unique_for_Andrea.csv'))
coords.dropna(subset=['x', 'y'], inplace=True)
branch_df = pd.read_csv(
    os.path.join(
        path_data, 
        'Filteredmutations_14061_Sample_subset_snv_assigned_to_branches.txt'
    ),
    sep='\t'
)


##


# Brain
organ = 'Brain'

# Calculate euclidean distances in physical space
samples = set(coords.query('Bulk_phenotype == @organ')['Sample_ID'].values)
spatial_coords = coords.query('Sample_ID in @samples').set_index('Sample_ID')[['x', 'y']]
samples = set(spatial_coords.index.values)

# Subset mut_table
mut_table.columns
muts = ( 
    mut_table
    .query('Sample_ID in @samples')
    .pivot_table(index='Sample_ID', columns='mutation_id', values='VAF', fill_value=0)
)
muts = muts.loc[:,(muts>0).any(axis=0)]

# Create sparse co-occuring weight matrix for soft cosine distance calculation
branch_df['mut_id'] = branch_df['Chr'] + '_' + \
                      branch_df['Pos'].astype(str) + '_' + \
                      branch_df['Ref'] + '_' + \
                      branch_df['Alt']
mut_ids = branch_df['mut_id'].astype('category')
branches = branch_df['Branch'].astype('category')
W = csr_matrix(
    (np.ones(len(branch_df)), (mut_ids.cat.codes, branches.cat.codes)),
    shape=(len(mut_ids.cat.categories), len(branches.cat.categories))
)
W = (W @ W.T) > 0
W = (W.toarray()).astype(np.uint8)
W = pd.DataFrame(W, index=mut_ids.cat.categories, columns=mut_ids.cat.categories)
W = W.loc[muts.columns, muts.columns]

# Check
np.all(muts.index == spatial_coords.index)

# Calculate genetic distances
D = pairwise_soft_cosine(muts.values, W.values)
D = pd.DataFrame(D, index=muts.index, columns=muts.index)
D.to_csv(os.path.join(path_results, f'{organ}_genetic_distances.csv'))

# Calculate euclidean distances in physical space
D_xyz = pairwise_distances(spatial_coords.values, metric='euclidean')
D_xyz = pd.DataFrame(D_xyz, index=muts.index, columns=muts.index)
D_xyz.to_csv(os.path.join(path_results, f'{organ}_physical_distances.csv'))


##


# Keep un-rescaled distances: genetic ones for the neighborhood effect sizes
# (rescaling shifts the scale, distorting ratios), physical ones for reporting microns
resolution = 0.44186  # Microns per pixel (LCM image resolution)
D_raw = D.values.copy()
D_xyz_um = resolution * D_xyz.values

# Rescale distances first
D = rescale_distances(D.values)
D_xyz = rescale_distances(D_xyz.values)

# AOC + local genetic compactness of physical neighborhoods
n_max = min(30, muts.shape[0]//2)  # Need at least k non-neighbors to sample the null
L = []
for k in np.arange(2,n_max,1):
    df_k = spatial_AOC(D, D_xyz, D_gen_raw=D_raw, D_phys_um=D_xyz_um, k=k, n_trials=1000)
    # Drop samples whose neighborhood is too poorly resolved to be informative,
    # then correct only within the retained ones (they are the tests we report)
    df_k = power_filter(df_k, min_power=0.8)
    keep = df_k['keep'].values
    df_k['FDR'] = np.nan
    df_k['FDR_gen'] = np.nan
    if keep.any():
        df_k.loc[keep,'FDR'] = multipletests(df_k.loc[keep,'p'], alpha=0.05, method="fdr_bh")[1]
        df_k.loc[keep,'FDR_gen'] = multipletests(df_k.loc[keep,'gen_p'], alpha=0.05, method="fdr_bh")[1]
    df_k.index = muts.index
    L.append(df_k.assign(k=k))

df_AOC = pd.concat(L)

# Save results
df_AOC.to_csv(os.path.join(path_results, f'{organ}_AOC_table.csv'))

# Summarise per k, over the retained samples only. n_kept / frac_kept document how
# much of the organ was actually evaluable at that k.
def summarise(x):
    kept = x.loc[x['keep']]
    return pd.Series({
        'n_kept' : x['keep'].sum(),
        'frac_kept' : x['keep'].mean(),
        'median_gen_ratio' : kept['gen_ratio'].median(),
        'median_AOC' : kept['AOC'].median(),
        'frac_sig' : np.mean(kept['FDR_gen']<=0.05),
        'frac_fail_power' : x['fail_power'].mean(),
        'frac_fail_sep' : x['fail_sep'].mean(),
        'median_sep_ratio' : kept['sep_ratio'].median(),
        'median_mde_ratio' : kept['mde_ratio'].median(),
        'median_d_kNN_um' : kept['d_kNN_um'].median(),
        'rho_AOC_gen_ratio' : spearmanr(kept['AOC'], kept['gen_ratio'])[0],
    })

summary = df_AOC.groupby('k').apply(summarise, include_groups=False)
summary.to_csv(os.path.join(path_results, f'{organ}_AOC_summary.csv'))
print(summary)

# Sensitivity: does the conclusion depend on where the power threshold is set?
sens = []
for target_ratio in [0.7, 0.8, 0.9]:
    for min_power in [0.5, 0.8, 0.9]:
        for k, x in df_AOC.groupby('k'):
            kept = power_filter(x, target_ratio=target_ratio, min_power=min_power).query('keep')
            sens.append({
                'target_ratio' : target_ratio, 'min_power' : min_power, 'k' : k,
                'frac_kept' : len(kept)/len(x),
                'median_gen_ratio' : kept['gen_ratio'].median(),
                'median_AOC' : kept['AOC'].median(),
            })
sens = pd.DataFrame(sens)
sens.to_csv(os.path.join(path_results, f'{organ}_AOC_power_sensitivity.csv'), index=False)


##


# Quantify kNN in physical distance as average surface distance
resolution = 0.44186  # Microns per pixel (LCM image resolution)

# Re-calculate unscaled distances
D_xyz_unscaled = pairwise_distances(spatial_coords.values, metric='euclidean')
d = {}
for k in [3, 5, 10, 15]:
    idx, D, _ = mt.pp.kNN_graph(D=D_xyz, k=k, from_distances=True)
    d[k] = np.mean([ 
        np.mean(resolution * D_xyz_unscaled[i, idx[i,:]]) \
        for i in range(D.shape[0]) 
    ])
print(d)