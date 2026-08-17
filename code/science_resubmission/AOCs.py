"""
Local genetic structure of physical neighbourhoods.

Reads the two tables built by build_metadata_table.py - data/metadata_table.csv (counts)
and data/sample_annotations.csv (coordinates already in microns, isotropic) - so nothing
here has to know about pixel scales. Writes to results/AOC/.

TWO READOUTS

1. KERNEL AOC (kernel_AOC), per sample, for HEART, KIDNEY_LEFT, KIDNEY_RIGHT, LIVER.
   Every other sample enters a sample's neighbourhood with weight
   w_ij = exp(-d_ij^2 / 2 sigma^2), sigma fixed in microns; the statistic is the
   kernel-weighted mean genetic distance, tested against permutations of that sample's
   genetic-distance row. Run over a 9-point sigma grid (100 um to 1600 um), so the result
   is a distance-decay curve rather than a single thresholded number. Every sample is
   reported - nothing is filtered - with per-sample power diagnostics next to the effect
   and the p-value. See STATISTICS REPORTED below for what every column means.

2. LABEL-LEVEL PERMUTATION TEST (label_permutation_test), per anatomical grouping, for
   the BRAIN and the HEART. Is genetic distance smaller between samples sharing an
   anatomical label than between samples with different labels? One number per grouping,
   tested by permuting the labels. This asks the same biological question at a coarser
   grain and never requires a single sample to have a local neighbourhood, which is why
   it is the only readout the brain's sampling design can support.

The rank-based AOC of Weng et al. 2024 (mt.ut.AOC) and its power-filter wrapper have been
REMOVED. Two reasons. Its null draws k-subsets of the n-1-k non-neighbours without
replacement, so the variance of the null carries the finite-population factor (N-k)/(N-1)
and collapses as k approaches the pool size: at n=26 (liver) and k=12 there are 13
possible subsets, null_sd falls to 15% of its k=2 value, and 57% of samples turn
"significant" while the effect itself stays flat at gen_diff = 0.012. And it needed a
keep/drop power filter to handle sparse sampling, which discarded data and required a
pre-specified target effect. The kernel has neither problem: its null is a permutation of
n-1 values (support (n-1)!), and sparsity becomes a reported quantity (ess) instead of a
verdict.

SIGMA IS FIXED IN MICRONS, NEVER ADAPTIVE
The Hotspot-style adaptive bandwidth used in autocorrelation_single_muts.py (sigma_i = i's
own distance to its ceil(k/3)-th neighbour) renormalises every sample to look equally well
sampled. Measured here it reports min ESS 3.9-6.5 for EVERY organ, brain (4.6) and liver
(6.5) included, at a kernel radius of ~1.0-1.1 mm: it manufactures a 7-11 sample
neighbourhood where locality does not physically exist. Under fixed sigma the same organs
honestly report ESS 1.0-2.2. Adaptive weighting is fine for a global statistic; it must
never stand in for a sparsity diagnostic.

STATISTICS REPORTED, per sample and per sigma
  Effect size (read these first; both are on the unrescaled soft-cosine scale)
    ratio      d_gen_kern / d_gen_all. The primary effect size. < 1 = the physical
               neighbourhood is genetically tighter than the organ as a whole; 1 = no
               local structure. Dimensionless, so comparable across sigma, samples and
               organs. 0.93 means "neighbours are 7% genetically closer than average".
    diff       d_gen_all - d_gen_kern. The same contrast additively, in soft-cosine
               units. Needed because a ratio hides how much absolute signal there is:
               two samples can share ratio = 0.93 with diff differing tenfold.
    d_gen_kern kernel-weighted mean genetic distance from the sample (the observed value).
    d_gen_all  plain mean genetic distance to all other samples. This is the exact centre
               of the permutation null, so it is also the null expectation of d_gen_kern.
  Significance (all one-sided: the alternative is "neighbours are CLOSER")
    p          permutation p, (1 + #{null <= observed}) / (n_trials + 1). Floor 1/1001,
               so no sample can report below 1e-3 at the default n_trials.
    FDR        Benjamini-Hochberg over the n samples, within organ and sigma. That is the
               family that matters: one test per sample.
    z          (observed - mean(null)) / sd(null). More sensitive than ratio, and the
               quantity to compare between organs at matched sigma, but its magnitude
               also tracks the spread of that sample's own distances, so it is not an
               effect size. Negative = neighbours closer.
  Power diagnostics (they say whether a null result is informative; NOTHING is filtered)
    ess        Kish effective sample size, (sum w)^2 / sum w^2 - how many samples the
               neighbourhood effectively averages over. ESS ~ 1 means the statistic is
               essentially the distance to the single nearest sample: honest but
               uninformative. Read every null result against this. As a rule of thumb
               ESS < 3 means "this sample had no neighbourhood at this sigma", and the
               summary reports both the fraction of such samples and the effect restricted
               to the rest.
    d_kern_um  kernel-weighted mean physical distance, i.e. the radius the statistic
               actually refers to. Always compare to sigma: d_kern_um >> sigma means the
               kernel found nothing nearby and fell back on distant samples.
    null_sd    sd of the permutation null. The resolution of the test for that sample.
    mde_ratio  minimum detectable effect on the ratio scale, quantile(null, alpha) /
               d_gen_all: the least extreme ratio that would still have reached p <= alpha.
               A sample with mde_ratio 0.95 can resolve a 5% tightening and no less.
    power      probability this sample would have detected a pre-specified effect of
               TARGET_RATIO, Phi((mde_ratio - TARGET_RATIO) * d_gen_all / null_sd). This is
               the power calculation the old power_filter() used to gate on, kept here
               purely as a diagnostic: low power explains a null, it does not disqualify a
               sample.

WHY THE BRAIN IS EXCLUDED FROM THE PER-SAMPLE READOUT
Not because the result was unwelcome - because the sampling geometry cannot support the
statistic at any setting, which sampling_structure() quantifies and writes to disk:

  - The 86 brain samples form 29 islands (single-linkage at 3x the median nearest-
    neighbour distance) with 56% of samples sitting in islands of fewer than 5, spread
    over a 29 x 13 mm brain. The nearest sample from a different island is 3.7 mm away.
    Heart and the kidneys have 0-14% of samples in such islands.
  - Pooled, that leaves no usable scale. At small sigma the brain reports the largest
    apparent effect of any organ (ratio 0.877 at sigma = 200 um) on a median ESS of 1.2 -
    i.e. a one-nearest-neighbour statistic - and frac_sig 0. At larger sigma ESS becomes
    respectable only by absorbing other islands: 20% of every k=10 neighbourhood is in
    the OTHER HEMISPHERE.
  - Stratifying by brain_group does not rescue it, it makes it worse. Groups of 3 are
    excluded outright, k is capped at 2-4, sep_ratio rises to 0.34 (inside an island there
    is no scale separation left - the island IS the neighbourhood), the null degenerates
    (1 distinct subset for a 7-sample group at k=3), and at the only setting with tolerable
    retention the effect is simply gone (gen_ratio 1.012).
  - Splitting by hemisphere changes nothing: at k=10 it reproduces the pooled number to
    three decimals (0.974 vs 0.974), because the islands survive inside each hemisphere.

So the brain is analysed with readout 2 only, where its sampling design is not a problem
and where it does show real structure: a 4% within-hemisphere tightening at p = 3e-4.

THE LIVER IS INCLUDED, WITH A CAVEAT ATTACHED TO IT
n=26, mostly one blob of 21 samples, so the geometry is sound but thin: at sigma = 400 um
62% of samples have ESS < 3 and frac_sig is 0 at every sigma. Report the liver effect size
and the decay curve; do not make a significance claim from it.
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm
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


def kernel_AOC(
    D_gen, D_phys_um, sigma, n_trials=1000, alpha=0.05, target_ratio=0.9,
    random_state=1234
    ):
    """
    Kernel AOC: is a sample genetically closer to its physical surroundings than to the
    organ as a whole? No neighbour count, no filtering - every other sample contributes
    with weight w_ij = exp(-d_ij^2 / 2 sigma^2), normalised to sum to 1 over j != i.

    D_gen       : (n,n) genetic distances, UNRESCALED (soft cosine). Every ratio below is
                  taken on this scale; a rescaled matrix would shift the zero and distort
                  them.
    D_phys_um   : (n,n) physical distances in microns.
    sigma       : kernel bandwidth in microns, FIXED across samples (see module docstring).
    target_ratio: the effect the `power` column is computed against, pre-specified.
                  0.9 = 'neighbours 10% genetically closer'.

    Returns one row per sample; see STATISTICS REPORTED in the module docstring for the
    meaning of every column. Grouped there as effect size (ratio, diff, d_gen_kern,
    d_gen_all), significance (p, z; FDR added by the caller, who owns the family) and
    power diagnostics (ess, d_kern_um, null_sd, mde_ratio, power).
    """

    n = D_gen.shape[0]
    rng = np.random.default_rng(random_state)

    keys = ['d_gen_kern', 'd_gen_all', 'ratio', 'diff', 'z', 'p', 'ess', 'd_kern_um',
            'null_sd', 'mde_ratio', 'power']
    res = { key : np.zeros(n) for key in keys }

    for i in range(n):

        others = np.delete(np.arange(n), i)
        d_phys = D_phys_um[i,others]
        d_gen = D_gen[i,others]

        w = np.exp(-d_phys**2 / (2*sigma**2))
        if not w.sum() > 0:  # Nothing within reach of the kernel: fall back to flat
            w = np.ones_like(w)
        w = w / w.sum()

        obs = w @ d_gen
        d_all = d_gen.mean()
        # Permute which sample sits at which genetic distance, weights held fixed. Support
        # is (n-1)!, so unlike a k-subset null this one cannot collapse at small n.
        null = rng.permuted(np.tile(d_gen, (n_trials,1)), axis=1) @ w
        null_sd = null.std(ddof=1)
        mde = np.quantile(null, alpha) / d_all

        res['d_gen_kern'][i] = obs
        res['d_gen_all'][i] = d_all
        res['ratio'][i] = obs / d_all
        res['diff'][i] = d_all - obs
        res['z'][i] = (obs-null.mean()) / null_sd
        res['p'][i] = (np.sum(null <= obs) + 1) / (n_trials + 1)
        res['ess'][i] = 1 / np.sum(w**2)
        res['d_kern_um'][i] = w @ d_phys
        res['null_sd'][i] = null_sd
        res['mde_ratio'][i] = mde
        res['power'][i] = norm.cdf((mde-target_ratio) * d_all / null_sd)

    return pd.DataFrame(res)


##


def label_permutation_test(D_gen, labels, D_phys_um=None, focal=None, n_perm=10000,
                           random_state=1234):
    """
    Is genetic distance smaller between samples sharing an anatomical label than between
    samples with different labels? One number per grouping, tested by permuting the labels
    over the samples.

    This is the coarse-grained version of the kernel AOC's question, and the only version
    that survives a sampling design made of far-apart islands: it never asks a single
    sample to have a local neighbourhood, so it does not care that the brain's islands
    hold 3-15 samples each. Use it wherever ess says the per-sample statistic has nothing
    to work with.

    D_gen     : (n,n) unrescaled genetic distances.
    labels    : (n,) anatomical grouping. Samples with a missing label are dropped, unless
                `focal` is given.
    D_phys_um : (n,n) physical distances, optional. If given, the same within/between
                contrast is reported for PHYSICAL distance (phys_* columns). Read it
                alongside the genetic result: a label whose groups are also spatially
                compact (phys_ratio << 1) is partly re-measuring physical proximity, so
                its genetic result is not independent of the kernel AOC. A label that is
                spatially mixed (phys_ratio ~ 1) but genetically tight is the informative
                case - anatomy predicting clonal identity beyond mere adjacency.
    focal     : if given, test that ONE group against everything else: 'within' pairs are
                those with both samples in the focal group, 'between' pairs are all the
                rest. Missing labels are then kept as part of the comparison set, which is
                what makes a 7-sample stratum like heart_strata_dmp testable at all.

    Returns a dict: mean within- and between-label distance, their ratio and difference,
    the one-sided permutation p (alternative: within < between), the pair counts behind
    each mean, and the physical-distance companion if D_phys_um was given.
    """

    labels = pd.Series(labels).reset_index(drop=True)
    if focal is None:
        ok = labels.notna().values
        D_gen, labels = D_gen[np.ix_(ok, ok)], labels[ok].values
        D_phys_um = None if D_phys_um is None else D_phys_um[np.ix_(ok, ok)]
    else:
        labels = (labels == focal).values

    iu = np.triu_indices(len(D_gen), 1)
    d = D_gen[iu]
    rng = np.random.default_rng(random_state)

    def contrast(lab):
        if focal is None:
            same = (lab[:,np.newaxis] == lab[np.newaxis,:])[iu]
        else:
            same = (lab[:,np.newaxis] & lab[np.newaxis,:])[iu]
        if same.sum() < 1 or (~same).sum() < 1:
            return np.nan, same
        return d[same].mean() - d[~same].mean(), same

    obs, same = contrast(labels)
    null = np.array([ contrast(rng.permutation(labels))[0] for _ in range(n_perm) ])

    out = dict(
        n=len(labels), n_groups=(2 if focal is not None else len(np.unique(labels))),
        d_within=d[same].mean(), d_between=d[~same].mean(),
        ratio=d[same].mean()/d[~same].mean(), diff=obs,
        p=(np.sum(null <= obs) + 1) / (n_perm + 1),
        n_pairs_within=int(same.sum()), n_pairs_between=int((~same).sum()),
    )
    if D_phys_um is not None:
        dp = D_phys_um[iu]
        out.update(phys_within_um=dp[same].mean(), phys_between_um=dp[~same].mean(),
                   phys_ratio=dp[same].mean()/dp[~same].mean())

    return out


##


def sampling_structure(coords, gap_factor=3, small_island=5):
    """
    Is 'physical neighbourhood' a meaningful notion in this organ's sampling design?

    Samples are linked when they sit closer than gap_factor x the median nearest-neighbour
    distance, and the connected components of that graph are the sampling islands. An
    organ sampled as one contiguous field gives one big island; an organ sampled as
    scattered patches gives many small ones, and a sample alone in a 3-member island
    cannot have a local neighbourhood at any bandwidth.

    This is what justifies including or excluding an organ from the per-sample readout,
    so it is computed and written out rather than asserted.
    """
    X = coords[['x','y','z']].values
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    med = np.median(nn)

    n_islands, lab = connected_components(
        csr_matrix((D < gap_factor*med).astype(int)), directed=False
    )
    sizes = np.bincount(lab)
    same = lab[:,np.newaxis] == lab[np.newaxis,:]
    cross = np.where(~same, D, np.inf).min(axis=1)

    return pd.Series({
        'n' : len(X),
        'nn_median_um' : med,
        'nn_p10_um' : np.percentile(nn, 10),
        'nn_p90_um' : np.percentile(nn, 90),
        'n_islands' : n_islands,
        'largest_island' : sizes.max(),
        f'n_islands_lt{small_island}' : int((sizes < small_island).sum()),
        f'frac_in_islands_lt{small_island}' : float((sizes[lab] < small_island).mean()),
        'median_gap_to_other_island_um' : (np.median(cross[np.isfinite(cross)])
                                           if n_islands > 1 else np.nan),
        'extent_x_mm' : (X[:,0].max()-X[:,0].min())/1000,
        'extent_y_mm' : (X[:,1].max()-X[:,1].min())/1000,
        'extent_z_mm' : (X[:,2].max()-X[:,2].min())/1000,
    })


##


def at_matched_ess(summary, target_ess=10, cols=('median_ratio','frac_sig','median_power',
                                                 'median_d_kern_um','frac_ess_lt3')):
    """
    Each organ's statistics at a MATCHED effective neighbourhood size, interpolated across
    the sigma grid.

    Comparing organs at one fixed sigma is not a fair comparison of their biology: at
    sigma = 400 um the same bandwidth buys a median ESS of 29.7 in the right kidney and
    14.1 in the heart, because the right kidney is sampled twice as densely. An organ
    averaging over more neighbours reports a weaker ratio for purely geometric reasons -
    the kernel reaches further into the organ-wide mean - so a cross-organ ranking at
    fixed sigma partly ranks sampling density.

    Interpolating each organ to the sigma where its median ESS equals target_ess puts them
    on the same footing: same number of effective neighbours, different physical radius,
    and that radius (median_d_kern_um) becomes itself a reportable quantity - the physical
    scale over which each organ's clonal neighbourhood extends.

    Interpolation is linear in log(sigma) against log(median ESS), both of which are
    close to linear over this grid. Organs whose ESS never reaches target_ess return NaN.
    """
    out = {}
    for organ, s in summary.groupby('organ'):
        s = s.reset_index().sort_values('sigma')
        ess, sig = s['median_ess'].values, s['sigma'].values.astype(float)
        if ess.max() < target_ess or ess.min() > target_ess:
            out[organ] = pd.Series({'sigma_at_ess':np.nan,
                                    **{c:np.nan for c in cols}})
            continue
        x = np.log(ess)
        row = {'sigma_at_ess' : float(np.exp(np.interp(np.log(target_ess), x, np.log(sig))))}
        for c in cols:
            row[c] = float(np.interp(np.log(target_ess), x, s[c].values))
        out[organ] = pd.Series(row)

    return pd.DataFrame(out).T.rename_axis('organ')


##


def build_W(muts_columns, path_branches):
    """
    Soft-cosine feature weights: two mutations count towards each other's similarity if
    they ever co-occur on a phylogeny branch, i.e. W = (M M^T) > 0 over the
    mutation x branch incidence. Restricted to the mutations present in this organ.
    """
    branch_df = pd.read_csv(path_branches, sep='\t')
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

    return W.loc[muts_columns, muts_columns]


##


def load_organ(organ, df, ann, path_branches):
    """
    (VAF matrix, annotation, genetic distances, physical distances, n unmeasured) for one
    organ.

    Coordinates come from sample_annotations.csv already in microns and isotropic (x/y/z,
    z = 0 for the 2D organs), so physical distances are a plain Euclidean call.

    A handful of (sample, mutation) pairs have DP == 0, i.e. not measured rather than not
    mutated; pivot_table turns them into 0, the least harmful option available here
    (dropping the mutation would cost a whole column for one sample) and 11 of 6.9M rows
    embryo-wide. Counted and reported rather than left silent.
    """
    sub = df.query('organ == @organ')
    n_unmeasured = int((sub['DP'] == 0).sum())

    muts = sub.pivot_table(index='sample', columns='MUT', values='AF', fill_value=0)
    muts = muts.loc[:,(muts>0).any(axis=0)]
    coords = ann.loc[muts.index]
    assert np.all(muts.index == coords.index)

    D_gen = pairwise_soft_cosine(muts.values, build_W(muts.columns, path_branches).values)
    D_phys_um = pairwise_distances(coords[['x','y','z']].values, metric='euclidean')

    return muts, coords, D_gen, D_phys_um, n_unmeasured


##


# Paths
path_main = '/Users/cossa/Desktop/projects/embryo_bottlenecks/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'AOC')
os.makedirs(path_results, exist_ok=True)
path_branches = os.path.join(
    path_data, 'Filteredmutations_14061_Sample_subset_snv_assigned_to_branches.txt'
)

# Kernel bandwidths, microns: 9 points, roughly geometric, spanning from below the
# densest organ's nearest-neighbour distance (heart, 137 um) to well beyond the sparsest
# organ's (liver, 495 um). Pre-specified, not tuned. Reported as a curve, so no single
# value has to carry the result; 400 um is quoted as the reference point because it is
# ~2-3x the nearest-neighbour distance of the three densely sampled organs.
SIGMAS = [100, 150, 200, 300, 400, 600, 800, 1200, 1600]
SIGMA_REFERENCE = 400

# Organs whose sampling design supports a per-sample statistic. The brain does not; see
# WHY THE BRAIN IS EXCLUDED in the module docstring and results/AOC/sampling_structure.csv
ORGANS_KERNEL = ['Heart', 'Kidney_left', 'Kidney_right', 'Liver']

# Groupings for the label-level test, as (organ, column, focal). focal=None tests all
# groups against each other; a focal value tests that one stratum against everything else,
# which is how a stratum with a single annotated value (heart_strata_dmp) becomes testable.
#
# Two SEPARATE families, each BH-corrected on its own, because they ask different
# questions and must not share a correction:
#   'region'    - anatomical / positional identity: does where a sample sits in the organ
#                 predict its clonal relatedness?
#   'histology' - cell-type identity: does what a sample is made of predict it? Tested
#                 with `histo`, the one annotation available for all five organs, which
#                 makes it the only grouping comparable across organs.
LABEL_TESTS = [
    ('Brain', 'side', None, 'region'),
    ('Brain', 'brain_group', None, 'region'),
    ('Heart', 'heart_group_B', None, 'region'),
    ('Heart', 'heart_strata_ivs', None, 'region'),
    ('Heart', 'heart_strata_ncc', 'NCC+R', 'region'),
    ('Heart', 'heart_strata_dmp', 'CloseR2', 'region'),
    ('Heart', 'heart_section_round', None, 'region'),  # negative control
] + [ (organ, 'histo', None, 'histology')
      for organ in ['Brain', 'Heart', 'Kidney_left', 'Kidney_right', 'Liver'] ]

N_TRIALS = 1000
N_PERM_LABEL = 10_000
ALPHA = 0.05
TARGET_RATIO = 0.9  # the effect the `power` column is computed against
TARGET_ESS = 10     # effective neighbourhood size at which organs are compared (see at_matched_ess)


##


# Read the two tables built by build_metadata_table.py
df = pd.read_csv(os.path.join(path_data, 'metadata_table.csv'))
ann = pd.read_csv(os.path.join(path_data, 'sample_annotations.csv')).set_index('sample')
organs = sorted(df['organ'].unique())


##


# Sampling geometry of every organ, brain included: this is the evidence on which the
# brain is left out of the per-sample readout
structure = pd.DataFrame({
    organ : sampling_structure(ann.loc[ann['organ'] == organ])
    for organ in organs
}).T.rename_axis('organ')
structure.to_csv(os.path.join(path_results, 'sampling_structure.csv'))
pd.set_option('display.width', 250)
print('===== SAMPLING GEOMETRY (islands: single-linkage at 3x median NN distance)')
print(structure.round(2).to_string())
print(f'\n--> per-sample kernel AOC run for: {", ".join(ORGANS_KERNEL)}')
print(f'--> excluded from it: {", ".join(o for o in organs if o not in ORGANS_KERNEL)}'
      ' (see module docstring)')


##


D_gen_cache, kernel_all = {}, []

for organ in organs:

    muts, coords, D_gen, D_phys_um, n_unmeasured = load_organ(organ, df, ann, path_branches)
    D_gen_cache[organ] = (D_gen, D_phys_um, coords)
    print(f'\n=== {organ}: n={muts.shape[0]} samples, {muts.shape[1]} mutations '
          f'({n_unmeasured} unmeasured (sample, mutation) pairs set to 0)')

    pd.DataFrame(D_gen, index=muts.index, columns=muts.index).to_csv(
        os.path.join(path_results, f'{organ}_genetic_distances.csv'))
    pd.DataFrame(D_phys_um, index=muts.index, columns=muts.index).to_csv(
        os.path.join(path_results, f'{organ}_physical_distances_um.csv'))

    if organ not in ORGANS_KERNEL:
        print('  per-sample kernel AOC skipped (sampling geometry, see docstring)')
        continue

    L = []
    for sigma in SIGMAS:
        k_df = kernel_AOC(D_gen, D_phys_um, sigma=sigma, n_trials=N_TRIALS, alpha=ALPHA,
                          target_ratio=TARGET_RATIO)
        # BH within (organ, sigma): the family is one test per sample
        k_df['FDR'] = multipletests(k_df['p'], alpha=ALPHA, method='fdr_bh')[1]
        k_df.index = muts.index
        L.append(k_df.assign(sigma=sigma))
    kern = pd.concat(L)
    kern.to_csv(os.path.join(path_results, f'{organ}_kernel_AOC_table.csv'))

    def summarise_kernel(x):
        informative = x.loc[x['ess']>=3]
        return pd.Series({
            'n' : len(x),
            # effect size
            'median_ratio' : x['ratio'].median(),
            'q25_ratio' : x['ratio'].quantile(.25),
            'q75_ratio' : x['ratio'].quantile(.75),
            'median_diff' : x['diff'].median(),
            # significance
            'median_z' : x['z'].median(),
            'frac_sig' : np.mean(x['FDR']<=ALPHA),
            'n_sig' : int(np.sum(x['FDR']<=ALPHA)),
            # power diagnostics
            'median_ess' : x['ess'].median(),
            'min_ess' : x['ess'].min(),
            'frac_ess_lt3' : np.mean(x['ess']<3),
            'median_d_kern_um' : x['d_kern_um'].median(),
            'median_null_sd' : x['null_sd'].median(),
            'median_mde_ratio' : x['mde_ratio'].median(),
            'median_power' : x['power'].median(),
            'frac_power_ge80' : np.mean(x['power']>=0.8),
            # the same, restricted to samples that actually had a neighbourhood
            'n_ess_ge3' : len(informative),
            'median_ratio_ess_ge3' : informative['ratio'].median(),
            'frac_sig_ess_ge3' : np.mean(informative['FDR']<=ALPHA),
        })

    kern_summary = kern.groupby('sigma').apply(summarise_kernel, include_groups=False)
    kern_summary.to_csv(os.path.join(path_results, f'{organ}_kernel_AOC_summary.csv'))
    kernel_all.append(kern_summary.assign(organ=organ).reset_index())
    print(kern_summary[['median_ratio','median_z','frac_sig','median_ess','frac_ess_lt3',
                        'median_d_kern_um','median_mde_ratio','median_power']].round(3).to_string())


##


# Label-level tests, on the same genetic distances
rows = []
for organ, col, focal, family in LABEL_TESTS:
    D_gen, D_phys_um, coords = D_gen_cache[organ]
    labels = coords[col]
    if labels.notna().sum() < 4:
        continue
    row = label_permutation_test(D_gen, labels.values, D_phys_um=D_phys_um,
                                 focal=focal, n_perm=N_PERM_LABEL)
    rows.append({'organ':organ, 'family':family, 'labels':col,
                 'focal':(focal or 'all groups'), **row})
labels_all = pd.DataFrame(rows)
# BH within family, never across: 'region' and 'histology' are different questions
labels_all['FDR'] = np.nan
for family, idx in labels_all.groupby('family').groups.items():
    labels_all.loc[idx,'FDR'] = multipletests(
        labels_all.loc[idx,'p'], alpha=ALPHA, method='fdr_bh')[1]
labels_all = labels_all.sort_values(['family','p'])
labels_all.to_csv(os.path.join(path_results, 'label_permutation_tests.csv'), index=False)


##

kernel_all = pd.concat(kernel_all).set_index(['organ','sigma'])
kernel_all.to_csv(os.path.join(path_results, 'kernel_AOC_summary_all_organs.csv'))

##


# ---- The three tables to report ----------------------------------------------------
# T1  per-organ effect, significance and power at a matched effective neighbourhood size
# T2  the decay of the effect with physical distance (tidy, one row per organ x sigma)
# T3  the label-level tests, two families

T1 = pd.concat([
    at_matched_ess(kernel_all, target_ess=TARGET_ESS).add_prefix('ess10_'),
    kernel_all.xs(SIGMA_REFERENCE, level='sigma')[
        ['n','median_ratio','q25_ratio','q75_ratio','median_diff','median_z','frac_sig',
         'n_sig','median_ess','frac_ess_lt3','median_mde_ratio','median_power',
         'frac_power_ge80','median_ratio_ess_ge3']
    ].add_prefix(f'sig{SIGMA_REFERENCE}_'),
    structure[['nn_median_um','n_islands','frac_in_islands_lt5']],
], axis=1)
T1.to_csv(os.path.join(path_results, 'REPORT_T1_organ_summary.csv'))

T2 = kernel_all.reset_index()[
    ['organ','sigma','median_ratio','q25_ratio','q75_ratio','median_diff','median_z',
     'frac_sig','n_sig','median_ess','frac_ess_lt3','median_d_kern_um','median_power']
]
T2.to_csv(os.path.join(path_results, 'REPORT_T2_distance_decay.csv'), index=False)

labels_all.to_csv(os.path.join(path_results, 'REPORT_T3_label_tests.csv'), index=False)

print(f'\n\n===== T1: per organ, at matched ESS={TARGET_ESS} and at sigma={SIGMA_REFERENCE} um')
print(T1.round(3).T.to_string())
print(f'\n===== KERNEL AOC, full summary at sigma={SIGMA_REFERENCE} um')
print(kernel_all.xs(SIGMA_REFERENCE, level='sigma').round(3).T.to_string())
print('\n===== T2: DISTANCE DECAY: median ratio by bandwidth (um)')
print(kernel_all['median_ratio'].unstack('sigma').round(3).to_string())
print('\n===== effective neighbourhood size (median ESS) by bandwidth (um)')
print(kernel_all['median_ess'].unstack('sigma').round(1).to_string())
print('\n===== fraction of samples with ESS < 3 by bandwidth (um)')
print(kernel_all['frac_ess_lt3'].unstack('sigma').round(2).to_string())
print('\n===== significant fraction (BH within organ x sigma) by bandwidth (um)')
print(kernel_all['frac_sig'].unstack('sigma').round(3).to_string())
print('\n===== T3: LABEL-LEVEL PERMUTATION TESTS (BH within family)')
for family, x in labels_all.groupby('family'):
    print(f'\n-- family: {family}')
    print(x.drop(columns=['family','diff','n_pairs_within','n_pairs_between'])
           .round(4).to_string(index=False))
