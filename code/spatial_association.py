"""
Manas' Embryo project Spatial Association analyses.
"""

import os
import numpy as np
import pandas as pd
import mito as mt
from scipy.sparse import csr_matrix
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


# Paths
path_main = '/Users/cossa/Desktop/projects/manas_embryo/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results')


##


# Sample
sample = 'Heart'

# Read mutations table
mut_table = pd.read_csv(os.path.join(path_data, f'{sample}_metadata.csv'))
muts = ( 
    mut_table 
    .pivot_table(index='Sample_ID', columns='mutation_id', values='VAF', fill_value=0)
)
muts = muts.loc[:,(muts>0).any(axis=0)]

# Create sparse co-occuring weight matrix for soft cosine distance calculation
branch_df = pd.read_csv(
    os.path.join(
        path_data, 
        'Filteredmutations_14061_Sample_subset_snv_assigned_to_branches.txt'
    ),
    sep='\t'
)
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

# Mock data for soft cosine calculations
# X = np.array([[.0001,.000002,0],[.1,0,.0000001],[.1,.00000004,.1],[.1,.2,.0000001]])
# X1 = np.array([[1,2,3],[3,4,3],[5,6,7],[1,1,1]])
# W = np.array([[1,1,0],[1,1,1],[0,1,1]])
# W.T == W

# Calculate genetic distances
D = pairwise_soft_cosine(muts.values, W.values)
D = pd.DataFrame(D, index=muts.index, columns=muts.index)
D.to_csv(os.path.join(path_results, f'{sample}_genetic_distances.csv'))


##


# Physical distances
spatial_coords = (
    pd.read_csv(os.path.join(path_data, f'{sample}_final_coorindates_135.csv'))
    # pd.read_csv(os.path.join(path_data, f'{sample}_coordinates.csv'))
    .set_index('name')
    .loc[muts.index] # Ensure proper registration with mut samples
)

# Check 
np.all(spatial_coords.index==muts.index)

# Calculate euclidean distances in physical space
D_xyz = pairwise_distances(spatial_coords.values, metric='euclidean')
D_xyz = pd.DataFrame(D_xyz, index=muts.index, columns=muts.index)
D_xyz.to_csv(os.path.join(path_results, f'{sample}_physical_distances.csv'))


##


# AOC analysis

"""
Genetic distance --> Cas 9 distance (D->D1)
Spatial neighbors --> MT-SNVs kNNs

Observed rank: the rank across all (ordered) sample-i genetic distances 
of the mean genetic distances between sample-i and its k nearest 
neighbors (in physical space)
Random rank: the rank across all (ordered) sample-i genetic distances 
of the mean genetic distances between sample-i and a random selection o k sample

AOC: mean (across random trials) difference between a random rank and the 
observed rank normalized by the number of total samples

p: p_value = np.sum(random_ranks < obs_rank) / n_trials
"""

# Rescale distances first
D = D.values
D_xyz = D_xyz.values

# AOC
n = 30
L = []
for i,k in enumerate(np.arange(2,n,1)):           
    aoc, p = mt.ut.AOC(D, D_xyz, k=k, n_trials=1000)
    FDR = multipletests(p, alpha=0.05, method="fdr_bh")[1] # Multiple testing correction
    L.append(
        pd.DataFrame({'AOC':aoc, 'p':p, 'FDR':FDR}, index=muts.index)
        .assign(k=k)
    )

# Save results
pd.concat(L).to_csv(os.path.join(path_results, f'{sample}_AOC_table.csv'))


##


# Quantify kNN in physical distance as average surface distance
resolution = 0.44186  # Microns per pixel (LCM image resolution)

# Re-calculate unscaled distances
D_xyz_unscaled = pairwise_distances(spatial_coords.values, metric='euclidean')
d = {}
for k in [3, 5, 10, 15, 20, 25]:
    idx, D, _ = mt.pp.kNN_graph(D=D_xyz, k=k, from_distances=True)
    d[k] = np.mean([ 
        np.mean(resolution * D_xyz_unscaled[i, idx[i,:]]) \
        for i in range(D.shape[0]) 
    ])
print(d)


##