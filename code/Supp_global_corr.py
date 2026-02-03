"""
Fig Supp: right kidney soft cosine distances
"""

import os
import numpy as np
import pandas as pd
import plotting_utils as plu
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import pearsonr
matplotlib.use('macOSX')
plu.set_rcParams()


##


# Utils
def rescale_distances(D):
    """
    Rescale (row-wise) pairwise distances to [0,1].
    """
    min_dist = D[~np.eye(D.shape[0], dtype=bool)].min()
    max_dist = D[~np.eye(D.shape[0], dtype=bool)].max()
    D = (D-min_dist)/(max_dist-min_dist)
    np.fill_diagonal(D, 0)
    return D


##


# Paths
path_main = '/Users/cossa/Desktop/projects/manas_embryo/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results')
path_figures = os.path.join(path_main, 'figures')


##



# Names
samples = ['Heart', 'PD53943o', 'PD53943w']
names = {'PD53943o':'Left kidney', 'PD53943w':'Right kidney', 'Heart':'Heart'}

# Viz
fig, axs = plt.subplots(1,3,figsize=(7,2.5),sharey=True)

for i, sample in enumerate(samples):

    ax = axs[i]

    # Read distances
    D = pd.read_csv(os.path.join(path_results, f'{sample}_genetic_distances.csv'), index_col=0)
    D = rescale_distances(D.values)
    D_xyz = pd.read_csv(os.path.join(path_results, f'{sample}_physical_distances.csv'), index_col=0)
    D_xyz = rescale_distances(D_xyz.values)

    # Global correlation
    d = D.flatten()
    test = d>0
    d = d[test]
    d_xyz = D_xyz.flatten()
    d_xyz = d_xyz[test]
    r, p = pearsonr(d, d_xyz)
    r, p

    # Plot
    ax.plot(d, d_xyz, 'ko', alpha=0.1, markersize=.5)
    sns.regplot(x=d, y=d_xyz , ax=ax, scatter=False)
    plu.format_ax(
        ax=ax, 
        xlabel='Genetic distance', 
        ylabel='Physical distance' if i==0 else '', 
        title=names[sample]
    )
    ax.text(0.7, 0.8, f'r={r:.2f}\np={p:.2f}', 
            transform=ax.transAxes, fontsize=8)
            
fig.tight_layout()
fig.savefig(os.path.join(path_figures, f'Supp_corr_genetic_vs_physical_distances.pdf'))


##


