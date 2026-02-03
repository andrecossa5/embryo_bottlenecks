"""
Fig 4a: Visualization of clustered soft cosine genetic distances.
"""

import os
import numpy as np
import pandas as pd
import plotting_utils as plu
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib.pyplot as plt
import matplotlib
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


# ======== Fig 4a: clustered genetic distances

SAMPLES = ['Heart', 'PD53943o']
names = {'PD53943o': 'Left kidney', 'Heart': 'Heart'}


# Plot
fig, axs = plt.subplots(2,1,figsize=(3.5,5))

for i,sample in enumerate(SAMPLES):

    ax = axs[i]

    # Read genetic distances and rescale
    D = pd.read_csv(
        os.path.join(path_results, f'{sample}_genetic_distances.csv'), 
        index_col=0
    )
    D = rescale_distances(D.values)
    order = leaves_list(linkage(squareform(D), method='average'))

    # Plot
    ax.imshow(D[np.ix_(order, order)], cmap='Spectral', vmin=.2, vmax=.8)
    plu.format_ax(
        ax=ax, 
        xlabel='LCM samples', ylabel='LCM samples', title=names[sample],
        xticks=[], yticks=[]
    )
    plu.add_cbar(
        D.flatten(), palette='Spectral', 
        ax=ax, vmin=.2, vmax=.8, label='Genetic distance'
    )

fig.subplots_adjust(left=.1, right=0.85, hspace=.3)
fig.savefig(os.path.join(path_figures, 'Fig4a.png'), dpi=1000)
plu.save_best_pdf_quality(fig, (3.5,5), path_figures, 'Fig4a.pdf', 1000)


##
