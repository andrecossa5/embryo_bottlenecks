"""
Fig Supp: right kidney soft cosine distances
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


# Read data
D = pd.read_csv(os.path.join(path_results, 'PD53943w_genetic_distances.csv'), index_col=0)
D = rescale_distances(D.values)

# Plot
fig, ax = plt.subplots(figsize=(4,3.5))

# Hierarchical clustering
order = leaves_list(linkage(squareform(D), method='average'))
ax.imshow(D[np.ix_(order, order)], cmap='Spectral', vmin=.2, vmax=.8)
plu.format_ax(
    ax=ax, 
    xlabel='LCM samples', ylabel='LCM samples', title='Right kidney',
    xticks=[], yticks=[]
)
plu.add_cbar(
    D.flatten(), palette='Spectral', 
    ax=ax, vmin=.2, vmax=.8, label='Genetic distance'
)
fig.tight_layout()

# Save
fig.savefig(os.path.join(path_figures, 'Supp_Right_kidney_distances.png'), dpi=1000)
plu.save_best_pdf_quality(fig, (4,3.5), path_figures, 'Supp_RSupp_Right_kidney_distancesight_kidney_distances.pdf', 1000)


##
