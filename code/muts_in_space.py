"""
Fig Supp: Individual mutations first division spatial distribution
"""

import os
import numpy as np
import pandas as pd
import plotting_utils as plu
from mito.pp.filters import moran_I
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('macOSX')
plu.set_rcParams()


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


##


# Early cell division mutations
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
interesting_muts = pd.DataFrame({'mutation_id': ids, 'Branch': branches})


##


# Paths
path_main = '/Users/cossa/Desktop/projects/manas_embryo/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results')
path_figures = os.path.join(path_main, 'figures')


##


# Sample
sample = 'PD53943o'
names = {'PD53943o':'Left kidney', 'PD53943w':'Right kidney', 'Heart':'Heart'}

# Reload mutations table
mut_table = pd.read_csv(os.path.join(path_data, f'{sample}_metadata.csv'))
muts = ( 
    mut_table 
    .pivot_table(index='Sample_ID', columns='mutation_id', values='VAF', fill_value=0)
)
muts = muts[interesting_muts['mutation_id'].values]   # Take only earliest mutations

# Load spatial coordinates and physical distances
D_xyz = pd.read_csv(os.path.join(path_results, f'{sample}_physical_distances.csv'), index_col=0)
D_xyz = rescale_distances(D_xyz.values)
spatial_coords = pd.read_csv(os.path.join(path_data, f'{sample}_coordinates.csv'))

# Visualize VAF and calculate Moran's I
fig = plt.figure(figsize=(8,5))

for i in range(interesting_muts.shape[0]):
    
    ID = interesting_muts['mutation_id'][i]
    branch = interesting_muts['Branch'][i]
    ax = fig.add_subplot(3, 6, i+1)
    _, p = moran_I(1-D_xyz, muts[ID].values, 1000)
    spatial_coords['mut'] = muts[ID].values
    plu.scatter(
        spatial_coords, 'x', 'y', by='mut', continuous_cmap='Blues', ax=ax,
        size=5, kwargs={'edgecolor':'k', 'linewidth':.01}
    )
    plu.format_ax(ax=ax, title=f'{ID}\n {branch} (p={p:.3f})', title_size=7)
    ax.axis('off')

fig.suptitle(names[sample])
fig.subplots_adjust(top=.85, bottom=.1, left=.1, right=.9, wspace=.8, hspace=.5)
fig.savefig(os.path.join(path_figures, f'{sample}_spatial_distribution_early_mutations.pdf'))


##


# Cbar 
fig, ax = plt.subplots(figsize=(2,2))
plu.add_cbar(
    np.array([0,.25,.05]), palette='Blues', ax=ax,
    label='AF', layout=( (.1,.5,.8,.05), 'top', 'horizontal' ) 
)
ax.axis('off')
fig.tight_layout()
fig.savefig(os.path.join(path_figures, f'cbar.pdf'))


##