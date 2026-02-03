"""
Fig 4de: Example of AOC neighborhoods.
"""

import os
import numpy as np
import pandas as pd
import mito as mt
import plotting_utils as plu
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('macOSX')
plu.set_rcParams()


##


# Utils
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


#========= Fig 4c: AOC examples on left kidney

# Read data
sample = 'PD53943o'
D = pd.read_csv(os.path.join(path_results, f'{sample}_genetic_distances.csv'), index_col=0)
D_xyz = pd.read_csv(os.path.join(path_results, f'{sample}_physical_distances.csv'), index_col=0)
spatial_coords = (
    pd.read_csv(os.path.join(path_data, f'{sample}_coordinates.csv'))
    .set_index('name')
    .loc[D.index]
)
df_aoc = pd.read_csv(os.path.join(path_results, f'{sample}_AOC_table.csv'))


##




# Compute kNN graph
idx = mt.pp.kNN_graph(D=D_xyz.values, k=30, from_distances=True)[0]
D = rescale_distances(D.values)

# Find examples of good and bad AOC
df_aoc.query('k==5').sort_values('AOC', ascending=False)
df_aoc.query('k==5 and AOC<0.05 and AOC>-0.05').sort_values('AOC', ascending=True)
df_ = spatial_coords.join(df_aoc.query('k==5').set_index('Sample_ID')[['AOC', 'FDR']])


##


# Plot AOC in x/y coordinates
fig, ax = plt.subplots(figsize=(2.5,2.5))
plu.scatter(
    df_, x='x', y='y', by='AOC', 
    continuous_cmap='Reds', ax=ax, size=20, alpha=.7,
    kwargs={'edgecolors':'k', 'linewidths':0.1}
)
plu.add_cbar(df_['AOC'], palette='Reds', ax=ax, label='AOC, k=5')
ax.axis('off')
fig.tight_layout()
fig.savefig(os.path.join(path_figures, 'Fig4c.pdf'))


##


# Examples zoom-ins
fig, ax = plt.subplots(figsize=(2,1.5))

# lcm = 'PD53943o_lo0229'
# lcm = 'PD53943o_lo0090'
lcm = 'PD53943o_lo0104'

lcm_idx = D_xyz.index.get_loc(lcm)
neighbors = [ D_xyz.index[i] for i in idx[lcm_idx] ]

ax.scatter(
    df_.loc[neighbors[:5], 'x'], 
    df_.loc[neighbors[:5], 'y'],
    s=15, c=D[lcm_idx, idx[lcm_idx][:5]], cmap='Spectral', vmin=0, vmax=1,
    edgecolors='k', linewidths=0.8
)
ax.scatter(
    df_.loc[neighbors[5:], 'x'], 
    df_.loc[neighbors[5:], 'y'],
    s=15, c=D[lcm_idx, idx[lcm_idx][5:]], cmap='Spectral', vmin=0, vmax=1,
    edgecolors='k', linewidths=0.1
)
ax.plot(
    df_.loc[lcm, 'x'], 
    df_.loc[lcm, 'y'],
    'ko', markersize=4, markeredgecolor='k', markeredgewidth=0.5
)
plu.add_cbar(D[lcm_idx, idx[lcm_idx]], palette='Spectral', ax=ax, label='d', ticks_size=6)
plu.format_ax(
    ax=ax,
    title=f'{lcm}:\n AOC={df_.loc[lcm, "AOC"]:.2f}; FDR={df_.loc[lcm, "FDR"]:.2f}', 
    title_size=6
)

fig.tight_layout()
ax.axis('off')
fig.savefig(os.path.join(path_figures, f'Fig4d_{lcm}.pdf'))


##
