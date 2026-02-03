"""
Fig Supp: breakdown of genetic vs physical distance trends in samples
with significant and non-significant AOC results (Heart).
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


# Paths
path_main = '/Users/cossa/Desktop/projects/manas_embryo/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results')
path_figures = os.path.join(path_main, 'figures')


##


# Read AOCs
df_aoc = pd.concat(
    [pd.read_csv(os.path.join(path_results, f'PD53943o_AOC_table.csv')).assign(sample='Left kidney'),
    pd.read_csv(os.path.join(path_results, f'PD53943w_AOC_table.csv')).assign(sample='Right kidney'),
    pd.read_csv(os.path.join(path_results, f'Heart_AOC_table.csv')).assign(sample='Heart')]
)

# Read AOCs Heart
sample = 'Heart'
df = df_aoc.query(f'sample=="{sample}"').set_index('Sample_ID')

# Read distances
D = pd.read_csv(os.path.join(path_results, f'{sample}_genetic_distances.csv'), index_col=0)
D_xyz = pd.read_csv(os.path.join(path_results, f'{sample}_physical_distances.csv'), index_col=0)    
samples = D.index

# Take out arrays and rescale physical distances in microns
D = D.values
resolution = 0.44186 
D_xyz = D_xyz.values * resolution

# Find significant and non-significant AOCs at k=5
significant_aoc = df[(df['k']==5) & (df['FDR']<0.05) & (df['AOC']>0)].index
non_significant_aoc = df[(df['k']==5) & (df['FDR']>0.05) & (df['AOC']>0)].index

#Specify n of 30 neighbors to visualize
n = 10

# Visualize
fig, axs = plt.subplots(1,2,figsize=(5,2.5))

x_ = []
y_ = []
for sample_id in significant_aoc:
    sample_idx = np.where(samples==sample_id)[0][0]
    physical_idx = D_xyz[sample_idx,:].argsort()[1:n]
    x = D_xyz[sample_idx, physical_idx]
    y = D[sample_idx, physical_idx]
    x_.extend(x)
    y_.extend(y)
    axs[0].plot(x, y, 'ko', markersize=.5, alpha=0.5)

sns.regplot(x=x_, y=y_, scatter=False, ax=axs[0])
plu.format_ax(
    ax=axs[0], xlabel='Physical distance (um)', ylabel='Genetic distance', 
    title='Significant AOC',
    reduced_spines=True
)
r, p = pearsonr(x_, y_)
axs[0].text(0.1, 0.8, f'r={r:.2f}\np={p:.2e}', transform=axs[0].transAxes, fontsize=6)

x_ = []
y_ = []
for sample_id in non_significant_aoc:
    sample_idx = np.where(samples==sample_id)[0][0]
    physical_idx = D_xyz[sample_idx,:].argsort()[-n+1:]
    x = D_xyz[sample_idx, physical_idx]
    y = D[sample_idx, physical_idx]
    x_.extend(x)
    y_.extend(y)
    axs[1].plot(x, y, 'ko', markersize=.5, alpha=0.5)

sns.regplot(x=x_, y=y_, scatter=False, ax=axs[1])
plu.format_ax(
    ax=axs[1], xlabel='Physical distance (um)', 
    title='Non-significant AOC',
    reduced_spines=True
)
r, p = pearsonr(x_, y_)
axs[1].text(0.1, 0.8, f'r={r:.2f}\np={p:.2e}', transform=axs[1].transAxes, fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(path_figures, f'Supp_Heart_significant_vs_nonsignificant_AOC.pdf'))


##


#======================== Supp: spatial sampling density differences among samples

D_xyz_left = pd.read_csv(os.path.join(path_results, f'PD53943o_physical_distances.csv'), index_col=0) 
D_xyz_right = pd.read_csv(os.path.join(path_results, f'PD53943w_physical_distances.csv'), index_col=0) 
D_xyz_heart = pd.read_csv(os.path.join(path_results, 'Heart_physical_distances.csv'), index_col=0) 

k = 10
def find_mean_distances(D, k):
    dist = np.zeros(D.shape[0])
    for i in range(D.shape[0]):
        idx = np.argsort(D[i,:])[1:k+1]
        dist[i] = D[i, idx].mean()
    return dist

dist_left = find_mean_distances(D_xyz_left.values, k)
dist_right = find_mean_distances(D_xyz_right.values, k)
dist_heart = find_mean_distances(D_xyz_heart.values, k) 
 
resolution = 0.44186
df = pd.DataFrame({
    'd': np.concatenate([dist_left, dist_right, dist_heart]) * resolution,
    'sample': ['Left kidney']*len(dist_left) + ['Right kidney']*len(dist_right) + ['Heart']*len(dist_heart)
})

fig, ax = plt.subplots(figsize=(2.5,2.7))
plu.bar(df, x='sample', y='d', ax=ax, color='white',
        categorical_cmap=plu.create_palette(df, 'sample', 'Set1'),
        x_order=['Left kidney', 'Right kidney', 'Heart'])
plu.format_ax(
    ax=ax, 
    xlabel='', 
    ylabel=f'Mean physical distance\nto the top {k} NNs (um)',
    reduced_spines=True,
    rotx=90
)
fig.tight_layout()
fig.savefig(os.path.join(path_figures, f'Supp_spatial_sampling_density_across_samples.pdf'))


##
