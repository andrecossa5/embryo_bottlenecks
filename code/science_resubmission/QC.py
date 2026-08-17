"""
Small figure.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_utils as plu
import matplotlib
from scipy.spatial.distance import pdist
plu.set_rcParams()
matplotlib.use('macOSX')



##


def mean_organ_distance(df, organ):
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

    muts = sub.pivot_table(index='sample', columns='MUT', values='AF', fill_value=0)
    muts = muts.loc[:,(muts>0).any(axis=0)]
    coords = ann.loc[muts.index]
    assert np.all(muts.index == coords.index)
    D_phys_um = pdist(coords[['x','y','z']].values, metric='euclidean')

    return D_phys_um.mean()




path_main = '/Users/cossa/Desktop/projects/embryo_bottlenecks/'
path_results = os.path.join(path_main, 'results', 'AOC')
path_data = os.path.join(path_main, 'data')
path_figures = os.path.join(path_main, 'figures')

# Read the two tables built by build_metadata_table.py
df = pd.read_csv(os.path.join(path_data, 'metadata_table.csv'))
ann = pd.read_csv(os.path.join(path_data, 'sample_annotations.csv')).set_index('sample')
organs = sorted(df['organ'].unique())




fig, axs = plt.subplots(1,2,figsize=(4,2.5))

df_ = (
    df
    .groupby('organ')['sample']
    .nunique()
    .to_frame('n')
    .reset_index()
    .sort_values('n', ascending=False)
)
plu.bar(df_, 'organ', 'n', x_order=df_['organ'].values, ax=axs[0])
plu.format_ax(ax=axs[0], reduced_spines=True, xlabel='', ylabel='n LCMs', rotx=90)

L = []
for organ in df['organ'].unique():
    L.append([organ, mean_organ_distance(df, organ)])

df_ = (
    pd.DataFrame(L, columns=['organ', 'mean_D'])
    .sort_values('mean_D', ascending=False)
)
plu.bar(df_, 'organ', 'mean_D', x_order=df_['organ'].values, ax=axs[1])
plu.format_ax(ax=axs[1], reduced_spines=True, xlabel='', ylabel='Mean dist (um)', rotx=90)

fig.tight_layout()
fig.savefig(os.path.join(path_figures, 'QC.pdf'))



