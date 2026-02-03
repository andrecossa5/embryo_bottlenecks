"""
Fig Supp: heart clustering and association with anatomical structures
"""

import os
import numpy as np
import pandas as pd
import mito as mt
import plotting_utils as plu
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import fisher_exact
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


def compute_enrichment(df, clone_column, state_column, target_state):

    n = df.shape[0]
    clones = np.sort(df[clone_column].unique())
    target_ratio_array = np.zeros(clones.size)
    oddsratio_array = np.zeros(clones.size)
    pvals = np.zeros(clones.size)

    for i, clone in enumerate(clones):

        test_clone = df[clone_column] == clone
        test_state = df[state_column] == target_state

        clone_size = test_clone.sum()
        clone_state_size = (test_clone & test_state).sum()
        target_ratio = clone_state_size / clone_size
        target_ratio_array[i] = target_ratio
        other_clones_state_size = (~test_clone & test_state).sum()

        # Fisher
        oddsratio, pvalue = fisher_exact(
            [
                [clone_state_size, clone_size - clone_state_size],
                [other_clones_state_size, n - other_clones_state_size],
            ],
            alternative='greater',
        )
        oddsratio_array[i] = oddsratio
        pvals[i] = pvalue

    return clones,target_ratio_array, oddsratio_array, pvals, -np.log10(pvals)


##


# Paths
path_main = '/Users/cossa/Desktop/projects/manas_embryo/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results')
path_figures = os.path.join(path_main, 'figures')


##


# Define sample
sample = 'Heart'

# Read mutations table
mut_table = pd.read_csv(os.path.join(path_data, f'{sample}_metadata.csv'))
muts = ( 
    mut_table 
    .pivot_table(index='Sample_ID', columns='mutation_id', values='VAF', fill_value=0)
)
muts = muts.loc[:,(muts>0).any(axis=0)]

# Choose a clustering solution
fig, axs = plt.subplots(1,4,figsize=(8,2))

for i,k in enumerate([5,10,15,20]):

    D = pd.read_csv(os.path.join(path_results, f'{sample}_genetic_distances.csv'), index_col=0)
    D = rescale_distances(D.values)
    _,_,conn = mt.pp.kNN_graph(D=D, k=k, from_distances=True)

    silh = []
    n_clusters = []
    for r in np.linspace(0.1, 1.0, 10):
        labels = mt.tl.leiden_clustering(conn, res=r)
        if len(np.unique(labels)) >= 2:
            s = silhouette_score(D, metric='precomputed', labels=labels)
            silh.append(s)
            n_clusters.append(len(np.unique(labels)))
        else:
            silh.append(np.nan)
            n_clusters.append(len(np.unique(labels)))   

    ax = axs[i]
    ax.plot(np.linspace(0.1, 1.0, 10), silh)
    plu.format_ax(ax=ax, xlabel='Leiden resolution', ylabel='Silhouette Score', title=f'k={k}', reduced_spines=True)

fig.tight_layout()
fig.savefig(os.path.join(path_figures, 'Leiden_clustering.pdf'))


##


# PCA
model = PCA(n_components=100, random_state=1234)
X_pca = model.fit_transform(muts.values)
# model.explained_variance_ratio_.cumsum()

# Clustering (final choice)
_,_,conn = mt.pp.kNN_graph(D=D, k=15, from_distances=True)

# Assemble for plotting
df_clustering = pd.DataFrame(X_pca[:,:3], index=muts.index, columns=['PC1', 'PC2', 'PC3'])
df_clustering['leiden'] = mt.tl.leiden_clustering(conn, res=0.8).astype(str)


## 


# Cluster colors
colors = plu.create_palette(df_clustering, var='leiden', palette=plu.darjeeling)

# Fig 4d: Genetic clusters in PCA space
fig, ax = plt.subplots(figsize=(2.25,2.25))

plu.scatter(df_clustering, x='PC1', y='PC2', by='leiden', 
            categorical_cmap=colors, ax=ax, size=10, alpha=.7)
plu.add_legend(ax=ax, label='Clonal groups', colors=colors,
               bbox_to_anchor=(0,1), loc='upper left', ncols=1,
               ticks_size=6, label_size=8, artists_size=6)
ax.axis('off')

fig.tight_layout()
fig.savefig(os.path.join(path_figures, 'Supp_clusters_PCA.pdf'))


##


# Supp 4e: Mutations driving genetic variation
muts_to_plot = muts.columns[np.hstack([model.components_.argmax(axis=1)[:3], model.components_.argmin(axis=1)[:3]])]

fig, axs = plt.subplots(1,6,figsize=(8,8/6))

for i,mut in enumerate(muts_to_plot):
    ax = axs[i]
    plu.scatter(df_clustering.join(muts[[mut]]),
                x='PC1', y='PC2', by=mut, 
                continuous_cmap='Blues', ax=axs[i], size=2, alpha=.7,
                kwargs={'edgecolor':'k', 'linewidth':.1})
    plu.format_ax(ax=axs[i], title=mut, title_size=6)
    if i==5:
        plu.add_cbar(muts[mut], palette='Blues', ax=axs[i], label='AF')
    axs[i].axis('off')

fig.tight_layout()
fig.savefig(os.path.join(path_figures, 'Supp_highest_loadings_mutations.pdf'))


##


# Fig 4f: Enrichment pseudoclones in anatomical structures

# Read anatomical structure info
meta = (
    pd.read_csv(os.path.join(path_data, f'Heart_metadata.csv'))
    [['Histo', 'Sample_ID']]
    .drop_duplicates().set_index('Sample_ID')[['Histo']]
)
df_clustering = df_clustering.join(meta)
cross = pd.crosstab(df_clustering['Histo'], df_clustering['leiden'])


##


# Enrichment calculations
df_list = []
for target_state in df_clustering['Histo'].unique():
    L = compute_enrichment(df_clustering, 'leiden', 'Histo', target_state)
    df = pd.DataFrame(L).T
    df.columns = ['cluster', 'perc_in_target_state', 'odds_ratio', 'pvals', 'enrichment']
    df_list.append(df.assign(target_state=target_state))

# Plot
df_enrichment = pd.concat(df_list)
order = df_clustering['Histo'].value_counts(ascending=False).index

# Viz enrichments 
fig, axs = plt.subplots(2,1,figsize=(3.5,3), sharex=True)

plu.bar(
    df_enrichment, 
    x='target_state', 
    y='perc_in_target_state', 
    by='cluster',
    categorical_cmap=colors,
    x_order=order,
    ax=axs[0]
)
plu.format_ax(ax=axs[0], xlabel='', ylabel='% in region', rotx=90, reduced_spines=True)
plu.bar(
    df_enrichment, 
    x='target_state', 
    y='enrichment', 
    by='cluster',
    categorical_cmap=colors,
    x_order=order,
    ax=axs[1]
)
axs[1].axhline(-np.log10(0.05), color='k', linestyle='--')
plu.format_ax(ax=axs[1], xlabel='', ylabel='-log10(p)', rotx=90, reduced_spines=True)

fig.tight_layout()
fig.savefig(os.path.join(path_figures, 'Fig4f.pdf'))


##


# Supp: composition (n of samples) of genetic clusters across anatomical structures
fig, ax = plt.subplots(figsize=(3.5,2.5))

order = cross.sum(axis=1).sort_values(ascending=False).index
data = cross.loc[order]
data_cum = data.cumsum(axis=1)
ys = data.index
    
for i, x in enumerate(data.columns):
    widths = data.values[:,i]
    starts = data_cum.values[:,i] - widths
    ax.barh(ys, widths, left=starts, height=0.95, label=x, color=colors[x])

plu.format_ax(ax=ax, xlabel='n LCM samples', reduced_spines=True)
fig.tight_layout()
fig.savefig(os.path.join(path_figures, 'Supp_heart_clonal_composition.pdf'))


##


