"""
Fig 4f: Breakdown of AOC across organs and histo-anatomical structures.
"""

import os
import numpy as np
import pandas as pd
import plotting_utils as plu
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('macOSX')
plu.set_rcParams()


##


# Paths
path_main = '/Users/cossa/Desktop/projects/embryo_bottlenecks/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results')
path_figures = os.path.join(path_main, 'figures')


##


#======================== Fig 4d: AOC acoss organs and histo-anatomical structures

# Read AOCs
df_aoc = (
    pd.concat(
        [pd.read_csv(os.path.join(path_results, f'PD53943o_AOC_table.csv')).assign(sample='Left Kidney'),
         pd.read_csv(os.path.join(path_results, f'Heart_AOC_table.csv')).assign(sample='Heart')]
    )
    .query('k==5')
    .set_index('Sample_ID')
)
df_meta = (
    pd.concat(
        [pd.read_csv(os.path.join(path_data, f'PD53943o_metadata.csv')).assign(sample='Left Kidney'),
         pd.read_csv(os.path.join(path_data, f'Heart_metadata.csv')).assign(sample='Heart')]
    )
    [['Sample_ID', 'Histo']].drop_duplicates()
    .set_index('Sample_ID')
)
df = df_meta.join(df_aoc)
df['status'] = np.where(df['FDR']<0.05, 'significant', 'non-significant')

##

fig, axs = plt.subplots(1,2,figsize=(6,3), width_ratios=[18,4], sharey=True)

order = [
    "Left atria",
    "Right ventricle",
    "Right DMP trab",
    "Right IVS",
    "Centre base IVS",
    "Centre tip IVS",
    "Centre IVS",
    "Left IVS",
    "Left ventricle",
    "Left ventricle + DMP",
    "LV trab + compact",
    "Left DMP trab",
    "Left trabeculae",
    "Apex",
    "DMP",
    "Aortic Valve",
    "Aorta",
    "Pulmonary trunk"
]
colors = {'non-significant':'grey', 'significant':'red'}
ax = axs[0]
df_ = df.query('sample=="Heart"')
plu.box(df_, x='Histo', y='AOC', color='white', x_order=order, ax=ax)
sns.stripplot(
    data=df_, x='Histo', y='AOC', ax=ax, 
    order=order, 
    dodge=False,
    hue='status',
    palette=colors,
    edgecolor='k',
    linewidth=0.1,
    size=3
)
ax.get_legend().remove()
plu.format_ax(ax=ax, xlabel='', rotx=90, reduced_spines=True, title='Heart')
ax.axhline(0, color='red', linestyle='--', linewidth=1)

ax = axs[1]
df_ = df.query('sample=="Left Kidney"')
order = df_.groupby('Histo')['AOC'].median().sort_values(ascending=False).index
plu.box(df_, x='Histo', y='AOC', color='white', x_order=order, ax=ax)
sns.stripplot(
    data=df_, x='Histo', y='AOC', ax=ax, 
    order=order, 
    dodge=False,
    hue='status',
    palette=colors,
    edgecolor='k',
    linewidth=0.1,
    size=2
)
plu.format_ax(ax=ax, xlabel='', rotx=90, reduced_spines=True, title='Left Kidney')
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.get_legend().remove()
plu.add_legend(ax=ax, colors=colors, loc='upper left', label='Status', 
               artists_size=6, label_size=8, ticks_size=6)

fig.subplots_adjust(right=0.7, left=0.2, top=.8, bottom=0.45, hspace=0.2)
fig.savefig(os.path.join(path_figures, 'Fig4f.pdf'))


##