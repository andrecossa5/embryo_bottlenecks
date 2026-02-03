"""
Fig 4f: association AOC and clonal complexity
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
path_main = '/Users/cossa/Desktop/projects/manas_embryo/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results')
path_figures = os.path.join(path_main, 'figures')


##


# Read AOCs
df_aoc = (
    pd.concat(
        [pd.read_csv(os.path.join(path_results, f'PD53943w_AOC_table.csv')).assign(sample='Right Kidney'),
         pd.read_csv(os.path.join(path_results, f'PD53943o_AOC_table.csv')).assign(sample='Left Kidney'),
         pd.read_csv(os.path.join(path_results, f'Heart_AOC_table.csv')).assign(sample='Heart')]
    )
    .query('k==5')
    .set_index('Sample_ID')
)
df_meta = (
    pd.concat(
        [pd.read_csv(os.path.join(path_data, f'PD53943w_metadata.csv')),
         pd.read_csv(os.path.join(path_data, f'PD53943o_metadata.csv')),               
         pd.read_csv(os.path.join(path_data, f'Heart_metadata.csv'))]
    )
    [['Sample_ID', 'Histo']].drop_duplicates()
    .set_index('Sample_ID')
)
df_n_clones = (
    pd.concat(
        [pd.read_csv(os.path.join(path_results, f'PD53943w_n_clones.csv'), index_col=0),
         pd.read_csv(os.path.join(path_results, f'PD53943o_n_clones.csv'), index_col=0),
         pd.read_csv(os.path.join(path_results, f'Heart_n_clones.csv'), index_col=0)]
    )
)


df = df_meta.join(df_aoc).join(df_n_clones.set_index('samples'))
df['status'] = np.where(df['FDR']<0.05, 'significant', 'non-significant')

# Stats
df.groupby(['status', 'n_clones']).size()
df.groupby('status')['n_muts'].describe()
df.groupby('status')['highest_VAF'].describe()


## ...


