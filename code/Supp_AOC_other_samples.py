"""
Fig Supp: AOC right kidney
"""

import os
import numpy as np
import pandas as pd
import plotting_utils as plu
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


# Read AOC results
df = pd.read_csv(os.path.join(path_results, f'PD53943w_AOC_table.csv'))


## 


# Plot AOC, p-values, proportion AOC>0 with 95% CIs
fig, axs = plt.subplots(1,3,figsize=(5,2), constrained_layout=True)

# AOC plot with 95% CI
grouped_aoc = df.groupby('k')['AOC']
mean_aoc = grouped_aoc.mean()
sem_aoc = grouped_aoc.sem()
ci_lower_aoc = mean_aoc - 1.96 * sem_aoc
ci_upper_aoc = mean_aoc + 1.96 * sem_aoc
axs[0].plot(mean_aoc, label='Mean')
axs[0].fill_between(mean_aoc.index, ci_lower_aoc, ci_upper_aoc, alpha=0.3)
axs[0].set_ylabel('Agreement of\n Closeness (AOC)')
axs[0].set_xlabel('k')

# % significant AOC
grouped_prop = df.groupby('k').apply(lambda x: pd.Series({
    'prop': (x['FDR']<0.05).sum() / x.shape[0],
    'n': x.shape[0]
}))
grouped_prop['se'] = np.sqrt(grouped_prop['prop'] * (1 - grouped_prop['prop']) / grouped_prop['n'])
grouped_prop['ci_lower'] = grouped_prop['prop'] - 1.96 * grouped_prop['se']
grouped_prop['ci_upper'] = grouped_prop['prop'] + 1.96 * grouped_prop['se']
axs[1].plot(grouped_prop.index, grouped_prop['prop'])
axs[1].fill_between(grouped_prop.index, grouped_prop['ci_lower'], grouped_prop['ci_upper'], alpha=0.3)
axs[1].set_ylabel('FDR<0.05 fraction')
axs[1].set_xlabel('k')
axs[1].set_xticks([5,15,25])

# Fraction AOC>0 with 95% CI
grouped_prop = df.groupby('k').apply(lambda x: pd.Series({
    'prop': (x['AOC']>0).sum() / x.shape[0],
    'n': x.shape[0]
}))
grouped_prop['se'] = np.sqrt(grouped_prop['prop'] * (1 - grouped_prop['prop']) / grouped_prop['n'])
grouped_prop['ci_lower'] = grouped_prop['prop'] - 1.96 * grouped_prop['se']
grouped_prop['ci_upper'] = grouped_prop['prop'] + 1.96 * grouped_prop['se']
axs[2].plot(grouped_prop.index, grouped_prop['prop'])
axs[2].fill_between(grouped_prop.index, grouped_prop['ci_lower'], grouped_prop['ci_upper'], alpha=0.3)
axs[2].set_ylabel('AOC>0 fraction')
axs[2].set_xlabel('k')
axs[2].set_xticks([5,15,25])

fig.suptitle('Right kidney')
# fig.savefig(os.path.join(path_figures, 'Fig4b.pdf'))
fig.savefig(os.path.join(path_figures, f'Supp_AOC_Right_kidney.pdf'))


##

