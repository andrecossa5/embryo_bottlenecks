"""
Figures for the kernel AOC analysis. Reads results/AOC/, writes one standalone PDF per
figure into figures/.

  AOC_decay.pdf          median ratio vs kernel bandwidth, one line per organ, IQR ribbon.
                         The main result: the effect decays with physical distance and
                         converges on 1 (no local structure) as the kernel reaches organ
                         scale.
  AOC_ess.pdf            the companion needed to read the decay honestly: effective
                         neighbourhood size at the same bandwidths, with the ESS < 3 band
                         shaded. The left end of the decay curve is only meaningful where
                         this curve sits above that band.
  AOC_significance.pdf   fraction of samples with FDR <= 0.05, same x axis.
  AOC_power.pdf          median per-sample power to detect a 10% effect, same x axis.
                         Kept as its own figure rather than sharing a twin axis with
                         significance; together the two show effect and power moving in
                         opposite directions with sigma, so no single bandwidth is "best".
  AOC_effect_vs_mde.pdf  per sample, at the reference bandwidth: observed ratio against
                         that sample's own detection limit. The identity line IS the
                         significance boundary - a sample is significant exactly when its
                         ratio falls below its MDE. Effect, power and significance in one
                         geometry, with no aggregation.
  AOC_label_tests.pdf    label-level tests, reduced to the four that carry the argument
                         (see LABEL_KEEP). Genetic ratio (filled) joined to the physical
                         ratio of the same grouping (open): a test far left in genetics
                         but near 1 in physics is anatomy acting independently of
                         proximity.

Colours are the dataviz reference palette, assigned to organs in fixed order and never
cycled or reassigned between figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_utils as plu
from matplotlib.lines import Line2D


##


path_main = '/Users/cossa/Desktop/projects/embryo_bottlenecks/'
path_results = os.path.join(path_main, 'results', 'AOC')
path_figures = os.path.join(path_main, 'figures')
os.makedirs(path_figures, exist_ok=True)

SIGMA_REFERENCE = 400
ESS_FLOOR = 3

# The label tests worth a figure: the brain, whose only readout this is, and the two heart
# groupings that reached significance - heart_group_B (region family) and histo (histology
# family). The rest stay in REPORT_T3_label_tests.csv: heart_strata_ncc / _dmp are
# underpowered, heart_section_round is a negative control, kidney and liver histo are null.
LABEL_KEEP = [
    ('Heart', 'heart_group_B'),
    ('Brain', 'side'),
    ('Brain', 'brain_group'),
    ('Heart', 'histo'),
]

# Categorical palette, fixed order (dataviz reference instance). Organ -> slot.
COLORS = {
    'Heart'        : '#2a78d6',  # slot 1 blue
    'Kidney_left'  : '#eb6834',  # slot 2 orange
    'Kidney_right' : '#1baf7a',  # slot 3 aqua
    'Liver'        : '#eda100',  # slot 4 yellow
    'Brain'        : '#e87ba4',  # slot 5 magenta (label tests only)
}
LABELS = {'Heart':'Heart', 'Kidney_left':'Kidney L', 'Kidney_right':'Kidney R',
          'Liver':'Liver', 'Brain':'Brain'}
NICE_LABEL = {'heart_group_B':'anatomical group', 'histo':'cell type',
              'side':'hemisphere', 'brain_group':'sampling cluster'}
ORDER = ['Heart', 'Kidney_left', 'Kidney_right', 'Liver']

ANNOT    = 8       # in-plot annotations, matching the journal tick-label size
INK      = '#0b0b0b'
INK_2    = '#52514e'
INK_MUTE = '#8a8984'
GRID     = '#e4e3df'

# Journal defaults from plotting_utils. The only override is the PDF font type, which is
# an output requirement (keeps text editable in Illustrator) rather than a style choice;
# type sizes, tick geometry, line and marker defaults are all left as plotting_utils sets
# them. Where a figure needs more room for that type, it asks for it with figsize.
plu.set_rcParams({'pdf.fonttype' : 42})

# Figure sizes: plotting_utils' default (3.5 x 3.5 in, one column) is used for every
# single-axes figure; only the two multi-panel / wide-label figures override it.
FIGSIZE_FACETS = (5.4, 5.4)   # 2 x 2 organ facets
FIGSIZE_LABELS = (6.0, 2.8)   # horizontal test names need the width


def style(ax, xlabel=None, ylabel=None, title=None, rotx=0):
    """plotting_utils formatting, plus a recessive y grid behind the data."""
    plu.format_ax(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, rotx=rotx,
                  reduced_spines=True)
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    return ax


def sigma_axis(ax, sigmas):
    """Log x with one labelled tick per bandwidth (rotation is applied by style())."""
    ax.set_xscale('log')
    ax.set_xticks(sigmas)
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.minorticks_off()


def save(fig, name):
    fig.savefig(os.path.join(path_figures, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.join(path_figures, f'{name}.pdf'))


##


T2 = pd.read_csv(os.path.join(path_results, 'REPORT_T2_distance_decay.csv'))
T3 = pd.read_csv(os.path.join(path_results, 'REPORT_T3_label_tests.csv'))
per_sample = {
    organ : pd.read_csv(os.path.join(path_results, f'{organ}_kernel_AOC_table.csv'))
    for organ in ORDER
}
sigmas = sorted(T2['sigma'].unique())
SIGMA_LABEL = 'Kernel bandwidth $\\sigma$ ($\\mu$m)'


##


# ---- distance decay -------------------------------------------------------------------
fig, ax = plt.subplots()
for organ in ORDER:
    d = T2.query('organ == @organ').sort_values('sigma')
    ax.fill_between(d['sigma'], d['q25_ratio'], d['q75_ratio'], color=COLORS[organ],
                    alpha=0.13, lw=0, zorder=1)
    ax.plot(d['sigma'], d['median_ratio'], color=COLORS[organ], lw=1.6, marker='o',
            ms=4, mec='white', mew=0.5, zorder=3, label=LABELS[organ])
ax.axhline(1, color=INK_MUTE, lw=0.8, ls=(0,(4,3)), zorder=2)
ax.text(sigmas[0]*1.02, 1.004, 'no local structure', color=INK_MUTE, fontsize=ANNOT,
        va='bottom')
sigma_axis(ax, sigmas)
ax.set_ylim(0.72, 1.045)
style(ax, SIGMA_LABEL, 'Genetic distance ratio\n(neighbourhood / organ)',
      'Effect decays with physical distance', rotx=45)
ax.legend(frameon=False, loc='lower right', ncol=2, handlelength=1.4, columnspacing=0.9,
          borderpad=0.2)
save(fig, 'AOC_decay')


# ---- effective neighbourhood size -----------------------------------------------------
fig, ax = plt.subplots()
ax.axhspan(0.8, ESS_FLOOR, color=GRID, alpha=0.9, lw=0, zorder=0)
ax.text(sigmas[-1], ESS_FLOOR*0.93, 'no neighbourhood', color=INK_MUTE, fontsize=ANNOT,
        va='top', ha='right')
for organ in ORDER:
    d = T2.query('organ == @organ').sort_values('sigma')
    ax.plot(d['sigma'], d['median_ess'], color=COLORS[organ], lw=1.6, marker='o', ms=4,
            mec='white', mew=0.5, zorder=3)
    dy = {'Kidney_right':6, 'Heart':-1, 'Kidney_left':-9}.get(organ, 0)
    ax.annotate(LABELS[organ], (d['sigma'].iloc[-1], d['median_ess'].iloc[-1]),
                xytext=(5, dy), textcoords='offset points', color=COLORS[organ],
                fontsize=ANNOT, va='center')
ax.set_yscale('log')
ax.set_yticks([1, 3, 10, 30, 100]); ax.set_yticklabels(['1','3','10','30','100'])
sigma_axis(ax, sigmas)
ax.set_xlim(sigmas[0]*0.9, sigmas[-1]*2.6)
style(ax, SIGMA_LABEL, 'Effective neighbourhood\nsize (ESS)',
      'How many neighbours the effect averages', rotx=45)
save(fig, 'AOC_ess')


# ---- significance and power, one figure each ------------------------------------------
for name, col, ttl, ylab, loc in [
        ('AOC_significance', 'frac_sig', 'Significant samples',
         'Fraction FDR $\\leq$ 0.05', 'upper left'),
        ('AOC_power', 'median_power', 'Power to detect a 10% effect',
         'Median power', 'lower right')]:
    fig, ax = plt.subplots()
    for organ in ORDER:
        d = T2.query('organ == @organ').sort_values('sigma')
        ax.plot(d['sigma'], d[col], color=COLORS[organ], lw=1.6, marker='o', ms=4,
                mec='white', mew=0.5, zorder=3, label=LABELS[organ])
    if col == 'median_power':
        ax.axhline(0.8, color=INK_MUTE, lw=0.8, ls=(0,(4,3)), zorder=2)
        ax.text(sigmas[0]*1.02, 0.815, 'power 0.8', color=INK_MUTE, fontsize=ANNOT,
                va='bottom')
    ax.set_ylim(-0.03, 1.02)
    sigma_axis(ax, sigmas)
    style(ax, SIGMA_LABEL, ylab, ttl, rotx=45)
    ax.legend(frameon=False, loc=loc, ncol=2, handlelength=1.4, columnspacing=0.9,
              borderpad=0.2)
    save(fig, name)


# ---- per-sample effect vs its own detection limit -------------------------------------
fig, axs = plt.subplots(2, 2, figsize=FIGSIZE_FACETS)
for j, (organ, ax) in enumerate(zip(ORDER, axs.ravel())):
    d = per_sample[organ].query('sigma == @SIGMA_REFERENCE')
    sig = d['FDR'] <= 0.05
    lims = (0.55, 1.25)
    ax.plot(lims, lims, color=INK_MUTE, lw=0.8, ls=(0,(4,3)), zorder=1)
    ax.scatter(d.loc[~sig,'mde_ratio'], d.loc[~sig,'ratio'], s=13, facecolor='white',
               edgecolor=COLORS[organ], lw=0.7, zorder=2)
    ax.scatter(d.loc[sig,'mde_ratio'], d.loc[sig,'ratio'], s=16, color=COLORS[organ],
               edgecolor='white', lw=0.4, zorder=3)
    ax.axhline(1, color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xticks([0.6, 0.8, 1.0, 1.2]); ax.set_yticks([0.6, 0.8, 1.0, 1.2])
    ax.set_aspect('equal')
    ax.set_title(f'{LABELS[organ]}  ({int(sig.sum())}/{len(d)} sig.)', loc='left',
                 color=COLORS[organ])
    style(ax, 'Detection limit (MDE)' if j > 1 else None,
          'Observed ratio' if j % 2 == 0 else None)
    if j == 0:
        ax.text(1.22, 0.60, 'significant\nbelow the line', fontsize=ANNOT,
                color=INK_MUTE, ha='right', va='bottom')
fig.subplots_adjust(wspace=0.28, hspace=0.45)
save(fig, 'AOC_effect_vs_mde')


# ---- label-level tests, reduced -------------------------------------------------------
T3p = pd.DataFrame(LABEL_KEEP, columns=['organ','labels']).merge(
    T3, on=['organ','labels'], how='left')
assert T3p['ratio'].notna().all(), 'a kept label test is missing from REPORT_T3'

fig, ax = plt.subplots(figsize=FIGSIZE_LABELS)
XMIN = 0.62
for y, r in T3p.iterrows():
    phys, off = r['phys_ratio'], r['phys_ratio'] < XMIN
    phys_plot = max(phys, XMIN)
    ax.plot([phys_plot, r['ratio']], [y, y], color=GRID, lw=1.0, zorder=1)
    if off:
        # off-scale to the left: caret at the boundary, value written beside it
        ax.scatter(phys_plot, y, s=34, marker='<', facecolor='white', edgecolor=INK_MUTE,
                   lw=0.7, zorder=2, clip_on=False)
        ax.text(phys_plot+0.008, y-0.25, f'{phys:.2f}', fontsize=ANNOT, color=INK_MUTE,
                va='center', ha='left')
    else:
        ax.scatter(phys, y, s=30, facecolor='white', edgecolor=INK_MUTE, lw=0.7, zorder=2)
    ax.scatter(r['ratio'], y, s=44, color=COLORS[r['organ']], edgecolor='white', lw=0.5,
               zorder=3)
    star = '***' if r['FDR'] <= 0.001 else '**' if r['FDR'] <= 0.01 else \
           '*' if r['FDR'] <= 0.05 else 'n.s.'
    ax.text(1.185, y, star, fontsize=ANNOT, color=INK if star != 'n.s.' else INK_MUTE,
            va='center', ha='left')
ax.axvline(1, color=INK_MUTE, lw=0.8, ls=(0,(4,3)), zorder=1)
ax.set_yticks(np.arange(len(T3p)))
ax.set_yticklabels([f"{LABELS[r['organ']]}  ·  {NICE_LABEL.get(r['labels'], r['labels'])}"
                    for _, r in T3p.iterrows()])
for tick, (_, r) in zip(ax.get_yticklabels(), T3p.iterrows()):
    tick.set_color(COLORS[r['organ']])
ax.invert_yaxis()
ax.set_xlim(XMIN, 1.25)
ax.set_ylim(len(T3p)-0.45, -0.55)
style(ax, 'Within-group / between-group ratio', None,
      'Genetic cohesion vs physical proximity')
ax.grid(axis='y', lw=0)
ax.legend(handles=[
    Line2D([],[], marker='o', ls='', mfc=INK_2, mec='white', ms=6, label='genetic distance'),
    Line2D([],[], marker='o', ls='', mfc='white', mec=INK_MUTE, ms=6, label='physical distance'),
], frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.34), ncol=2,
   handletextpad=0.3, columnspacing=1.6)
save(fig, 'AOC_label_tests')
