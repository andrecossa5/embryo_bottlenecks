"""
Figures for autocorrelation_muts.py. Reads results/autocorrelation/, writes to figures/.
Kept separate from the analysis so a figure can be re-cut without re-running five minutes
of permutations.

  AC_volcano_<organ>.pdf        effect size C against significance, over every testable
                                mutation in the genome-wide scan. The top 5 hits are
                                labelled (textalloc placement, via plu.volcano) and the 18
                                first-division mutations are ringed, so "are the early
                                mutations unusually structured?" is answerable by eye
                                against the rest of the callset.
  AC_map_<organ>_<mut>.pdf      one mutation per figure: its AF drawn in physical space.
                                This is what the statistic claims - high-AF samples sitting
                                next to each other. The heart is 3D (the only specimen
                                sectioned that way), every other organ 2D.

Every figure is square, title and colorbar included: they are saved at their declared size
rather than with bbox_inches='tight', so a panel keeps its aspect when placed.

The 3D style - corner arrows instead of a box, transparent panes, no tick labels - and the
afmhot_r AF ramp are taken from manas_heart/code/6.spatial_analysis.py, so the two
projects' spatial figures look like they belong together. All formatting goes through
plotting_utils.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_utils as plu
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform


##


class Arrow3D(FancyArrowPatch):
    """A 3D arrow drawn as a FancyArrowPatch (clean 2D-style arrowhead)."""

    def __init__(self, x0, y0, z0, x1, y1, z1, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz0 = (x0, y0, z0)
        self._xyz1 = (x1, y1, z1)

    def do_3d_projection(self, renderer=None):
        (x0, y0, z0), (x1, y1, z1) = self._xyz0, self._xyz1
        xs, ys, zs = proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


##


path_main = '/Users/cossa/Desktop/projects/embryo_bottlenecks/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results', 'autocorrelation')
path_figures = os.path.join(path_main, 'figures')
os.makedirs(path_figures, exist_ok=True)

FDR_SCAN = 0.10        # the scan's reporting cut, matching autocorrelation_muts.py
GRAPH = 'knn'          # the graph the volcano and the examples are read from
N_EXAMPLES = 2         # example mutations per organ
N_LABELS = 5           # mutations labelled in a volcano
MIN_POS_EXAMPLE = 10   # carriers an example must have to be worth drawing

AF_CMAP = 'afmhot_r'   # sequential, as in manas_heart: AF is a magnitude
COLORS = {
    'Heart'        : '#2a78d6',
    'Kidney_left'  : '#eb6834',
    'Kidney_right' : '#1baf7a',
    'Liver'        : '#eda100',
    'Brain'        : '#e87ba4',
}
LABELS = {'Heart':'Heart', 'Kidney_left':'Kidney L', 'Kidney_right':'Kidney R',
          'Liver':'Liver', 'Brain':'Brain'}
ORDER = ['Heart', 'Kidney_left', 'Kidney_right', 'Brain', 'Liver']

GREY = '#bfbebb'
INK_MUTE = '#8a8984'

plu.set_rcParams({'pdf.fonttype' : 42})
SQUARE_3D = (3.5, 3.5)
SQUARE_2D = (4.5, 3.5)


def short(mut):
    """chr7_93731124_C_G -> chr7:93.73Mb C>G, which fits in a label."""
    c, pos, ref, alt = mut.split('_')
    return f'{c}:{int(pos)/1e6:.2f}Mb {ref}>{alt}'


def save(fig, name):
    # No bbox_inches='tight': the declared size IS the figure, title and colorbar included
    fig.savefig(os.path.join(path_figures, f'{name}.pdf'))
    plt.close(fig)
    print('wrote', os.path.join(path_figures, f'{name}.pdf'))


##


scan = pd.read_csv(os.path.join(path_results, 'scan_all_muts.csv'))
df = pd.read_csv(os.path.join(path_data, 'metadata_table.csv'))
ann = pd.read_csv(os.path.join(path_data, 'sample_annotations.csv')).set_index('sample')

scan = scan.query('stratum == "All" and graph == @GRAPH')
organs = [o for o in ORDER if o in set(scan['organ'])]


##


# ---- volcanoes -------------------------------------------------------------------------
for organ in organs:

    d = scan.query('organ == @organ').copy()
    d['nlp'] = -np.log10(d['Pval_perm'].clip(lower=1e-6))
    d['label'] = d['mutation_id'].map(short)
    d = d.set_index('label')
    sig = d['FDR_perm'] <= FDR_SCAN

    # Label the strongest hits only; textalloc places them without overlaps
    labels = d.loc[sig].nlargest(N_LABELS, 'C').index.tolist()

    # plu.volcano splits the points into labelled and everything else, so the
    # significant/not-significant distinction is carried by a per-point colour array
    # handed to it as cmap['others'].
    not_labelled = ~d.index.isin(labels)
    others_colors = np.where(sig, COLORS[organ], GREY)[not_labelled]

    fig, ax = plt.subplots(figsize=SQUARE)
    # Limits are set before plotting, with headroom at the top, so textalloc has somewhere
    # to put the labels; autoscaling off keeps the scatters from taking it back.
    pad_x, pad_y = .06*np.ptp(d['C']), .05*np.ptp(d['nlp'])
    ax.set_xlim(d['C'].min()-pad_x, d['C'].max()+pad_x)
    ax.set_ylim(-pad_y, d['nlp'].max() + 6*pad_y)
    ax.set_autoscale_on(False)

    plu.volcano(
        d, x='C', y='nlp', labels=labels, ax=ax, fig=fig,
        cmap={'labelled':COLORS[organ], 'others':others_colors},
        kwargs_labelled={'s':26, 'edgecolor':'k', 'linewidths':.4},
        kwargs_others={'s':5, 'alpha':.6},
        kwargs_text={'textsize':6, 'linecolor':INK_MUTE, 'linewidth':.4,
                     'max_distance':.35, 'min_distance':.02, 'nbr_candidates':400},
    )
    # the 18 pre-specified first-division mutations, ringed wherever they fall
    early = d.loc[d['is_early'].fillna(False)]
    ax.scatter(early['C'], early['nlp'], s=30, marker='D', facecolor='none',
               edgecolor='k', linewidths=.8, zorder=5)
    # the FDR cut, drawn where it actually falls on the p axis
    if sig.any():
        ax.axhline(-np.log10(d.loc[sig, 'Pval_perm'].max()), color=INK_MUTE, lw=.8,
                   ls=(0,(4,3)), zorder=1)

    plu.format_ax(ax=ax, xlabel='Autocorrelation effect size C',
                  ylabel='$-\\log_{10}$ permutation $p$',
                  title=f'{LABELS[organ]}  ({int(sig.sum())}/{len(d)} at FDR {FDR_SCAN:g})',
                  reduced_spines=True)
    # marker shape carries 'first-division', so this legend needs handles, not swatches
    ax.legend(handles=[
        Line2D([],[], marker='o', ls='', mfc=COLORS[organ], mec='none', ms=4,
               label=f'FDR $\\leq$ {FDR_SCAN:g}'),
        Line2D([],[], marker='o', ls='', mfc=GREY, mec='none', ms=4, label='n.s.'),
        Line2D([],[], marker='D', ls='', mfc='none', mec='k', mew=.8, ms=5,
               label='first-division'),
    ], frameon=False, loc='lower right', fontsize=6.5, handletextpad=.2, borderpad=.1)
    fig.subplots_adjust(left=.19, right=.96, top=.92, bottom=.14)
    save(fig, f'AC_volcano_{organ}')


##


# ---- example maps ----------------------------------------------------------------------
def pick_examples(organ):
    """
    The mutations worth drawing: significant, and carried by enough samples to look like
    anything. Ranking on C alone picks rare mutations - the heart's top hit has 8 carriers
    out of 135, so its map is two dark dots on a pale field, which illustrates nothing.
    First-division mutations are preferred where they qualify, since they are the
    pre-specified question; remaining slots go to the strongest other hits.
    """
    d = scan.query('organ == @organ and FDR_perm <= @FDR_SCAN and '
                   'n_positive >= @MIN_POS_EXAMPLE')
    if d.empty:
        d = scan.query('organ == @organ and FDR_perm <= @FDR_SCAN')
    early = d.loc[d['is_early'].fillna(False)].nlargest(N_EXAMPLES, 'C')
    rest = d.loc[~d['mutation_id'].isin(early['mutation_id'])].nlargest(N_EXAMPLES, 'C')
    return pd.concat([early, rest]).head(N_EXAMPLES)


def style_3d_axes(ax, X, Y, Z, pad=.12, gap=.06):
    """
    manas_heart's 3D look: no box, no ticks, transparent panes, a faint grid, and x/y/z
    arrows out of the back-bottom corner.
    """
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['grid'].update(color=(0, 0, 0, .12), linewidth=.2)
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_linewidth(0)
        axis._axinfo['tick']['inward_factor'] = 0
        axis._axinfo['tick']['outward_factor'] = 0
    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    # A flat slab seen edge-on wastes most of the panel; a higher elevation gives the
    # point cloud a squarer footprint without misrepresenting the geometry.
    ax.view_init(elev=38, azim=45)

    x0, y0, z0 = X.min(), Y.min(), Z.min()
    ends = {
        'x' : (X.max() + np.ptp(X)*pad, y0, z0),
        'y' : (x0, Y.max() + np.ptp(Y)*pad, z0),
        'z' : (x0, y0, Z.max() + max(np.ptp(Z), .2)*pad),
    }
    for lab, (xe, ye, ze) in ends.items():
        ax.add_artist(Arrow3D(x0, y0, z0, xe, ye, ze, mutation_scale=9, lw=.5,
                              arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0, zorder=0))
        ax.text(x0 + (xe-x0)*(1+gap), y0 + (ye-y0)*(1+gap), z0 + (ze-z0)*(1+gap),
                lab, fontsize=7, ha='center', va='center')


for organ in organs:

    examples = pick_examples(organ)
    if examples.empty:
        print(f'  {organ}: no mutation passes FDR {FDR_SCAN:g}, maps skipped')
        continue

    sub = df.query('organ == @organ')
    xyz = ann.loc[sorted(sub['sample'].unique()), ['x','y','z']]
    is3d = xyz['z'].nunique() > 1
    X, Y, Z = xyz['x'].values/1000, xyz['y'].values/1000, xyz['z'].values/1000

    for _, r in examples.iterrows():

        mut = r['mutation_id']
        af = (sub.query('MUT == @mut').set_index('sample')
              .reindex(xyz.index).eval('AD / DP').fillna(0).values)
        vmax = float(np.nanmax(af)) or 1.0

        if is3d:

            fig = plt.figure(figsize=SQUARE_3D)
            ax = fig.add_subplot(projection='3d')
            ax.computed_zorder = False  # respect zorder so the arrows sit behind the dots
            ax.scatter(X, Y, Z, s=22, c=af, cmap=AF_CMAP, vmin=0, vmax=vmax,
                       edgecolor='#bbbbbb', linewidth=.3, depthshade=False, zorder=5)
            # z spans ~1 mm against 5-7 in x and y; drawn to scale it collapses to a plane
            ax.set_box_aspect((np.ptp(X), np.ptp(Y), max(np.ptp(Z), .45*np.ptp(X))),
                              zoom=1.05)
            style_3d_axes(ax, X, Y, Z)
            fig.subplots_adjust(left=.0, right=.80, top=.94, bottom=.0)
        else:
            fig = plt.figure(figsize=SQUARE_2D)
            ax = fig.add_subplot()
            ax.scatter(X, Y, s=26, c=af, cmap=AF_CMAP, vmin=0, vmax=vmax,
                       edgecolor='#bbbbbb', linewidth=.3)
            ax.set_aspect('equal')
            plu.format_ax(ax=ax, xlabel='x (mm)', ylabel='y (mm)', reduced_spines=True)
            fig.subplots_adjust(left=.15, right=.80, top=.86, bottom=.14)

        plu.add_cbar(af, ax=ax, label='AF', palette=AF_CMAP, vmin=0, vmax=vmax,
                     label_size=7, ticks_size=6)
        fig.suptitle(f'{LABELS[organ]}  ·  {short(mut)}\n'
                     f'C = {r["C"]:.2f}, FDR = {r["FDR_perm"]:.3f}, '
                     f'{int(r["n_positive"])}/{len(xyz)} carriers',
                     fontsize=7.5, y=.98)
        save(fig, f'AC_map_{organ}_{mut}')



