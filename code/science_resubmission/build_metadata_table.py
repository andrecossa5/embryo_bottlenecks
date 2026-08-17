"""
Build the two tables every spatial script should read:

  data/metadata_table.csv     long, one row per (sample, mutation):
                              sample, organ, MUT, AD, DP, AF, x, y, z
  data/sample_annotations.csv one row per sample: coordinates + every annotation
                              carried over from the source files (493 rows)

Sample-level annotation is deliberately NOT repeated inside metadata_table.csv: it would
be duplicated 14,064 times per sample and triple the file for nothing. Join on `sample`.

WHY THIS EXISTS
The inputs are scattered over eight files with three different sample-ID columns, two
mutation-ID spellings and - critically - three different physical scales. Every
downstream script re-did that join with its own idea of the pixel size. Here a spatial
distance matrix is just pairwise_distances() on x/y/z, with no further conversion.

SAMPLE INCLUSION - samples with BOTH counts and coordinates:

    Kidney_left   109   PD53943o_metadata.csv      + PD53943o_coordinates.csv
    Kidney_right  137   PD53943w_metadata.csv      + PD53943w_coordinates.csv
    Heart         135   Heart_metadata.csv         + 11pcw_104_coordinates_FULL_v2.xlsx
    Brain          86   Final_Dataframe_*.csv      + FINAL_df_brain_liver_annotations_*.csv
    Liver          26   Final_Dataframe_*.csv (11) + FINAL_df_brain_liver_annotations_*.csv
                        LongData_NV_NR_VAF_*.csv (15)

BRAIN SIDE AND GROUPS
`side` (Left 43 / Right 43 among the samples with coordinates) is the hemisphere, and it
agrees with the mutation table's own Site column on every sample. `brain_group` is a
finer label: 10 spatially disjoint sampling clusters, groups 1-5 all Left and groups 6-10
all Right - not 1=Left / 2=Right. The clusters are genuinely far apart: within-group
nearest-neighbour distance is 298 um, but the nearest sample from another group sits
3.7 mm away, over a 29 x 13 mm brain. That gap is why the brain AOC has almost no power
at low k, and it makes brain_group the natural stratum (or a reason to run the two
hemispheres separately) rather than treating the brain as one field.

The three per-organ metadata files hold exactly the coordinate-matched sample sets, so
they are used instead of the 2.7 GB embryo-wide table (checked: 135/109/137 samples,
14,064 mutations each - identical panel).

Liver merges TWO disjoint count sources: lo0004-lo0019 (11 samples) exist only in the
embryo-wide table, lo0020-lo0035 (16) only in LongData. No sample overlap, same 14,064
mutations, so they concatenate cleanly. lo0031 is dropped (row present, NaN x/y); brain
loses 35 of 107 samples the same way.

COORDINATE SCALE - all x/y/z in MICRONS, isotropic, ready for Euclidean distances
    kidneys, brain, liver : x,y = pixels * 0.46            (LCM image resolution)
    heart                 : x,y = pixels * 0.46 * 16 = 7.36
                            z   = Section_number * 16      (16 um LCM sections)
    z = 0 for every 2D organ, so the column is always safe to pass to
    pairwise_distances(): within a 2D organ it contributes nothing.

The heart needs its own factor because its coordinates were exported from a 16x
downsampled image, not the full-resolution one. Proof, from Biopsy_Size in the xlsx
(median cut area 47,020 um^2 = 245 um equivalent diameter): at 0.46 um/unit, 85% of all
1,007 within-section sample pairs would physically overlap, which is impossible. At
7.36 um/unit only 3.7% do, and non-overlap brackets the factor at ~6.7-8.2 um/unit.
0.46 * 16 is the value inside that bracket consistent with the LCM resolution.

HEART z - AN ASSUMPTION INHERITED FROM THE SOURCE FILE
Section_number * 16 reproduces Heart_final_coorindates_135.csv exactly (asserted below).
But sections 1, 2 and 3 are three separate cutting rounds (dates 19012023, 10102022,
15022023 - the last being the PD53943w one), whereas sections 5-59 are the systematic
B1-C3-S* series. Their 16 um spacing was assigned by whoever built the sheet, not read
off a section count, and 61 of 135 heart samples (45%) live there. If those rounds are
really further apart, the local neighbourhoods of those samples change. Flagged, not
fixed: `heart_section_round` marks them so a sensitivity check is one filter away.

Eight heart samples are annotated on two consecutive sections (all pulmonary trunk,
23,551 um^2). The lower section is kept, reproducing the existing file;
`heart_n_section_rows` marks them.
"""

import os
import numpy as np
import pandas as pd


##


# Paths
path_main = '/Users/cossa/Desktop/projects/embryo_bottlenecks/'
path_data = os.path.join(path_main, 'data')

# Physical scales (see docstring)
RES = 0.46             # um per pixel, LCM images
RES_HEART = RES * 16   # um per unit, heart x/y (16x downsampled export)
SECTION_UM = 16        # um per LCM section, i.e. per heart Section_number step

COUNT_COLS = ['sample', 'organ', 'MUT', 'AD', 'DP', 'AF', 'x', 'y', 'z']

# Canonical count-column names of the two file layouts
MAP_WIDE = {'Sample_ID':'sample', 'mutation_id':'MUT', 'NV':'AD', 'NR':'DP'}
MAP_LONG = {'Sample':'sample', 'Mutation_ID':'MUT', 'NV':'AD', 'NR':'DP'}

# Sample-level annotation to carry over from the count tables, renamed to snake_case
ANNOT_WIDE = {
    'Sample_ID':'sample', 'Organ':'slide_description', 'Histo':'histo',
    'Bulk_phenotype':'bulk_phenotype', 'Site':'site', 'Original_ID':'original_id',
    'Project_ID':'project_id', 'Germ_layer':'germ_layer', 'Germ_Layer2':'germ_layer2',
}
ANNOT_LONG = {
    'Sample':'sample', 'Organ':'slide_description', 'Histo':'histo',
    'Bulk_phenotype':'bulk_phenotype', 'Site':'site', 'Original_ID':'original_id',
    'Project_ID':'project_id',
}

BIG = 'Final_Dataframe_heart_annotations_raw_trophoblasts.csv'

# organ -> [(count file, count map, annotation map)]; >1 source is concatenated
SOURCES = {
    'Kidney_left'  : [('PD53943o_metadata.csv', MAP_WIDE, ANNOT_WIDE)],
    'Kidney_right' : [('PD53943w_metadata.csv', MAP_WIDE, ANNOT_WIDE)],
    'Heart'        : [('Heart_metadata.csv', MAP_WIDE, ANNOT_WIDE)],
    'Brain'        : [(BIG, MAP_WIDE, ANNOT_WIDE)],
    'Liver'        : [(BIG, MAP_WIDE, ANNOT_WIDE),
                      ('LongData_NV_NR_VAF_with_MutationID.csv', MAP_LONG, ANNOT_LONG)],
}


##


def parse_heart_xlsx():
    """
    Heart coordinates and annotation from 11pcw_104_coordinates_FULL_v2.xlsx.

    Sheet Q1 holds two vertically stacked tables with DIFFERENT column alignment, so the
    columns are taken positionally per block rather than from a header:
      rows   3-84  : 82 rows / 74 samples, the systematic HEART PD53943V B1-C3-S* series
      rows  90-151 : 61 rows / 61 samples, incl. all 15 PD53943w, with NDPI annotation
    The last three columns of both are X coord / Y coord / Section number.

    Sheets Q2/Q3/Q4 add one pre-specified anatomical grouping each over the same
    manifest, keyed by PDID; they are carried over as heart_strata_*.
    """
    path = os.path.join(path_data, '11pcw_104_coordinates_FULL_v2.xlsx')
    raw = pd.read_excel(path, 'Q1', header=None)

    # (block rows, {position -> name})
    blocks = [
        (slice(3, 85), {21:'sample', 8:'heart_slide_description', 10:'cutting_date',
                        13:'block', 14:'slide', 15:'section', 16:'cutting_well',
                        17:'tissue_type', 19:'biopsy_area_um2',
                        30:'X', 31:'Y', 32:'section_number'}),
        (slice(90, 152), {22:'sample', 8:'cutting_date', 11:'block', 12:'slide',
                          13:'section', 14:'cutting_well', 15:'tissue_type',
                          16:'biopsy_area_um2', 18:'oxford_id',
                          20:'anatomical_annotation', 21:'comments',
                          30:'X', 31:'Y', 32:'section_number'}),
    ]
    L = []
    for rows, cols in blocks:
        block = raw.iloc[rows]
        L.append(pd.DataFrame({name: block[pos].values for pos, name in cols.items()}))
    df = pd.concat(L, ignore_index=True).dropna(subset=['sample', 'X', 'Y', 'section_number'])

    for col in ['X', 'Y', 'section_number', 'biopsy_area_um2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # One LCM cut can be annotated on more than one section: keep the lowest section,
    # which is what Heart_final_coorindates_135.csv did
    df['heart_n_section_rows'] = df.groupby('sample')['sample'].transform('size')
    df = df.sort_values(['sample', 'section_number']).drop_duplicates('sample', keep='first')

    # Sections 1-3 are separate cutting rounds, 5-59 the systematic series (see docstring)
    df['heart_section_round'] = np.where(
        df['section_number'] <= 3, 'early_round', 'B1-C3-S_series'
    )
    df = df.rename(columns={'section_number':'heart_section_number',
                            'biopsy_area_um2':'heart_biopsy_area_um2',
                            'tissue_type':'heart_tissue_type',
                            'anatomical_annotation':'heart_anatomical_annotation',
                            'oxford_id':'heart_oxford_id',
                            'cutting_date':'heart_cutting_date',
                            'block':'heart_block', 'slide':'heart_slide',
                            'section':'heart_section', 'cutting_well':'heart_cutting_well'})

    # Pre-specified anatomical strata, one per extra sheet
    for sheet, col, name in [('Q2', 'Right left IVS', 'heart_strata_ivs'),
                             ('Q3', 'Neural Crest', 'heart_strata_ncc'),
                             ('Q4', 'DMP and Right ventricle', 'heart_strata_dmp')]:
        s = pd.read_excel(path, sheet)[['PDID', col]].dropna()
        # A twice-sectioned sample is annotated twice; the two rows must agree
        assert s.groupby('PDID')[col].nunique().le(1).all(), f'{sheet}: conflicting {col}'
        s = s.drop_duplicates('PDID').set_index('PDID')[col].rename(name)
        df = df.join(s, on='sample')

    # ...and the grouping already used downstream
    g = pd.read_csv(os.path.join(path_data, 'HeartCoords_HEART_assignments.csv'))
    df = df.join(g.drop_duplicates('sample_id').set_index('sample_id')['group_B']
                  .rename('heart_group_B'), on='sample')

    df['x'] = df['X'] * RES_HEART
    df['y'] = df['Y'] * RES_HEART
    df['z'] = df['heart_section_number'] * SECTION_UM

    # The rebuild must reproduce the file the previous analyses ran on
    ref = (
        pd.read_csv(os.path.join(path_data, 'Heart_final_coorindates_135.csv'))
        .rename(columns={'name':'sample'}).set_index('sample')
    )
    chk = df.set_index('sample').reindex(ref.index)
    assert len(ref) == len(df) == 135, f'expected 135 heart samples, got {len(df)}'
    assert np.allclose(chk['X'], ref['x']) and np.allclose(chk['Y'], ref['y']), \
        'heart x/y disagree with Heart_final_coorindates_135.csv'
    assert np.allclose(chk['z'], ref['z']), 'Section_number * 16 != z of the reference file'

    return df.drop(columns=['X','Y','comments']).set_index('sample')


##


def load_coords():
    """
    {organ: DataFrame indexed by sample} with x/y/z in microns plus whatever
    coordinate-file annotation exists. All scale reconciliation happens here.
    """
    coords = {}

    # Kidneys: 2D, LCM pixels
    for organ, specimen in [('Kidney_left','PD53943o'), ('Kidney_right','PD53943w')]:
        df = (
            pd.read_csv(os.path.join(path_data, f'{specimen}_coordinates.csv'), encoding='utf-8-sig')
            .rename(columns={'name':'sample'}).dropna(subset=['x','y']).set_index('sample')
        )
        coords[organ] = pd.DataFrame({'x':df['x']*RES, 'y':df['y']*RES, 'z':0.0})

    # Heart: 3D, own x/y scale, z from Section_number
    coords['Heart'] = parse_heart_xlsx()

    # Brain and liver: 2D, LCM pixels, one shared annotation file. Use the FINAL_ version:
    # same 134 samples and identical coordinates where both files have them, but it fills
    # in x/y for 14 further brain samples (86 usable, not 72) and populates Brain_Group,
    # which the earlier file left entirely empty.
    annot = pd.read_csv(
        os.path.join(path_data, 'FINAL_df_brain_liver_annotations_unique_for_Andrea.csv'),
        encoding='utf-8-sig'
    ).dropna(subset=['x','y'])
    keep = {'Germ_layer':'germ_layer_annot', 'Germ_Layer2':'germ_layer2_annot',
            'Histo':'histo_annot', 'Brain_Group':'brain_group', 'Side':'side'}
    keep = {k:v for k, v in keep.items() if annot[k].notna().any()}
    for organ in ['Brain', 'Liver']:
        df = annot.query('Bulk_phenotype == @organ').set_index('Sample_ID')
        coords[organ] = pd.concat(
            [pd.DataFrame({'x':df['x']*RES, 'y':df['y']*RES, 'z':0.0}),
             df[list(keep)].rename(columns=keep)], axis=1
        )

    for organ, df in coords.items():
        assert not df.index.duplicated().any(), f'{organ}: duplicated sample coordinates'

    return coords


##


def load_counts(path_csv, samples, count_map, annot_map, chunksize=2_000_000):
    """
    (long counts, per-sample annotation) for `samples` only. The embryo-wide table is
    2.7 GB, so it is read in chunks and filtered on the way in.

    AF is recomputed from AD/DP rather than taken from the file's VAF column, so it is
    defined identically across the two file layouts.
    """
    cols = sorted(set(count_map) | set(annot_map))
    L = []
    for chunk in pd.read_csv(path_csv, usecols=cols, chunksize=chunksize):
        L.append(chunk[chunk[[c for c in count_map if count_map[c]=='sample'][0]].isin(samples)])
    df = pd.concat(L, ignore_index=True)

    counts = df[list(count_map)].rename(columns=count_map)
    counts['AF'] = np.where(counts['DP']>0, counts['AD']/counts['DP'], np.nan)

    annot = df[list(annot_map)].rename(columns=annot_map).drop_duplicates()
    assert not annot['sample'].duplicated().any(), \
        f'{os.path.basename(path_csv)}: annotation is not constant within a sample'

    return counts[['sample','MUT','AD','DP','AF']], annot.set_index('sample')


##


coords = load_coords()

# Brain and liver share one pass over the 2.7 GB table: read once for the union of the
# samples they need, then split. Every other organ reads its own, much smaller file.
big_samples = set(coords['Brain'].index) | set(coords['Liver'].index)
print(f'Reading {BIG} for {len(big_samples)} brain/liver samples...')
big_counts, big_annot = load_counts(os.path.join(path_data, BIG), big_samples,
                                    MAP_WIDE, ANNOT_WIDE)

L_counts, L_annot = [], []
for organ, sources in SOURCES.items():
    xyz = coords[organ]
    c_parts, a_parts = [], []
    for fname, count_map, annot_map in sources:
        if fname == BIG:
            c_parts.append(big_counts[big_counts['sample'].isin(xyz.index)])
            a_parts.append(big_annot.reindex(big_annot.index.intersection(xyz.index)))
        else:
            print(f'Reading {fname} for {organ}...')
            c, a = load_counts(os.path.join(path_data, fname), set(xyz.index),
                               count_map, annot_map)
            c_parts.append(c)
            a_parts.append(a)

    counts = pd.concat(c_parts, ignore_index=True)
    counts = counts[counts['sample'].isin(xyz.index)]
    counts['organ'] = organ
    counts = counts.join(xyz[['x','y','z']], on='sample')

    annot = pd.concat(a_parts)
    annot = annot.reindex(xyz.index)                 # coordinate set is the reference
    annot.insert(0, 'organ', organ)
    annot['specimen'] = annot.index.str.split('_').str[0]
    annot = annot.join(xyz)                          # coords + coordinate-file annotation

    assert not counts.duplicated(['sample','MUT']).any(), f'{organ}: duplicated (sample, MUT)'
    assert counts[['x','y','z']].notna().all().all(), f'{organ}: missing coordinates'
    assert annot.index.equals(xyz.index), f'{organ}: annotation/coordinate mismatch'

    L_counts.append(counts[COUNT_COLS])
    L_annot.append(annot)
    print(f'  {organ}: {counts["sample"].nunique()} samples x {counts["MUT"].nunique()} mutations')

df = pd.concat(L_counts, ignore_index=True)
ann = pd.concat(L_annot).rename_axis('sample')

# The brain/liver annotation file repeats histo and germ layer, which the count tables
# also carry. Verified identical wherever both exist, so fold them into one column: the
# annotation file is the only source of germ layer for the 15 new LongData liver samples.
for col, col_annot in [('germ_layer','germ_layer_annot'), ('germ_layer2','germ_layer2_annot'),
                       ('histo','histo_annot')]:
    both = ann[col].notna() & ann[col_annot].notna()
    assert (ann.loc[both, col] == ann.loc[both, col_annot]).all(), f'{col} disagrees with {col_annot}'
    ann[col] = ann[col].fillna(ann[col_annot])
ann = ann.drop(columns=['germ_layer_annot','germ_layer2_annot','histo_annot'])


##


# Every organ must carry the same mutation panel, or genetic distances are not
# comparable across organs
panel = df.groupby('organ')['MUT'].nunique()
assert panel.nunique() == 1, f'mutation panel differs between organs:\n{panel}'
assert len(ann) == df['sample'].nunique() == 493, 'unexpected number of samples'

df['AF'] = df['AF'].round(6)
df[['x','y','z']] = df[['x','y','z']].round(2)
df.to_csv(os.path.join(path_data, 'metadata_table.csv'), index=False)

ann[['x','y','z']] = ann[['x','y','z']].round(2)
front = ['organ', 'specimen', 'x', 'y', 'z']
ann = ann[front + [c for c in ann.columns if c not in front]]
ann.to_csv(os.path.join(path_data, 'sample_annotations.csv'))

print(f'\nmetadata_table.csv    : {len(df)} rows, {df["sample"].nunique()} samples')
print(f'sample_annotations.csv: {len(ann)} rows, {ann.shape[1]} columns')
print(ann.groupby('organ').agg(n=('specimen','size'), specimens=('specimen', lambda x: '/'.join(sorted(set(x))))).to_string())
print('\nannotation coverage (non-null per organ):')
print(ann.notna().groupby(ann['organ']).sum().T.to_string())
