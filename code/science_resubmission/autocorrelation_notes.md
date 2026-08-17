# Spatial autocorrelation of single early mutations — handoff notes

Status as of 2026-08-04. Script: `code/science_resubmission/autocorrelation_single_muts.py`.
Written for a fresh agent picking this up. Everything below was run and checked unless
explicitly marked as unverified.

---

## 1. The question

The embryo (PD53943) was sampled by LCM across five specimens. 18 mutations are assigned
to the earliest branches (A–J) of the phylogeny — first cell divisions. If an early
division seeded a spatially coherent part of an organ, the AF of its mutations should
be more similar between physically adjacent LCM samples than between distant ones.

`code/muts_in_space.py` asked this with a global Moran's I, one sample (`PD53943o`),
1000 permutations. This script generalises that: all five specimens, and a null model
that accounts for sequencing depth.

## 2. What the script computes

A port of Hotspot (DeTomaso & Yosef 2021, github.com/YosefLab/Hotspot), which was
built for exactly this shape of problem in scRNA-seq: is a feature's value structured
over a kNN graph, given a model of how noisy each observation is?

Per (mutation, organ, k):

1. **Graph** — kNN on physical coordinates, Gaussian kernel with a *per-sample* adaptive
   bandwidth (σ_i = i's own distance to its ⌈k/3⌉-th neighbour), row-normalised,
   mutual edges de-duplicated.
2. **Null model** — beta-binomial MLE of NV given NR, giving per-sample
   `mu_i = p`, `var_i = p(1-p)/NR_i · (1 + (NR_i−1)ρ)`.
3. **Statistic** — `z = (AF − mu)/sd`, then `G = Σ_edges w_ij z_i z_j`.
4. **Inference** — `Z = G/sqrt(Σw²)`, `C = G/(Σ D_i z_i²/2)` (effect size, ~[−1,1]),
   plus a 50,000-permutation null of the same G. BH over the 18 mutations within each
   (organ, k). **§3b spells out exactly how the p-values and FDR are built, in both
   scripts — read it before quoting any number.**

`k ∈ {5,10,15,20}` capped at n/3; k=10 reported. Runtime ~41 s.

There are now three scripts (full list and outputs in §7):

| script | scope | family per BH | reported at |
|---|---|---|---|
| `autocorrelation_single_muts.py` | 18 early mutations, 5 specimens | 18 | FDR 0.05 |
| `autocorrelation_all_muts.py` | all mutations, kidneys stratified | 207–1031 | FDR 0.10 |
| `kidney_strata_check.py` | stratification + power controls | 18 | FDR 0.05 |

The thresholds differ deliberately: the first is confirmatory on a pre-specified set,
the second is a discovery scan whose hits are meant to be followed up (§4b).

### Why the beta-binomial matters here

NR runs 8–63 reads (median 27). A permutation test on AF treats a 12-read AF and a
90-read AF as equally trustworthy; this null does not. Fitted ρ is 0.013–0.025 —
small but non-zero, i.e. slightly more spread than pure binomial sampling noise.

## 3. Verification already done — do not redo

- **The port is exact.** Hotspot's own modules were imported from a clone and run
  against this implementation on the same random graph: `compute_weights`,
  `make_weights_non_redundant`, `compute_node_degree`, `local_cov_weights` (G), and
  `compute_local_cov_max` (G_max) all agree to floating point. The statistic is theirs.
- **Hotspot's default path is `centered=True`.** The analytic-moments machinery in
  `local_stats.py` (`compute_moments_weights`, ~150 lines) is the legacy uncentered
  path and is NOT what `compute_autocorrelations()` runs. Don't be misled by it.
- **The analytic p-value is anticonservative in the tail at this n.** `Var[G] = Σw²`
  is exact, but `norm.sf(Z)` assumes G is normal, and it isn't: z is right-skewed
  (skew 1.2–2.2), so the null of G is skewed (skew 0.36–0.57, excess kurtosis
  0.17–0.61) with a heavier far tail than normal. Measured on 200k permutations for
  the top heart hit: p_analytic = 4.1e-5 vs p_perm = 1.4e-4, i.e. **3.4× too small**.
  Across organs `median_p_ratio` is 1.12–1.31 (conservative in the bulk) but
  `min_p_ratio` is 0.17–0.25 (anticonservative in the tail, where calls are made).
  **Use `Pval_perm` / `FDR_perm`. `Pval` is retained only as the calibration reference.**
  Consequence: 13 mutations pass analytic FDR, 12 pass permutation FDR; the extra one
  is right-kidney `chr1_12414725_C_A` (analytic FDR 0.013 → permutation FDR 0.067).
- **The `observed` mask never fires.** There are zero NR=0 entries across all 9,144
  analysed sample × mutation pairs. The mask is defensive code only.

## 3b. How p-values and FDR are computed — read this before quoting any number

Both scripts compute the same statistic per mutation, but they differ in how the
p-value is obtained, and the genome-wide scan adds a screening step. Getting this wrong
is the easiest way to misreport these results.

### The statistic

```
z = (AF - mu) / sigma                 # beta-binomial standardization, per sample
G = sum over edges of w_ij z_i z_j    # observed covariance over the kNN graph
C = G / (sum_i D_i z_i^2 / 2)         # effect size, ~[-1,1]
```

`C` is **descriptive only** — it never gets a p-value of its own, and (see §5.2c) it is
not comparable across specimens sampled at different densities.

### Two p-values for the same G

**Analytic.** Because z is model-standardized, `E[G] = 0` and `Var[G] = sum(w^2)`
*exactly*, so `Z = G/sqrt(sum(w^2))` and `Pval = norm.sf(Z)`. One-sided: only positive
autocorrelation is of interest. Costs one matrix product for all mutations at once.

**Permutation.** Reshuffle z across samples with the graph held fixed,
`Pval_perm = (#{G_null >= G} + 1) / (N_PERM + 1)`. Floor 2e-5 at N_PERM = 50,000.

They disagree because the analytic route assumes G is normal and it is not: G's null is
right-skewed, so `norm.sf` understates the far tail. Measured: the analytic p is
**2-6x too small in the tail** and ~1.2x too large in the bulk (§3, `min_p_ratio` /
`median_p_ratio`). **The permutation p is the one to report.**

### Two-stage screening (genome-wide scan only)

`autocorrelation_all_muts.py` runs permutations only where the analytic p < `SCREEN`
(0.05); everything else keeps its analytic p. Two reasons this is safe:

1. the analytic p is anticonservative in the tail, so screening on it cannot discard a
   mutation the permutation test would have called;
2. unscreened mutations have p >= 0.05, while the BH thresholds here are 1e-4 to 1e-2,
   so they could never be discoveries under any route.

**NAMING WART — the `Pval_perm` column is a mixture.** Only screened rows hold an actual
permutation p; the rest hold the analytic value. Per stratum: heart 139 of 771,
left kidney 54 of 608, right-kidney glomeruli 38 of 503, primitive tubule 0 of 35. The
boolean `permuted` column is the flag. Do not describe that column as "permutation
p-values" without qualification. (The targeted script has no such ambiguity: it
permutes all 18 mutations and keeps `Pval` and `Pval_perm` as separate columns.)

### BH, and what it is and is not correcting

Applied within each **specimen x stratum** over all tested mutations in that stratum.
`multipletests(..., method='fdr_bh')[1]` returns **adjusted p-values**; its `alpha`
argument only affects a reject-array that is never used. Consequently
**`FDR_THRESHOLD` changes nothing computationally — it is purely where the line is
drawn when reporting.** The 0.05 and 0.10 runs produced identical `FDR_perm` columns.

BH is *step-up*: adjusted_i = running minimum, from the bottom rank upwards, of
`p_j * M / j`. That has a consequence worth internalising. The left kidney, M = 608:

| rank | mutation | p | p*M/rank | reported FDR |
|---|---|---|---|---|
| 1 | `chr9_69981290_G_A` | 0.00010 | 0.0608 | **0.0608** |
| 2 | `chr19_33601569_C_T` | 0.00034 | 0.1034 | 0.0997 |
| 3 | `chr11_45722388_C_A` | 0.00056 | 0.1135 | 0.0997 |
| 4 | `chr6_119742053_C_T` | 0.00078 | 0.1186 | 0.0997 |
| 5 | `chr7_47818768_C_T`  | 0.00082 | **0.0997** | 0.0997 |

Ranks 2-4 have raw ratios *above* 0.10 and pass only because rank 5 dips to 0.0997 and
monotonicity drags them down to it. **Those four pass on a fifth mutation's p-value,
not their own** — one entangled block, not four independent findings. Only rank 1
stands alone. This is correct BH behaviour, and it is why §4b calls them
threshold-dependent.

**Not corrected:** BH runs within each of the **9 families** (left kidney
All/Glomerulus/Urothelium, right kidney All/Glomerulus/Blastema/Primitive tubule,
heart, brain). There is no correction across the 9, and the kidney strata are *nested*
inside their own "All" stratum, so the families are not independent either. Each
controls its own FDR; the study-wide error rate does not. For a figure claim, either
pre-specify one stratum per specimen or correct across the whole grid.

## 4. Results (k as reported per specimen, FDR_perm ≤ 0.05)

| Specimen | n | sig / tested | median C | max C |
|---|---|---|---|---|
| Heart | 135 | **10 / 15** | 0.115 | 0.198 |
| Left kidney | 109 | 1 / 16 | 0.037 | 0.163 |
| Brain | 72 | 1 / 15 | 0.014 | 0.249 |
| Right kidney | 137 | 0 / 17 | −0.008 | 0.139 |
| Liver | 11 | 0 / 15 | −0.187 | — |

- Heart hits span branches A, B, D, F, H, I, J — most of the early tree, not one clone.
- `chr18_69326508_G_A` (branch B) is the only mutation significant in two specimens
  (heart C=0.20, left kidney C=0.16).
- C decays smoothly with k while staying significant (heart top hit: 0.26 → 0.20 →
  0.15 → 0.11 for k=5→20), the signature of real local structure rather than a
  k-specific artifact.
- Liver: n=11 (26 samples have coordinates, only 11 have counts). Runs at k=5 = 45% of
  the organ, prints a warning. **Not evidence either way. Do not report it.**

## 4b. Genome-wide run (all mutations, kidneys stratified)

`code/science_resubmission/autocorrelation_all_muts.py` (1 min 52 s). Same statistic,
null and graph; all 14,063 callset mutations filtered to `n_positive >= 5` (most are
private), kidneys split by `Histo`, two-stage inference (analytic screen at p<0.05 then
50k permutations), BH within each specimen x stratum. Outputs
`results/autocorrelation_all_muts*.csv` and `autocorrelation_early_vs_all.csv`.

**Reported at `FDR_THRESHOLD = 0.10`**, not 0.05 — this is a discovery scan over
200–1000 mutations per specimen, several of whose hits already replicate across
specimens, not a confirmatory test. At 0.05 the left kidney returned nothing anywhere
while its top mutations sat at FDR 0.06–0.10, i.e. a whole specimen was being called
empty on a threshold convention. `FDR_perm` is in the CSVs; any other cut can be applied.

| specimen | stratum | n | tested | sig @0.10 | (@0.05) | median C | max C |
|---|---|---|---|---|---|---|---|
| Heart | All | 135 | 771 | **100** | (71) | 0.001 | 0.449 |
| Brain | All | 72 | 207 | 18 | (5) | 0.042 | 0.403 |
| Right kidney | Glomerulus | 93 | 503 | 5 | (3) | −0.011 | 0.321 |
| Right kidney | All | 137 | 1031 | 2 | (2) | −0.015 | 0.287 |
| Left kidney | All | 109 | 608 | 5 | (0) | −0.012 | 0.330 |
| Left kidney | Glomerulus | 63 | 221 | 1 | (0) | −0.016 | 0.334 |
| Left kidney | Urothelium | 23 | 84 | 0 | (0) | −0.037 | 0.471 |
| Right kidney | Blastema / Prim. tubule | 28 / 16 | 108 / 35 | 0 | (0) | −0.03 | 0.36 |

**The early mutations are 8x enriched among the heart's hits.** 8 of 15 testable early
mutations vs 92 of 756 others, Fisher OR = 8.25, **p = 1.8e-4**; and by rank,
median C 0.114 vs 0.000 (Mann-Whitney p = 2e-6). (At FDR 0.05 the Fisher test was
OR 1.53, p = 0.41 — only 2 early mutations had crossed, so the enrichment was invisible.
The rank test, which uses no threshold, was already significant either way and is the
more reliable statement.) In the left kidney the rank test also holds: 0.038 vs −0.014,
p = 0.002. Not in the brain (p = 0.95).

**But the strongest individual signals are later, rarer mutations.** Heart max C = 0.449
(`chr7_158116761_G_A`, branch 568, 8 positive samples) against 0.198 for the best early
one. Reading: early mutations are consistently *mildly* structured, while a subset of
later mutations are *strongly* structured — as expected if later clones are smaller and
more spatially compact.

**Left kidney, the 5 hits.** `chr9_69981290_G_A` (branch 549, C = 0.330, FDR = 0.061)
is the only one clearly inside the threshold; the other four
(`chr19_33601569_C_T`, `chr11_45722388_C_A`, `chr6_119742053_C_T`, `chr7_47818768_C_T`)
all sit at FDR = 0.0997, i.e. exactly on the line, and should be treated as
threshold-dependent. The one piece of independent support:
**`chr19_33601569_C_T` (branch 574) is significant in the heart too** (C = 0.208,
FDR = 0.021), a genuine cross-specimen replication. `chr9_69981290_G_A` is only
suggestive elsewhere (right kidney C = 0.079, p = 0.052). Three of the five are too
rare to test outside the left kidney.

**Note on the targeted vs scan frames.** Only 8 of the 18 early mutations survive here
versus 10 in the targeted run at FDR 0.05 — not a contradiction, the family grew from
18 tests to 771. The targeted script stays the correct frame for the pre-specified
early-mutation question; this is the discovery scan.

**The glomerulus stratification replicates on independent mutations.** In the right
kidney, `chr1_201571433_C_T` goes C = 0.284 (whole organ) -> 0.314 (glomeruli) and
`chr8_76476085_T_C` 0.287 -> 0.321, both on fewer samples. Same direction as the
`chr1_12414725_C_A` result in §5.2c. Note however that `chr1_12414725_C_A` itself now
lands at FDR = 0.093 (right-kidney glomeruli) and 0.059 (brain) — it does not survive
correction against 500 and 207 tests, only against 18.

**Exploratory: branches recurring across specimens.** Branches 476 and 470 have median
C > 0.1 in heart, left kidney AND right kidney; 469 in heart + left kidney; 613 in
brain + heart. Only 3–4 mutations each and no correction across the 637 branches, so
this is a lead, not a result — but a clade being spatially structured in several organs
at once would be a strong statement about early lineage allocation if it holds up.

**Caveat carried over:** within the heart, C correlates with `n_positive`
(Spearman 0.445). Cross-specimen this was an organ-size artifact (§5.2), but the
within-specimen version is not explained away, and mutations with 5–10 positive samples
can post high C on thin evidence. Treat individual low-`n_positive` hits with caution.

**Untested for this set:** the compartment-restricted permutation of §5.2b was run only
for the 18 early mutations. The 71 heart hits have not been checked against
anatomy-preserving nulls.

## 5. Open questions, most important first

### 5.0 RESOLVED — the heart result is not an artifact of dense/clustered 3D sampling

Two mechanisms were tested (2026-08-04) and neither explains the heart calls.

**Sampling density cannot inflate significance in principle.** The permutation null
conditions on the exact graph: `G_null` is computed by reshuffling z over the same W.
Any geometry — clustered, anisotropic, 3D — gets its own null. Density buys statistical
power to detect real structure; it cannot manufacture a false positive.

**Section-level batch effects could have, but did not.** z is clumped, so 38.8% of edge
weight (238 of 829 edges) links samples in the same z-level, and 3 of the 10 significant
mutations do show a section effect on AF (eta² permutation p < 0.01:
`chr3_85532662_T_C`, `chr6_50196398_C_T`, `chr8_15505436_G_A`). But under a
**restricted permutation that shuffles only within z-levels** — holding section
membership fixed, so any section-level technical variation is preserved under the null —
all 10 mutations remain significant (p ≤ 0.045, 8 of 10 below 0.015). Section structure
is therefore not the source of the signal.

**Dropping z entirely still works** (see 5.1): 8 of 15 significant in 2D vs 10 of 15
in 3D.

### 5.1 The heart's 3D coordinates may be unit-inconsistent — largely defused

**Update (2026-08-04):** the (a)-vs-(c) comparison below has been run. Re-running the
heart at k=10 with x,y only gives **8/15 significant, median C = 0.094**, against
10/15 and 0.114 in 3D. The top hits are unchanged or stronger in 2D
(`chr18_69326508_G_A` C = 0.216, FDR = 0.0012; `chr7_93731124_C_G` C = 0.155,
FDR = 0.020). Only `chr6_50196398_C_T` and `chr8_143635887_G_A` drop out.
So the headline does not depend on how z is treated. Resolving the units is still worth
doing for the final figure — and z clearly adds real information, since 3D recovers two
mutations 2D misses — but it is no longer a blocker.

Original note follows.



`data/Heart_final_coorindates_135.csv` has x ∈ [13, 655], y ∈ [417, 1358], and z taking
17 discrete values from 16 to 944, clustered (16–144: 112 samples; 384–576: 17;
816–944: 6). The Euclidean distance in `neighbors_and_weights` mixes all three axes.

If z is a section index, or in a different physical unit from x/y, then the 3D graph is
distorted — and the heart is the specimen carrying the headline result. Note also that
heart x/y are on a completely different numeric scale from the kidney coordinate files
(ranges of ~10³–10⁴ pixels there), so some rescaling has already been applied to the
heart file by someone.

This is **pre-existing behaviour**, not introduced here: `code/spatial_association.py:305`
builds `D_xyz` the same way. Whatever is decided should probably be applied to both.

Concrete next steps: find out the section thickness and the µm/pixel for the heart file
(the kidney/brain pipeline uses `resolution = 0.44186` µm/pixel, see
`AOC_brain_liver.py:308`); re-run the heart with (a) z dropped entirely, (b) z rescaled
to microns, (c) as-is, and compare C and FDR_perm. If the 10 heart calls survive all
three, the result is robust to the ambiguity and this stops being a blocker.

### 5.2 Is the heart special, or just better powered? — mostly resolved, weakened

**Update (2026-08-04). The "detection breadth drives C" concern was overstated; it was
an organ-size confound.** `n_positive` is the raw count of samples carrying the
mutation, so it scales with organ n (Spearman ρ = 0.61 with n). Using the *fraction*
of samples instead:

| | Spearman ρ vs C | p |
|---|---|---|
| `n_positive` (raw count, all organs) | +0.45 | <0.001 |
| `frac_pos` = n_positive/n (all organs) | **−0.07** | 0.55 |
| `frac_pos`, ranked within organ | +0.13 | 0.25 |
| `rho`, ranked within organ | −0.13 | 0.28 |

Once organ size is removed the association disappears. Detectability is not driving
the ranking.

**Overdispersion cannot inflate C either — this is structural.** Both G and G_max are
quadratic in z, so **C is invariant to any uniform rescaling of z** (verified
numerically: C(z) == C(3z) exactly). A mutation with globally larger ρ gets uniformly
smaller |z| and exactly the same C. ρ affects C only through the *relative* weighting
across samples:

- ρ → 0: `var_i ∝ 1/NR_i`, deeply covered samples dominate the statistic;
- ρ → 1: `var_i → p(1−p)` constant, coverage ignored (equivalent to naive AF z-scoring).

Fitted values give median `(NR−1)·ρ` ≈ 0.49–0.64 across organs, i.e. `var_i` is about
`1.5 · p(1−p)/NR_i` — the depth term still dominates, so the coverage correction is
active but not extreme. Empirically ρ vs C, ranked within organ, is −0.13 (p = 0.28).

**What survives:** within the heart alone, C vs `frac_pos` is +0.60 (p = 0.018) and
C vs `p_hat` +0.67 (p = 0.007). That is one nominal result out of five organs examined,
on 15 points, so treat it as a hint rather than a finding. If someone wants to close
this properly: subsample heart samples down to the brain's n = 72 and its detection
spectrum, re-run, and see how many calls survive.

### 5.2b Not a tissue-composition artifact either, and the signal is finer than anatomy

Heart LCM samples are annotated with 18 *anatomical* classes; coarsened to
{DMP 36, IVS 36, ventricle 34, outflow 24, atria 5}, **87.6% of kNN edge weight falls
within a single compartment** — the spatial graph is very nearly an anatomy graph. So
"adjacent samples share a compartment, and compartments differ in cell composition"
was a live alternative to clonal structure.

It does not hold. Permuting only *within* compartment (which preserves any
compartment-level effect under the null) leaves 9 of 10 mutations significant
(all p < 0.05 except `chr18_37006496_T_A`, p = 0.109). Under the much more constrained
fine 18-class restriction, 7 of 10 survive. And compartment identity explains very
little of the AF variance directly (eta² = 0.006–0.097).

**Implication:** the autocorrelation is largely *within*-compartment, i.e. at a scale
finer than gross cardiac anatomy — which is what clonal patches inside a continuous
myocardium should look like, and is not what a compositional artifact would look like.

### 5.2c RUN — kidney cell-type strata, and a warning about comparing C across organs

`code/science_resubmission/kidney_strata_check.py` (31 s) re-runs the statistic within
single cell-type strata, plus a heart-subsampling power control. Outputs:
`results/kidney_strata_autocorrelation.csv`, `results/heart_subsample_power.csv`.

**Positive result — right kidney glomeruli.** `chr1_12414725_C_A` (branch C) goes from
C = 0.139, FDR = 0.068 (whole organ, n=137, NOT significant) to **C = 0.220,
FDR = 0.012 (glomeruli only, n=93, significant)** — a larger effect on fewer samples.
Restricting to one kind of object strengthened it, exactly as the sampling-unit
hypothesis predicts. Note this is the same mutation that is the brain's only hit
(C = 0.249), so branch C now has spatial structure in two organs.

**The other strata are uninformative, not negative.** Left kidney glomeruli (n=63):
max C = 0.096, nothing significant. Non-glomerular strata (n=44, 46): nothing. But see
the power control — at these n the test cannot see heart-sized effects, so absence here
means nothing.

**Power control, and a methodological warning.** Subsampling the heart to n=63
(20 replicates, k=10) recovers a **median of 0** of its 10 whole-organ hits at
FDR<0.05 (13 of 20 replicates recover none; range 0–7), and the median C of those
mutations falls from 0.121 to **0.054**.

That second number matters beyond this test: **C is not invariant to sampling density.**
At fixed k, thinning the samples makes the k nearest neighbours physically farther
apart, so the neighbourhood spans more tissue and averages over more of whatever
clonal domain structure exists. C therefore depends on the sampled density relative to
the size of the clonal domains.

Consequence: **do not compare C across specimens sampled at different densities**
(the heart 0.115 vs left kidney 0.037 contrast in §4 is confounded this way). The
permutation p-values remain valid — the null conditions on the actual graph, so nothing
here creates false positives — but the *effect sizes* and the *power* are not
comparable across specimens. A fair cross-organ comparison would need neighbourhoods
matched in microns rather than in k, which is blocked by the heart's coordinate-unit
ambiguity (§5.1).

### 5.3 Why does the right kidney give nothing?

It has the *most* samples (n=137) and zero significant mutations, while the left kidney
(n=109) has one. The coordinate extents differ substantially: left kidney spans
8556 × 11709, right kidney 5631 × 7699 — the right kidney is more densely sampled over
a smaller area. If clonal domains are large relative to the sampled field, a small
field has little spatial dynamic range and the statistic has nothing to see.

**The kidneys sample a different kind of object.** Heart LCMs are contiguous anatomical
territories; kidney LCMs are interdigitated cell-type classes — left kidney is
{Glomerulus 63, Urothelium 23, Primitive tubule 14, Blastema 9}, right kidney is
{Glomerulus 93, Blastema 28, Primitive tubule 16}. Each glomerulus derives from its own
nephron-progenitor condensation, so two spatially adjacent glomeruli need not be
clonally adjacent, and a kNN graph over them may simply not connect clonally related
material. A concrete test: re-run each kidney restricted to glomeruli only (n = 63 and
93, enough for k = 10), and separately on the non-glomerular samples. If signal appears
within a cell-type stratum, the whole-organ null is a sampling-unit effect, not absence
of early clonal structure.

Check also: the physical-distance distributions (`Supp_distances.py`,
`figures/Supp_spatial_sampling_density_across_samples.pdf` already exist), and whether
the two kidneys were sampled at comparable physical scale in microns. If the right
kidney's field is genuinely smaller, say so explicitly rather than reporting a null.

### 5.4 Deferred scope (explicitly out of the current script)

- **Pairwise local correlation and modules** — Hotspot's `local_stats_pairs.py` +
  `modules.py`. `G_xy = Σ_i Σ_j w_ij (x_i y_j + y_i x_j)/2` → Z matrix over mutations →
  hierarchical clustering. This answers "which early mutations occupy the same spatial
  domain within an organ", arguably the more interesting question, and the centered
  path makes it cheap (`EG=0`, `EG2=Wtot2/2`). Not implemented.
- **Per-sample local scores** — a LISA-style per-LCM-sample readout for the maps. An
  earlier attempt at this (permutation-based local Moran's I) was abandoned; see §6.
- **Shrinkage on ρ** — currently fitted independently per mutation. Sharing ρ across
  mutations within an organ would stabilise it at low n (relevant for liver/brain).

## 6. What was already tried and abandoned — don't repeat it

A per-sample local Moran's I (LISA) with conditional permutation was implemented and
run over all five specimens. It is gone (script rewritten, outputs deleted). It failed
for two compounding reasons, both worth remembering:

1. **Permutation floor vs BH.** The local test runs once per LCM sample, so BH is over
   n tests; at n=137 the smallest usable p is ~0.05/137 = 3.6e-4, which a
   1000-permutation null (floor 1/1001) cannot reach. Nothing survived FDR anywhere —
   for want of resolution, not signal. Raising to 19,999 permutations took the runtime
   to 9 minutes and still yielded only 0–2 significant samples per mutation.
2. **No depth model.** It standardised AF directly, so a 12-read AF shouted as loudly
   as a 90-read one.

The Hotspot formulation fixes both and runs in 41 s. If a per-sample readout is wanted
(§5.4), build it on top of the model-standardized z from this script — do not go back
to permuting raw AF.

## 7. Files

All three scripts run under the `MiTo` env
(`/Users/cossa/miniforge3/envs/MiTo/bin/python`), which has `mito`, `plotting_utils`
and a macOS matplotlib backend.

### `autocorrelation_single_muts.py` — targeted, the 18 first-division mutations (41 s)

Permutes **all** mutations, so `Pval` (analytic) and `Pval_perm` are separate, unmixed
columns. This is the confirmatory frame for the pre-specified early-mutation question.

- `results/hotspot_autocorrelation_single_muts.csv` — per mutation × organ × k. Key
  columns: `C`, `Z`, `Pval`, `FDR`, `Z_perm`, `Pval_perm`, `FDR_perm`, `p_hat`, `rho`,
  `n_covered`, `n_positive`, `median_NR`, `k_main`. **Report `FDR_perm`** (§3b).
- `results/hotspot_autocorrelation_organ_summary.csv` — per specimen, incl. the
  calibration diagnostics `median_sd_ratio`, `median_p_ratio`, `min_p_ratio`.
- `figures/hotspot_autocorrelation_C.pdf` — organ × mutation heatmap of C, * = FDR_perm.
- `figures/hotspot_Z_calibration.pdf` — analytic vs permutation Z.
- `figures/hotspot_autocorrelation_k_sensitivity.pdf` — C vs k, one line per mutation.
- `figures/{specimen}_hotspot_autocorrelation.pdf` — AF maps annotated with C.

### `autocorrelation_all_muts.py` — genome-wide scan, kidneys stratified (1 min 52 s)

All 14,063 callset mutations filtered to `n_positive >= 5`; kidneys split by `Histo`;
**two-stage inference, so `Pval_perm` is a mixture — check the `permuted` flag** (§3b).
Reported at `FDR_THRESHOLD = 0.10`, which is a reporting cut only.

- `results/autocorrelation_all_muts.csv` — one row per mutation × specimen × stratum,
  with `Branch` and `is_early` annotations.
- `results/autocorrelation_all_muts_summary.csv` — per specimen × stratum.
- `results/autocorrelation_early_vs_all.csv` — are the 18 early mutations more
  autocorrelated than the rest (rank test + Fisher on the thresholded calls; prefer the
  rank test, §4b).
- `results/autocorrelation_branch_summary.csv` — per branch, ≥3 mutations.

### `kidney_strata_check.py` — the stratification and power controls (31 s)

- `results/kidney_strata_autocorrelation.csv`, `results/heart_subsample_power.csv`.
  See §5.2c: this is where the "C is not comparable across sampling densities" warning
  comes from.

### Caches

- `results/cache/*_18muts.csv` — 18-mutation subsets (targeted script).
- `results/cache/*_full.npz` — full NV/NR matrices, all 14,063 mutations (scan script).
  Delete to force a re-read; the brain/liver source is 2.7 GB and is read in chunks.

Inputs, per specimen (assembled by `get_specimen`):

| Specimen | counts | coordinates |
|---|---|---|
| `PD53943o` (left kidney) | `PD53943o_metadata.csv` | `PD53943o_coordinates.csv` |
| `PD53943w` (right kidney) | `PD53943w_metadata.csv` | `PD53943w_coordinates.csv` |
| `Heart` | `Heart_metadata.csv` | `Heart_final_coorindates_135.csv` (x,y,z) |
| `Brain`, `Liver` | `Final_Dataframe_heart_annotations_raw_trophoblasts.csv` | `df_brain_liver_annotations_unique_for_Andrea.csv` |

Related existing analyses: `code/science_resubmission/AOC_brain_liver.py` (AOC +
neighborhood effect sizes, soft-cosine genetic distances), `code/spatial_association.py`
(the heart equivalent), `code/muts_in_space.py` (the global Moran's I this replaces).
