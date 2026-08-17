# Spatial analyses — statistical definitions

Every quantity produced by `AOCs.py` and `autocorrelation_muts.py`, with the equation
behind it and the figure it appears in. Written so a methods section can be assembled
from it without reading the code.

Two analyses share the same inputs and the same permutation logic but answer different
questions:

| | question | unit of analysis | script |
|---|---|---|---|
| **Kernel AOC** | is a *sample* genetically closer to its physical neighbourhood than to its organ? | one test per LCM sample | `AOCs.py` |
| **Mutation autocorrelation** | is a *mutation's* allele fraction spatially structured within an organ? | one test per mutation × organ | `autocorrelation_muts.py` |

Common notation:

| symbol | meaning |
|---|---|
| $n$ | LCM samples in the organ (or stratum) |
| $i,j$ | sample indices, $1 \dots n$ |
| $d^{\text{phys}}_{ij}$ | physical distance between samples $i$ and $j$, **microns** |
| $d^{\text{gen}}_{ij}$ | genetic (soft-cosine) distance between samples $i$ and $j$ |
| $w_{ij}$ | spatial weight from $i$ to $j$ |
| $\alpha$ | per-test level, 0.05 |
| $B$ | permutation replicates |

---

## 0. Inputs common to both analyses

### 0.1 Physical distances

Coordinates are converted to microns once, in `build_metadata_table.py`, and are isotropic
thereafter ($z=0$ for the four 2D organs):

$$
(x,y)_{\text{kidneys, brain, liver}} = 0.46 \cdot (x,y)_{\text{px}},\qquad
(x,y)_{\text{heart}} = 7.36 \cdot (x,y)_{\text{px}},\qquad
z_{\text{heart}} = 16\,\cdot\,\text{section number}
$$

so that

$$
d^{\text{phys}}_{ij} = \lVert \mathbf{r}_i - \mathbf{r}_j \rVert_2 ,
\qquad \mathbf{r}_i = (x_i, y_i, z_i) \ [\mu m].
$$

The heart factor is $0.46 \times 16$: its coordinates were exported from a 16×-downsampled
image. Section thickness is 16 µm, so its $z$ is already in microns.

### 0.2 Genetic distance (kernel AOC only)

Let $X \in \mathbb{R}^{n \times m}$ hold the VAF of $m$ mutations in $n$ samples, and let
$M \in \{0,1\}^{m \times b}$ be the mutation × phylogeny-branch incidence. Two mutations
count towards each other's similarity when they co-occur on a branch:

$$
W^{\text{feat}} = \mathbf{1}\!\left[ M M^{\top} > 0 \right] \in \{0,1\}^{m \times m}
$$

The **soft cosine** similarity and distance are

$$
G^{\text{soft}} = X\,W^{\text{feat}}X^{\top}, \qquad
S_{ij} = \frac{G^{\text{soft}}_{ij}}{\sqrt{G^{\text{soft}}_{ii}\,G^{\text{soft}}_{jj}}},
\qquad
d^{\text{gen}}_{ij} = 1 - S_{ij}.
$$

$S$ is clipped to $[0,1]$ and rounded to 7 decimals for numerical symmetry. **Distances are
never rescaled**: every ratio below is taken on this scale, and min–max rescaling would
shift the zero and distort them.

---

## 1. Kernel AOC — `AOCs.py`

### 1.1 Neighbourhood weights

For sample $i$, every other sample contributes with a Gaussian weight of **fixed**
bandwidth $\sigma$ (microns), normalised to sum to one:

$$
\tilde{w}_{ij} = \exp\!\left(-\frac{(d^{\text{phys}}_{ij})^{2}}{2\sigma^{2}}\right),
\qquad
w_{ij} = \frac{\tilde{w}_{ij}}{\sum_{k \neq i} \tilde{w}_{ik}},
\qquad j \neq i .
$$

$\sigma$ is scanned over $\{100, 150, 200, 300, 400, 600, 800, 1200, 1600\}$ µm;
$\sigma = 400$ µm is the reference. The bandwidth is deliberately **not** adaptive — see §4.

### 1.2 Effect size

$$
\underbrace{\bar d^{\,\text{kern}}_i = \sum_{j \neq i} w_{ij}\, d^{\text{gen}}_{ij}}_{\texttt{d\_gen\_kern}}
\qquad
\underbrace{\bar d^{\,\text{all}}_i = \frac{1}{n-1}\sum_{j \neq i} d^{\text{gen}}_{ij}}_{\texttt{d\_gen\_all}}
$$

$$
\boxed{\ \texttt{ratio}_i = \frac{\bar d^{\,\text{kern}}_i}{\bar d^{\,\text{all}}_i}\ }
\qquad
\texttt{diff}_i = \bar d^{\,\text{all}}_i - \bar d^{\,\text{kern}}_i
$$

`ratio` < 1 means the physical neighbourhood is genetically tighter than the organ as a
whole; `ratio` = 1 means no local structure. It is dimensionless, hence comparable across
$\sigma$, samples and organs. `diff` carries the absolute magnitude the ratio hides.

### 1.3 Null distribution and significance

The genetic distances of row $i$ are permuted across the other samples with the weights
held fixed. For $b = 1 \dots B$ ($B = 1000$), with $\pi_b$ a uniform random permutation of
$\{j : j \neq i\}$:

$$
\bar d^{\,(b)}_i = \sum_{j \neq i} w_{ij}\, d^{\text{gen}}_{i\,\pi_b(j)}
$$

Because $\sum_j w_{ij} = 1$, the null is centred exactly on the plain mean:
$\mathbb{E}\big[\bar d^{\,(b)}_i\big] = \bar d^{\,\text{all}}_i$. Then

$$
\texttt{null\_sd}_i = \mathrm{sd}_b\big(\bar d^{\,(b)}_i\big),
\qquad
\texttt{z}_i = \frac{\bar d^{\,\text{kern}}_i - \overline{\bar d^{\,(b)}_i}}{\texttt{null\_sd}_i},
$$

$$
\boxed{\ \texttt{p}_i = \frac{1 + \#\{b : \bar d^{\,(b)}_i \le \bar d^{\,\text{kern}}_i\}}{B + 1}\ }
$$

One-sided: the alternative is "neighbours are *closer*". The support of this null is
$(n-1)!$, so it cannot collapse at small $n$ — unlike a $k$-subset null (§4).

**FDR.** Benjamini–Hochberg over the $n$ samples, **within (organ, $\sigma$)** — one test
per sample is the family:

$$
\texttt{FDR}_{(r)} = \min_{s \ge r}\ \min\!\left(1,\ \frac{n}{s}\, p_{(s)}\right)
$$

### 1.4 Power diagnostics

Nothing is filtered on these; they say whether a null result is informative.

**Effective neighbourhood size** (Kish), the continuous analogue of "how many neighbours":

$$
\boxed{\ \texttt{ess}_i = \frac{\left(\sum_{j} w_{ij}\right)^{2}}{\sum_{j} w_{ij}^{2}}
= \frac{1}{\sum_{j} w_{ij}^{2}}\ }\qquad (\text{since } \textstyle\sum_j w_{ij}=1)
$$

$\texttt{ess}\approx 1$ means the statistic is essentially "distance to my single nearest
sample". Convention used throughout: $\texttt{ess} < 3$ = "no neighbourhood at this $\sigma$".

**Physical radius actually used**, to be read against $\sigma$:

$$
\texttt{d\_kern\_um}_i = \sum_{j \neq i} w_{ij}\, d^{\text{phys}}_{ij}
$$

**Minimum detectable effect**, on the `ratio` scale — the least extreme ratio that would
still have reached $p \le \alpha$:

$$
\boxed{\ \texttt{mde\_ratio}_i = \frac{Q_{\alpha}\!\left(\bar d^{\,(b)}_i\right)}{\bar d^{\,\text{all}}_i}\ }
$$

with $Q_\alpha$ the $\alpha$-quantile of the permutation null. The identity

$$
p_i \le \alpha \iff \texttt{ratio}_i \le \texttt{mde\_ratio}_i
$$

is what makes the diagonal of figure `AOC_effect_vs_mde.pdf` the significance boundary.

**Power** against a pre-specified effect $\rho_0$ (= 0.9, "neighbours 10 % closer"),
normal approximation to the null:

$$
\boxed{\ \texttt{power}_i = \Phi\!\left(
\frac{\left(\texttt{mde\_ratio}_i - \rho_0\right)\,\bar d^{\,\text{all}}_i}{\texttt{null\_sd}_i}
\right)}
$$

Sanity check: at $\rho_0 = 1$ (no effect) this returns $\alpha$, as it must.

### 1.5 Cross-organ comparison at matched ESS

At fixed $\sigma$ the same bandwidth buys different neighbourhood sizes in differently
sampled organs (ESS 29.7 in the right kidney vs 14.1 in the heart at 400 µm), and a wider
neighbourhood mechanically pushes `ratio` towards 1. Each organ is therefore also reported
at a common $\text{ESS}^{\ast} = 10$, interpolated linearly in logs over the $\sigma$ grid:

$$
\log \sigma^{\ast} = \mathrm{interp}\big(\log \text{ESS}^{\ast};\ \log \widetilde{\text{ess}}(\sigma),\ \log \sigma\big),
\qquad
\texttt{ratio}^{\ast} = \mathrm{interp}\big(\log \text{ESS}^{\ast};\ \log \widetilde{\text{ess}}(\sigma),\ \widetilde{\texttt{ratio}}(\sigma)\big)
$$

where $\widetilde{\cdot}$ denotes the per-$\sigma$ median over samples. $\sigma^{\ast}$ — the
bandwidth each organ needs to reach 10 effective neighbours — is itself reportable: it is
the physical scale of that organ's sampled neighbourhood.

### 1.6 Sampling geometry (organ inclusion)

Samples are linked when closer than 3× the median nearest-neighbour distance, and the
**islands** are the connected components of that graph:

$$
A_{ij} = \mathbf{1}\!\left[d^{\text{phys}}_{ij} < 3\,\mathrm{median}_i\big(\min_{j\neq i} d^{\text{phys}}_{ij}\big)\right]
$$

Reported: number of islands, largest island, fraction of samples in islands of $<5$, and
the median distance to the nearest sample of another island. This is the criterion on which
the brain is excluded from the per-sample readout (29 islands, 56 % of samples in islands
of $<5$, 3.7 mm to the next island).

### 1.7 Label-level test (anatomy vs adjacency)

For a grouping $\ell_i$ (hemisphere, anatomical group, cell type), over unordered pairs
$\mathcal{P} = \{(i,j) : i < j\}$:

$$
\bar d^{\text{within}} = \frac{\sum_{(i,j) \in \mathcal{P}} \mathbf{1}[\ell_i = \ell_j]\, d^{\text{gen}}_{ij}}{\sum_{(i,j)\in\mathcal{P}} \mathbf{1}[\ell_i = \ell_j]},
\qquad
\bar d^{\text{between}} = \frac{\sum_{(i,j)\in\mathcal{P}} \mathbf{1}[\ell_i \neq \ell_j]\, d^{\text{gen}}_{ij}}{\sum_{(i,j)\in\mathcal{P}} \mathbf{1}[\ell_i \neq \ell_j]}
$$

$$
\boxed{\ \texttt{ratio} = \frac{\bar d^{\text{within}}}{\bar d^{\text{between}}}\ },
\qquad
T = \bar d^{\text{within}} - \bar d^{\text{between}},
\qquad
p = \frac{1 + \#\{b : T^{(b)} \le T\}}{B+1}
$$

with $B = 10{,}000$ permutations **of the labels** over samples. For a single stratum tested
against everything else (`focal`), $\mathbf{1}[\ell_i = \ell_j]$ is replaced by
$\mathbf{1}[\ell_i = \ell^{\ast}]\cdot\mathbf{1}[\ell_j = \ell^{\ast}]$.

The identical contrast computed on $d^{\text{phys}}$ gives `phys_ratio`, which separates
anatomy from mere proximity: `phys_ratio` ≈ 1 with `ratio` ≪ 1 means the grouping predicts
clonal relatedness *at equal physical distance*. BH is applied **within family**
(`region`, `histology`), never across.

---

## 2. Mutation autocorrelation — `autocorrelation_muts.py`

A port of Hotspot (DeTomaso & Yosef 2021) with a beta-binomial measurement model.

### 2.1 Measurement model and standardisation

For one mutation, sample $i$ has alt reads $NV_i$ out of depth $NR_i$. The null is a
beta-binomial with organ-wide AF $p$ and overdispersion $\rho$:

$$
NV_i \sim \mathrm{BetaBin}\!\left(NR_i,\ a,\ b\right),
\qquad a = p\,s,\quad b = (1-p)\,s,\quad s = \frac{1-\rho}{\rho}
$$

$(p,\rho)$ are fitted by maximum likelihood on the covered samples (Nelder–Mead on the
logit scale, multi-start over $\rho_0 \in \{0.01, 0.1, 0.5\}$):

$$
(\hat p, \hat\rho) = \arg\max_{p,\rho} \sum_{i : NR_i > 0} \log \mathrm{BetaBin}\!\left(NV_i \mid NR_i, p, \rho\right)
$$

giving per-sample moments **on the AF scale**

$$
\mu_i = \hat p,
\qquad
\boxed{\ \sigma^2_i = \frac{\hat p (1-\hat p)}{NR_i}\Big(1 + (NR_i - 1)\hat\rho\Big)\ }
$$

and the standardised value

$$
\boxed{\ z_i = \frac{AF_i - \mu_i}{\sigma_i},\qquad AF_i = \frac{NV_i}{NR_i},\qquad
z_i := 0 \ \text{ if } NR_i = 0\ }
$$

$NR_i = 0$ is a *missing measurement*, not an AF of 0: such samples contribute neither to
the statistic nor to its normalisation, but stay in the graph so their neighbours keep
theirs. This is what makes a 12-read AF weigh less than a 90-read one, and it is the reason
for using this model rather than permuting AF directly. Fitted $\hat\rho$ gives median
$(NR-1)\hat\rho \approx 0.5$–0.6, i.e. $\sigma^2_i \approx 1.5 \times$ the binomial value:
the depth term still dominates.

### 2.2 Spatial graph

**(a) Adaptive kNN** (Hotspot's, used for continuity with the earlier results). With
$d_{i(r)}$ the distance from $i$ to its $r$-th nearest neighbour and
$r_0 = \lceil k/3 \rceil$:

$$
\sigma_i = d_{i(r_0)},
\qquad
\tilde{w}_{ij} = \exp\!\left(-\frac{(d^{\text{phys}}_{ij})^{2}}{\sigma_i^{2}}\right)\ \text{ for } j \in \mathcal{N}_k(i),
\qquad
P_{ij} = \frac{\tilde w_{ij}}{\sum_{j'} \tilde w_{ij'}}
$$

**(b) Fixed bandwidth** (the cross-organ comparable graph), over *all* pairs:

$$
\tilde{w}_{ij} = \exp\!\left(-\frac{(d^{\text{phys}}_{ij})^{2}}{2\sigma^{2}}\right),
\qquad P_{ij} = \frac{\tilde w_{ij}}{\sum_{j' \neq i} \tilde w_{ij'}}
$$

Both are then made **non-redundant** — each undirected edge carried once, in the upper
triangle:

$$
W_{ij} = \begin{cases} P_{ij} + P_{ji} & i < j \\ 0 & \text{otherwise}\end{cases}
\qquad
D_i = \sum_j W_{ij} + \sum_j W_{ji}
\qquad
W_{\text{tot}2} = \sum_{i<j} W_{ij}^{2}
$$

ESS is computed from the row-normalised $P$ exactly as in §1.4.

### 2.3 Statistic, effect size, analytic significance

$$
\boxed{\ G = \sum_{i<j} W_{ij}\, z_i z_j \ = \ \mathbf{z}^{\top} W \mathbf{z}\ }
$$

Because $\mathbf{z}$ is model-standardised, $\mathbb{E}[G] = 0$ and
$\mathrm{Var}[G] = W_{\text{tot}2}$ **exactly**, giving

$$
Z = \frac{G}{\sqrt{W_{\text{tot}2}}},
\qquad
\texttt{Pval} = 1 - \Phi(Z) \quad\text{(one-sided)}
$$

and the effect size, normalised by the largest value $G$ could take for that $\mathbf{z}$:

$$
\boxed{\ C = \frac{G}{G_{\max}},\qquad G_{\max} = \frac{1}{2}\sum_i D_i z_i^{2}\ }
\qquad C \in [-1, 1] \ \text{approximately}
$$

$C$ is **descriptive only** — it never receives a p-value of its own. It is invariant to any
uniform rescaling of $\mathbf{z}$ (both $G$ and $G_{\max}$ are quadratic in $\mathbf{z}$),
so a globally larger $\hat\rho$ cannot inflate it.

### 2.4 Permutation significance

$\mathbf{z}$ is reshuffled across samples with the graph held fixed
($B = 50{,}000$, floor $2\times10^{-5}$):

$$
G^{(b)} = \big(\Pi_b \mathbf{z}\big)^{\top} W \big(\Pi_b \mathbf{z}\big),
\qquad
\boxed{\ \texttt{Pval\_perm} = \frac{1 + \#\{b : G^{(b)} \ge G\}}{B+1}\ }
$$

**Report `Pval_perm` / `FDR_perm`, not the analytic pair.** $G$'s null is right-skewed
(because $z$ is), so $1-\Phi(Z)$ understates the far tail — measured 2–6× too small exactly
where calls are made, while being ~1.2× conservative in the bulk. The analytic p is kept as
the calibration reference (`median_p_ratio`, `min_p_ratio` = median and min of
$\texttt{Pval}/\texttt{Pval\_perm}$).

**Two-stage screening** (genome-wide scan only): permutations are run only where
$\texttt{Pval} < 0.05$; the rest keep their analytic value. Safe because the analytic p is
anticonservative in the tail, so screening on it cannot discard a mutation the permutation
test would have called. ⚠️ The resulting `Pval_perm` column is therefore a **mixture** —
the boolean `permuted` flags the rows holding a true permutation p.

**FDR.** BH as in §1.3, with family = all mutations tested within one
(organ × stratum × graph) for the scan, and the 18 first-division mutations within
(organ × graph × graph parameter) for the targeted analysis.

### 2.5 Early mutations vs the rest

Let $\mathcal{E}$ be the 18 first-division mutations and $\mathcal{R}$ the remainder of the
testable callset in that organ. Threshold-free (**preferred**):

$$
U\text{-test}: \quad H_1 : C_{\mathcal{E}} \succ C_{\mathcal{R}} \quad \text{(Mann–Whitney, one-sided)}
$$

and, on the thresholded calls, Fisher's exact test on

$$
\begin{pmatrix}
\#\{\mathcal{E} : \texttt{FDR\_perm} \le 0.10\} & \#\{\mathcal{E} : \texttt{FDR\_perm} > 0.10\}\\
\#\{\mathcal{R} : \texttt{FDR\_perm} \le 0.10\} & \#\{\mathcal{R} : \texttt{FDR\_perm} > 0.10\}
\end{pmatrix}
$$

The rank test is quoted because the Fisher table inherits the instability of calls sitting
near the threshold.

### 2.6 Power control

The heart is subsampled to $n_{\text{sub}} = 63$ (the size of a kidney stratum),
$R = 20$ replicates. For each replicate the graph is rebuilt on the subsample, the
statistic recomputed for the organ's whole-organ hits, and

$$
\text{recovered}^{(r)} = \#\{\text{hits with } \texttt{FDR\_perm} \le 0.05 \text{ in replicate } r\}
$$

reported as a median and range, together with the median $C$ of those mutations — which
also quantifies the warning that **$C$ is not invariant to sampling density**: at fixed $k$,
thinning the samples widens the neighbourhood and averages over more of whatever domain
structure exists.

---

## 3. What each figure plots

### Kernel AOC (`AOC_figures.py`)

| figure | x | y | encoding |
|---|---|---|---|
| `AOC_decay` | $\sigma$ (log) | median$_i$ `ratio` | line per organ; ribbon = per-sample IQR (q25–q75); dashed line at `ratio` = 1 |
| `AOC_ess` | $\sigma$ (log) | median$_i$ `ess` (log) | shaded band `ess` < 3 = "no neighbourhood" |
| `AOC_significance` | $\sigma$ (log) | $\frac{1}{n}\#\{i : \texttt{FDR}_i \le 0.05\}$ | — |
| `AOC_power` | $\sigma$ (log) | median$_i$ `power` (§1.4, $\rho_0 = 0.9$) | dashed line at 0.8 |
| `AOC_effect_vs_mde` | `mde_ratio`$_i$ | `ratio`$_i$ | one point per sample at $\sigma$ = 400 µm; **the identity line is the $\alpha$ = 0.05 boundary** (§1.4); filled = `FDR` ≤ 0.05 |
| `AOC_label_tests` | `ratio` (filled) and `phys_ratio` (open) | test | §1.7; stars from BH within family |

⚠️ In `AOC_effect_vs_mde` the diagonal is the **unadjusted** $\alpha$-level boundary while
the fill marks BH significance, so filled points are a subset of the points below the line.
That gap *is* the multiple-testing correction, and is worth stating in the legend.

### Mutation autocorrelation (`autocorrelation_figures.py`)

| figure | x | y | encoding |
|---|---|---|---|
| `AC_volcano_<organ>` | $C$ (§2.3) | $-\log_{10}$ `Pval_perm` | colour = `FDR_perm` ≤ 0.10; open diamonds = the 18 first-division mutations; dashed line at the largest $p$ among mutations passing BH, i.e. where the FDR cut actually falls; top 5 by $C$ labelled |
| `AC_map_<organ>_<mut>` | $x$ (mm) | $y$ (mm), $z$ (mm) in 3D | colour = $AF_i = NV_i/NR_i$, `afmhot_r`, $[0, \max_i AF_i]$ |

Two properties of the volcano to state in the caption: it is a **half** volcano (the test is
one-sided, only positive autocorrelation is of interest), and the $y$ axis saturates at
$-\log_{10}(2\times10^{-5}) = 4.7$, the permutation floor at $B = 50{,}000$.

---

## 4. Why the design is what it is

**Fixed bandwidth, never adaptive.** An adaptive $\sigma_i$ renormalises every sample to its
own neighbour distances, which reports a healthy neighbourhood where locality does not
physically exist: measured here it gives min ESS 3.9–6.5 in *every* organ — brain and liver
included — at a kernel radius of ~1 mm. Under fixed $\sigma$ the same organs report ESS
1.0–2.2. Adaptive weighting is acceptable for a global statistic (§2.2a, kept for
continuity); it must never stand in for a sparsity diagnostic.

**Why the kernel replaced the $k$-NN AOC.** The earlier AOC drew its null as $k$-subsets of
the $n-1-k$ non-neighbours, so the null variance carries the finite-population factor

$$
\mathrm{Var}\big[\bar d^{(b)}\big] \propto \frac{N-k}{N-1},\qquad N = n-1-k
$$

which collapses as $k \to N$. At $n = 26$ (liver) and $k = 12$ there are $\binom{13}{12} = 13$
distinct subsets: `null_sd` falls to 15 % of its $k=2$ value and 57 % of samples turn
"significant" while the effect itself stays flat. The permutation-of-values null used here
has support $(n-1)!$ and cannot degenerate. Sparsity then stops being a keep/drop verdict
(the old `power_filter`) and becomes a reported quantity, `ess`.

**Why `frac_sig` must not be compared across $\sigma$ or $k$.** Effect and power move in
opposite directions with neighbourhood size: widening the kernel raises `ess` and power but
pushes `ratio` towards 1. The significant fraction therefore peaks at an intermediate
bandwidth (heart: 0.33 at 400 µm, 0.14 at 1600 µm) and its maximum is not a result.
