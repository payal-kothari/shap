# SHAP mental model — grounded in `payal-kothari/shap` (fork of `shap/shap`)

Built from the actual source in this repo, not from memory. File/line references
are current as of commit `998b5105`.

---

## 1. List of explainers (`shap/explainers/__init__.py`)

**TreeExplainer** (`_tree.py`) — Exact, polynomial-time SHAP values for tree
ensembles (XGBoost, LightGBM, CatBoost, sklearn forests/GBMs, etc.). Walks the
tree structure directly rather than treating the model as a black box, using
the recursive algorithm in `cext/tree_shap.h`. Default `feature_perturbation`
is `tree_path_dependent` (no background data needed — baseline comes from
per-leaf training-sample counts baked into the tree) or `interventional` (an
explicit background dataset, handles feature correlation via causal-inference
rules). This is the explainer we've used throughout our benchmarks.

**KernelExplainer** (`_kernel.py`) — Model-agnostic. Treats the model as a
pure function; explains it by masking random feature subsets, observing the
output shift, and fitting a specially-weighted linear regression whose
coefficients are the Shapley values. Works on anything with a `predict`-style
call, at the cost of many model evaluations per instance. This is the "true
black-box" baseline we compared TreeSHAP against.

**GPUTreeExplainer** (`_gpu_tree.py`) — Literally `class GPUTreeExplainer(TreeExplainer)`
— same algorithm, CUDA-accelerated. Requires a source build with
`CUDA_PATH` set. Marked experimental in its own docstring.

**GradientExplainer** (`_gradient.py`) — "Expected gradients," an extension of
Integrated Gradients to an expectation over a background dataset. Differentiable
models only. Framed explicitly in the repo as an approximation to Shapley
values via Aumann-Shapley (infinite-player) game theory, not exact Shapley.

**LinearExplainer** (`_linear.py`) — Closed-form SHAP values for linear models.
Interventional case is exact and cheap: `coef[i] * (x[i] - mean(x)[i])`. Also
supports accounting for inter-feature correlation via sampling, trading
simplicity for correctness under collinearity.

**PartitionExplainer** (`_partition.py`) — Model-agnostic, but exploits a
feature-hierarchy (partition tree) to get from KernelExplainer's exponential
cost down to quadratic. Produces **Owen values**, a generalization of Shapley
values for grouped/hierarchical features (e.g., correlated feature clusters
treated as a unit). Good middle ground when you have structure to exploit.

**CoalitionExplainer** (`_coalition.py`) — Produces **Winter values** (a.k.a.
recursive Owen values), generalizing Shapley values further for explicit
coalition/hierarchy structures (e.g., temporal or multimodal groupings:
demographic block, financial block, etc.). Newer, more specialized than
PartitionExplainer; textual/image data not yet supported per its own docstring.

**ExactExplainer** (`_exact.py`) — Brute-force-but-optimized enumeration of
exact Shapley values. Practical ceiling is ~15 features (standard Shapley) or
~100 features (Owen values via hierarchical masker). Uses Gray codes to order
masking sets so consecutive evaluations differ by one feature at a time —
minimizes redundant model calls.

**PermutationExplainer** (`_permutation.py`) — Model-agnostic approximation via
random feature permutations, evaluated forward and backward (antithetic
sampling) for variance reduction. One full permutation pair already gives
exact values for models with up to second-order interactions; more permutations
improve estimates for higher-order interaction models. Supports partition
trees (unlike Kernel/Sampling).

**SamplingExplainer** (`_sampling.py`) — `class SamplingExplainer(KernelExplainer)`.
Implements the older Shapley-sampling / IME method (Štrumbelj & Kononenko,
2010) under a feature-independence assumption. Preferred over KernelExplainer
when you want a large background sample rather than one reference point.

**AdditiveExplainer** (`_additive.py`) — For generalized additive models
(first-order effects only, by the model's own docstring). Explicitly warns:
applying it to a model with 2nd/3rd-order effects gives **wrong** answers
that silently fail additivity — a nice real-world echo of "don't trust output
that isn't validated."

---

## 2. List of models (grouped by what `TreeExplainer` actually dispatches on,
`_tree.py` lines ~1120–1520 — this is a real, exhaustive `safe_isinstance` chain,
not a generic list)

**sklearn tree ensembles** — `RandomForestClassifier/Regressor`,
`ExtraTreesClassifier/Regressor`, `GradientBoostingClassifier/Regressor` (incl.
older `sklearn.ensemble.gradient_boosting.*` module paths for back-compat),
`HistGradientBoostingClassifier/Regressor` (what our own benchmarks used),
`IsolationForest`, plus `DummyClassifier/Regressor` as trivial baselines.
Single `DecisionTreeClassifier/Regressor` also supported directly.

**XGBoost** — `xgboost.core.Booster` (the low-level trained model),
`XGBClassifier`, `XGBRegressor`, `XGBRanker` (sklearn-style wrappers), and
`xgboost.core.DMatrix` as an input data type. Has its own model-loader class
(`XGBTreeModelLoader`) to parse the raw booster dump into SHAP's internal
tree representation.

**LightGBM** — `lightgbm.basic.Booster`, `LGBMClassifier/Regressor/Ranker`.

**CatBoost** — `catboost.core.CatBoost/CatBoostClassifier/CatBoostRegressor`,
with its own loader (`CatBoostTreeModelLoader`) — CatBoost's oblivious-tree
structure differs enough from the others to need dedicated parsing logic.

**GPBoost** — `gpboost.basic.Booster` (Gaussian-process boosting; a less
common library, included for completeness).

**PySpark** — `model_type == "pyspark"` handled as a distinct branch
(distributed training framework; own additivity-check subtlety at line 528).

**PyOD** — `pyod.models.iforest.IForest` (anomaly-detection isolation forest,
a different framing of the same tree structure TreeSHAP already handles).

`shap/models/` (a *different*, much smaller module) is not this list — it's
wrappers for **generative/NLP models**: `TeacherForcing`, `TextGeneration`,
`TopKLM`, `TransformersPipeline` — used by `PartitionExplainer` and friends
when explaining language models, not tree ensembles.

---

## 3. Math topics

### Shapley value theory (game-theoretic foundation, applies to all explainers)

A Shapley value answers: "of the total payoff from `n` players cooperating,
how much should player `i` get, given they may join in any order?" Formally,
for feature set `N`, feature `i`'s value is the size-weighted average of its
marginal contribution `v(S ∪ {i}) − v(S)` over every subset `S ⊆ N \ {i}`,
weighted by `|S|!(n−|S|−1)!/n!` (the probability of that subset forming if
players join in a uniformly random order). This weighting is what the `pweight`
field in `cext/tree_shap.h`'s `PathElement` struct directly encodes — it's the
running "how much of all possible orderings pass through this path" value.

Four axioms uniquely characterize this solution (Shapley, 1953): **efficiency**
(contributions sum exactly to `v(N) − v(∅)` — this is our `f(x) = E[f(X)] +
Σφᵢ`), **symmetry** (two features with identical marginal contributions in
every coalition get equal credit), **dummy/null player** (a feature that never
changes the output gets zero credit), and **linearity** (Shapley values of a
sum of games is the sum of Shapley values). Every "exact" explainer in this
repo (Tree, Exact, Linear-interventional) satisfies all four by construction;
approximate ones (Kernel, Permutation, Sampling, Gradient) only satisfy them
in expectation, which is exactly why our benchmark measured KernelSHAP/LIME
having variance across repeated runs on the same instance.

Why this generalizes beyond features: **Owen values** (PartitionExplainer)
and **Winter values** (CoalitionExplainer) are Shapley's same four-axiom
solution concept applied to a *quotient game* on groups-of-features first,
then features within a group — same math, different player structure.

### TreeSHAP optimizations (the actual algorithm in `cext/tree_shap.h`)

Naive Shapley requires evaluating the model on `2^n` feature subsets — for
tree ensembles specifically, the 2018 TreeSHAP paper (arXiv:1802.03888, cited
at the top of `tree_shap.h`) shows this collapses to `O(T·L·D²)` (trees ×
leaves × depth²) by walking each tree exactly once and maintaining a
compressed representation of "all coalitions consistent with the path taken
so far," rather than ever materializing individual coalitions.

The mechanism, traced directly from the code:
- **`extend_path`** (line 348): each time the recursion descends past a real
  split, it updates every entry in the current path's `pweight` array in
  O(depth) time — incorporating the new split's `zero_fraction`/`one_fraction`
  (what portion of training data would go each way) into the running weighted
  count of "coalitions consistent with this path so far."
- **`unwind_path`** / **`unwound_path_sum`** (lines 363, 390): compute what the
  path weight *would have been* had one specific earlier feature been excluded
  from the coalition — this is the mechanism that isolates a single feature's
  marginal contribution without recomputing from scratch.
- **`tree_shap_recursive`** (line 411): the actual recursive walk — for each
  leaf reached, it calls `unwound_path_sum` once per feature on the path to
  get that feature's contribution at that leaf, weighted correctly by how many
  training rows support that path (`node_sample_weight`).

Two further optimizations sit on top of this exact algorithm:
- **`feature_perturbation` modes** — `tree_path_dependent` (baseline from the
  tree's own leaf-sample-counts, zero extra cost) vs. `interventional`
  (baseline from an explicit background sample, handles correlated features
  via causal-graph-consistent interventions, cost scales with background size).
- **`approximate=True`** (`_cext.dense_tree_saabas`) — a *different*, older,
  biased method (a single fixed feature ordering, no coalition weighting at
  all) traded in when speed matters more than the Shapley consistency
  guarantees; SHAP's own docstring warns it "places too much weight on lower
  splits in the tree."

### Quadrature-TreeSHAP (arXiv:2605.04497, Wettenstein/Mitchell/Yu 2026)

My initial guess — that "quadrature" meant Integrated Gradients' path integral
— was **wrong**. This paper is about something different and more interesting:
a numerically stable, depth-independent *reformulation of exact Path-Dependent
TreeSHAP itself* (the same algorithm we traced in `cext/tree_shap.h`), not an
approximation technique borrowed from gradient methods.

**The problem it solves.** Section 8 of the paper ("Numerical stability")
shows real TreeSHAP becomes numerically unstable past tree depth ~32 — the
efficiency property (`Σφᵢ = f(x) − E[f(X)]`, i.e. our `assert_additivity`
check) starts failing by orders of magnitude. The root cause, per the paper's
own diagnosis: "per leaf, TreeSHAP works in the *monomial basis*, whose
coefficients scale binomially, and recovers each Shapley value as a
Bernstein-weighted alternating sum of those coefficients, so any rounding in
the inflated monomial terms surfaces as catastrophic cancellation." This maps
directly onto the real code we read: the `PathElement.pweight` field, updated
by `extend_path`'s recursive multiply/divide-by-`(unique_depth+1)` arithmetic
(`tree_shap.h` lines 348–360), *is* that monomial-basis computation — repeated
multiplication and division of path fractions across many tree levels is
exactly the kind of operation that accumulates floating-point cancellation
error at high depth.

**Step 1 — reframe as a polynomial (§2, building on prior "Linear TreeSHAP").**
For one leaf $v$, define `q_i^v` = the "marginal multiplier" of feature $i$ on
$v$'s root-to-leaf path (Eq. 2: product of `1/edge_weight` along edges that
split on $i$, or $0$ if $x$ doesn't satisfy the split). Package all of a
leaf's path information into one polynomial in a dummy variable $y$:
$$G_v(y) := R_\emptyset^v \prod_{j \in M(v)} (q_j^v + y)$$
where $R_\emptyset^v$ is the leaf's *empty prediction* (leaf value × product
of all edge weights on the path — this is literally the tree-path-dependent
baseline contribution of that leaf, same concept as `node_sample_weight`
weighting we already understood). Feature $i$'s Shapley contribution at leaf
$v$ becomes a specific linear functional of this polynomial's coefficients
(Eq. 5) — so exact TreeSHAP is *equivalent to* extracting coefficients from a
per-leaf polynomial, rather than the path-weight bookkeeping in `pweight`.

**Step 2 — switch games from Shapley to weighted Banzhaf (§3).** Banzhaf
values are a classical alternative to Shapley values: instead of averaging a
feature's marginal contribution over a *uniformly random* coalition size (the
$\binom{m-1}{s}$ weighting that makes Shapley's exact formula binomial and
numerically fragile), Banzhaf values fix an inclusion probability $p$ for
every other feature and take a plain expectation (Eq. 6). The paper's key
insight (Theorem 1): for a tree leaf, the weighted-Banzhaf interaction value
$B_p^{(S)}(v)$ *also* has a clean closed form — a product of simple factors
$(1-p+pq_j^v)$ — and critically, **this polynomial in $p$ has no binomial
blow-up**, because Banzhaf's fixed-$p$ formulation avoids the alternating
binomial coefficients that cause Shapley's cancellation.

**Step 3 — recover Shapley by integrating Banzhaf over $p$ (§4, Theorem 6,
proved via a Beta-function identity in Appendix D).** The paper shows
$$\text{SII}(S) = \int_0^1 B_p^{(S)}(f_v)\, dp$$
— i.e., if you sweep the Banzhaf inclusion probability $p$ from 0 to 1 and
integrate, you get back the exact Shapley (interaction) value. This is the
crux: Shapley's cancellation-prone formula gets replaced by *an integral of a
numerically well-behaved function*.

**Step 4 — quadrature, at last.** Since (Theorem 2, proved in Appendix C) the
Banzhaf-in-$p$ integrand is a low-degree polynomial (degree ≤ tree depth $d$,
and *lower* for higher interaction orders — order-$s$ interactions cancel $s$
linear factors, dropping the degree to $d-s$), the integral can be computed
**exactly** with **Gauss–Legendre quadrature**: a numerical-integration method
that evaluates a polynomial at a small fixed set of specially-chosen points
and takes a weighted sum, and is *exact* (not approximate) for polynomials up
to degree $2n-1$ given $n$ points. Empirically the paper finds **8 fixed
quadrature points are enough to reach machine precision at any depth** —
because every factor evaluated is bounded and positive (no cancellation is
possible), unlike the original monomial-basis computation.

**Why this belongs in your syllabus alongside Shapley theory and TreeSHAP
optimizations:** it's the same object (`R_∅^v`, `q_i^v`, tree paths) we already
understood from `tree_shap.h`, reframed through a different game-theoretic
lens (Banzhaf) purely to get numerically nicer math, then patched back to
exact Shapley via one classical numerical-methods tool (Gauss–Legendre
quadrature). It also generalizes cleanly to **any-order Shapley interactions**
(pairwise, triple-wise, etc. — not just single-feature attributions), which
`TreeExplainer`'s existing `shap_interaction_values` (line 801 in `_tree.py`)
only computes pairwise; this paper's Algorithm 1 extends to arbitrary order
$s$ at complexity $O(MTNd^{s-1})$, a large improvement over the prior
state-of-the-art TreeSHAP-IQ's $O(MTLD\binom{F-1}{s-1})$.

**Reported gains (§8, XGBoost benchmarks, Tables 2–4):** 1.06×–10.59× faster
than TreeSHAP on CPU for single-feature attributions, 3.80×–58.11× faster for
pairwise interactions, up to 1209× faster than TreeSHAP-IQ at 6th-order
interactions — and, per Figure 1, TreeSHAP's efficiency-check error grows to
$>10$ (completely wrong) past depth 32, while Quadrature-TreeSHAP stays at
the float32 noise floor ($\sim10^{-7}$) at every depth tested up to 55. This
is already merged into XGBoost's C++ backend (footnote 1) and will ship in
the next XGBoost release after 3.2.0 — meaning `TreeExplainer(xgb_model)`
in *this very repo* will silently start using this algorithm once XGBoost
updates, with no API change (`pred_contribs=True` stays the same call).
