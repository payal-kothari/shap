# Your syllabus, in plain ML terms

Companion to `MENTAL_MODEL_BEGINNER.md` (read that first if you haven't —
it explains *why* SHAP exists and what problem it solves at all). This
covers your three syllabus items — explainers, models, math — the same
way: plain language first, jargon only after it's been explained.

---

## 1. List of explainers — "different ways to compute the same fair-share
number"

Recall from the beginner doc: an **explainer** is just a piece of code that
takes (a trained model, one data point) and hands back "how much did each
feature contribute to this prediction." Every explainer below is trying to
compute the *same* fair-share (Shapley) idea — they just differ in **how
much they're allowed to peek inside the model**, and what shortcut they use
because of that.

**TreeExplainer** — for decision-tree-based models only (Random Forest,
XGBoost, LightGBM, CatBoost). Peeks all the way inside the tree's actual
yes/no questions. Gets the mathematically exact fair-share answer, fast,
using the "walk the tree once" shortcut from the beginner doc. This is the
one we've used in all our benchmarks so far.

**KernelExplainer** — works on *any* model, even ones you can't peek
inside at all (a locked black box you can only feed inputs and read
outputs from). Since it can't peek, it can't use the tree shortcut — it
has to approximate: randomly hide some features, see how the prediction
changes, repeat many times, then solve a bit of math (a weighted linear
regression) to estimate each feature's contribution. Slower, and not
exact — repeat it twice on the same row and you might get a slightly
different answer.

**GPUTreeExplainer** — exactly TreeExplainer's same exact algorithm, just
rewritten to run on a graphics card (GPU) instead of a regular processor
(CPU), for extra speed on big workloads. Nothing different mathematically.

**GradientExplainer** — for neural networks specifically (models made of
layers you can compute derivatives/slopes through, i.e. "differentiable"
models). Instead of trees, it uses the model's gradients (how much the
output changes if you nudge one input slightly) to estimate feature
contributions. An approximation, not the exact fair-share answer, but
tailored to how neural nets work.

**LinearExplainer** — for plain linear models (e.g., "prediction = 3×age +
2×income − 500"). For these simple models, the fair-share answer has a
clean, exact formula you can compute directly — no approximation, no tree
walk needed, just arithmetic.

**PartitionExplainer** — for any model, but assumes you can group related
features together first (e.g., "these 5 features are all about employment,
treat them as one group"). By working with groups instead of every single
feature individually, it becomes much faster than KernelExplainer while
still being a fair-share method — just fair-share *among groups first, then
within a group*.

**CoalitionExplainer** — same grouping idea as PartitionExplainer, but for
more complex group structures (e.g., "these features cluster by time
period, and within each time period they cluster by data type"). Handles
nested/hierarchical groupings that PartitionExplainer's simpler grouping
can't.

**ExactExplainer** — brute force, but smart about it. Actually tries every
relevant combination of features (not an approximation), which only stays
practical for a small number of features (roughly under 15-20). Useful as
a "ground truth" to check faster/approximate methods against.

**PermutationExplainer** — approximates the fair-share answer by actually
trying out random orderings of the features (recall Step 1 of the beginner
doc — "average over every possible order features are revealed"). Doesn't
try *all* orderings (too many), but a smart random sample of them, in both
forward and backward direction to cancel out some error.

**SamplingExplainer** — an older, simpler cousin of KernelExplainer.
Similar random-sampling approach, but assumes features don't interact with
each other in complicated ways. Good when you have a lot of background/
reference data to sample from.

**AdditiveExplainer** — for models that are literally built by adding up
simple per-feature effects (no complex feature-interactions at all, e.g.
"prediction = f(age) + f(income) + f(education)", each computed
separately). If your model actually has interactions between features
(most models do) and you use this explainer anyway, you'll get **wrong**
numbers — it's built assuming a simplifying condition that must be true for
it to work at all.

**The one-line summary:** exact-and-fast methods exist only when you can
peek inside the model's structure (trees → TreeExplainer, linear models →
LinearExplainer, additive models → AdditiveExplainer). Everything else is
some flavor of "can't peek, so approximate via sampling/random orderings"
(Kernel, Permutation, Sampling, Gradient), or "peek at groups instead of
individual features to go faster" (Partition, Coalition), or "just brute
force it since there aren't too many features" (Exact).

---

## 2. List of models — "which trained models can TreeExplainer actually
read?"

This isn't a list of *ML concepts* — it's a practical list of **which
specific software libraries' trained tree models** TreeExplainer knows how
to open up and peek inside. Each of these libraries stores its trained
tree in its own internal format, so SHAP needs to know how to translate
each one into the "list of yes/no splits and leaf values" shape its
algorithm expects.

**scikit-learn (sklearn)** — the most common general-purpose Python ML
library. Its tree-based models: Random Forest (many trees, each trained on
a random subset of data, averaged together), Extra Trees (similar, with
extra randomness), Gradient Boosting (trees trained one after another,
each fixing the previous ones' mistakes — this is the family we used in
our own benchmark), Isolation Forest (used for spotting anomalies/outliers
rather than predicting a label), and plain single Decision Trees.

**XGBoost** — a very popular, highly optimized gradient-boosting library
(same "trees fixing each other's mistakes" idea as sklearn's Gradient
Boosting, but faster and more feature-rich). One of the most common real-
world choices for tabular data.

**LightGBM** — Microsoft's gradient-boosting library, similar purpose to
XGBoost, different internal tree-growing strategy (grows trees leaf-by-
leaf instead of level-by-level, generally faster on large datasets).

**CatBoost** — Yandex's gradient-boosting library, notable for handling
categorical features (like "city name" or "job title") natively without
needing to convert them to numbers first the way other libraries require.

**GPBoost** — a less common library combining gradient boosting with
Gaussian Processes (a statistical modeling technique); included for
completeness, not something you're likely to run into often.

**PySpark** — not a model type itself, but a distributed/big-data
computing framework; this entry means SHAP can explain tree models that
were trained using Spark's distributed machine learning tools.

**PyOD** — a library specifically for anomaly/outlier detection; its
Isolation Forest implementation is supported here.

**Not on this list, but easy to confuse:** `shap/models/` (a different,
smaller folder in the codebase) is not "more model types TreeExplainer
supports" — it's a completely separate set of helpers for explaining
**language models** (the kind that generate text), used by different
explainers (Partition/Coalition-style), not TreeExplainer at all.

---

## 3. Math topics — see `MENTAL_MODEL_BEGINNER.md` for the full plain-
English build-up. Quick recap of what each topic *is*, in one line:

- **Shapley value theory** — the "fair way to split credit among
  teammates" idea from 1950s game theory, applied to input features
  instead of teammates.
- **TreeSHAP optimizations** — the shortcut that lets you compute that
  fair-share number *exactly*, for tree models, by walking the tree once
  instead of checking every possible ordering of features.
- **Quadrature (from the paper you shared)** — a fix for a rounding-error
  bug in that shortcut on very deep trees, done by computing a numerically
  safer cousin quantity (Banzhaf values) at a handful of well-chosen points
  and averaging them, which turns out to give back the exact original
  answer.

---

## How to actually use these two documents together

- `MENTAL_MODEL_BEGINNER.md` = the *story*, read once, top to bottom, no
  skipping.
- This file = the *reference*, come back to whenever you're about to touch
  a specific explainer or model type, to remind yourself in one paragraph
  what it's for and why it exists.
- `MENTAL_MODEL.md` (the original, code-and-equation-heavy one) = what to
  read once each of the above feels solid and you want the precise
  file/line/formula version.
