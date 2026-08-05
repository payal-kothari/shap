# Same topics, explained with basic ML words (simple version)

Uses real ML terms (model, prediction, feature) but explains each one
simply the first time it shows up. Pairs with `MENTAL_MODEL.md`, which has
the full technical/code version of these same topics.

---

## First, what are we even talking about?

A **model** is a computer program that has learned, from lots of past
examples, how to make a **prediction** about something new. You give it some
**features** (pieces of information about a person or thing — like age,
income, whether they like soccer), and it predicts something — like "this
person will probably like this product."

The model is usually right pretty often! But it doesn't tell you **why** it
predicted that. Was it the age? The soccer feature? Nobody knows just from
looking at the prediction — the model just outputs a number.

Our whole job is: **figure out, after the fact, how much each feature
contributed to that one prediction.** Like saying "the soccer feature was
worth 60% of why it predicted that, income was worth 30%, and age barely
mattered."

That's the whole subject — **feature attribution**. Everything below is
just different ways people have invented to actually compute "how much did
each feature contribute."

---

## 0. What kinds of models are we even trying to explain?

Before we get to "how do we explain a model," it helps to know that
**"model" isn't one single thing** — it's a general word for lots of
different kinds of programs that all do the same job (take in features,
output a prediction), but are built completely differently inside. Which
explainer you're allowed to use depends heavily on which kind of model you
have. Here are the main kinds, in plain words:

**Linear models** — the simplest kind. The prediction is just a simple
formula, like `prediction = 3 × age + 2 × income − 500`. Each feature gets
its own fixed multiplier, you multiply and add them all up, done. Easy to
understand, but only works well when the real-world pattern is actually
that simple (it usually isn't, for complicated things).

**Decision trees** — a chain of yes/no questions ending in an answer:
"Is age over 30? Yes → Is income over 50k? No → predict 0.8." One tree by
itself is usually not very accurate, but it's easy to look at and
understand exactly why it gave the answer it did.

**Tree ensembles** (also called tree-based models) — instead of just one
decision tree, you build **hundreds of separate trees**, each asking its
own chain of yes/no questions, and add up (or average) all their answers
together for the final prediction. This is one of the most common and most
accurate kinds of model for "normal" data like spreadsheets (names, ages,
prices, etc.). XGBoost, LightGBM, CatBoost, and Random Forest are all
popular tools for building this kind of model — this is exactly the kind
of model TreeExplainer/TreeSHAP is built for (see section 2 below).

**Random Forest, specifically** — a good concrete example of a tree
ensemble, worth unpacking on its own. One decision tree by itself has a
real weakness: it tends to **overfit**, meaning it can end up memorizing
quirky little patterns in the exact data it was trained on that don't
actually hold up in general — like if, purely by coincidence, every tall
person in the training data happened to also like pizza, a single tree
might latch onto height as a pizza-predictor, even though there's no real
connection. It's too trusting of the one dataset it saw.

Random Forest's fix: build **many separate trees** (often hundreds) and
combine their answers (majority vote, or average, depending on the task) —
but deliberately make the trees different from each other using two tricks:

1. **Each tree studies a different random slice of the training data** —
   like giving each of 100 students a different random sample of practice
   problems instead of the identical set.
2. **Each tree is only allowed to look at a random subset of features at
   each yes/no question**, instead of every feature being available every
   time. This forces different trees to discover different useful splits,
   instead of all of them independently finding the same "obvious" one
   first.

Because each tree saw slightly different data and considered different
features, each tree ends up making *different* mistakes — one tree might
be fooled by the height/pizza coincidence, but most of the other 99 won't
be. When you average across all the trees, individual quirky mistakes
mostly cancel out, and what's left is the real underlying pattern most
trees agree on. This is often called "the wisdom of the crowd": many
independent, imperfect guessers combined tend to beat one single guesser.

**XGBoost, LightGBM, CatBoost — a different way to combine many trees
(boosting)** — Random Forest grows its trees **independently and in
parallel**: each tree gets a random slice of data/features, does its best
on its own, and all the trees' guesses get averaged at the end, with no
tree knowing what the others are doing. XGBoost, LightGBM, and CatBoost
belong to a different family, called **gradient boosting**, that grows
trees **one at a time, in order, where each new tree's entire job is to fix
the mistakes the previous trees already made.**

Here's the idea, step by step. Say you're guessing house prices. You build
**Tree 1** first — it's not perfect, so for each house you can work out its
**error** (actual price minus Tree 1's guess). Instead of throwing that
away, you build **Tree 2** whose whole job is to predict *Tree 1's errors* —
not the house price itself, but "how wrong was Tree 1, and in which
direction, for this house?" Add Tree 2's correction on top of Tree 1's
guess, and the combined guess is now a bit better. Then you look at what's
*still* wrong after Tree 1 + Tree 2, and build **Tree 3** to fix that
leftover error. Repeat this hundreds of times — each tree is small and only
fixes a little, but hundreds of small corrections stacked on top of each
other add up to a very accurate combined prediction. This is fundamentally
sequential: you can't build Tree 5 without already knowing what Trees 1-4
got wrong.

XGBoost, LightGBM, and CatBoost are all doing this same "fix the leftover
mistakes" idea — their differences are about speed and engineering
choices, not the core concept:

- **XGBoost** — one of the first hugely popular boosting tools, carefully
  engineered for speed, with built-in safeguards against overfitting (it
  slightly penalizes trees that get too complicated).
- **LightGBM** — built by Microsoft, focused on being extra fast on very
  large datasets. Its trick: instead of growing a tree level-by-level
  (finishing every question at depth 1 before starting depth 2), it grows
  leaf-by-leaf, always expanding whichever single branch helps the most
  next. Often reaches a good answer with fewer total splits.
- **CatBoost** — built by Yandex, standout feature is handling
  **categorical features** (things like "city name" or "job title" —
  categories, not numbers) natively, without needing to manually convert
  them into numbers first the way the other tools usually require.

All three are still tree ensembles — no matter whether the many trees were
grown independently (Random Forest) or sequentially to fix each other's
mistakes (XGBoost/LightGBM/CatBoost), the *final shape* is the same: a pile
of yes/no question-chains whose leaf values get added up. That's exactly
why TreeExplainer/TreeSHAP works identically on all of them (section 2).

**Neural networks** — inspired loosely by how brains work. Here's the
simple way to picture one.

Say you're trying to guess a house's price from 3 clues: its size, how many
bedrooms it has, and its neighborhood. A linear model would just do
`price = 1000 × size + 5000 × bedrooms + 20000 × neighborhood_score` — one
formula, straight from clues to answer. But real life is often not that
simple: maybe a big house is only worth a lot extra if it's *also* in a good
neighborhood — a huge house in a bad neighborhood doesn't get the full
bonus. A single "multiply and add" formula can't capture that kind of "it
depends on the combination" pattern.

A neural network handles this by not jumping straight from clues to answer.
Instead, imagine a **small team of helpers** in between: each helper looks
at the same 3 original clues, but combines them using their own private set
of "how much do I care about each clue" numbers (these private numbers are
called **weights**), and produces one opinion-number of their own. For
example:

- **Helper 1** might specialize in noticing "is this a big house in a good
  neighborhood?" — combining size and neighborhood in their own way.
- **Helper 2** might specialize in noticing "is this a small house with
  lots of bedrooms squeezed in?" — a different combination of the same
  clues.
- **Helper 3** notices something else entirely — maybe a combination none
  of us would have thought to look for.

None of the helpers were told what to specialize in — each one just ends up
finding its own useful combination of the clues, purely from being trained
on lots of past examples. Then a **final helper** looks at all these
helpers' opinions (not the original clues anymore, just their summaries)
and combines *those*, again with its own weights, into the final price
guess.

That's a neural network: **layers of helpers, each combining the previous
layer's outputs using their own weights**, stacked one after another, until
you reach a final answer. (A "layer" is literally one row of helpers.)
Stacking many layers of many helpers lets a neural network learn really
complicated "it depends on combinations of things" patterns — which is why
they're so good at recognizing pictures or understanding language.

But here's the catch: with a decision tree, you can point at the exact
yes/no question that mattered. With a neural network, the final answer came
from hundreds or thousands of helpers, each mixing things their own private
way, several layers deep — there's no single clean question to point to.
That's exactly why methods like GradientExplainer exist, specifically for
this kind of model: instead of reading yes/no splits like TreeSHAP does, it
nudges the network's weights slightly and watches how much the final answer
wiggles, to work backward toward how much each original clue mattered.

**"Black box" models** — this isn't a different way of *building* a model,
it's about **what you're allowed to see**. Any of the model types above can
be a "black box" to you if you're only given a way to feed it inputs and
read outputs, without being allowed to see its actual internal structure —
like a bank's proprietary scoring system. When a model is a black box to
you, you're stuck using the "poke it from outside and estimate" style of
explainer (KernelExplainer, LIME), even if the model happens to secretly be
a tree ensemble underneath, which normally would've let you use the much
faster, exact TreeSHAP shortcut instead.

The rest of this document mostly focuses on **tree ensembles**, since
that's what TreeSHAP (the paper you shared) is specifically about.

---

## 1. The different ways to compute "how much did each feature contribute"
(this replaces "list of explainers")

These are called **explainers** — different tools/methods for computing
feature attribution. Think of them as different strategies, because they
differ in how much they're allowed to look inside the model.

**TreeExplainer** — works when the model is a "tree" type (we'll get to what
that means in section 2). If you're allowed to look inside and see the
model's actual structure, you can compute the *exact* contribution of each
feature, fast, every time.

**KernelExplainer** — works on ANY model, even a locked black box where you
can only feed it inputs and read outputs (you can't look inside at all). It
works by feeding the model different combinations of features — sometimes
hiding one — and watching how the prediction changes. Do this many times and
you can *estimate* each feature's contribution. Slower, and not exact —
run it twice and you might get a slightly different answer both times.

**GPUTreeExplainer** — the exact same method as TreeExplainer, just run on
a graphics card (GPU) instead of a regular processor, so it finishes
quicker on big jobs. Same answer, just faster.

**GradientExplainer** — for a different kind of model (neural networks)
that's built out of adjustable numbers called "weights." You can nudge a
weight slightly and see exactly how much the prediction moves — that
tells you a feature's contribution. Only works on this kind of model.

**LinearExplainer** — some models are so simple that the prediction is just
"3 × age + 2 × income," with nothing more complicated going on. For models
like this, you don't need to estimate anything — you can read each
feature's contribution straight off the simple math. Instant, exact.

**PartitionExplainer / CoalitionExplainer** — if you have a LOT of
features, checking every possible combination takes forever. Instead, you
group related features into teams first (like "all the demographic
features" as one team), figure out how much each team contributed, then
split credit within the team. Much faster than checking every single
feature by itself. CoalitionExplainer is the fancier version, for teams
that have teams inside them.

**ExactExplainer** — for a small number of features (fewer than ~15-20),
you can just try every single possible combination and get a perfectly
correct answer. Doesn't work once you have too many features, but great
for double-checking that the faster methods above got it right.

**PermutationExplainer** — keeps randomly reordering the features (imagine
revealing them to the model in a different random order each time) and
watches how the prediction builds up. Average this over many random orders
and you get close to the true contribution, without checking every single
possible order.

**SamplingExplainer** — an older, simpler version of the random-reordering
idea, which assumes features don't interact with each other in complicated
ways.

**AdditiveExplainer** — only works if the model itself was literally built
by adding up separate per-feature pieces (like "3 points for age" + "2
points for income," with no feature changing how another one counts). If
you use this explainer on a model that does NOT work that way, it gives you
the **wrong answer** and has no way of knowing it's wrong.

---

## 2. What kinds of models can TreeExplainer actually look inside?
(this replaces "list of models")

TreeExplainer (the fast, exact one) only works on models built a certain
way — as a big pile of separate **decision trees**. A decision tree is just
a chain of yes/no questions ending in a number: "Is age over 30? Yes → Is
income over 50k? No → predict 0.8."

A real model isn't just one tree — it's usually **hundreds of separate
trees**, each with its own full chain of yes/no questions. To make one
prediction, your data point walks down *every single tree*, lands on one
number in each tree, and all those numbers get added together for the
final prediction.

Different companies build tools that create these tree-based models, each
with their own name but built the same underlying way:

- **A common library** builds models as a big pile of separate trees, where
  you run through all of them and add up every tree's answer — this is the
  most common setup, and the one we've used in our own work.
- Some tools support a **single simpler tree**, not a whole pile.
- One tool's trees are specifically built for **spotting an "outlier"**
  (something unusual) in a group, instead of predicting a normal value.
- A few other, less common libraries make basically the same style of
  model, just built by different companies with different settings.

TreeExplainer knows how to open up all of these, because even though each
one is packaged differently, inside they're all the same shape: a pile of
yes/no question-chains ending in numbers.

**Important distinction:** there's a completely separate part of the SHAP
toolkit for a *different* kind of model — ones that generate text/language
instead of predicting a number. That's not related to trees at all; don't
mix the two up.

---

## 3. The math ideas — same three topics, simple version

### Idea 1: The fair credit-sharing rule (Shapley values)

Three friends work together on something and it turns out great. How do you
fairly decide who gets how much credit?

Here's the fair way: imagine every possible order the three friends could
have joined in. For each order, ask "how much extra value did this friend
add, right when they joined, given who was already there?" Do this for
every possible order, then average it. That average is each friend's fair
share of the credit.

This exact rule is called a **Shapley value**, from game theory (a branch
of math about how people/things cooperate and split rewards fairly). SHAP
applies this same rule to model features instead of friends: each feature
is a "player," the prediction is the "reward," and a feature's Shapley
value is its fair share of credit for that one prediction — averaged over
every possible order the features could've been revealed to the model.

**Worked example, with real numbers.** You and two friends run a lemonade
stand. Alice makes the lemonade, Bob yells to bring customers over, Charlie
brought ice. Revenue by group:

| Who's there | Revenue |
|---|---|
| Nobody | $0 |
| Alice alone | $10 (lemonade exists, but warm, nobody knows) |
| Bob alone | $0 (yelling with no lemonade) |
| Charlie alone | $0 (just ice, no lemonade) |
| Alice + Bob | $40 (good lemonade, good crowd, but warm) |
| Alice + Charlie | $20 (cold lemonade, nobody notices) |
| Bob + Charlie | $0 (yelling around ice, no lemonade) |
| All three | $100 (cold, delicious, popular) |

Watch Charlie's contribution change depending on who's already there: if he
joins Alice+Bob (already at $40), the total jumps to $100 — Charlie added
**+$60**. If he joins Bob alone (at $0), the total stays $0 — Charlie added
**$0**. Same person, wildly different contribution, purely because of
context. That's exactly why you can't just look at one scenario — you check
every possible order the three could've joined in, and average Charlie's
contribution across all of them, to get his one true fair share (his
Shapley value).

### Idea 2: A fast shortcut for tree-based models (TreeSHAP)

If you have a lot of features, checking every possible order they could be
revealed in takes way too long — for more than ~20 features, it's more
combinations than you could ever compute, even with a huge computer.

But if the model is a decision tree (section 2), there's a shortcut. As
your data point walks down one yes/no question chain, at every fork, the
tree already remembers — from when it was trained — roughly what fraction
of training examples went left vs. right at that exact fork. Using those
already-known fractions, you can correctly work out every feature's fair
share **while walking the tree just once**, instead of separately checking
every possible order. Same fair-share (Shapley) answer as Idea 1, computed
via a much faster shortcut. This shortcut is called **TreeSHAP**.

### Idea 3: Fixing a rounding-error problem in that shortcut (quadrature)

Here's a real problem: TreeSHAP's shortcut involves a lot of multiplying
and dividing numbers together as you walk down a really long chain of
yes/no questions (30+ levels deep). When a computer multiplies and divides
tiny numbers together over and over, **small rounding mistakes pile up** —
like a game of telephone, where a message whispered around a big circle of
100 kids comes back garbled, even though everyone tried to whisper it
correctly.

When this happens with TreeSHAP on very deep trees, the feature
contributions stop adding up to the right total — breaking the one rule
that matters most (the numbers must add up exactly, nothing lost, nothing
extra).

**The fix a group of researchers found (the paper you shared):** there's a
close cousin of the Shapley/fair-share rule, called a **Banzhaf value**,
that computes credit slightly differently and doesn't have this
rounding-error problem — it stays accurate even on very deep trees. It's
not quite the number you actually want, though. So here's the trick: the
Banzhaf version has a hidden "dial," a probability you can set anywhere
from 0 to 1. The researchers proved that **if you compute the Banzhaf
answer at every possible dial setting from 0 to 1, and average all of
those answers together, you get back the exact original Shapley answer** —
not an estimate, the literal correct number.

Checking every dial setting from 0 to 1 sounds like it should also take
forever (infinite settings in between). But it turns out you only need to
check **8 specific, carefully chosen dial settings** to get the perfect
answer, no matter how deep the tree is. That "only need to check a handful
of clever spots to know the full answer" trick is what the math term
**quadrature** means — a smart, small set of points that lets you compute
an average/integral almost exactly, without checking every possible value.

---

## 4. Other basic ML concepts worth knowing (beyond explainers/models/math)

Everything above is about explaining a model's predictions. These are the
basic ideas about **building and judging** a model in the first place —
worth knowing since they come up constantly once you're working with any
model, SHAP or not.

- **Training set vs. test set** — you never judge a model on the same data
  it learned from, because that's like grading a test using the answer key
  the student copied from. You split your data in two: a **training set**
  (the model learns from this) and a **test set** (kept completely hidden
  during training, used only afterward to check whether the model actually
  learned a real pattern or just memorized the training examples).

- **Overfitting** (we already met this with Random Forest) — when a model
  latches onto quirky coincidences in the training data that don't hold up
  in general, instead of learning the real underlying pattern. A model that
  does great on the training set but poorly on the test set is a classic
  sign of overfitting.

- **Classification vs. regression** — the two basic flavors of "what is the
  model predicting." **Regression** predicts a number (a house price, a
  revenue amount). **Classification** predicts a category (will default /
  won't default, spam / not spam). This matters for SHAP directly: recall
  that our very first walkthrough's `base_value` was a raw score, not a
  probability — that distinction comes from this being a classification
  model under the hood.

- **Loss function** — a single number that measures "how wrong was the
  model," which the model tries to make as small as possible while
  learning. This is the concept underneath boosting's "error" from the
  XGBoost/LightGBM explanation above — Tree 2 wasn't just predicting Tree
  1's raw mistake, it was working to minimize a specific loss function.

- **Accuracy isn't the whole story (precision and recall)** — "% of
  predictions that were correct" can be dangerously misleading. Example: if
  only 1% of loan applicants actually default, a model that always guesses
  "won't default" is 99% accurate and completely useless — it never
  catches a single real default. **Precision** and **recall** are two more
  careful ways to measure quality: precision asks "of the times the model
  said 'default,' how often was it right?"; recall asks "of all the actual
  defaults, how many did the model catch?" You usually have to trade one
  off against the other.

- **Bias vs. variance** — the tension between a model being too simple
  (misses real patterns — called "bias") and too complicated (memorizes
  noise instead of the real pattern — called "variance," another name for
  overfitting). Random Forest's whole "average many independent trees"
  trick is specifically a variance-reduction technique, seen from this
  angle.

- **Feature engineering / feature scaling** — raw inputs often need
  transforming before a model can use them well — like turning a category
  such as "job title" into numbers, or making sure "income" (in the tens
  of thousands) and "age" (under 100) are on comparable scales so one
  doesn't dominate purely by being a bigger number. We did a small version
  of this ourselves, using an encoder to turn categorical columns into
  numbers before training our own benchmark model.

- **Cross-validation** — a more thorough version of the train/test split:
  instead of splitting the data just once, you split it multiple different
  ways and test each time, then combine the results — giving a more
  reliable estimate of how good the model really is, instead of trusting
  one lucky (or unlucky) split.

- **Hyperparameters** — the "dial settings" you choose *before* training
  starts (like how many trees to grow, how deep each tree is allowed to
  get, or how big a correction each boosting step makes) — as opposed to
  what the model *learns* during training. We actually set some of these
  directly in our own benchmark code (`n_estimators`, `max_depth`) without
  naming them as such at the time.

- **Regularization** — a general technique, across many model types, for
  fighting overfitting by discouraging the model from becoming too
  complicated. This is the broader idea behind XGBoost's "penalize overly
  complicated trees" behavior mentioned earlier — regularization is the
  general concept, XGBoost's penalty is one specific example of it.

---

## The whole thing in one breath

*A model gives you a prediction but not a reason, so feature attribution
methods (explainers) exist to fairly split credit for that prediction among
the input features — the fairest version of this, the Shapley value, works
by checking every possible order features could be revealed and averaging,
borrowed straight from how you'd fairly split credit for teamwork. For
tree-based models, TreeSHAP gets the same exact fair answer via a much
faster shortcut — walk the tree once instead of checking every order — but
on very deep trees that shortcut's repeated multiplying/dividing causes
tiny computer rounding errors to pile up and break the answer. The fix:
compute a rounding-error-proof cousin quantity (Banzhaf values) at just 8
carefully chosen points and average them — quadrature — which turns out to
give back the exact original Shapley answer, cheaply and without ever
getting corrupted, no matter how deep the tree is.*
