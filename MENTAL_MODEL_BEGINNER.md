# Beginner's path into the Quadrature-TreeSHAP paper

This is the on-ramp. Read this before `MENTAL_MODEL.md`. No equations you
haven't seen defined in plain English first.

---

## Step 0: What problem are we even solving?

You have a trained ML model. It looks at a person's data (age, income,
education, etc.) and spits out a prediction — say, "this person will default
on a loan" or "this person will click this ad."

The model is usually a **decision tree ensemble** (hundreds of small
yes/no-question trees whose answers get added up — think "is age > 30? if
yes, go one way; if no, go another way; eventually land on a number").

**The problem:** the model gives you a *number*, but not a *reason*. A bank
can't tell a customer "you were denied a loan" without saying *why*. We need
to take one prediction and break it into: "age contributed +0.2, income
contributed -0.5, education contributed +0.1," etc. — one number per input
feature, telling you how much that feature pushed the prediction up or down.

This "breaking a prediction into per-feature pieces" is called **feature
attribution**, and the leading method for doing it is called **SHAP**. That's
the whole subject. Everything else in this document is just: *how do you
actually compute those per-feature numbers, correctly and fast?*

---

## Step 1: Where do the "correct" attribution numbers come from? (Game theory,
in plain terms)

Here's the intuition, no formulas yet.

Imagine 3 coworkers split a $300 bonus their team earned. How much should
each person get? One fair way to decide: imagine every possible order the
three people could have "joined the project" (there are 6 orderings). For
each ordering, ask "how much extra value did this person add, given who was
already there when they joined?" Average that "how much extra value did I
add" number across all 6 orderings. That average is that person's fair share.

This exact idea — "average the extra value you add, across every possible
order people could join in" — is called a **Shapley value**. It comes from
1950s economics/game-theory research on how to fairly split a group payoff.

**SHAP applies this same idea to ML features instead of coworkers.** Each
input feature (age, income, education...) is a "player." The "payoff" is the
model's prediction. A feature's SHAP value is: *how much did this feature
add to the prediction, averaged over every possible order you could have
"revealed" the features to the model?*

This is why it's called a fair, principled method rather than a heuristic —
it inherits guarantees from that 1950s economics theory, most importantly:
**if you add up every feature's SHAP value, you get back exactly the gap
between this prediction and an average/baseline prediction.** Nothing is
lost, nothing double-counted. That's the one property to hold onto — we'll
call it "the numbers must add up exactly."

### Worked example: the lemonade stand

Let's make Step 1 fully concrete with real numbers, before we ever touch a
model. This is the same "coworkers splitting a bonus" idea, just worked all
the way through by hand.

**The setup.** You and two friends run a lemonade stand. Your team of 3
"players" (stand-ins for *features* later):

- **Alice** — has the secret recipe and mixing skills
- **Bob** — has a loud voice, brings customers over
- **Charlie** — brought a giant block of ice

Goal: total revenue (the stand-in for *the model's prediction*).

**Step A: List every possible coalition.** A **coalition** is just any
subset of the team working together — some show up, some don't. With 3
people there are 8 possible coalitions (2³, since each person is either "in"
or "out"):

| Coalition | Revenue | Why |
|---|---|---|
| {} (nobody shows up) | $0 | no stand at all |
| {Alice} | $10 | lemonade exists but is warm, nobody knows about it |
| {Bob} | $0 | yelling at an empty street, no lemonade to sell |
| {Charlie} | $0 | just a melting ice block, no lemonade |
| {Alice, Bob} | $40 | great lemonade, lots of customers — but warm |
| {Alice, Charlie} | $20 | cold lemonade, but nobody notices the stand |
| {Bob, Charlie} | $0 | yelling around cold ice, still no lemonade |
| {Alice, Bob, Charlie} | $100 | cold, delicious lemonade *and* a crowd |

**Step B: Marginal contribution — what one person adds to a coalition
that's already formed.** This is the "how much extra value did this person
add, given who was already there" question from Step 1, now with numbers.
Crucially, **the same person's contribution changes depending on who's
already there** — that's the whole reason we need to average over many
scenarios rather than trust just one.

Take **Charlie (the ice)** and watch his contribution swing wildly:

- *Charlie joins {Alice, Bob}:* they had $40 without him, $100 with him →
  Charlie added **+$60**. (Ice is huge once there's lemonade *and*
  customers already — nothing melts before it's sold.)
- *Charlie joins {Bob} alone:* they had $0 without him, $0 with him →
  Charlie added **$0**. (Ice is worthless if nobody's making lemonade.)
- *Charlie joins {Alice} alone:* they had $10 without him, $20 with him →
  Charlie added **+$10**.

Same person, three completely different contributions, depending purely on
context. This is exactly why "just look at one scenario" isn't a fair way
to credit anyone.

**Step C: The Shapley value = average Charlie's contribution over *every*
order the team could have assembled.** Instead of picking one scenario, list
every order the 3 people could join in (there are 3! = 6 such orders),
work out Charlie's marginal contribution at the moment *he* joins in each
one, and average those 6 numbers. That average is Charlie's Shapley value —
his single, fair, context-independent share of the $100.

**Mapping this back to ML, in one line each:**

- **Coalition** → any subset of *features* the model is given at once
  (e.g., using only `[Age, Income]` and ignoring the rest).
- **Marginal contribution** → how much the prediction jumps when one more
  feature (e.g., `Credit Score`) is added to a subset that's already there.
- **Shapley value** → the average of that feature's marginal contribution,
  taken over every possible order the features could have been "revealed"
  to the model — exactly like averaging Charlie's $0/$10/$60 swings above,
  just with features standing in for people and "prediction" standing in
  for "revenue."

---

## Step 2: Why is this hard to actually compute?

The literal definition ("average over every possible order the features
could be revealed") sounds simple, but if you have 20 features, there are
20! (a gigantic number) possible orderings. For 100 features it's
astronomically larger. You cannot literally check every ordering.

So the entire research area — including everything in the paper you shared
— is about **finding clever shortcuts that get you the exact same correct
answer without literally checking every ordering.** That's the whole game.
Different shortcuts work for different situations:

- If your model is a black box (you can only feed it inputs and read
  outputs, e.g., a bank's proprietary scoring API) — you're stuck doing
  something approximate and slow. This is what **KernelExplainer** and
  **LIME** do: randomly try a bunch of orderings/combinations, not all of
  them, and estimate.

  **Real example — Lightbox partner models.** Some of our Lightbox partner
  models are, under the hood, tree ensembles just like the ones we score
  ourselves — meaning the *fast, exact* TreeSHAP shortcut would technically
  work on them if we could see the actual tree structure (the yes/no splits
  and leaf values). But partnership/data-access terms only give us a
  scoring API for those models, not the underlying tree — so even though
  the "peek inside" shortcut is technically possible on a model like this,
  we're not permitted to use it. We're contractually stuck treating it as a
  black box, meaning KernelExplainer/LIME-style approximation is the only
  option available to us for these specific models, even though it's the
  slower and less exact path.

- **If your model is specifically a decision tree ensemble (like XGBoost,
  LightGBM, Random Forest) — you get to peek inside the model, see its
  actual yes/no splits, and there's a mathematical shortcut that gives you
  the exact answer in reasonable time, without ever checking all those
  orderings. This shortcut is called TreeSHAP, and it's the main character
  of the paper you shared.**

---

## Step 3: What does "peeking inside a tree" actually let you skip?

Here's the intuition for TreeSHAP's shortcut, still no formulas.

A decision tree is a chain of yes/no questions ending in a number (the
"leaf value"). When your specific data point walks down the tree, it takes
one specific path — say: "age > 30? yes → income > 50k? no → predict 0.8."

The trick: at each yes/no question (**split**) on that path, the tree
itself already recorded, from its training data, *what fraction of training
examples went left vs. right*. That fraction is a stand-in for "if we didn't
know this feature's value yet, what would we expect on average?" You don't
need to try every ordering of features — you can walk the tree **once** and,
at each split your data point passes through, use those already-known
training fractions to correctly account for "what if this feature came
first vs. last in the ordering," all at once, using some clever bookkeeping
math.

That's the entire content of "TreeSHAP optimizations" — it's an algorithm
that walks each tree exactly once (instead of checking every ordering) and
still gets the mathematically exact, fair-share answer from Step 1.

---

## Step 4: So what's the problem the paper you shared is fixing?

That "clever bookkeeping" from Step 3, it turns out, involves a lot of
multiplying and dividing numbers together as you go deeper into a tree.
On very deep trees (very long chains of yes/no questions — 30+ levels
deep), doing lots of multiplication and division on a computer causes
**tiny rounding errors that pile up** — the kind of thing where
`0.1 + 0.2` doesn't quite equal `0.3` in computer math, except here it
happens repeatedly and compounds.

When it compounds enough, the "the numbers must add up exactly" guarantee
from Step 1 — the one thing SHAP promises you — **breaks**. The paper shows
real examples where, past a certain tree depth, TreeSHAP's per-feature
numbers stop adding up to the actual prediction at all. That's a serious
correctness bug, not just a slowness issue.

---

## Step 5: The paper's fix, in plain English

The authors noticed something clever: there's a **sibling concept** to
Shapley values called a **Banzhaf value** (another, older way from the same
game-theory field to split credit fairly among players, with a slightly
different fairness rule). Banzhaf values, it turns out, can be computed for
a tree using math that **doesn't have the multiply/divide-heavy structure**
that causes TreeSHAP's rounding problem — Banzhaf's version of the
bookkeeping is numerically well-behaved even at extreme tree depth.

The catch: you actually want *Shapley* values (that's the standard everyone
uses and trusts), not Banzhaf values. So here's the move:

1. Compute the tree's contribution using the numerically-safe Banzhaf math.
2. But do it while treating one part of that Banzhaf formula as a "dial"
   you can turn from 0 to 1 (a probability knob).
3. It turns out — and this is the paper's proven result — if you
   **average that Banzhaf answer over every possible setting of the dial
   from 0 to 1**, you get back the *exact* Shapley value. Not an
   approximation of it — the literal correct answer.

So instead of ever doing the fragile Shapley bookkeeping directly, they
compute a numerically-safe quantity (Banzhaf) at many dial-settings and
average — and that average *is* the number you actually wanted all along.

---

## Step 6: Where does "quadrature" come in?

"Averaging a formula over every possible setting of a dial from 0 to 1" is,
mathematically, computing an **integral** (the area-under-the-curve type of
average, from calculus). Computing exact integrals directly can be
expensive or impossible in general.

**Numerical quadrature** is just the technical term for: *"a smart set of
sample points, plus weights for each point, that lets you compute an
integral extremely accurately (even exactly, for simple enough functions)
by evaluating the function at only a handful of points and taking a
weighted sum."** ("Quadrature" is an old word for "finding an area" —
literally, finding a square with the same area as a curved shape.)

The paper proves that the specific formula they're integrating (from the
Banzhaf/dial trick) is simple enough — a low-degree polynomial, in math
terms — that **only 8 sample points** are needed to get the exact Shapley
answer, at *any* tree depth, with no rounding blow-up. That's the whole
punchline: 8 fixed evaluation points, always numerically stable, and it's
faster than old TreeSHAP too because 8 fixed points is less work than the
deep recursive multiply/divide bookkeeping from Step 3.

---

## The one-paragraph version, if you forget everything else

*ML models don't explain themselves, so we use SHAP to break one prediction
into a "how much did each feature contribute" score, based on a fair-split
idea borrowed from game theory. For tree-based models we can compute this
exactly and fast by walking the tree once (TreeSHAP) instead of checking
every possible ordering — but that walk's bookkeeping breaks down
numerically on very deep trees. The paper fixes this by computing a
close-cousin quantity (Banzhaf values) that doesn't have the same rounding
problem, and proves that averaging Banzhaf values over a probability dial
from 0 to 1 gives back the exact Shapley answer — and that this averaging
(an integral) can be done almost for free using just 8 well-chosen sample
points (Gauss-Legendre quadrature), instead of struggling with the fragile
original math.*

---

## Suggested next step

Once this clicks, go re-read `MENTAL_MODEL.md`'s "Quadrature-TreeSHAP"
section — the equations there will now be attaching to concepts you already
have, instead of being the first time you're meeting them. If any single
step above (0 through 6) doesn't feel solid, tell me which one and we'll
slow down on just that step — ideally by running real code again, the way
we did for the original SHAP baseline/additivity walkthrough.
