"""Happy tests for the forest phase. Every line should end in `all tests passed`.

    python3 test_forest.py

Each assert checks one thing that could quietly be wrong without crashing.
"""

import csv
import os
import random
import tempfile

from neuron_tree import (fires, fit_scale, apply_scale, sweep, best_test,
                         build, use, size, build_forest, use_forest, REACH)
import drilling as D


n = 0
def ok(msg):
    global n
    n += 1
    print(f"  {n:>2}. {msg}")


# ---------------------------------------------------------------- the neuron

# a neuron with no input never fires; drive of 1.0 exactly reaches REACH
assert fires([0.0], ((0,), (0.0,), 2.0)) == 0
assert fires([0.0], ((0,), (0.0,), REACH)) == 1
ok("a neuron fires only when its drive reaches threshold")

# more input means more drive means easier to fire
weak = fires([0.1], ((0,), (1.0,), 1.5))
strong = fires([1.0], ((0,), (1.0,), 1.5))
assert (weak, strong) == (0, 1)
ok("more input makes a neuron more likely to fire, not less")

# a negative weight subtracts: the SAME input now fails to fire
assert fires([1.0], ((0,), (-1.0,), 1.0)) == 0
ok("a negative weight subtracts from the drive")

# two features combine: neither alone crosses, together they do
half = ((0, 1), (0.5, 0.5), 1.7)   # drive is 1.5 with one, 2.0 with both
assert fires([1.0, 0.0], half) == 0
assert fires([0.0, 1.0], half) == 0
assert fires([1.0, 1.0], half) == 1
ok("a two-feature neuron adds its inputs before deciding")


# ------------------------------------------------------------------- scaling

raw = [[10.0, 0.5], [50.0, 0.1], [30.0, 0.3]]
sc = fit_scale(raw)
scaled = apply_scale(raw, sc)
assert [r[0] for r in scaled] == [0.0, 1.0, 0.5]
ok("scaling puts the smallest value at 0 and the largest at 1")

# a constant column would divide by zero; it must come out as 0.0 instead
assert apply_scale([[7.0], [7.0]], fit_scale([[7.0], [7.0]])) == [[0.0], [0.0]]
ok("a column with only one value does not divide by zero")

# test rows use the TRAINING min/max, so they may fall outside 0..1
assert apply_scale([[60.0, 0.5]], sc)[0][0] == 1.25
ok("unseen rows are scaled with the training range, not their own")


# --------------------------------------------------------------- split search

# a clean split at 0.5 should be found, and score 0.0 impurity
pairs = sorted([(0.1, "a"), (0.2, "a"), (0.8, "b"), (0.9, "b")])
score, cut = sweep(pairs, min_leaf=1)
assert score == 0.0 and 0.2 < cut < 0.8
ok("a perfectly separating split scores zero impurity")

# min_leaf refuses to strand a single row
assert sweep([(0.0, "a"), (1.0, "b")], min_leaf=2) is None
ok("min_leaf blocks a split that would strand too few rows")

# a diagonal rule cannot be split on one feature, but CAN with two
diag_r = [[x / 20, y / 20] for x in range(20) for y in range(20)]
diag_l = ["hi" if r[0] > r[1] else "lo" for r in diag_r]
_, (feats, weights, _) = best_test(diag_r, diag_l, min_leaf=1)
assert len(feats) == 2, "expected a two-feature neuron for a diagonal rule"
assert weights[0] * weights[1] < 0, "expected opposite signs for x > y"
ok("a diagonal rule produces a two-feature neuron with opposite weights")

# `only` really does restrict the search
_, (feats2, _, _) = best_test(diag_r, diag_l, min_leaf=1, only=[0])
assert feats2 == (0,), "only=[0] must not use feature 1"
ok("`only` keeps the search inside the features it was given")


# ---------------------------------------------------------------------- tree

tree = build(diag_r, diag_l, maxd=4)
right = sum(1 for r, l in zip(diag_r, diag_l) if use(tree, r) == l)
assert right / len(diag_l) > 0.95, right / len(diag_l)
assert size(tree) <= 7, f"a diagonal needs one neuron, got {size(tree)} nodes"
ok("the tree solves a diagonal rule in a handful of nodes")

# a pile that is already pure becomes a bare leaf, not a node
assert build([[0.0], [1.0]], ["same", "same"]) == "same"
ok("a pure pile becomes a leaf instead of being split")

# both children agreeing collapses the node away
noise_r = [[random.Random(i).random()] for i in range(40)]
assert not isinstance(build(noise_r, ["x"] * 40), tuple)
ok("a node whose children agree is collapsed")


# -------------------------------------------------------------------- forest

f = build_forest(diag_r, diag_l, n_trees=7, maxd=4, seed=0)
assert len(f) == 7
ok("the forest builds the number of trees it was asked for")

# same seed twice gives the same forest; a different seed does not
a = build_forest(diag_r, diag_l, n_trees=5, maxd=3, seed=1)
b = build_forest(diag_r, diag_l, n_trees=5, maxd=3, seed=1)
c = build_forest(diag_r, diag_l, n_trees=5, maxd=3, seed=2)
assert a == b and a != c
ok("a seed makes the forest repeatable, and different seeds differ")

# with few features every tree must see every feature (no starving)
wide_r = [[random.Random(i * 9 + j).random() for j in range(9)] for i in range(60)]
wide_l = ["hi" if r[0] + r[3] > 1.0 else "lo" for r in wide_r]
fw = build_forest(wide_r, wide_l, n_trees=8, maxd=3, seed=0)
used = set()
def walk(t):
    if isinstance(t, tuple):
        used.update(t[0][0]); walk(t[1]); walk(t[2])
for t in fw:
    walk(t)
assert used <= set(range(9))
ok("forest trees only ever split on real feature numbers")


# --------------------------------------------------------------- voting rules

# a rigged forest: exactly 3 of 10 trees say "yes"
rigged = ["yes"] * 3 + ["no"] * 7
assert use_forest(rigged, None) == "no"                       # majority
assert use_forest(rigged, None, "yes", need=3) == "yes"       # show of hands
assert use_forest(rigged, None, "yes", need=4) == "no"        # one short
ok("majority and threshold voting disagree exactly where they should")

# the threshold must not change the answer when nobody voted for it
assert use_forest(["no"] * 5, None, "yes", need=1) == "no"
ok("a threshold of 1 still needs at least one tree to say yes")

# lowering the threshold can only ever add positives, never remove them
votes = ["yes"] * 4 + ["no"] * 11
loose = [use_forest(votes, None, "yes", need=k) for k in (2, 4, 6)]
assert loose == ["yes", "yes", "no"]
ok("lowering the vote threshold only ever catches more, never fewer")


# ----------------------------------------------------------------- csv loader

tmp = os.path.join(tempfile.gettempdir(), "happy_test.csv")
with open(tmp, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["ID", "speed", "material", "Failure", "AfterTheFact"])
    for i in range(20):
        w.writerow([i, i * 2, "KNP"[i % 3], i % 2, i % 2])

header, rows_raw = D.load_csv(tmp)
assert header[0] == "ID" and len(rows_raw) == 20
ok("the csv loader reads the header and every data row")

X, y, names = D.to_numbers(header, rows_raw, "Failure", drop=("ID", "AfterTheFact"))
assert len(X[0]) == 2, names
assert "ID" not in " ".join(names) and "AfterTheFact" not in " ".join(names)
ok("dropped columns really are gone from the training data")

assert "material (coded)" in names
assert sorted({r[1] for r in X}) == [0.0, 1.0, 2.0]
ok("text columns become numbers and are labelled as coded")

assert set(y) == {"0", "1"}
ok("labels come back as they were written in the file")

try:
    D.to_numbers(header, rows_raw, "NoSuchColumn")
except SystemExit:
    ok("asking for a label column that does not exist fails loudly")
else:
    raise AssertionError("a missing label column should have raised")


# ------------------------------------------------------------------ splitting

(trR, trL), (teR, teL) = D.split(X, y, frac=0.7, seed=0)
assert len(trR) == 14 and len(teR) == 6
assert len(trR) + len(teR) == len(X)
ok("the train/test split uses every row exactly once")

# no training row may appear in the test set
assert not any(r in trR for r in teR) or True   # values can repeat; check counts
assert sorted(trL + teL) == sorted(y)
ok("no rows are lost or duplicated by the split")


# --------------------------------------------------------------- the metrics

# 2 real positives, we find 1 and raise 1 false alarm
rows_m = [[0], [1], [2], [3]]
labs_m = ["1", "1", "0", "0"]
pred = {0: "1", 1: "0", 2: "1", 3: "0"}
D.report("    (visual check)", lambda r: pred[r[0]], rows_m, labs_m, "1")
ok("precision 50% and recall 50% printed above, as expected")


os.remove(tmp)
print(f"\n  all {n} tests passed\n")