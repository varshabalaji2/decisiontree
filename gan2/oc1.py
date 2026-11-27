"""Hill-climb the split's weight vector instead of sweeping eight angles.

    python3 oc1.py
    python3 oc1.py --iters 20 --cands 16 --seeds 3

FINDINGS names this as the cheap door to the one real weakness in the split
search:

    The principled fix for the greedy search itself is non-greedy joint
    optimisation of all splits -- Norouzi et al., NIPS 2015 -- not
    implemented here: it needs gradients, and this project is pure Python.
    OC1-style randomised hill-climbing on the weight vector (Murthy, Kasif
    & Salzberg) is the version that fits without gradients and remains
    untried.

This is that. It does not fix greediness BETWEEN nodes -- each node is still
chosen without regard to its descendants, and only Norouzi-style joint
optimisation addresses that. What it fixes is greediness WITHIN a node.

neuron_tree.best_test searches two features at a time at eight fixed angles:

    w1*x_a + w2*x_b,  (w1, w2) = (cos th, sin th),  th in 8 steps

So a split can only ever use TWO features, and only at one of eight
directions. OC1 searches the whole k-dimensional weight vector by coordinate
hill-climbing, so a split may combine six features at any weights.

HOW A COEFFICIENT IS CHOSEN, which is the part worth copying from OC1. Given
current weights w and threshold t, row i sits exactly on the hyperplane when
coefficient j takes the value

    U_i = (t - w.x_i + w_j * x_ij) / x_ij

so the U_i are precisely the values of w_j at which row i changes side.
Everything between two consecutive U_i gives the same partition, which makes
the midpoints the only candidates worth trying. Sorting all n of them is
O(n^2) per node, so a random sample of them is taken instead -- the standard
practical shortcut, and `--cands` sets how many.

THE BUDGETS ARE COUNTED AND PRINTED, because a richer search that is also
given more evaluations has not been shown to be better, only better fed. One
sweep is one O(n log n) pass over a projection, and both arms report theirs.
The defaults are set to put them within about 10% of each other on the image
task; if a run shows otherwise, read the sweep column before the accuracy.

WHAT THIS IS TESTING. Three times in this project a richer menu of split
options made the greedy search WORSE -- it picked locally good, globally bad
splits. That is the observation OC1 should speak to, and the tasks below are
where it was seen, not C-MAPSS. If a wider search inside each node makes a
richer option set pay instead of cost, that is a result about this project's
own documented pathology.

MEASURED, 3 seeds, forests of 3, 10 moves x 9 candidates:

            search      nodes  sweeps   one tree   forest
    drilling  angles       15     297      95.5%    94.8%
              oc1          21    1010      93.9%    94.3%
    diagonal  angles        3      10      99.8%    99.8%
              oc1           4     124      98.8%    99.0%
    images    angles       21    2000      94.8%    97.8%
              oc1          24    1810      95.2%    97.2%

At the full forest of 15 over 3 seeds the images row is +1.4% to OC1 at
1.05x the sweeps, and the two low-dimensional tasks lose while spending 3x
and 12x the evaluations.

THE GAIN IS THE STRATEGY, NOT THE BUDGET, and it took a control to know it.
The images gain moves monotonically with OC1's budget -- -0.3% at 0.80x,
+0.4% at 0.91x, +1.4% at 1.05x, +2.7% at 1.15x -- which looks exactly like
"more evaluations win". The angles arm has a fixed budget, so --angles
exists to raise it and ask the same question of the other side. Doubling it:

    images, single tree        sweeps   accuracy
      angles, 8 angles           2000      94.8%
      angles, 16 angles          3200      94.2%
      oc1                        2121      96.5%

MORE OPTIONS MADE THE GRID WORSE, and OC1 wins at two thirds of its budget.
The angles search has no rng, so its single-tree numbers are deterministic
and directly comparable. Budget is not the explanation.

AND THAT IS THE PATHOLOGY, ISOLATED. Three times in this project a richer
menu of split options made the greedy search worse. This is the fourth, and
the first with everything else held fixed. It sharpens the finding:

    A richer MENU hurts the greedy search. A richer SEARCH helps it.

Sixteen angles is more independent candidates, each scored alone, with the
best-looking one taken -- so the search overfits its own menu. OC1 is a
directed walk toward a better split. Same extra compute, opposite sign, and
the project had been treating the two as one thing.

BUT THE FOREST ERASES IT. On the images at 15 trees the single-tree gap of
+1.4% becomes 99.8% against 99.7% -- nothing. Bagging already recovers
whatever the better in-node search was buying, which is the same result
shapes.py reached about split geometries. The win is real and it is a
single-tree win.

WHY A FIXED GRID BEATS A RANDOM SEARCH HERE. The diagonal task's optimum is
exactly 45 degrees, which is one of the eight ANGLES -- so the grid lands on
the answer and OC1, sampling candidate values at random, lands near it. A
grid wins whenever the answer is on the grid, and only loses when the space
is too large to grid at all. At six features it is not yet large enough.

And the prediction that it would at least buy FEWER NODES was wrong: it used
more in all three tasks.

On C-MAPSS specifically, expect nothing. Three algorithms already agree to
within 0.1 RMSE there -- this forest 15.9, sklearn's 16.0, gradient
boosting 16.0 -- which says the ceiling is in the columns, not in how the
search moves through them. A better optimiser inside a node adds no
information, and this is the fourth measurement saying so.

WHAT IT DOES NOT TOUCH, and the reason this is not a verdict on non-greedy
trees: OC1 fixes greediness WITHIN a node. The three occasions in this
project where a richer option menu made things WORSE are about choices made
at one node damaging its descendants, which is greediness BETWEEN nodes.
Only joint optimisation of all splits at once -- Norouzi et al. -- addresses
that, and it needs gradients this project does not have. This result closes
the cheap door and leaves the expensive one open.
"""

import math
import random
import sys

from neuron_tree import (REACH, sweep, mixed, majority, fires, use, size,
                         tree_depth, fit_scale, apply_scale, drill, diagonal,
                         images, use_forest)

# neuron_tree fixes this at 8. Made settable here for one reason: the angles
# arm otherwise has a FIXED budget, so there is no way to ask whether it
# would also improve given OC1's extra evaluations. Without that control,
# "OC1 wins at 1.05x the sweeps" cannot be separated from "more sweeps win".
N_ANGLES = 8


class Budget:
    """One sweep is one O(n log n) pass over a projection. Counted so a
    richer search cannot quietly win by being given more of them."""

    def __init__(self):
        self.n = 0

    def sweep(self, pairs, min_leaf):
        self.n += 1
        return sweep(pairs, min_leaf)


def _rank(rows, labels, cols, min_leaf, bud):
    singles = []
    for f in cols:
        got = bud.sweep(sorted((r[f], l) for r, l in zip(rows, labels)),
                        min_leaf)
        if got:
            singles.append((got[0], f, got[1]))
    singles.sort()
    return singles


def angles_test(rows, labels, min_leaf, top_k, only, bud, rng, iters, cands):
    """neuron_tree.best_test, re-implemented only so its sweeps are counted.
    Identical search: every feature alone, then pairs of the top k at eight
    fixed angles."""
    cols = range(len(rows[0])) if only is None else only
    singles = _rank(rows, labels, cols, min_leaf, bud)
    if not singles:
        return None
    best = (singles[0][0], ((singles[0][1],), (1.0,), singles[0][2]))

    keep = [f for _, f, _ in singles[:top_k]]
    angles = [i * math.pi / N_ANGLES for i in range(N_ANGLES)]
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            fa, fb = keep[a], keep[b]
            for th in angles:
                w1, w2 = math.cos(th), math.sin(th)
                got = bud.sweep(sorted((w1 * r[fa] + w2 * r[fb], l)
                                       for r, l in zip(rows, labels)),
                                min_leaf)
                if got and got[0] < best[0]:
                    best = (got[0], ((fa, fb), (w1, w2), got[1]))
    s, (feats, weights, cut) = best
    return s, (feats, weights, (1.0 + cut) * REACH)


def oc1_test(rows, labels, min_leaf, top_k, only, bud, rng, iters, cands):
    """Coordinate hill-climbing on the full weight vector over the top k
    features, started from the best single feature."""
    cols = range(len(rows[0])) if only is None else only
    singles = _rank(rows, labels, cols, min_leaf, bud)
    if not singles:
        return None
    feats = tuple(f for _, f, _ in singles[:top_k])
    if len(feats) < 2:
        s, f, c = singles[0]
        return s, ((f,), (1.0,), (1.0 + c) * REACH)

    proj = [[r[f] for f in feats] for r in rows]

    def score(w):
        got = bud.sweep(sorted((sum(wi * xi for wi, xi in zip(w, x)), l)
                               for x, l in zip(proj, labels)), min_leaf)
        return got            # (impurity, cut) or None

    # start from the best single feature, which is where best_test starts too
    w = [0.0] * len(feats)
    w[0] = 1.0
    got = score(w)
    if got is None:
        s, f, c = singles[0]
        return s, ((f,), (1.0,), (1.0 + c) * REACH)
    best_s, best_cut, best_w = got[0], got[1], list(w)

    for _ in range(iters):
        j = rng.randrange(len(feats))
        # values of w[j] at which some row crosses the current hyperplane
        us = []
        for x in proj:
            if abs(x[j]) < 1e-12:
                continue
            p = sum(wi * xi for wi, xi in zip(w, x))
            us.append((best_cut - p + w[j] * x[j]) / x[j])
        if len(us) < 2:
            continue
        us.sort()
        # midpoints between consecutive crossings are the only distinct
        # partitions; sample rather than try all n of them
        mids = [(us[i] + us[i + 1]) / 2.0 for i in range(len(us) - 1)]
        for v in rng.sample(mids, min(cands, len(mids))):
            trial = list(w)
            trial[j] = v
            got = score(trial)
            if got and got[0] < best_s:
                best_s, best_cut, best_w = got[0], got[1], trial
        w = list(best_w)

    nz = [i for i, x in enumerate(best_w) if abs(x) > 1e-12]
    if not nz:
        s, f, c = singles[0]
        return s, ((f,), (1.0,), (1.0 + c) * REACH)
    return best_s, (tuple(feats[i] for i in nz),
                    tuple(best_w[i] for i in nz),
                    (1.0 + best_cut) * REACH)


def build(rows, labels, search, bud, rng, depth=0, maxd=8, min_leaf=1,
          only=None, top_k=6, iters=12, cands=10):
    if mixed(labels) == 0 or depth >= maxd or len(labels) < 2 * min_leaf:
        return majority(labels)
    got = search(rows, labels, min_leaf, top_k, only, bud, rng, iters, cands)
    if got is None:
        return majority(labels)
    _, test = got

    lr, ll, rr, rl = [], [], [], []
    for r, l in zip(rows, labels):
        if fires(r, test):
            rr.append(r); rl.append(l)
        else:
            lr.append(r); ll.append(l)
    if not ll or not rl:
        return majority(labels)
    lo = build(lr, ll, search, bud, rng, depth + 1, maxd, min_leaf, only,
               top_k, iters, cands)
    hi = build(rr, rl, search, bud, rng, depth + 1, maxd, min_leaf, only,
               top_k, iters, cands)
    if not isinstance(lo, tuple) and lo == hi:
        return lo
    return (test, lo, hi)


def build_forest(rows, labels, search, bud, n_trees=15, maxd=8, min_leaf=1,
                 seed=0, top_k=6, iters=12, cands=10):
    rng = random.Random(seed)
    n, nfeat = len(rows), len(rows[0])
    take = nfeat if nfeat <= 6 else nfeat - max(4, nfeat // 4)
    out = []
    for _ in range(n_trees):
        pick = [rng.randrange(n) for _ in range(n)]
        cols = rng.sample(range(nfeat), take)
        out.append(build([rows[i] for i in pick], [labels[i] for i in pick],
                         search, bud, random.Random(rng.randrange(1 << 30)),
                         maxd=maxd, min_leaf=min_leaf, only=cols,
                         top_k=top_k, iters=iters, cands=cands))
    return out


def run(seeds=1, n_trees=15, iters=12, cands=10, top_k=6, n_angles=8):
    global N_ANGLES
    N_ANGLES = n_angles
    tasks = [("drilling", drill(600, 1), drill(600, 2), 4),
             ("diagonal", diagonal(400, 3), diagonal(400, 4), 4),
             ("images",   images(400, 7), images(400, 8), 8)]

    print(f"\n  {n_angles} fixed angles over feature PAIRS, against OC1 "
          f"hill-climbing the\n  whole weight vector over the top {top_k} "
          f"features. {iters} coordinate moves,\n  {cands} candidate values "
          f"each. Sweeps are counted: a wider search that\n  also gets more "
          f"evaluations has not been shown better, only better fed.")
    if n_angles != 8:
        print(f"  --angles {n_angles} raises the ANGLES arm's budget too, "
              f"which is the control for\n  'did OC1 win on strategy or on "
              f"evaluations'.")
    print()
    print(f"  {'':<10}{'search':<10}{'nodes':>7}{'depth':>7}{'sweeps':>9}"
          f"{'one tree':>10}{'forest':>9}")

    verdict = {}
    for name, tr, te, maxd in tasks:
        trR, trL = tr
        teR, teL = te
        sc = fit_scale(trR)
        trN, teN = apply_scale(trR, sc), apply_scale(teR, sc)
        row = {}
        for tag, search in (("angles", angles_test), ("oc1", oc1_test)):
            accs, nodes, deeps, sweeps, facc = [], [], [], [], []
            for s in range(seeds):
                bud = Budget()
                t = build(trN, trL, search, bud, random.Random(s),
                          maxd=maxd, top_k=top_k, iters=iters, cands=cands)
                accs.append(sum(1 for r, l in zip(teN, teL)
                                if use(t, r) == l) / len(teL))
                nodes.append(size(t))
                deeps.append(tree_depth(t))
                sweeps.append(bud.n)

                fbud = Budget()
                f = build_forest(trN, trL, search, fbud, n_trees=n_trees,
                                 maxd=maxd, seed=s, top_k=top_k,
                                 iters=iters, cands=cands)
                facc.append(sum(1 for r, l in zip(teN, teL)
                                if use_forest(f, r) == l) / len(teL))
            m = lambda xs: sum(xs) / len(xs)
            row[tag] = (m(nodes), m(deeps), m(sweeps), m(accs), m(facc))
            print(f"  {name if tag == 'angles' else '':<10}{tag:<10}"
                  f"{m(nodes):>7.0f}{m(deeps):>7.0f}{m(sweeps):>9.0f}"
                  f"{m(accs):>10.1%}{m(facc):>9.1%}", flush=True)
        verdict[name] = row

    print()
    for name in verdict:
        a, o = verdict[name]["angles"], verdict[name]["oc1"]
        dacc, dnode = o[3] - a[3], o[0] - a[0]
        budget = o[2] / a[2] if a[2] else 1.0
        if budget > 1.15:
            note = f" -- but on {budget:.1f}x the sweeps, so discount it"
        elif budget < 0.87:
            note = f" -- on only {budget:.2f}x the sweeps"
        else:
            note = f" -- at {budget:.2f}x the sweeps, a fair comparison"
        if dacc > 0.01:
            print(f"  {name}: OC1 WINS {dacc:+.1%} on one tree{note}.")
        elif dacc < -0.01:
            print(f"  {name}: OC1 LOSES {dacc:+.1%} on one tree{note}.")
        elif dnode < -0.5:
            print(f"  {name}: a tie on accuracy, but OC1 gets there in "
                  f"{-dnode:.0f} fewer nodes{note}.")
        else:
            print(f"  {name}: a tie, {dacc:+.1%} and {dnode:+.0f} nodes"
                  f"{note}.")

    print(f"\n  OC1 fixes greediness WITHIN a node, not between nodes. Each "
          f"split is\n  still chosen without regard to its descendants, "
          f"which is what Norouzi\n  et al. address and what no amount of "
          f"hill-climbing here can reach.\n")


if __name__ == "__main__":
    f = sys.argv[1:]

    def opt(name, cast, default):
        return cast(f[f.index(name) + 1]) if name in f else default

    run(seeds=opt("--seeds", int, 1), n_trees=opt("--trees", int, 15),
        iters=opt("--iters", int, 12), cands=opt("--cands", int, 10),
        top_k=opt("--topk", int, 6), n_angles=opt("--angles", int, 8))
