"""A split test where the leak constant genuinely matters.

    python3 temporal.py

In neuron_tree.py the whole row arrives at once and stays constant for twenty
timesteps. A leaky integrator fed a constant settles at that constant, so the
leak only rescales the threshold and cancels against it -- replacing the
twenty-step loop with one comparison gives a bit-identical tree.

Here the features arrive ONE PER TIMESTEP instead:

    v = 0
    for each feature f in a chosen order:
        v = v + k * (x[f] - v)          <- the neuron, unchanged
        if v >= threshold: fire

Two things change, and both matter:

  the leak means something   v ends up a weighted sum of the features with
                             weights k(1-k)^(T-t) -- exponentially decaying
                             with how long ago the feature arrived. k sets
                             how much the early features still count for, so
                             different k give genuinely different splits
                             rather than rescalings of one split.

  the test is not linear     firing on the running PEAK rather than the final
                             value means the test is max over prefixes, which
                             no single weighted sum can express.

The order is part of the model: the same features in a different order are a
different test. Here the search tries best-first and worst-first, which is
crude but enough to show the mechanism.
"""

import random
from collections import Counter

from neuron_tree import fit_scale, apply_scale, sweep, mixed, majority, images
import neuron_tree as N


KS = (0.15, 0.3, 0.5, 0.75, 0.95, 1.0)


def peak(row, feats, k):
    """Highest the membrane ever reaches while the features stream past."""
    v, top = 0.0, 0.0
    for f in feats:
        v = v + k * (row[f] - v)
        if v > top:
            top = v
    return top


def fires_t(row, test):
    """The neuron itself: integrate, and fire the moment it crosses."""
    feats, k, threshold = test
    v = 0.0
    for f in feats:
        v = v + k * (row[f] - v)
        if v >= threshold:
            return 1
    return 0


def best_test_t(rows, labels, min_leaf=1, top_k=6, only=None):
    """Try each leak and each feature order; sweep the threshold on the peak."""
    cols = range(len(rows[0])) if only is None else list(only)

    # rank the features on their own, to get a sensible order to stream them in
    singles = []
    for f in cols:
        got = sweep(sorted((r[f], l) for r, l in zip(rows, labels)), min_leaf)
        if got:
            singles.append((got[0], f, got[1]))
    if not singles:
        return None
    singles.sort()

    # a plain one-feature split is always a candidate
    best = (singles[0][0], ((singles[0][1],), 1.0, singles[0][2]))

    order = [f for _, f, _ in singles[:top_k]]
    for feats in (tuple(order), tuple(reversed(order))):
        for k in KS:
            got = sweep(sorted((peak(r, feats, k), l)
                               for r, l in zip(rows, labels)), min_leaf)
            if got and got[0] < best[0]:
                best = (got[0], (feats, k, got[1]))
    return best


def build_t(rows, labels, depth=0, maxd=8, min_leaf=1, only=None):
    if mixed(labels) == 0 or depth >= maxd or len(labels) < 2 * min_leaf:
        return majority(labels)
    got = best_test_t(rows, labels, min_leaf, only=only)
    if got is None:
        return majority(labels)
    _, test = got

    lr, ll, rr, rl = [], [], [], []
    for r, l in zip(rows, labels):
        if fires_t(r, test):
            rr.append(r); rl.append(l)
        else:
            lr.append(r); ll.append(l)
    if not ll or not rl:
        return majority(labels)

    lo = build_t(lr, ll, depth + 1, maxd, min_leaf, only)
    hi = build_t(rr, rl, depth + 1, maxd, min_leaf, only)
    if not isinstance(lo, tuple) and lo == hi:
        return lo
    return (test, lo, hi)


def use_t(tree, row):
    while isinstance(tree, tuple):
        test, lo, hi = tree
        tree = hi if fires_t(row, test) else lo
    return tree


def leaks_used(tree, out=None):
    """Which leak constants the tree actually chose. If k did nothing, one
    value would do as well as any other and this would be uniform noise."""
    out = Counter() if out is None else out
    if isinstance(tree, tuple):
        (feats, k, _), lo, hi = tree
        if len(feats) > 1:
            out[k] += 1
        leaks_used(lo, out)
        leaks_used(hi, out)
    return out


if __name__ == "__main__":
    trR, trL = images(400, 7)
    teR, teL = images(400, 8)
    sc = fit_scale(trR)
    trN, teN = apply_scale(trR, sc), apply_scale(teR, sc)

    print("\n  does the leak constant change the answer now?\n")
    print(f"  {'k fixed at':>12}{'nodes':>8}{'test':>9}")
    for k in KS:
        saved = KS
        globals()['KS'] = (k,)
        t = build_t(trN, trL, maxd=6)
        acc = sum(1 for r, l in zip(teN, teL) if use_t(t, r) == l) / len(teL)
        print(f"  {k:>12}{N.size(t):>8}{acc:>9.1%}")
        globals()['KS'] = saved

    t = build_t(trN, trL, maxd=6)
    acc = sum(1 for r, l in zip(teN, teL) if use_t(t, r) == l) / len(teL)
    print(f"\n  free to choose per node: {N.size(t)} nodes, {acc:.1%}")
    print(f"  leaks it picked: {dict(leaks_used(t))}")

    flat = N.build(trN, trL, maxd=6)
    facc = sum(1 for r, l in zip(teN, teL) if N.use(flat, r) == l) / len(teL)
    print(f"\n  the weighted-pair version for comparison: "
          f"{N.size(flat)} nodes, {facc:.1%}\n")
