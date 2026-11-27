"""A decision tree whose split tests are real spiking neurons."""

import math
import random
from collections import Counter


# ---------------------------------------------------------------- the neuron

def fires(row, test, k=0.3, steps=20):
    """One leaky integrate-and-fire neuron, exactly as in step 5 of the tutorial.

    It receives a weighted sum of several features, integrates it over `steps`
    timesteps with leak `k`, and returns 1 if it ever crosses threshold.
    """
    feats, weights, threshold = test
    drive = 1.0 + sum(w * row[f] for f, w in zip(feats, weights))
    v = 0.0
    for _ in range(steps):
        v = v + k * (drive - v)          # <-- the neuron, one line
        if v >= threshold:
            return 1
    return 0


K, STEPS = 0.3, 20
REACH = 1.0 - (1.0 - K) ** STEPS         # highest v reachable, as a fraction
                                         # of drive, within STEPS timesteps


# -------------------------------------------------- the dynamic stopping rule
#
# FINDINGS lists "depth was chosen by reading the test set" as a known flaw
# of this project. This is the fix. At every node,
#
#     eps = sqrt(ln(1/delta) / (2n))         n = rows at THIS node
#     split only if  gain_runner_up / gain_best + eps < 1
#
# where a gain is the drop in Gini this candidate buys. The ratio is in
# [0, 1], so the Hoeffding bound applies with R = 1. n shrinks as the tree
# deepens, so eps grows, so deep nodes need a clearer winner. The tree stops
# where the evidence runs out and maxd stops being a knob.
#
# FIMT-DD (Ikonomovska, Gama & Dzeroski 2011) / VFDT (Domingos & Hulten,
# KDD 2000). The streaming half of that work is deliberately not taken: it
# solves not storing the data, which is not a problem here, and its
# guarantee is that it MATCHES a batch tree, so it could at best tie.
#
# THE RUNNER-UP IS PER ATTRIBUTE, NOT PER CANDIDATE. Here a candidate is a
# feature tuple AND an angle, and adjacent angles on one pair score almost
# identically. Letting a pair compete with itself pins the ratio near 1 and
# refuses every split -- an earlier version did exactly that and returned
# single-node trees. So scores are reduced to the best per FEATURE TUPLE
# first, and the top two tuples are what the bound compares.
#
# MEASURED, AND IT DOES NOT WORK HERE. Run this file to see it:
#
#                       nodes  depth  one tree   forest
#     drilling  fixed       15      4     95.5%    95.0%
#               dynamic      3      1     90.0%    91.0%
#     diagonal  fixed        3      1     99.8%    99.8%
#               dynamic      3      1     99.8%    99.8%
#     images    fixed       21      5     94.8%    99.8%
#               dynamic      1      0     50.2%    50.2%   <- never split
#
# The reason is arithmetic, not a bug. eps < tau needs
#
#     n > ln(1/delta) / (2 tau^2) = 13.8 / 0.005 = 2763 rows
#
# so above ~2800 rows per node the rule always splits and below it the
# ratio test bites. C-MAPSS has 53,000 training rows, so in cmapss.py the
# test only engages near the leaves and the result was a tie with 23% fewer
# nodes. These tasks have 400-600 rows, so the test applies at the ROOT,
# and on the images the 80 features are correlated enough that the top two
# attributes sit within 13% of each other -- so it refuses to split at all
# and returns a coin flip.
#
# delta and tau are NOT tuned to fix this. Tuning them against these
# numbers would be reading the test set, which is the flaw the rule exists
# to remove. The honest statement is that a Hoeffding bound is built for
# streams of unbounded length and earns its keep in the tens of thousands
# of rows, not the hundreds.
DELTA, TAU, HARD_CAP = 1e-6, 0.05, 30


def clear_winner(n, best, second, delta=DELTA, tau=TAU):
    """Is the best split beating the runner-up by more than n rows could
    explain by chance?"""
    if best <= 0.0:
        return False
    eps = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    if eps < tau:           # already close; more rows would settle nothing
        return True
    return (second / best) + eps < 1.0


# ------------------------------------------------------- putting rows on 0..1

def fit_scale(rows):
    cols = list(zip(*rows))
    return [(min(c), max(c)) for c in cols]


def apply_scale(rows, scale):
    out = []
    for r in rows:
        out.append([0.0 if hi == lo else (x - lo) / (hi - lo)
                    for x, (lo, hi) in zip(r, scale)])
    return out


# ------------------------------------------------------------ the split search

def sweep(pairs, min_leaf):
    """Best threshold on an already-sorted list of (value, label). O(n)."""
    n = len(pairs)
    right, left = Counter(l for _, l in pairs), Counter()
    best = None
    for i in range(n - 1):
        v, l = pairs[i]
        left[l] += 1
        right[l] -= 1
        if right[l] == 0:
            del right[l]
        if pairs[i + 1][0] == v:
            continue
        nl, nr = i + 1, n - i - 1
        if nl < min_leaf or nr < min_leaf:
            continue
        gl = 1.0 - sum((c / nl) ** 2 for c in left.values())
        gr = 1.0 - sum((c / nr) ** 2 for c in right.values())
        s = nl / n * gl + nr / n * gr
        if best is None or s < best[0]:
            best = (s, (v + pairs[i + 1][0]) / 2.0)
    return best


ANGLES = [i * math.pi / 8 for i in range(8)]


def best_test(rows, labels, min_leaf, top_k=6, only=None, want_second=False):
    """Search axis-aligned splits first, then two-feature neuron splits.

    want_second=False returns exactly what it always did, so every earlier
    result reproduces. want_second=True also returns the best and runner-up
    GAINS, reduced per feature tuple, for the Hoeffding test.
    """

    cols = range(len(rows[0])) if only is None else only
    # 1. every feature on its own (this is also the axis-aligned tree's search)
    singles = []
    for f in cols:
        got = sweep(sorted((r[f], l) for r, l in zip(rows, labels)), min_leaf)
        if got:
            singles.append((got[0], f, got[1]))
    if not singles:
        return None
    singles.sort()
    best = (singles[0][0], ((singles[0][1],), (1.0,), singles[0][2]))

    # best score per FEATURE TUPLE, so the bound compares attributes rather
    # than letting one pair's eight angles compete with each other
    by_attr = {}
    if want_second:
        for s, f, _ in singles:
            k = (f,)
            if k not in by_attr or s < by_attr[k]:
                by_attr[k] = s

    # 2. pairs of the most promising features, at eight angles
    keep = [f for _, f, _ in singles[:top_k]]
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            fa, fb = keep[a], keep[b]
            for th in ANGLES:
                w1, w2 = math.cos(th), math.sin(th)
                got = sweep(sorted((w1 * r[fa] + w2 * r[fb], l)
                                   for r, l in zip(rows, labels)), min_leaf)
                if not got:
                    continue
                if want_second:
                    k = (fa, fb)
                    if k not in by_attr or got[0] < by_attr[k]:
                        by_attr[k] = got[0]
                if got[0] < best[0]:
                    best = (got[0], ((fa, fb), (w1, w2), got[1]))

    # 3. convert the cut on `drive` into the neuron's threshold
    s, (feats, weights, cut) = best
    out = (s, (feats, weights, (1.0 + cut) * REACH))
    if not want_second:
        return out
    parent = mixed(labels)
    gains = sorted((parent - v for v in by_attr.values()), reverse=True)
    return out + (gains[0] if gains else 0.0,
                  gains[1] if len(gains) > 1 else 0.0)


# ------------------------------------------------------------------- the tree

def mixed(labels):
    if not labels:
        return 0.0
    n = len(labels)
    return 1.0 - sum((c / n) ** 2 for c in Counter(labels).values())


def majority(labels):
    return Counter(labels).most_common(1)[0][0]


def build(rows, labels, depth=0, maxd=8, min_leaf=1, only=None,
          dynamic=False):
    """dynamic=False is the original tree, unchanged. dynamic=True ignores
    maxd and stops on clear_winner instead, so no depth is tuned."""
    n = len(labels)
    if (mixed(labels) == 0 or depth >= (HARD_CAP if dynamic else maxd)
            or n < 2 * min_leaf):
        return majority(labels)

    got = best_test(rows, labels, min_leaf, only=only, want_second=dynamic)
    if got is None:
        return majority(labels)
    if dynamic:
        _, test, gain, second = got
        if not clear_winner(n, gain, second):
            return majority(labels)
    else:
        _, test = got

    left_r, left_l, right_r, right_l = [], [], [], []
    for r, l in zip(rows, labels):
        if fires(r, test):
            right_r.append(r); right_l.append(l)
        else:
            left_r.append(r); left_l.append(l)
    if not left_l or not right_l:
        return majority(labels)

    lo = build(left_r,  left_l,  depth + 1, maxd, min_leaf, only, dynamic)
    hi = build(right_r, right_l, depth + 1, maxd, min_leaf, only, dynamic)
    if not isinstance(lo, tuple) and lo == hi:      # both sides agree -> prune
        return lo
    return (test, lo, hi)


def build_forest(rows, labels, n_trees=15, maxd=8, min_leaf=1, seed=0,
                 dynamic=False):
    """Many trees, each shown a different slice of the same data.

    Two sources of disagreement, and you need BOTH -- one alone leaves the
    trees too similar for voting to fix anything:

      rows     each tree gets its own bootstrap sample: len(rows) rows drawn
               WITH replacement, so it sees roughly 63% of them, some twice.
      columns  each tree may only split on a random quarter of the features,
               so a strong feature cannot dominate every tree.

    Column bagging needs columns to spare. With only a handful of features,
    hiding some leaves trees that physically cannot express the rule -- so
    below the cutoff every tree sees every feature and only the rows differ.

    A quarter, not the textbook sqrt(nfeat). sqrt assumes every feature is
    about as informative as every other. On the image task 16 of the 80
    features carry nearly all the signal, so a sqrt-sized sample of 9 holds
    fewer than two of them on average and most trees are starved: measured
    97.8% at sqrt against 99.7% at a quarter.
    """
    rng = random.Random(seed)
    n, nfeat = len(rows), len(rows[0])
    take = nfeat if nfeat <= 6 else nfeat-max(4, nfeat // 4)
    forest = []
    for _ in range(n_trees):
        pick = [rng.randrange(n) for _ in range(n)]          # bootstrap rows
        cols = rng.sample(range(nfeat), take)                # random columns
        forest.append(build([rows[i] for i in pick],
                            [labels[i] for i in pick],
                            maxd=maxd, min_leaf=min_leaf, only=cols,
                            dynamic=dynamic))
    return forest


def use_forest(forest, row, positive=None, need=None):
    """Ask every tree, then combine the answers.

    By default the majority wins, which is right when every tree learned
    roughly the same thing and the voting is there to cancel out noise.

    It is WRONG when the thing being predicted is rare and made of several
    separate patterns. Then each tree learns only some of the patterns, and
    a tree answering "normal" mostly means "I was never shown that pattern"
    rather than "I checked and it is absent" -- so the ignorant trees
    outvote the informed ones. Measured on drilling-shaped data, where the
    failure label is an OR of four rare modes and individual tree recall
    ranges from 23% to 99%:

        majority (8 of 15)   recall 74.5%   precision 100.0%
        5 of 15              recall 97.9%   precision  94.5%
        3 of 15              recall 99.3%   precision  87.5%

    Pass `positive` (the rare label) and `need` (how many trees must call
    it) to switch from majority rule to a show of hands. A third of the
    trees is a reasonable starting point.
    """
    answers = [use(t, row) for t in forest]
    if positive is None or need is None:
        return Counter(answers).most_common(1)[0][0]
    if answers.count(positive) >= need:
        return positive
    rest = [a for a in answers if a != positive]
    return Counter(rest).most_common(1)[0][0] if rest else positive


def use(tree, row):
    while isinstance(tree, tuple):
        test, lo, hi = tree
        tree = hi if fires(row, test) else lo
    return tree


def size(tree):
    return 1 if not isinstance(tree, tuple) else 1 + size(tree[1]) + size(tree[2])


def tree_depth(tree):
    return (0 if not isinstance(tree, tuple)
            else 1 + max(tree_depth(tree[1]), tree_depth(tree[2])))


def show(tree, names, depth=0):
    pad = "  " * (depth + 1)
    if not isinstance(tree, tuple):
        print(f"{pad}-> {tree}")
        return
    (feats, weights, thr), lo, hi = tree
    terms = " + ".join(f"{w:+.2f}*{names[f]}" for f, w in zip(feats, weights))
    kind = "neuron" if len(feats) > 1 else "  axis"
    print(f"{pad}{kind}: does 1{terms} fire? (thr {thr:.3f})")
    show(lo, names, depth + 1)
    show(hi, names, depth + 1)


# ------------------------------------------------------------------ test data

def drill(n, seed):
    rng = random.Random(seed)
    R, L = [], []
    for _ in range(n):
        speed, feed = rng.uniform(10, 50), rng.uniform(0.1, 0.6)
        sharp = rng.uniform(0, 100)
        fail = 1 if (sharp < 40 and feed > 0.35) else 0
        if rng.random() < 0.05:
            fail = 1 - fail
        R.append([speed, feed, sharp]); L.append(fail)
    return R, L


def make(kind, rng):
    img = [[0] * 8 for _ in range(8)]
    if kind == "H":
        r = rng.randint(1, 6)
        for c in range(1, 7): img[r][c] = 1
    else:
        c = rng.randint(1, 6)
        for r in range(1, 7): img[r][c] = 1
    for _ in range(4):
        img[rng.randrange(8)][rng.randrange(8)] ^= 1
    rs = [sum(r) for r in img]
    cs = [sum(img[r][c] for r in range(8)) for c in range(8)]
    return [p for row in img for p in row] + rs + cs


def images(n, seed):
    rng = random.Random(seed)
    R, L = [], []
    for _ in range(n):
        kind = "H" if rng.random() < 0.5 else "V"
        R.append(make(kind, rng)); L.append(kind)
    return R, L


def diagonal(n, seed):
    """A rule no axis-aligned split can draw: is x above the line y = x?"""
    rng = random.Random(seed)
    R, L = [], []
    for _ in range(n):
        x, y = rng.uniform(0, 1), rng.uniform(0, 1)
        R.append([x, y]); L.append(1 if x > y else 0)
    return R, L


# ---------------------------------------------------------------------- run it


def evaluate(name, tr, te, maxd, names=None, dynamic=False):
    trR, trL = tr
    teR, teL = te
    scale = fit_scale(trR)
    trN, teN = apply_scale(trR, scale), apply_scale(teR, scale)
    t = build(trN, trL, maxd=maxd, dynamic=dynamic)
    a = sum(1 for r, l in zip(trN, trL) if use(t, r) == l) / len(trL)
    b = sum(1 for r, l in zip(teN, teL) if use(t, r) == l) / len(teL)
    print(f"  {name:<10} nodes {size(t):>4}   train {a:>6.1%}   test {b:>6.1%}")
    return t


if __name__ == "__main__":
    print(f"\n  fixed depth against the dynamic stopping rule, same data,")
    print(f"  same split search, differing only in when the tree stops.\n")
    print(f"  {'':<10}{'':<14}{'nodes':>7}{'depth':>7}{'one tree':>10}"
          f"{'forest':>9}")
    for name, tr, te, maxd in [
        ("drilling", drill(600, 1), drill(600, 2), 4),
        ("diagonal", diagonal(400, 3), diagonal(400, 4), 4),
        ("images",   images(400, 7), images(400, 8), 8),
    ]:
        trR, trL = tr
        teR, teL = te
        sc = fit_scale(trR)
        trN, teN = apply_scale(trR, sc), apply_scale(teR, sc)

        for tag, dyn in ((f"fixed d={maxd}", False), ("dynamic", True)):
            t = build(trN, trL, maxd=maxd, dynamic=dyn)
            a = sum(1 for r, l in zip(teN, teL) if use(t, r) == l) / len(teL)

            f = build_forest(trN, trL, n_trees=15, maxd=maxd, dynamic=dyn)
            b = sum(1 for r, l in zip(teN, teL)
                    if use_forest(f, r) == l) / len(teL)

            flag = "   <- never split" if size(t) == 1 else ""
            print(f"  {name if not dyn else '':<10}{tag:<14}{size(t):>7}"
                  f"{tree_depth(t):>7}{a:>10.1%}{b:>9.1%}{flag}")
    print(f"\n  eps < tau needs n > ln(1/delta)/(2 tau^2) = "
          f"{math.log(1/DELTA) / (2 * TAU ** 2):.0f} rows, so on these"
          f"\n  400-600 row tasks the test applies at the root. It is built "
          f"for tens of\n  thousands of rows; see cmapss.py, where it ties "
          f"on 53,000.\n")
