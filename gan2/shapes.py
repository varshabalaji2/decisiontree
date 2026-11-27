"""Different trees, different split SHAPES, same data.

    python3 shapes.py                                    the H/V images
    python3 shapes.py --seeds 5 --trees 9 --maxd 6
    python3 shapes.py --csv "XAI Drilling.csv" --label Failure
    python3 shapes.py --csv data.csv --label Class --drop id,ID
    python3 shapes.py --csv steel.csv --onehot Pastry,Z_Scratch,K_Scatch,\
Stains,Dirtiness,Bumps,Other_Faults --per-class

--onehot collapses a set of 0/1 indicator columns into one label and keeps all
of them out of the features. --per-class then runs each class one-versus-rest
and reports which geometry catches it best.

That pairing is the useful one, because "is there a dataset with mixed
boundary geometry" cannot be looked up -- no repository tags a file that way.
It can be MEASURED: if different classes are best served by different shapes,
that dataset has mixed boundaries. Drilling has one failure physics, so one
shape swept every class; a file whose classes have unrelated causes need not.

The images are synthetic and their boundaries were designed rather than found,
so a geometry winning there may say no more than "stripes have round or
cornered boundaries". --csv runs the same four geometries on a real file,
which is what turns it into a finding about decision trees.

A forest works because its trees make different mistakes and averaging cancels
them. This one's trees all ask the same KIND of question -- is a weighted sum
above a threshold -- and differ only by which rows and columns they happened to
draw. Same way of thinking, so similar mistakes, so less for the averaging to
cancel.

This gives different trees different cutting tools:

    sum    w.x >= cut                           a straight line at any angle
    max    max(x[f] for f in feats) >= cut      an axis-aligned corner
    min    min(x[f] for f in feats) >= cut      the opposite corner
    rbf    ||x - c||^2 <= r^2                   a ball around a point

The sum shape fits its weights by least squares against +/-1 labels -- the
Adaline rule, in closed form -- rather than sweeping eight fixed angles over a
feature pair. Same geometry, an actual weight search instead of a coarse one.

RBF is the only neuron on the usual list that is not absorbed by a weight
search. Sigmoid, tanh and ReLU units all compute f(w.x) for monotone f, and
thresholding f(w.x) is identical to thresholding w.x at f-inverse of the
cut -- so they produce bit-identical trees to the sum shape, the same collapse
as the twenty-step integration loop. McCulloch-Pitts freezes the weights and
is strictly weaker. A ball is a genuinely different shape.

Everything else is unchanged: every tree sees all the data, bootstrapped as
usual, columns subsampled as usual, votes counted as usual. The only
difference is the shape each tree cuts with.

This is anchored rather than hopeful. temporal.py measured the max rule at
99.0% against the weighted pair's 94.8% on these images, with fewer nodes --
stripe boundaries are corner-shaped and hyperplanes fit corners badly. So the
mixture contains a tool already known to work here.

THE RISK, and the reason there are three arms. Mixing a strong learner with
weaker ones can land below the strong one alone. If max is simply better on
this data, a mixed forest carrying six weaker trees may lose to a pure-max
forest. "all max" is in the table to catch exactly that, and either outcome is
a result: mixed winning means split-shape diversity pays, pure max winning
means diversity does not pay when one geometry is clearly better.

Note on the sum rule: neuron_tree spends twenty integration steps computing
what one comparison computes, since a leaky integrator fed a constant settles
at that constant. The sum shape here is written in the collapsed form, which
is the same test without the dead loop.
"""

import math
import random
import statistics
import sys

from neuron_tree import (fit_scale, apply_scale, sweep, mixed, majority,
                         images, size)

ANGLES = [i * math.pi / 8 for i in range(8)]
KINDS = ("sum", "max", "min", "rbf")
RIDGE = 1e-6


def rank_singles(rows, labels, cols, min_leaf):
    """Score every column on its own, best first. Used to decide which
    features are worth combining -- the same idea neuron_tree uses."""
    out = []
    for f in cols:
        got = sweep(sorted((r[f], l) for r, l in zip(rows, labels)), min_leaf)
        if got:
            out.append((got[0], f, got[1]))
    out.sort()
    return out


def _solve(a, b):
    """Gaussian elimination with partial pivoting. Small systems only --
    top_k is 6, so this is a 6x6."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    w = [0.0] * n
    for r in range(n - 1, -1, -1):
        w[r] = (m[r][n] - sum(m[r][c] * w[c] for c in range(r + 1, n))) \
            / m[r][r]
    return w


def _adaline(rows, labels, feats, ridge=RIDGE):
    """Least-squares fit to +/-1 targets: the Adaline rule in closed form.

    Beats sweeping fixed angles because it fits all the chosen features at
    once and picks the actual best-fitting direction rather than one of eight.
    Multi-class is handled as majority-versus-rest, which is what a split has
    to decide anyway.
    """
    big = majority(labels)
    k = len(feats)
    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for r, l in zip(rows, labels):
        y = 1.0 if l == big else -1.0
        v = [r[f] for f in feats]
        for i in range(k):
            xty[i] += v[i] * y
            for j in range(k):
                xtx[i][j] += v[i] * v[j]
    for i in range(k):
        xtx[i][i] += ridge
    return _solve(xtx, xty)


def best_sum(rows, labels, singles, min_leaf, top_k):
    """A straight cut, with the direction fitted rather than guessed."""
    best = None
    order = [f for _, f, _ in singles[:top_k]]

    # the fitted direction over all the promising features
    if len(order) >= 2:
        w = _adaline(rows, labels, order)
        if w and any(abs(x) > 1e-12 for x in w):
            got = sweep(sorted((sum(wi * r[f] for wi, f in zip(w, order)), l)
                               for r, l in zip(rows, labels)), min_leaf)
            if got:
                best = (got[0], ("sum", tuple(order), tuple(w), got[1]))

    # the angle sweep over pairs, kept because the fit can be dragged off by
    # outliers while a swept angle cannot
    for i, f in enumerate(order):
        for g in order[i + 1:]:
            for a in ANGLES:
                c, sn = math.cos(a), math.sin(a)
                got = sweep(sorted((c * r[f] + sn * r[g], l)
                                   for r, l in zip(rows, labels)), min_leaf)
                if got and (best is None or got[0] < best[0]):
                    best = (got[0], ("sum", (f, g), (c, sn), got[1]))
    return best


def _one_ball(rows, labels, feats, min_leaf):
    big = majority(labels)
    picked = [r for r, l in zip(rows, labels) if l == big]
    if not picked or len(feats) < 2:
        return None
    c = [sum(r[f] for r in picked) / len(picked) for f in feats]
    d = [sum((r[f] - ci) ** 2 for f, ci in zip(feats, c)) for r in rows]
    got = sweep(sorted(zip(d, labels)), min_leaf)
    if not got:
        return None
    return (got[0], ("rbf", tuple(feats), tuple(c), got[1]))


def best_rbf(rows, labels, singles, min_leaf, top_k, cols):
    """A ball: centre on the majority class's centroid, sweep the radius.

    The only unit on the standard neuron list whose shape a weight search
    cannot reproduce, because a sphere is not a half-space.

    TWO FEATURE SETS ARE TRIED, and the second exists because the first
    version failed on the case a ball is for. Ranking features by how well
    each separates ON ITS OWN is wrong precisely when the boundary is round:
    on a ring in x1 and x2, neither coordinate separates anything alone, so
    both rank below the noise columns and the ball gets centred in the wrong
    space. It scored 87.8% against a hyperplane's 95.3% on exactly that task.

    So the marginal top-k is tried, and so is every column available at this
    node, and the better sweep wins. The top-k set still matters in high
    dimensions, where a distance over all columns drowns in noise.
    """
    order = [f for _, f, _ in singles[:top_k]]
    best = None
    # prefixes, because a ball over the WHOLE column set is diluted by every
    # irrelevant dimension it includes: on a ring in two coordinates plus two
    # junk columns, the junk contributes as much squared distance as the
    # signal and the split degrades. Widening from two features upward lets
    # the search stop where the dilution starts.
    for i in range(2, len(order) + 1):
        got = _one_ball(rows, labels, order[:i], min_leaf)
        if got and (best is None or got[0] < best[0]):
            best = got
    whole = _one_ball(rows, labels, list(cols), min_leaf)
    if whole and (best is None or whole[0] < best[0]):
        best = whole
    return best


def _best_extreme(rows, labels, singles, min_leaf, top_k, kind):
    """A corner cut: the largest (or smallest) of a growing set of features.

    Prefixes of the ranked order are tried, 2 features up to top_k, and the
    running extreme is carried forward so each extra feature costs one pass
    rather than a recomputation.
    """
    order = [f for _, f, _ in singles[:top_k]]
    if len(order) < 2:
        return None
    pick = max if kind == "max" else min
    run = [r[order[0]] for r in rows]
    best = None
    for i in range(1, len(order)):
        f = order[i]
        run = [pick(v, r[f]) for v, r in zip(run, rows)]
        got = sweep(sorted(zip(run, labels)), min_leaf)
        if got and (best is None or got[0] < best[0]):
            best = (got[0], (kind, tuple(order[:i + 1]), None, got[1]))
    return best


def best_test(rows, labels, kind, min_leaf=1, top_k=6, only=None):
    cols = range(len(rows[0])) if only is None else list(only)
    singles = rank_singles(rows, labels, cols, min_leaf)
    if not singles:
        return None

    # a plain one-feature cut is always available, whatever the shape
    best = (singles[0][0], ("one", (singles[0][1],), None, singles[0][2]))

    if kind == "sum":
        got = best_sum(rows, labels, singles, min_leaf, top_k)
    elif kind == "rbf":
        got = best_rbf(rows, labels, singles, min_leaf, top_k, cols)
    else:
        got = _best_extreme(rows, labels, singles, min_leaf, top_k, kind)
    if got and got[0] < best[0]:
        best = got
    return best


def fires(row, test):
    kind, feats, w, cut = test
    if kind == "one":
        v = row[feats[0]]
    elif kind == "sum":
        v = sum(wi * row[f] for wi, f in zip(w, feats))
    elif kind == "rbf":
        v = sum((row[f] - ci) ** 2 for f, ci in zip(feats, w))
    elif kind == "max":
        v = max(row[f] for f in feats)
    else:
        v = min(row[f] for f in feats)
    return v > cut


def build(rows, labels, kind, depth=0, maxd=6, min_leaf=1, only=None):
    if mixed(labels) == 0 or depth >= maxd or len(labels) < 2 * min_leaf:
        return majority(labels)
    got = best_test(rows, labels, kind, min_leaf, only=only)
    if got is None:
        return majority(labels)
    _, test = got

    a, al, b, bl = [], [], [], []
    for r, l in zip(rows, labels):
        if fires(r, test):
            b.append(r); bl.append(l)
        else:
            a.append(r); al.append(l)
    if not al or not bl:
        return majority(labels)

    lo = build(a, al, kind, depth + 1, maxd, min_leaf, only)
    hi = build(b, bl, kind, depth + 1, maxd, min_leaf, only)
    if not isinstance(lo, tuple) and lo == hi:
        return lo
    return (test, lo, hi)


def use(tree, row):
    while isinstance(tree, tuple):
        test, lo, hi = tree
        tree = hi if fires(row, test) else lo
    return tree


def nodes(t):
    return 1 + nodes(t[1]) + nodes(t[2]) if isinstance(t, tuple) else 0


def build_forest(rows, labels, kinds, n_trees=9, maxd=6, min_leaf=1, seed=0):
    """Bootstrap rows and subsample columns exactly as the project's forest
    does; the only new thing is that tree i uses kinds[i % len(kinds)]."""
    rng = random.Random(seed)
    n, nfeat = len(rows), len(rows[0])
    take = nfeat if nfeat <= 6 else nfeat - max(1, nfeat // 4)
    out = []
    for t in range(n_trees):
        pick = [rng.randrange(n) for _ in range(n)]
        cols = rng.sample(range(nfeat), take)
        out.append(build([rows[i] for i in pick], [labels[i] for i in pick],
                         kinds[t % len(kinds)], maxd=maxd, min_leaf=min_leaf,
                         only=cols))
    return out


def use_forest(forest, row):
    votes = {}
    for t in forest:
        v = use(t, row)
        votes[v] = votes.get(v, 0) + 1
    return max(votes, key=votes.get)


def load_csv_task(path, label_col, drop=(), onehot=None):
    """A real file, through the project's own CSV pipeline.

    With `onehot`, the label is which of those indicator columns is set, and
    every one of them is kept out of the features -- leaving any of them in
    would hand the model the answer, the same trap as the drilling file's
    four failure-mode columns.
    """
    try:
        from drilling import load_csv, to_numbers
    except ImportError:
        raise SystemExit("  --csv needs drilling.py beside this file")
    header, raw = load_csv(path)

    if onehot:
        missing = [c for c in onehot if c not in header]
        if missing:
            raise SystemExit(f"  columns not in this file: {missing}\n"
                             f"  columns are:\n    {', '.join(header)}")
        idx = [header.index(c) for c in onehot]
        labels = []
        for r in raw:
            hit = [c for c, i in zip(onehot, idx)
                   if r[i].strip() not in ("", "0", "0.0")]
            labels.append(hit[0] if hit else "none")
        skip = [c for c in drop if c in header] + list(onehot[1:])
        rows, _, names = to_numbers(header, raw, onehot[0], drop=skip)
        return rows, labels, names

    if label_col not in header:
        raise SystemExit(f"  no column {label_col!r}. Columns are:\n"
                         f"    {', '.join(header)}")
    skip = [c for c in drop if c in header]
    rows, labels, names = to_numbers(header, raw, label_col, drop=skip)
    return rows, labels, names


def one_arm(rows, labels, kinds, seeds, maxd, n_trees, positive):
    """Score one geometry at BOTH levels, returning (tree, forest) recall.

    Both, because a forest hides geometry differences: on the H/V images a
    single tree separated the shapes by 6.1 points (93.3% against 99.4%) while
    the forest separated them by 0.7, nine trees covering each other's
    shape-specific blind spots. A per-class verdict read only off forests
    therefore cannot distinguish "these boundaries share a shape" from
    "forests do not care about shape".
    """
    t_rec, f_rec = [], []
    for s in range(seeds):
        trR, trL, teR, teL = shuffled_split(rows, labels, seed=s)
        sc = fit_scale(trR)
        a, b = apply_scale(trR, sc), apply_scale(teR, sc)

        t = build(a, trL, kinds[0], maxd=maxd)
        t_rec.append(recall_f1([use(t, r) for r in b], teL, positive)[0])

        f = build_forest(a, trL, kinds, n_trees=n_trees, maxd=maxd, seed=s)
        f_rec.append(recall_f1([use_forest(f, r) for r in b],
                               teL, positive)[0])

    def ms(xs):
        return statistics.mean(xs), (max(xs) - min(xs)) / 2, list(xs)
    return ms(t_rec), ms(f_rec)


REFERENCE = "sum"


def _verdict(level, wins, margins):
    """Which classes are better served by something other than the default?

    The first version compared each winner against the RUNNER-UP, which is the
    wrong question and hid a real result: on steel plates Z_Scratch went to
    min at 93.0% against sum's 77.8%, fifteen points clear of a 4.7% spread,
    and was reported as "not mixed geometry" because min beat rbf by only 2.3.

    What matters for specialising is whether a class does better under some
    shape than under the one the forest already uses, so the margin is taken
    against REFERENCE and an exception counts only if it clears the spread.
    """
    tally = ", ".join(f"{k} x{v}" for k, v in sorted(wins.items(),
                                                     key=lambda kv: -kv[1]))
    print(f"\n  {level}: wins by shape -- {tally}")

    # PAIRED, because every geometry saw the identical split on each seed.
    # Comparing a mean gap against a RANGE is conservative in the wrong
    # direction and can hide a real effect -- and a range widens with more
    # seeds rather than narrowing, since more draws reach further into the
    # tail. The question is how often the alternative won, and by how much.
    solid, close = [], []
    for c, k, lead, noise, pairs in margins:
        if k == REFERENCE or not pairs:
            continue
        wins_n = sum(1 for d in pairs if d > 0)
        n = len(pairs)
        avg = statistics.mean(pairs)
        # a majority that would be unlikely from coin flips, and a gap worth
        # having: 8 of 10 or better, averaging at least 2 points
        if wins_n * 10 >= n * 8 and avg > 0.02:
            solid.append((c, k, avg, wins_n, n))
        else:
            close.append((c, k, avg, wins_n, n))

    if not wins:
        print(f"  nothing scored.")
    elif solid:
        for c, k, avg, w, n in sorted(solid, key=lambda t: -t[2]):
            print(f"  {c}: {k} beat {REFERENCE} on {w} of {n} seeds, "
                  f"averaging {avg:+.1%}")
        print(f"  MIXED GEOMETRY -- {len(solid)} class(es) do better under a "
              f"shape other than\n  {REFERENCE!r}, consistently across "
              f"seeds rather than on average alone.")
    else:
        if close:
            c, k, avg, w, n = max(close, key=lambda t: t[2])
            print(f"  closest: {c} to {k}, winning {w} of {n} seeds, "
                  f"averaging {avg:+.1%}")
        print(f"  NOTHING HOLDS UP ACROSS SEEDS. No class beats "
              f"{REFERENCE!r} both consistently\n  and by a worthwhile "
              f"margin, so these geometries are not distinguishable here.")
    return bool(solid)


def run_per_class(rows, labels, names, seeds=3, maxd=6, n_trees=9):
    """One class at a time, versus the rest. Which shape catches which fault?"""
    classes = sorted(set(labels), key=lambda c: -labels.count(c))
    print(f"\n  {len(rows)} rows, {len(names)} columns, "
          f"{len(classes)} classes, {seeds} seeds")
    print(f"  each class one-versus-rest; the numbers are RECALL for that "
          f"class")

    scored = {}
    for c in classes:
        n = labels.count(c)
        if n < 20 or n == len(labels):
            print(f"  skipping {c!r}: {n} rows, too few to score")
            continue
        binary = [c if l == c else "rest" for l in labels]
        scored[c] = (n, {k: one_arm(rows, binary, (k,), seeds, maxd,
                                    n_trees, c) for k in KINDS})

    found = {}
    for lvl, idx in (("single trees", 0), ("forests", 1)):
        print(f"\n  {lvl}")
        print(f"  {'class':<16}{'n':>6}" + "".join(f"{k:>8}" for k in KINDS)
              + f"{'spread':>9}{'best':>8}")
        wins, margins = {}, []
        for c, (n, arms) in scored.items():
            got = {k: arms[k][idx][0] for k in KINDS}
            sp = {k: arms[k][idx][1] for k in KINDS}
            best = max(got, key=got.get)
            wins[best] = wins.get(best, 0) + 1
            # measured against the DEFAULT shape, not the runner-up: the
            # question is whether this class wants something other than what
            # the forest already uses. The per-seed differences are carried
            # along so the verdict can pair them.
            lead = got[best] - got[REFERENCE]
            pairs = [a - b for a, b in zip(arms[best][idx][2],
                                           arms[REFERENCE][idx][2])]
            margins.append((c, best, lead, max(sp.values()), pairs))
            # one spread column, the widest across the four shapes: a lead
            # smaller than this is not a lead
            print(f"  {c:<16}{n:>6}"
                  + "".join(f"{got[k]:>8.1%}" for k in KINDS)
                  + f"{max(sp.values()):>9.1%}{best:>8}")
        found[lvl] = _verdict(lvl, wins, margins)

    print()
    if found["single trees"] and not found["forests"]:
        print(f"  THE FOREST IS HIDING IT. Single trees want different shapes "
              f"for different\n  classes; nine of them voting do not. That is "
              f"the case for specialising,\n  and it is invisible at forest "
              f"level -- exactly as on the images, where a\n  tree separated "
              f"the shapes by 6.1 points and a forest by 0.7.")
    elif found["forests"] and not found["single trees"]:
        print(f"  Mixed only at forest level, which is odd: voting should "
              f"blur shape\n  differences, not create them. Treat it as noise "
              f"unless more seeds hold it.")
    elif found["single trees"]:
        print(f"  MIXED AT BOTH LEVELS, which is the strongest form of the "
              f"result.")
    else:
        print(f"  NO MIXED GEOMETRY AT EITHER LEVEL. One shape serves every "
              f"class, so handing\n  different classes to different shapes "
              f"has no measured job on this data --\n  and the forest is not "
              f"what is hiding it.")
    print(f"\n  Recall, not accuracy: a rare class can be missed entirely "
          f"while accuracy\n  stays high.\n")


def rare_class(labels):
    """The minority class, which on failure data is the one that matters."""
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    if len(counts) != 2:
        return None
    rare = min(counts, key=counts.get)
    return rare if counts[rare] / len(labels) < 0.4 else None


def recall_f1(pred, truth, positive):
    """Recall and F1 for the rare class.

    Accuracy hides everything here: with 5% failures, a model that catches
    none of them still scores 95%. F1 is reported alongside because recall
    alone is gamed by predicting positive for everything, which scores 1.0.
    """
    tp = sum(1 for p, t in zip(pred, truth) if p == positive and t == positive)
    fp = sum(1 for p, t in zip(pred, truth) if p == positive and t != positive)
    fn = sum(1 for p, t in zip(pred, truth) if p != positive and t == positive)
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return rec, f1


def shuffled_split(rows, labels, frac=0.7, seed=0):
    """Split here rather than in drilling, so the seed is this file's to vary.

    NOTE: this shuffles rows. That is right only if the rows are independent
    observations. If they are successive operations on one wearing tool, a
    shuffle puts near-identical neighbours either side of the split and every
    score below is inflated -- the same mistake as splitting C-MAPSS by row
    instead of by engine.
    """
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    cut = int(len(order) * frac)
    tr, te = order[:cut], order[cut:]
    return ([rows[i] for i in tr], [labels[i] for i in tr],
            [rows[i] for i in te], [labels[i] for i in te])


def run(n=400, seeds=5, maxd=6, n_trees=9, top_k=6, csv=None, label=None,
        drop=(), onehot=None, per_class=False):
    data, rare = None, None
    if csv:
        rows, labels, names = load_csv_task(csv, label, drop, onehot)
        if per_class:
            return run_per_class(rows, labels, names, seeds, maxd, n_trees)
        data = (rows, labels)
        big = max(set(labels), key=labels.count)
        rare = rare_class(labels)
        print(f"\n  {csv}: {len(rows)} rows, {len(names)} columns, "
              f"{len(set(labels))} classes")
        print(f"  always guessing {big!r}: {labels.count(big)/len(labels):.1%}")
        if rare is not None:
            share = labels.count(rare) / len(labels)
            floor = 2 * share / (1 + share)
            print(f"  rare class {rare!r} is {share:.1%} of rows -- recall and "
                  f"F1 reported for it")
            print(f"  calling everything {rare!r}: recall 100%, F1 "
                  f"{floor:.1%} (the floor)")
        print(f"  depth {maxd}, {n_trees} trees, {seeds} seeds, "
              f"shuffled 70/30 splits\n")
    else:
        print(f"\n  H/V images, {n} train / {n} test, depth {maxd}, "
              f"{n_trees} trees, {seeds} seeds\n")
    wide = rare is not None
    head = f"  {'':<26}{'tree acc':>15}{'forest acc':>15}"
    if wide:
        head += f"{'recall':>15}{'F1':>15}"
    print(head)

    results, trees, spreads = {}, {}, []
    for tag, kinds in (("all weighted-sum (yours)", ("sum",)),
                       ("all max", ("max",)),
                       ("all min", ("min",)),
                       ("all rbf (a ball)", ("rbf",)),
                       ("mixed: all four", KINDS)):
        t_acc, f_acc, t_sz, f_rec, f_f1 = [], [], [], [], []
        for s in range(seeds):
            if data is None:
                trR, trL = images(n, 100 + s)
                teR, teL = images(n, 200 + s)
            else:
                trR, trL, teR, teL = shuffled_split(data[0], data[1], seed=s)
            sc = fit_scale(trR)
            a, b = apply_scale(trR, sc), apply_scale(teR, sc)

            t = build(a, trL, kinds[0], maxd=maxd)
            t_acc.append(sum(1 for r, l in zip(b, teL)
                             if use(t, r) == l) / len(teL))
            t_sz.append(nodes(t))

            f = build_forest(a, trL, kinds, n_trees=n_trees, maxd=maxd, seed=s)
            fp = [use_forest(f, r) for r in b]
            f_acc.append(sum(1 for p, l in zip(fp, teL)
                             if p == l) / len(teL))
            if rare is not None:
                r_, f1_ = recall_f1(fp, teL, rare)
                f_rec.append(r_); f_f1.append(f1_)

        results[tag] = statistics.mean(f_acc)
        trees[tag] = (statistics.mean(t_acc), (max(t_acc) - min(t_acc)) / 2)
        spreads.append((max(f_acc) - min(f_acc)) / 2)

        def show(xs):
            return (f"{statistics.mean(xs):.1%} "
                    f"+/-{(max(xs) - min(xs)) / 2:.1%}")

        note = "" if len(kinds) == 1 else "  (tree: first kind only)"
        line = f"  {tag:<26}{show(t_acc):>15}{show(f_acc):>15}"
        if wide:
            line += f"{show(f_rec):>15}{show(f_f1):>15}"
        print(line + note)

    print()
    mixed_f = results["mixed: all four"]
    best_pure = max(results[k] for k in results if k.startswith("all"))
    base = results["all weighted-sum (yours)"]
    # the noise floor is the data's own seed spread, not a number picked by
    # hand -- a gap smaller than that is not a gap
    tol = max(max(spreads), 0.002)
    print(f"  mixed over the current forest        {mixed_f - base:+.1%}")
    print(f"  mixed over the best single shape     {mixed_f - best_pure:+.1%}")
    print(f"  seed spread, the noise floor         {tol:.1%}")

    # A forest that is already at 99% has no room to show a difference,
    # whichever way the comparison would have gone. Say so rather than
    # reading a verdict off three tenths of a point.
    if min(results.values()) > 0.98:
        print(f"\n  SATURATED AT FOREST LEVEL: every arm is above 98%, so "
              f"voting cannot be\n  compared here. Re-run with --trees 3 or "
              f"--maxd 3 to open up headroom.")
        # the forest saturating says nothing about the single trees, and the
        # first version of this check ignored them and so missed the result
        pure = {k: v for k, v in trees.items() if k.startswith("all")}
        best_k = max(pure, key=lambda k: pure[k][0])
        base_t = trees["all weighted-sum (yours)"][0]
        gap = pure[best_k][0] - base_t
        wide = max(v[1] for v in pure.values())
        print(f"\n  But the SINGLE TREE column is not saturated, and that is "
              f"where the\n  shapes separate:")
        for k, (m, sp) in sorted(pure.items(), key=lambda kv: -kv[1][0]):
            print(f"    {k:<28}{m:>7.1%}  +/-{sp:.1%}")
        if gap > wide:
            print(f"\n  {best_k!r} beats the current split by {gap:+.1%}, "
                  f"against a spread of\n  {wide:.1%} -- a real difference in "
                  f"split geometry, measured on single trees.")
        else:
            print(f"\n  The best shape leads by {gap:+.1%} against a spread "
                  f"of {wide:.1%}, so the\n  single-tree comparison does not "
                  f"separate them either.")
    elif mixed_f > best_pure + tol:
        print(f"\n  DIVERSITY PAYS: mixing shapes beats every single shape by "
              f"more than the\n  seed spread, so the trees really are making "
              f"different mistakes and the\n  averaging is cancelling them.")
    elif mixed_f > base + tol:
        print(f"\n  MIXED BEATS THE CURRENT FOREST but not the best single "
              f"shape. The gain is\n  the better shape, not the variety -- "
              f"switch shapes rather than mix them.")
    else:
        print(f"\n  DIVERSITY DOES NOT PAY here: every gap is inside the seed "
              f"spread. One\n  geometry is as good as a mixture, so carrying "
              f"weaker trees alongside it\n  returns less than the "
              f"decorrelation costs.")
    if rare is not None:
        print(f"\n  Read the RECALL column, not accuracy: with the rare class "
              f"at a few\n  percent, a model that catches none of it still "
              f"scores well on accuracy.")
    print(f"\n  +/- is half the observed spread across seeds.\n")


if __name__ == "__main__":
    f = sys.argv[1:]

    def opt(name, cast, default):
        return cast(f[f.index(name) + 1]) if name in f else default

    csv = f[f.index("--csv") + 1] if "--csv" in f else None
    label = f[f.index("--label") + 1] if "--label" in f else None
    hot = (f[f.index("--onehot") + 1].split(",") if "--onehot" in f else None)
    drop = (f[f.index("--drop") + 1].split(",") if "--drop" in f
            else ("id", "ID", "Id"))
    if csv and not (label or hot):
        raise SystemExit("  --csv also needs --label <column> or --onehot "
                         "<col,col,...>")

    run(n=opt("--n", int, 400), seeds=opt("--seeds", int, 3 if "--per-class"
                                          in f else 5),
        maxd=opt("--maxd", int, 6), n_trees=opt("--trees", int, 9),
        csv=csv, label=label, drop=drop, onehot=hot,
        per_class="--per-class" in f)
