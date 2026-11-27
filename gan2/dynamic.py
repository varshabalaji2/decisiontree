"""Let the data decide when to stop splitting, instead of a tuned depth.

    python3 dynamic.py cmapss/train_FD002.txt cmapss/test_FD002.txt \
                       cmapss/RUL_FD002.txt --seeds 3

FINDINGS.md lists this under what is NOT true of this project:

    Depth was chosen by reading the test set. maxd=11 was picked by trying
    8, 11 and 14 against the official test RMSE. Mild, common, still wrong.

That is the flaw this file attacks. A Hoeffding-style test asks, at every
node, whether the best split is beating the runner-up by more than could be
explained by the number of rows that node actually holds:

    eps = sqrt(ln(1/delta) / (2n))          n = rows at THIS node
    split only if  sdr_runner_up / sdr_best + eps < 1

The ratio of the two standard-deviation reductions lives in [0, 1], so the
Hoeffding bound applies with R = 1. This is the FIMT-DD formulation
(Ikonomovska, Gama & Dzeroski 2011), which is the regression version of the
VFDT rule (Domingos & Hulten, KDD 2000). The plain VFDT bound needs a
bounded split criterion and variance reduction is not bounded, so the ratio
is what makes it legitimate here.

n shrinks as the tree deepens, so eps grows, so deep nodes need a clearer
winner to justify splitting. The tree stops itself where the evidence runs
out. No maxd, nothing read off the test set.

tau is the usual tie-breaker: once eps is small enough that two candidates
are genuinely close, waiting longer settles nothing, so take the best.

WHAT THIS IS NOT. The streaming half of the online-tree literature -- update
incrementally, never store the data -- solves a problem this project does
not have. C-MAPSS is 53k rows that fit in memory and do not drift. The
guarantee of those algorithms is that they MATCH a batch tree, so adopting
one would at best tie. Only the confidence test is worth taking.

THE ADAPTIVE-LEAK ARMS ARE A CONTROL, NOT A HOPE. conditions.py already
measured the adaptive leak as redundant (-0.2 RMSE, 1 of 3 seeds, and worse
NASA). The redundancy is in the FEATURE SPACE -- five fixed leaks already
span those timescales -- and a stopping rule does not change what is in the
columns. If the adaptive arms win here but lost there, something is wrong
with one of the two runs, not with the earlier conclusion.

Both arms share every line of tree code and differ only in stop(). Columns
are pre-binned into quantile buckets once, so a node's split search is a
histogram pass rather than a sort, which is what makes pure Python finish.
"""

import math
import random
import sys

from cmapss import load_cmapss, by_unit, drop_dead_columns, nasa_score, CAP
from conditions import (LEAKS, ADAPT, regime_stats, normalise, names_for,
                        featurise)

BINS = 32
DELTA = 1e-6
TAU = 0.05
HARD_CAP = 30          # the dynamic arm still needs a guard against runaway


def cut_points(rows, col, bins=BINS):
    """Quantile edges, so each bucket holds a similar number of rows."""
    xs = sorted(r[col] for r in rows)
    out, n = [], len(xs)
    for i in range(1, bins):
        v = xs[i * n // bins]
        if not out or v > out[-1]:
            out.append(v)
    return out


def binned(rows, cuts):
    """Every column replaced by its bucket index. Done once, up front."""
    out = []
    for r in rows:
        b = []
        for c, cut in enumerate(cuts):
            x, lo, hi = r[c], 0, len(cut)
            while lo < hi:                       # bisect, inlined
                mid = (lo + hi) // 2
                if x < cut[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            b.append(lo)
        out.append(b)
    return out


def sdr_scan(idx, B, ys, col, nbin, min_leaf):
    """Standard-deviation reduction for every cut of one column.

    Returns (best, runner_up, cut). The runner-up is what the Hoeffding test
    compares against, so it is tracked here rather than recomputed."""
    cnt = [0] * nbin
    s1 = [0.0] * nbin
    s2 = [0.0] * nbin
    for i in idx:
        b = B[i][col]
        y = ys[i]
        cnt[b] += 1
        s1[b] += y
        s2[b] += y * y

    n = len(idx)
    tot1 = sum(s1)
    tot2 = sum(s2)
    parent = math.sqrt(max(0.0, tot2 / n - (tot1 / n) ** 2))

    best = second = 0.0
    at = None
    cL = 0
    a1 = a2 = 0.0
    for b in range(nbin - 1):
        cL += cnt[b]
        a1 += s1[b]
        a2 += s2[b]
        cR = n - cL
        if cL < min_leaf or cR < min_leaf:
            continue
        sdL = math.sqrt(max(0.0, a2 / cL - (a1 / cL) ** 2))
        sdR = math.sqrt(max(0.0, (tot2 - a2) / cR - ((tot1 - a1) / cR) ** 2))
        gain = parent - (cL / n) * sdL - (cR / n) * sdR
        if gain > best:
            second = best
            best, at = gain, b
        elif gain > second:
            second = gain
    return best, second, at


def best_split(idx, B, ys, cols, nbin, min_leaf, rng, mtry):
    """Search a random subset of columns and return the best, plus the
    second-best COLUMN's gain.

    The runner-up has to be another column's best cut, not another cut of
    the winning column. Mixing the two in lets a column compete with itself:
    adjacent cuts of one column score almost identically, so the ratio sits
    at ~0.99 and the test refuses every split. That is what made the first
    version of this file return a single-node tree."""
    pick = rng.sample(cols, min(mtry, len(cols)))
    scored = []
    for c in pick:
        g, _, at = sdr_scan(idx, B, ys, c, nbin, min_leaf)
        if at is not None and g > 0.0:
            scored.append((g, c, at))
    if not scored:
        return (0.0, None, None), 0.0
    scored.sort(key=lambda t: -t[0])
    return scored[0], (scored[1][0] if len(scored) > 1 else 0.0)


def stop_fixed(depth, maxd, n, best, second):
    return depth >= maxd


def stop_hoeffding(depth, maxd, n, best, second):
    """Split only if the winner is clear given how many rows are here."""
    if depth >= HARD_CAP or best <= 0.0:
        return True
    eps = math.sqrt(math.log(1.0 / DELTA) / (2.0 * n))
    if eps < TAU:                       # too close to matter; take the best
        return False
    return (second / best) + eps >= 1.0


def build(idx, B, ys, cols, nbin, depth, maxd, min_leaf, rng, mtry, stop):
    n = len(idx)
    if n < 2 * min_leaf:
        return sum(ys[i] for i in idx) / n
    (gain, col, at), second = best_split(idx, B, ys, cols, nbin, min_leaf,
                                         rng, mtry)
    if col is None or stop(depth, maxd, n, gain, second):
        return sum(ys[i] for i in idx) / n
    left = [i for i in idx if B[i][col] <= at]
    right = [i for i in idx if B[i][col] > at]
    if not left or not right:
        return sum(ys[i] for i in idx) / n
    return (col, at,
            build(left, B, ys, cols, nbin, depth + 1, maxd, min_leaf, rng,
                  mtry, stop),
            build(right, B, ys, cols, nbin, depth + 1, maxd, min_leaf, rng,
                  mtry, stop))


def use(t, row):
    while isinstance(t, tuple):
        col, at, lo, hi = t
        t = lo if row[col] <= at else hi
    return t


def size(t):
    return 1 if not isinstance(t, tuple) else 1 + size(t[2]) + size(t[3])


def depth_of(t):
    return 0 if not isinstance(t, tuple) else 1 + max(depth_of(t[2]),
                                                      depth_of(t[3]))


def forest(B, ys, cols, nbin, n_trees, maxd, min_leaf, seed, stop):
    rng = random.Random(seed)
    mtry = max(1, int(len(cols) ** 0.5))
    out = []
    for t in range(n_trees):
        idx = [rng.randrange(len(ys)) for _ in range(len(ys))]   # bagging
        out.append(build(idx, B, ys, cols, nbin, 0, maxd, min_leaf,
                         random.Random(seed * 100 + t), mtry, stop))
    return out


def predict(f, row):
    return sum(use(t, row) for t in f) / len(f)


# -- data ------------------------------------------------------------------

def make(train_path, test_path, leaks, adapt, stats):
    raw = load_cmapss(train_path)
    units = normalise(by_unit(raw), stats)
    units, kept, keep = drop_dead_columns(units, names_for(len(raw[0])))
    trR, trY = [], []
    for u in sorted(units):
        rs = units[u]
        last = rs[-1][1]
        feats, names = featurise(rs, kept, leaks, (), adapt)
        trR += feats
        trY += [min(CAP, last - r[1]) for r in rs]

    tu = normalise(by_unit(load_cmapss(test_path)), stats)
    teR = []
    for u in sorted(tu):
        rs = [[r[i] for i in keep] for r in tu[u]]
        feats, _ = featurise(rs, kept, leaks, (), adapt)
        teR.append(feats[-1])
    return trR, trY, teR, len(names)


def run(train_path, test_path, rul_path, n_trees=9, maxd=8, seeds=3,
        min_leaf=20):
    stats, counts = regime_stats(by_unit(load_cmapss(train_path)))
    want = [min(CAP, float(x)) for x in open(rul_path).read().split()]

    def rm(p):
        return (sum((a - y) ** 2 for a, y in zip(p, want)) / len(want)) ** .5

    def ma(p):
        return sum(abs(a - y) for a, y in zip(p, want)) / len(want)

    def show(xs):
        if len(xs) == 1:
            return f"{xs[0]:.1f}"
        return f"{sum(xs) / len(xs):.1f} +/-{(max(xs) - min(xs)) / 2:.1f}"

    print(f"\n  {len(counts)} operating conditions, normalised per condition")
    print(f"  {len(want)} test engines, {n_trees} trees, {seeds} seed(s), "
          f"RUL capped at {CAP}")
    print(f"  fixed arm stops at depth {maxd}; dynamic arm stops when "
          f"sdr2/sdr1 + eps >= 1,")
    print(f"  delta={DELTA}, tau={TAU}, hard cap {HARD_CAP}, {BINS} quantile "
          f"bins per column\n")

    got, cols_n, nodes, deep = {}, {}, {}, {}
    for fname, leaks, adapt in (("memory", LEAKS, ()),
                                ("memory + adaptive", LEAKS, ADAPT)):
        print(f"    building {fname} features...", flush=True)
        trR, trY, teR, ncol = make(train_path, test_path, leaks, adapt, stats)
        cuts = [cut_points(trR, c) for c in range(ncol)]
        B = binned(trR, cuts)
        teB = binned(teR, cuts)
        cols = list(range(ncol))
        nbin = max(max(len(c) for c in cuts) + 1, 2)

        for sname, stop in (("fixed depth", stop_fixed),
                            ("dynamic stop", stop_hoeffding)):
            tag = f"{fname}, {sname}"
            got[tag], nodes[tag], deep[tag] = [], [], []
            cols_n[tag] = ncol
            for s in range(seeds):
                f = forest(B, trY, cols, nbin, n_trees, maxd, min_leaf, s,
                           stop)
                p = [predict(f, r) for r in teB]
                got[tag].append(p)
                nodes[tag].append(sum(size(t) for t in f) / len(f))
                deep[tag].append(max(depth_of(t) for t in f))
                print(f"      {tag:<28} seed {s}: {rm(p):>5.1f} RMSE, "
                      f"{nodes[tag][-1]:>6.0f} nodes, depth "
                      f"{deep[tag][-1]}", flush=True)

    print(f"\n  {'':<30}{'cols':>6}{'nodes':>8}{'depth':>7}"
          f"{'RMSE':>13}{'MAE':>13}{'NASA':>8}")
    for tag in got:
        ps = got[tag]
        print(f"  {tag:<30}{cols_n[tag]:>6}"
              f"{sum(nodes[tag]) / len(nodes[tag]):>8.0f}"
              f"{max(deep[tag]):>7}"
              f"{show([rm(p) for p in ps]):>13}"
              f"{show([ma(p) for p in ps]):>13}"
              f"{sum(nasa_score(p, want) for p in ps) / len(ps):>8.0f}")

    def avg(tag):
        return sum(rm(p) for p in got[tag]) / len(got[tag])

    def paired(a, b):
        d = [rm(y) - rm(x) for x, y in zip(got[a], got[b])]
        return sum(1 for x in d if x > 0), len(d), sum(d) / len(d)

    print()
    for fname in ("memory", "memory + adaptive"):
        fx, dy = f"{fname}, fixed depth", f"{fname}, dynamic stop"
        w, n, d = paired(dy, fx)
        if w * 10 >= n * 8 and d > 0.3:
            print(f"  DYNAMIC STOPPING WINS on {fname}: {d:+.1f} RMSE over a "
                  f"tuned depth\n  on {w} of {n} seeds, and the depth was "
                  f"not read off the test set.")
        elif abs(d) <= 0.3:
            print(f"  DYNAMIC STOPPING TIES on {fname}: {d:+.1f} RMSE, {w} of "
                  f"{n} seeds. It matches a\n  depth that WAS tuned on the "
                  f"test set, so it buys honesty rather than\n  accuracy -- "
                  f"which is the better trade, since maxd stops being a knob.")
        else:
            print(f"  DYNAMIC STOPPING LOSES on {fname}: {d:+.1f} RMSE on {w} "
                  f"of {n} seeds.")

    a = avg("memory, dynamic stop")
    b = avg("memory + adaptive, dynamic stop")
    print(f"\n  adaptive columns under the dynamic rule: {b - a:+.1f} RMSE. "
          f"conditions.py\n  measured -0.2 with a fixed depth. A stopping "
          f"rule cannot change what is in\n  the columns, so these two should "
          f"agree; if they do not, one run is wrong.")
    print()


if __name__ == "__main__":
    f = sys.argv[1:]
    a, i = [], 0
    while i < len(f):
        if f[i].startswith("--"):
            i += 2
        else:
            a.append(f[i])
            i += 1
    if len(a) < 3:
        raise SystemExit(__doc__)

    def opt(name, cast, default):
        return cast(f[f.index(name) + 1]) if name in f else default

    run(a[0], a[1], a[2], n_trees=opt("--trees", int, 9),
        maxd=opt("--maxd", int, 8), seeds=opt("--seeds", int, 3),
        min_leaf=opt("--minleaf", int, 20))
