"""Predict for the cost you are scored on, not the one you fit.

    python3 goalbias.py cmapss/train_FD002.txt cmapss/test_FD002.txt \
                        cmapss/RUL_FD002.txt
    python3 goalbias.py ... --seeds 3 --maxd 8

The tree minimises SQUARED ERROR: var_sweep ranks splits by variance
reduction and a leaf answers with the mean. Both directions of error cost
the same.

The metric this project reports does not agree:

    over-stating life   (said 40, was 10)    exp(d/10) - 1
    under-stating life  (said 10, was 40)    exp(-d/13) - 1

Over-stating is the engine that fails before you expected it, and NASA
charges for it on a steeper exponent. Nothing in the pipeline has ever been
told which mistake is worse.

THIS IS THE SHAPE OF A MEASURED WINNER. Class rebalancing told a classifier
to care about rare faults more than their frequency warrants and took steel
macro recall from 65.7% to 75.9%. This tells a regressor to care about
under-stating less than over-stating. Both change what the model is ASKED
TO OPTIMISE, which is a change to what it is told rather than what it is --
the side of this project's rule that keeps paying.

TWO KNOBS, and they are not the same thing:

AND THE BASELINE HAD TO BE CORRECTED. The q=0.50 row was labelled "mean
leaf" and it is not: at_q(vals, 0.50) returns the MEDIAN. On FD002 that is
worth about 0.4 RMSE, which is exactly the gap between this file's 16.3 and
the 15.9 reported elsewhere, and quoting the trade against the median made
the RMSE side of it look cheaper than it is. There are now three rows -- the
mean leaf (the model the project reports), the median leaf, and the chosen
pair -- all read off the same forest, so the mean/median step is visible
separately from the goal-biasing step.

  quantile   a leaf answers with its q-th percentile instead of its mean,
             so the correction is LOCAL: a leaf whose engines disagree
             widely backs off further than a leaf whose engines agree.

  shift      a flat number subtracted from every prediction. Global, blunt,
             and free.

Both are read off ONE forest. The leaves store their sorted values rather
than a mean, so any quantile is available at predict time without refitting
-- which is why sweeping seven quantiles costs one forest and not seven.
The SPLITS are still chosen by variance reduction; only the answer changes.

CHOSEN ON HELD-OUT TRAINING ENGINES, NEVER THE TEST SET. FINDINGS lists
maxd=11 as a number picked by watching the official test RMSE, and that
flaw is not being repeated. Engines are held out, truncated at a random
point to resemble the official protocol, and the pair (q, shift) that
minimises NASA there is the one applied to the test set once.

THE VALIDATION ENGINES HAVE TO BE TRUNCATED. A training engine runs to
failure, so its last row has RUL 0 -- nothing like a test engine, which is
cut off partway through. Scoring the pair on untruncated engines would tune
for a question nobody asks. Each validation engine is therefore cut at a
uniformly random point between VAL_LO and VAL_HI of its life. The real
distribution of test truncations is not published, so this is an
assumption, and a visible one.
"""

import random
import sys

from cmapss import (load_cmapss, by_unit, forest_reg, use_reg, nasa_score,
                    CAP)
from conditions import LEAKS, regime_stats, prepare, test_rows

QUANTS = (0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20)
SHIFTS = tuple(range(0, 31, 2))
VAL_LO, VAL_HI = 0.40, 0.95
VAL_FRAC = 0.25


def sorted_leaf(ys):
    """A leaf keeps its values, sorted, instead of collapsing to the mean.

    A LIST and not a tuple: use_reg finds leaves with isinstance(tree,
    tuple), so a tuple leaf would be walked as an internal node.
    """
    return sorted(ys)


def at_q(vals, q):
    return vals[min(len(vals) - 1, int(q * len(vals)))]


def predict_mean(forest, row, shift=0):
    """The MEAN of the leaf's values -- the model the project reports.

    at_q(vals, 0.50) is the MEDIAN, not the mean, so a q=0.50 row is not the
    headline model and labelling it 'mean leaf' was wrong. On FD002 the two
    differ by about 0.4 RMSE, which is the whole gap between this file's
    16.3 and FINDINGS' 15.9, and quoting the trade against the median made
    the RMSE cost look smaller than it is.

    The leaves store their values, so this reads the SAME forest: no refit,
    and the two baselines differ in nothing but how a leaf answers.
    """
    v = sum(sum(use_reg(t, row)) / len(use_reg(t, row)) for t in forest)
    return min(float(CAP), max(0.0, v / len(forest) - shift))


def predict_q(forest, row, q, shift):
    """Average the trees' q-th percentiles, then step back by shift.

    Clamped to [0, CAP] because the truth is: a negative remaining life is
    not a prediction, and the target is capped, so neither end can help."""
    v = sum(at_q(use_reg(t, row), q) for t in forest) / len(forest)
    return min(float(CAP), max(0.0, v - shift))


def rmse(pred, want):
    return (sum((a - y) ** 2 for a, y in zip(pred, want)) / len(want)) ** 0.5


def mae(pred, want):
    return sum(abs(a - y) for a, y in zip(pred, want)) / len(want)


def val_set(data, units, seed):
    """One row per validation engine, cut at a random point in its life,
    with the true RUL that was recorded there."""
    rng = random.Random(seed + 808)
    rows, want = [], []
    for u in units:
        feats, ruls = data[u]
        n = len(feats)
        if n < 10:
            continue
        k = rng.randrange(int(n * VAL_LO), max(int(n * VAL_HI), int(n * VAL_LO) + 1))
        # row k already carries the memory columns replayed over rows 0..k,
        # because prepare() built them in cycle order -- no re-featurising
        rows.append(feats[k])
        want.append(ruls[k])
    return rows, want


def choose(forest, rows, want, quants=QUANTS, shifts=SHIFTS):
    """The (quantile, shift) with the lowest NASA score on held-out engines."""
    best, table = None, []
    for q in quants:
        base = [sum(at_q(use_reg(t, r), q) for t in forest) / len(forest)
                for r in rows]
        for sh in shifts:
            p = [min(float(CAP), max(0.0, v - sh)) for v in base]
            s = nasa_score(p, want)
            table.append((q, sh, s, rmse(p, want)))
            if best is None or s < best[2]:
                best = (q, sh, s, rmse(p, want))
    return best, table


def run(train_path, test_path, rul_path, n_trees=9, maxd=8, seeds=3,
        leaks=LEAKS):
    stats, counts = regime_stats(by_unit(load_cmapss(train_path)))
    want = [min(CAP, float(x)) for x in open(rul_path).read().split()]
    data, names, keep, kept_names = prepare(train_path, leaks, stats)
    teR = test_rows(test_path, keep, kept_names, leaks, stats)
    engines = sorted(data)

    print(f"\n  {len(counts)} operating conditions, normalised per condition")
    print(f"  {len(engines)} training engines, {len(want)} test engines, "
          f"{len(names)} columns")
    print(f"  depth {maxd}, {n_trees} trees, {seeds} seed(s)")
    print(f"  quantiles {QUANTS}")
    print(f"  shifts {SHIFTS[0]}..{SHIFTS[-1]} cycles")
    print(f"  {int(VAL_FRAC * 100)}% of TRAINING engines held out, each cut "
          f"at a random point\n  between {VAL_LO:.0%} and {VAL_HI:.0%} of "
          f"its life, to choose (quantile, shift)\n")

    plain, median, biased, picks = [], [], [], []
    for s in range(seeds):
        order = list(engines)
        random.Random(s + 11).shuffle(order)
        cut = max(1, int(len(order) * (1 - VAL_FRAC)))
        fit_u, val_u = order[:cut], order[cut:]

        trR = [r for u in fit_u for r in data[u][0]]
        trY = [y for u in fit_u for y in data[u][1]]
        f = forest_reg(trR, trY, n_trees=n_trees, maxd=maxd, seed=s,
                       leaf=sorted_leaf)
        vR, vW = val_set(data, val_u, s)
        (q, sh, vn, vr), _ = choose(f, vR, vW)
        picks.append((q, sh))
        print(f"    seed {s}: {len(fit_u)} fit / {len(val_u)} validation "
              f"engines -> q={q:.2f}, shift={sh}"
              f"   (validation NASA {vn:.0f}, RMSE {vr:.1f})", flush=True)

        # refit on EVERY training engine, then apply the pair chosen above
        allR = [r for u in engines for r in data[u][0]]
        allY = [y for u in engines for y in data[u][1]]
        full = forest_reg(allR, allY, n_trees=n_trees, maxd=maxd, seed=s,
                          leaf=sorted_leaf)
        plain.append([predict_mean(full, r) for r in teR])
        median.append([predict_q(full, r, 0.5, 0) for r in teR])
        biased.append([predict_q(full, r, q, sh) for r in teR])

    def show(xs):
        if len(xs) == 1:
            return f"{xs[0]:.1f}"
        return f"{sum(xs) / len(xs):.1f} +/-{(max(xs) - min(xs)) / 2:.1f}"

    mean = sum(y for u in engines for y in data[u][1])
    mean /= sum(len(data[u][1]) for u in engines)
    base = rmse([mean] * len(want), want)

    print(f"\n  {'':<30}{'RMSE':>14}{'MAE':>14}{'NASA':>10}")
    print(f"  {'always the average':<30}{base:>14.1f}{'-':>14}{'-':>10}")
    for tag, ps in (("mean leaf (the headline model)", plain),
                    ("median leaf, no shift", median),
                    ("goal-biased", biased)):
        print(f"  {tag:<30}{show([rmse(p, want) for p in ps]):>14}"
              f"{show([mae(p, want) for p in ps]):>14}"
              f"{sum(nasa_score(p, want) for p in ps) / len(ps):>10.0f}")

    pr = sum(rmse(p, want) for p in plain) / len(plain)
    br = sum(rmse(p, want) for p in biased) / len(biased)
    pn = sum(nasa_score(p, want) for p in plain) / len(plain)
    bn = sum(nasa_score(p, want) for p in biased) / len(biased)
    wins = sum(1 for a, b in zip(plain, biased)
               if nasa_score(b, want) < nasa_score(a, want))

    mr = sum(rmse(p, want) for p in median) / len(median)
    mn_ = sum(nasa_score(p, want) for p in median) / len(median)
    print(f"\n  chosen per seed: {', '.join(f'q={q:.2f} shift={s}' for q, s in picks)}")
    print(f"  the median row is the q=0.50 end of the same sweep, shown "
          f"because {mr - pr:+.1f} RMSE\n  and {mn_ - pn:+.0f} NASA of what "
          f"follows is just the switch from mean to median")
    if len(set(q for q, _ in picks)) > 1:
        lo = min(q for q, _ in picks)
        hi = max(q for q, _ in picks)
        print(f"  THE PICK IS NOT STABLE: q ranges {lo:.2f} to {hi:.2f} "
              f"across seeds, so the validation\n  engines are not "
              f"identifying one operating point -- they are identifying a "
              f"band.\n  The trade below is the average of three different "
              f"models, not one tuned model.")
    if all(sh == 0 for _, sh in picks):
        print(f"  AND THE SHIFT KNOB DID NOTHING: every seed chose shift=0, "
              f"so the whole effect\n  is the quantile. A flat subtraction "
              f"off every prediction was available and\n  the validation "
              f"engines refused it, which says the correction has to be "
              f"LOCAL.")
    if bn < pn and wins * 10 >= len(plain) * 8:
        print(f"\n  THE TRADE IS REAL: NASA {pn:.0f} -> {bn:.0f} "
              f"({(pn - bn) / pn:.0%} lower) on {wins} of\n  {len(plain)} "
              f"seeds, for {br - pr:+.1f} RMSE. That is the exchange NASA's "
              f"formula asks\n  for, and the operating point was chosen on "
              f"held-out engines, not on this\n  test set.")
    elif bn < pn:
        print(f"\n  NASA {pn:.0f} -> {bn:.0f} for {br - pr:+.1f} RMSE, but "
              f"on only {wins} of {len(plain)} seeds,\n  so it is not "
              f"consistent. Raise --seeds before relying on it.")
    else:
        print(f"\n  NO TRADE AVAILABLE: NASA {pn:.0f} -> {bn:.0f}. The "
              f"validation engines picked an\n  operating point that does "
              f"not transfer, which usually means the random\n  truncation "
              f"between {VAL_LO:.0%} and {VAL_HI:.0%} does not resemble how "
              f"the official test\n  engines were cut.")
    print(f"\n  RMSE counts both directions the same; NASA does not. A model "
          f"tuned for one\n  should not be read on the other without saying "
          f"which it was tuned for.\n")


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
        maxd=opt("--maxd", int, 8), seeds=opt("--seeds", int, 3))
