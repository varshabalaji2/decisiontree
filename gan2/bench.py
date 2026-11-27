"""Seeds and a baseline, which is what the headline numbers are missing.

    python3 bench.py cmapss/train_FD002.txt cmapss/test_FD002.txt \
                     cmapss/RUL_FD002.txt
    python3 bench.py ... --seeds 5 --maxd 8 --trees 9

Two gaps this closes.

SEEDS. Most numbers in this project are one seed. The three FD001 runs came
out 18.3 / 18.3 / 17.6, a spread of 0.7, and the FD002 depth-11 vs depth-14
comparison (17.3 vs 17.0) sat inside that -- it was read as a gain and was
not one. A difference smaller than the spread is not a difference, so every
headline needs a mean and a spread beside it.

A BASELINE. There is no sklearn number anywhere on the same split, so 14.5
RMSE cannot be attributed. It may be the forest being good, or the
normalisation being good enough that any model reaches it. Both are
defensible findings and neither is currently established. This runs
RandomForestRegressor on the IDENTICAL features -- same rows, same columns,
same normalisation, same test replay -- so the only difference is the model.
If sklearn is not installed it says so and skips that arm.

The features are built ONCE per arm and only the forest's seed changes, which
is both faster and the right comparison: it isolates model variance rather
than mixing in variance from the feature pipeline.
"""

import statistics
import sys

from cmapss import forest_reg, predict, nasa_score, CAP
from conditions import (LEAKS, regime_stats, prepare, test_rows, by_unit,
                        load_cmapss)


def spread(xs):
    return max(xs) - min(xs) if len(xs) > 1 else 0.0


def fmt(xs, width=1):
    if len(xs) == 1:
        return f"{xs[0]:.{width}f}"
    return f"{statistics.mean(xs):.{width}f} +/- {spread(xs) / 2:.{width}f}"


def scores(pred, want):
    rmse = (sum((a - y) ** 2 for a, y in zip(pred, want)) / len(want)) ** 0.5
    mae = sum(abs(a - y) for a, y in zip(pred, want)) / len(want)
    return rmse, mae, nasa_score(pred, want)


def features(train_path, test_path, leaks, stats):
    """Everything up to the model: load, normalise, memory columns, RUL, and
    the one row per test engine at its last observed cycle."""
    data, names, keep, kept_names = prepare(train_path, leaks, stats)
    trR = [r for u in sorted(data) for r in data[u][0]]
    trY = [y for u in sorted(data) for y in data[u][1]]
    teR = test_rows(test_path, keep, kept_names, leaks, stats)
    return trR, trY, teR, len(names)


def say(msg):
    """Progress, flushed. Ten pure-Python forests on 50k rows is a long wait,
    and a run with no output for half an hour looks like a hang."""
    print(msg, flush=True)


def sklearn_arm(trR, trY, teR, seeds, trees, maxd):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return None
    out = []
    for s in seeds:
        m = RandomForestRegressor(n_estimators=trees, max_depth=maxd,
                                  min_samples_leaf=20, random_state=s,
                                  n_jobs=-1)
        m.fit(trR, trY)
        out.append(list(m.predict(teR)))
        say(f"      sklearn seed {s}  done")
    return out


def run(train_path, test_path, rul_path, n_seeds=5, trees=9, maxd=8,
        leaks=LEAKS):
    want = [min(CAP, float(x)) for x in open(rul_path).read().split()]
    seeds = list(range(n_seeds))

    stats, counts = regime_stats(by_unit(load_cmapss(train_path)))
    print(f"\n  {len(counts)} operating condition(s) in the training file")
    print(f"  {len(want)} test engines, depth {maxd}, {trees} trees, "
          f"{n_seeds} seed(s), RUL capped at {CAP}")

    rows = []
    for tag, st in (("raw sensors", None), ("normalised per condition", stats)):
        say(f"\n  {tag}: building features...")
        trR, trY, teR, ncol = features(train_path, test_path, leaks, st)
        say(f"      {len(trR)} training rows, {ncol} columns")

        preds = []
        for s in seeds:
            f = forest_reg(trR, trY, n_trees=trees, maxd=maxd, seed=s)
            p = [predict(f, r) for r in teR]
            preds.append(p)
            r, m, _ = scores(p, want)
            say(f"      this forest seed {s}  RMSE {r:.1f}  MAE {m:.1f}")
        rows.append((tag, ncol, [scores(p, want) for p in preds]))

        sk = sklearn_arm(trR, trY, teR, seeds, trees, maxd)
        if sk is None:
            rows.append(("  sklearn RF (not installed)", ncol, None))
        else:
            rows.append((f"  sklearn RF, same features", ncol,
                         [scores(p, want) for p in sk]))

    mean = sum(trY) / len(trY)
    base = (sum((mean - y) ** 2 for y in want) / len(want)) ** 0.5

    print(f"\n  {'':<30}{'cols':>6}{'RMSE':>16}{'MAE':>16}{'NASA':>8}")
    print(f"  {'always the average':<30}{'-':>6}{base:>16.1f}{'-':>16}{'-':>8}")
    for tag, ncol, sc in rows:
        if sc is None:
            print(f"  {tag:<30}{'-':>6}{'-':>16}{'-':>16}{'-':>8}")
            continue
        print(f"  {tag:<30}{ncol:>6}"
              f"{fmt([x[0] for x in sc]):>16}"
              f"{fmt([x[1] for x in sc]):>16}"
              f"{statistics.mean([x[2] for x in sc]):>8.0f}")

    print(f"\n  +/- is half the observed spread across seeds, not a standard")
    print(f"  error. A difference smaller than it is not a difference.\n")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    flags = sys.argv[1:]
    if len(a) < 3:
        raise SystemExit(__doc__)

    def opt(name, default):
        if name in flags:
            return int(flags[flags.index(name) + 1])
        return default

    run(a[0], a[1], a[2], n_seeds=opt("--seeds", 5),
        trees=opt("--trees", 9), maxd=opt("--maxd", 8))
