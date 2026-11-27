"""Trees trained on each other's mistakes, instead of all on the same question.

    python3 boosting.py cmapss/train_FD002.txt cmapss/test_FD002.txt \
                        cmapss/RUL_FD002.txt
    python3 boosting.py ... --rounds 300 --lr 0.05 --depth 4 --seeds 3

The forest in cmapss.py trains nine trees on the SAME question and averages
them, which cuts variance. Boosting asks a different question each round:

    round 1   predict RUL
    round 2   predict what round 1 got wrong
    round 3   predict what rounds 1 and 2 still get wrong
    answer    the sum of all of them

Same rows, same columns, different TARGETS -- so this is neither computing
across rows nor routing, but a third thing: changing what the next stage is
asked to learn. That is worth recording in FINDINGS.md's rule, because the
rule as written does not cover it.

The correction only flows during TRAINING. At test time the trees are simply
added up, with no labels anywhere, which is what separates this from "predict,
check what was right, pass it down" -- that needs the answer at test time, and
there isn't one.

HOW MANY ROUNDS IS THE ONLY DANGEROUS KNOB. Too few underfits, too many
memorises, and the number cannot be chosen by watching the official test set,
which is how maxd=11 was picked earlier in this project and is mild test-set
leakage. So some TRAINING ENGINES are held out as a validation set, rounds are
added until validation RMSE stops improving, and the extra rounds are thrown
away. Held out by engine, never by row, for the same reason everything else
in this project is.

Each round also sees a random part of the rows and the columns. That is
standard stochastic gradient boosting -- it decorrelates the trees and, here,
makes pure Python fast enough to finish.
"""

import random
import statistics
import sys

from cmapss import build_reg, use_reg, nasa_score, CAP
from conditions import (LEAKS, regime_stats, prepare, test_rows, by_unit,
                        load_cmapss)


def rmse(pred, want):
    return (sum((a - y) ** 2 for a, y in zip(pred, want)) / len(want)) ** 0.5


def mae(pred, want):
    return sum(abs(a - y) for a, y in zip(pred, want)) / len(want)


def boost(fitR, fitY, valR, valY, rounds=200, lr=0.1, maxd=4, min_leaf=20,
          seed=0, subsample=0.6, colsample=0.4, patience=15, report=None):
    """Fit trees to the running residual, stopping when the held-out engines
    stop improving. Returns (base, lr, trees, rounds_kept, best_val)."""
    rng = random.Random(seed)
    n, ncol = len(fitR), len(fitR[0])
    take_r = max(1, int(subsample * n))
    take_c = max(1, int(colsample * ncol))

    base = sum(fitY) / len(fitY)
    pf = [base] * n
    pv = [base] * len(valR)

    trees = []
    best_v, best_k = rmse(pv, valY), 0
    for r in range(rounds):
        resid = [y - p for y, p in zip(fitY, pf)]
        rows_i = rng.sample(range(n), take_r)
        cols = rng.sample(range(ncol), take_c)
        t = build_reg([fitR[i] for i in rows_i], [resid[i] for i in rows_i],
                      maxd=maxd, min_leaf=min_leaf, only=cols)
        trees.append(t)

        pf = [p + lr * use_reg(t, row) for p, row in zip(pf, fitR)]
        pv = [p + lr * use_reg(t, row) for p, row in zip(pv, valR)]
        v = rmse(pv, valY)
        if v < best_v - 1e-9:
            best_v, best_k = v, len(trees)
        if report and (r + 1) % report == 0:
            print(f"      round {r + 1:>4}  validation RMSE {v:.2f}  "
                  f"(best {best_v:.2f} at {best_k})", flush=True)
        if len(trees) - best_k >= patience:
            break

    return base, lr, trees[:best_k], best_k, best_v


def boost_predict(model, row):
    base, lr, trees = model[0], model[1], model[2]
    return base + lr * sum(use_reg(t, row) for t in trees)


def split_engines(engines, val_frac=0.25, seed=0):
    order = list(engines)
    random.Random(seed + 991).shuffle(order)
    cut = max(1, int(len(order) * (1 - val_frac)))
    return order[:cut], order[cut:]


def flat(data, units):
    rows = [r for u in units for r in data[u][0]]
    ys = [y for u in units for y in data[u][1]]
    return rows, ys


def run(train_path, test_path, rul_path, rounds=200, lr=0.1, depth=4,
        seeds=3, leaks=LEAKS, val_frac=0.25, report=50):
    stats, counts = regime_stats(by_unit(load_cmapss(train_path)))
    want = [min(CAP, float(x)) for x in open(rul_path).read().split()]

    data, names, keep, kept_names = prepare(train_path, leaks, stats)
    engines = sorted(data)
    teR = test_rows(test_path, keep, kept_names, leaks, stats)

    print(f"\n  {len(counts)} operating condition(s), normalised per condition")
    print(f"  {len(engines)} training engines, {len(want)} test engines, "
          f"{len(names)} columns")
    print(f"  depth {depth}, learning rate {lr}, up to {rounds} rounds, "
          f"{seeds} seed(s)")
    print(f"  {int(val_frac * 100)}% of TRAINING ENGINES held out to choose "
          f"the number of rounds\n")

    preds, kept = [], []
    for s in range(seeds):
        fit_u, val_u = split_engines(engines, val_frac, seed=s)
        fitR, fitY = flat(data, fit_u)
        valR, valY = flat(data, val_u)
        print(f"    seed {s}: {len(fit_u)} fit / {len(val_u)} validation "
              f"engines, {len(fitR)} rows")
        m = boost(fitR, fitY, valR, valY, rounds=rounds, lr=lr, maxd=depth,
                  seed=s, report=report)
        kept.append(m[3])
        p = [boost_predict(m, r) for r in teR]
        preds.append(p)
        print(f"      kept {m[3]} rounds, validation RMSE {m[4]:.2f}"
              f"  ->  test RMSE {rmse(p, want):.1f}", flush=True)

    rs = [rmse(p, want) for p in preds]
    ms = [mae(p, want) for p in preds]
    ns = [nasa_score(p, want) for p in preds]

    mean = sum(y for u in engines for y in data[u][1])
    mean /= sum(len(data[u][1]) for u in engines)
    base = rmse([mean] * len(want), want)

    def show(xs):
        if len(xs) == 1:
            return f"{xs[0]:.1f}"
        return f"{statistics.mean(xs):.1f} +/- {(max(xs) - min(xs)) / 2:.1f}"

    def show_int(xs):
        return str(xs[0]) if len(xs) == 1 else f"{min(xs)}-{max(xs)}"

    print(f"\n  {'':<28}{'rounds':>8}{'RMSE':>16}{'MAE':>16}{'NASA':>8}")
    print(f"  {'always the average':<28}{'-':>8}{base:>16.1f}{'-':>16}{'-':>8}")
    print(f"  {'boosting, normalised':<28}{show_int(kept):>8}"
          f"{show(rs):>16}{show(ms):>16}{statistics.mean(ns):>8.0f}")
    print(f"\n  for comparison, the forest of 9 on the same features reached")
    print(f"  14.4 RMSE on FD002 and 18.7 on FD004, both at depth 8, one seed.")
    print(f"\n  +/- is half the observed spread, not a standard error.\n")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    f = sys.argv[1:]
    if len(a) < 3:
        raise SystemExit(__doc__)

    def opt(name, cast, default):
        return cast(f[f.index(name) + 1]) if name in f else default

    run(a[0], a[1], a[2], rounds=opt("--rounds", int, 200),
        lr=opt("--lr", float, 0.1), depth=opt("--depth", int, 4),
        seeds=opt("--seeds", int, 3))
