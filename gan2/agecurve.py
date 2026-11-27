"""How wrong is the model, as a function of how much of the engine it has seen?

    python3 agecurve.py cmapss/train_FD002.txt
    python3 agecurve.py cmapss/train_FD002.txt --seeds 3 --maxd 8

Every number this project reports on C-MAPSS is ONE prediction per engine, at
its last observed cycle, averaged over 259 engines. That average silently
mixes an engine with 30 cycles of history and one with 300. In service you
never face the average: you face one engine, at one age, and the question is
whether THIS prediction is worth acting on.

So hold out whole engines -- where the full run is known, unlike the official
test file -- truncate each at 20%, 40%, 60%, 80% and 100% of its life, and
score the prediction made at each truncation. Same trees, same columns, same
everything. Only the question changes.

Console, Picardi & Theseider Dupre, "Temporal Decision Trees", JAIR 19 (2003),
make the framing explicit for on-board diagnosis: a decision has a TIME, later
decisions are better informed, and a deadline forces the choice before the
information is complete. That is the turbofan question exactly -- ground it
now, or wait ten more flights?

AND THE NASA SCORE IS ALREADY A DEADLINE COST. It punishes over-stating
remaining life far harder than under-stating it, because an engine that fails
before you expected it is the expensive case. Over-prediction IS waiting too
long. The asymmetry in that formula and the deadline in that paper are the
same idea, and this file is where they meet the data.

TWO CONTROLS, BOTH NECESSARY, because a low RMSE early would otherwise be
mistaken for skill:

  always the CAP     RUL is capped at 125, so an engine 20% through its life
                     usually has more than 125 cycles left and the correct
                     answer is simply the cap. A model that learned nothing
                     scores well there by saying 125. If the model only ties
                     this control, it knows nothing at that age.

  always the mean    the usual floor.

It also tests an assumption already hard-coded in cmapss.py. CAP = 125 asserts
that beyond 125 cycles out nothing is knowable. This measures whether that is
true, and where the curve actually flattens.

Nothing here changes the model. Same forest, same features -- it is measured
differently, which is the side of this project's rule that has never lost.

MEASURED, FD002, 3 seeds, 182 fit / 78 held-out engines:

    life seen  at cap  true RUL   RMSE   vs cap  vs mean    NASA
          20%     85%       124    5.1      2.7     37.5      35
          40%     37%       114   21.5     16.7     30.3     519
          60%      1%        83   25.1     45.5     17.9    1475
          80%      0%        42   19.2     83.9     46.1     929
         100%      0%         0    3.5    125.0     86.8      30

THE HEADLINE 15.9 IS AN AVERAGE OVER A CURVE FROM 3.5 TO 25.1, and the
model is WORSE THAN A CONSTANT until 80% of life. Not merely less useful:
at 60% it scores 25.1 where guessing the training mean scores 17.9.

Two honest qualifications on that:

  the 60% row flatters the control. The training mean is about 87 and the
  true average RUL at 60% is 83, so the constant is accidentally
  well-calibrated at exactly that point. It is not being clever. The model
  still adds 7 RMSE of noise around a number a constant gets right, which
  is bad whatever the reason.

  the 100% row is operationally vacuous. 100% of life observed means the
  engine has already failed and the true RUL is 0. No model is needed for
  that, and in service the event arrives first.

The 20% row is not alarming for the same reason the cap column exists: 85%
of those engines really are at the cap, so "always say 125" is nearly
right and the model is adding noise to an almost-constant target.

AND IT VALIDATES CAP = 125. The cap answers for 85% of engines at 20% of
life, 37% at 40%, and 1% at 60%. cmapss.py asserts exactly this in its
docstring to justify the constant; this is the measurement.

THE LIMITATION OF THIS FILE, which is mine and not the model's. FRACTION
OF LIFE IS NOT KNOWABLE AT PREDICTION TIME. At cycle 150 there is no way
to tell whether that is 20% or 80% of an engine's life, because the total
is what the model is trying to predict. So this curve is DIAGNOSTIC -- it
says how the model behaves -- and it is not a POLICY, because "trust it
after 80%" cannot be acted on.

The deployable version bins by something available at inference: the
model's OWN prediction. "When it says 40 cycles, how wrong is it? When it
says 125?" A low prediction is itself the end-of-life signal, so the same
shape should appear in a form an engineer can follow.
"""

import random
import sys

from cmapss import (load_cmapss, by_unit, drop_dead_columns, forest_reg,
                    predict, nasa_score, CAP)
from conditions import LEAKS, regime_stats, normalise, names_for, featurise

FRACS = (0.2, 0.4, 0.6, 0.8, 1.0)


def rmse(pred, want):
    return (sum((a - y) ** 2 for a, y in zip(pred, want)) / len(want)) ** 0.5


def mae(pred, want):
    return sum(abs(a - y) for a, y in zip(pred, want)) / len(want)


def one_seed(units_raw, names, seed, fracs, n_trees, maxd, val_frac):
    """Fit on one set of engines, score the rest at several truncations.

    Everything the model is allowed to know comes from the FIT engines: the
    per-condition means and spreads, and which columns are dead. Taking
    either from the held-out engines would be the same mistake as reading
    the test set, in a file written to measure honesty.
    """
    engines = sorted(units_raw)
    order = list(engines)
    random.Random(seed + 404).shuffle(order)
    cut = max(1, int(len(order) * (1 - val_frac)))
    fit_u, held_u = order[:cut], order[cut:]

    stats, _ = regime_stats({u: units_raw[u] for u in fit_u})
    units = normalise(units_raw, stats)
    _, kept, keep = drop_dead_columns({u: units[u] for u in fit_u}, names)
    units = {u: [[r[i] for i in keep] for r in rs] for u, rs in units.items()}

    trR, trY = [], []
    for u in fit_u:
        rs = units[u]
        last = rs[-1][1]
        feats, _ = featurise(rs, kept, LEAKS, (), ())
        trR += feats
        trY += [min(CAP, last - r[1]) for r in rs]
    forest = forest_reg(trR, trY, n_trees=n_trees, maxd=maxd, seed=seed)
    mean = sum(trY) / len(trY)

    out = {}
    for f in fracs:
        pred, want = [], []
        for u in held_u:
            rs = units[u]
            k = max(2, int(round(f * len(rs))))
            prefix = rs[:k]
            # the memory columns are replayed over the prefix ONLY, which is
            # all an engine at that age would have produced
            feats, _ = featurise(prefix, kept, LEAKS, (), ())
            pred.append(predict(forest, feats[-1]))
            want.append(min(CAP, rs[-1][1] - prefix[-1][1]))
        out[f] = (pred, want, mean)
    return out, len(fit_u), len(held_u)


def run(train_path, fracs=FRACS, seeds=3, n_trees=9, maxd=8, val_frac=0.3):
    raw = load_cmapss(train_path)
    units_raw = by_unit(raw)
    names = names_for(len(raw[0]))

    got = {f: [] for f in fracs}
    nfit = nheld = 0
    for s in range(seeds):
        res, nfit, nheld = one_seed(units_raw, names, s, fracs, n_trees,
                                    maxd, val_frac)
        for f in fracs:
            got[f].append(res[f])
        print(f"    seed {s}: {nfit} fit / {nheld} held-out engines",
              flush=True)

    print(f"\n  {len(units_raw)} engines, {nfit} fit / {nheld} held out per "
          f"seed, {seeds} seed(s)")
    print(f"  depth {maxd}, {n_trees} trees, per-condition normalised, "
          f"RUL capped at {CAP}")
    print(f"  each held-out engine is truncated at a fraction of its OWN "
          f"life and\n  predicted there, with the memory columns replayed "
          f"over that prefix only\n")

    def avg(f, fn, which):
        xs = []
        for pred, want, mean in got[f]:
            base = ([CAP] * len(want) if which == "cap"
                    else [mean] * len(want) if which == "mean" else pred)
            xs.append(fn(base, want))
        return sum(xs) / len(xs)

    print(f"  {'life seen':>10}{'at cap':>8}{'true RUL':>10}"
          f"{'RMSE':>9}{'vs cap':>9}{'vs mean':>9}{'MAE':>8}{'NASA':>8}")
    rows = []
    for f in fracs:
        pred, want, mean = got[f][0]
        capped = sum(1 for y in want if y >= CAP) / len(want)
        truth = sum(sum(w for w in want) / len(want)
                    for _, want, _ in got[f]) / len(got[f])
        r = avg(f, rmse, "model")
        rc = avg(f, rmse, "cap")
        rm = avg(f, rmse, "mean")
        na = sum(nasa_score(p, w) for p, w, _ in got[f]) / len(got[f])
        rows.append((f, r, rc, rm))
        print(f"  {f:>9.0%}{capped:>8.0%}{truth:>10.0f}"
              f"{r:>9.1f}{rc:>9.1f}{rm:>9.1f}"
              f"{avg(f, mae, 'model'):>8.1f}{na:>8.0f}")

    # -- a picture, because the shape is the point ------------------------
    hi = max(max(r, rc, rm) for _, r, rc, rm in rows)
    print(f"\n  RMSE by fraction of life observed")
    print(f"  # is the model, | is the better control. A # bar reaching "
          f"PAST the |\n  means the model is losing to guessing at that age.")
    for f, r, rc, rm in rows:
        best = min(rc, rm)
        w = int(round(40 * r / hi))
        wb = min(int(round(40 * best / hi)), 40)
        bar = ["#" if i < w else "." for i in range(max(w, wb + 1))]
        bar[wb] = "|"          # always visible, even when the control wins
        print(f"  {f:>5.0%} {''.join(bar)} {r:.1f}  (control {best:.1f})")

    # -- what it says -----------------------------------------------------
    useful = [f for f, r, rc, rm in rows if r < min(rc, rm) - 1.0]
    print()
    if not useful:
        print(f"  THE MODEL NEVER BEATS THE CONTROLS at any age, which means "
              f"this split\n  is too small or the features are not working. "
              f"Check before reading\n  anything else here.")
    else:
        first = min(useful)
        print(f"  FIRST USEFUL AT {first:.0%} OF LIFE. Below that the model "
              f"does not beat\n  guessing the cap or the mean by more than "
              f"an RMSE, so a prediction\n  there is not worth acting on.")
        last_r = rows[-1][1]
        first_r = next(r for f, r, _, _ in rows if f == first)
        print(f"  From {first:.0%} to 100% of life the error moves "
              f"{first_r:.1f} -> {last_r:.1f} RMSE.")

    caps = [(f, sum(1 for y in got[f][0][1] if y >= CAP) / len(got[f][0][1]))
            for f in fracs]
    flat = [f for f, c in caps if c < 0.5]
    if flat:
        print(f"\n  THE CAP ANSWERS FOR MOST ENGINES BELOW {min(flat):.0%} "
              f"OF LIFE. At or above it,\n  most engines are genuinely "
              f"within {CAP} cycles of failing, so the model has\n  to do "
              f"the work itself. cmapss.py asserts this in its docstring to "
              f"justify\n  CAP = {CAP}; this is the measurement, and the "
              f"'vs cap' column is why a low\n  RMSE at a young age is not "
              f"evidence of anything.")
    print(f"\n  The NASA column is the one to read for deciding WHEN: it "
          f"punishes\n  over-stating remaining life far harder than "
          f"under-stating it, so it is\n  the cost of having waited too "
          f"long.\n")


if __name__ == "__main__":
    f = sys.argv[1:]
    a, i = [], 0
    while i < len(f):
        if f[i].startswith("--"):
            i += 2
        else:
            a.append(f[i])
            i += 1
    if not a:
        raise SystemExit(__doc__)

    def opt(name, cast, default):
        return cast(f[f.index(name) + 1]) if name in f else default

    run(a[0], seeds=opt("--seeds", int, 3), n_trees=opt("--trees", int, 9),
        maxd=opt("--maxd", int, 8), val_frac=opt("--heldout", float, 0.3))
