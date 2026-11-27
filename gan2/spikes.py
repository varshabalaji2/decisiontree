"""The one part of a LIF neuron that is not a threshold test: the RESET.

    python3 spikes.py cmapss/train_FD002.txt cmapss/test_FD002.txt \
                      cmapss/RUL_FD002.txt
    python3 spikes.py ... --seeds 3 --maxd 8 --only reset,noreset

THE PREMISE OF THIS PROJECT IS THAT A LIF NEURON CAN BE A TREE SPLIT, AND
THAT IS SETTLED: IT CANNOT, BECAUSE IT COLLAPSES.

    v = v + k*(drive - v)   fed a constant settles at that constant, so the
                            20-step integration loop cancels exactly against
                            the threshold and produces a bit-identical tree.

    integrating a weighted sum equals the weighted sum of the integrated
    columns -- verified to 1e-16 -- so a FIXED LEAK IS A RENAME OF THE
    WEIGHTS and nothing more.

    w.x + b >= theta is the same test as w.x >= theta - b, so the bias term
    is inert too.

Four knobs, all of them absorbed into one number an ordinary oblique split
already has. That is the project's largest finding and it is a negative one
about its own premise.

BUT ONE PIECE OF A LIF NEURON SURVIVES THAT ARGUMENT, AND IT IS THE PIECE
THIS FILE IS ABOUT. After it fires, the neuron RESETS TO ZERO:

    v = v + k*(x_t - v)
    if v >= theta:  spike;  v = 0

An exponential moving average is a LINEAR filter -- a weighted sum of past
inputs -- which is exactly why the memory columns turned out to be a
reparameterisation. A leaky integrator WITH RESET is not a linear functional
of the input sequence at all. It cannot be written as a weighted sum of
EMAs, at any leak, ever. The reset is a genuine nonlinearity and it is the
only part of the neuron that no algebra removes.

AND IT PUTS THE NEURON ON THE PAYING SIDE OF THIS PROJECT'S RULE. The rule,
from thirteen measured nulls: a stage above the tree pays if it computes
ACROSS ROWS something the tree cannot otherwise reach, or if it changes the
TARGET; it loses if it only decides who answers. The tree sees ONE row --
an engine's last cycle. Its history is gone. How many times the leaky
integral of sensor 11 crossed threshold over two hundred past cycles is not
in that row and no column the tree can split on recovers it, because the
reset makes the quantity path-dependent.

FOUR ARMS, AND THE SECOND CONTROL IS THE ONE THAT DECIDES IT:

  memory only             the floor: per-condition normalised sensors plus
                          the EMA columns, which is the model the project
                          reports.

  + LIF spikes            the same columns plus spike features WITH reset.

  + crossings, no reset   spike features computed from the SAME leaky
                          integral with the reset removed, so it counts how
                          many past cycles the plain EMA sat above the
                          threshold. This is still a historical count the
                          tree cannot reach, so it is not a null arm -- it
                          isolates the RESET and nothing else. If this arm
                          matches the reset arm, the nonlinearity bought
                          nothing and the neuron is inert for a THIRD
                          reason. If the reset arm wins, the neuron is
                          finally doing something irreducible.

  + raw exceedances       the same features with NO INTEGRAL: a count of
                          the past cycles where the sensor itself was past
                          the threshold. This peels the leak off the way the
                          no-reset arm peels off the reset. If it ties the
                          no-reset arm, neither the leak nor the reset is
                          doing the work and the whole effect is the
                          plainest historical count there is.

  + noise columns         the same NUMBER of columns, filled with noise.
                          127 -> 235 columns was measured to make this model
                          WORSE, because a greedy split search fooled by
                          more candidates picks a fluke more often. Any new
                          feature here has to beat that headwind, and this
                          arm measures the headwind on its own.

TWO FEATURES PER (SENSOR, LEAK, THRESHOLD), AND THE CHOICE OF WHICH IS NOT
COSMETIC:

  rate      spikes so far divided by cycles so far.
  gap       cycles since the last spike.

RAW SPIKE COUNT IS DELIBERATELY NOT USED, and the reason is a null result
this project already paid for. A raw count only ever increases, so it is
monotone in engine age -- and the cycle number is already a column the tree
can split on. The fleet-age experiment failed for exactly that reason: it
offered the tree information the cycle column had already given it, and
returned -0.6 RMSE. A rate divides the age back out; a gap is not monotone
in age at all.

EVERYTHING IS CAUSAL AND EVERY STATISTIC COMES FROM TRAINING ENGINES ONLY.
Row t's features are computed from cycles 0..t, so they are available at the
moment that row was recorded. The thresholds are multiples of each
(sensor, leak) integral's spread measured on the TRAINING fleet, never on
the test engines -- the same discipline as regime_stats, and the reason this
file does not repeat FINDINGS' mistake of choosing maxd by watching the
official test RMSE.

WHY A LEAK CAN COST YOU, since that is the measured result and not an
obvious one. v = (1-k)v + k*x unrolls to a weighted average of the past, so
the newest reading carries weight k and the rest is history: at k=0.02 a
single 5-sigma spike moves the integral by 0.100 while a 2-sigma shift that
PERSISTS reaches 1.405. Thresholding that can only count sustained
departures. And SPIKE_LEAKS are both already in conditions.LEAKS, so the
floor arm holds the current value of the very filters this file integrates
-- which makes a slow integral's exceedance count close to collinear with a
column the tree already has, while a transient excursion eighty cycles back
is in no column at any leak. The EMA count was already reachable; the raw
count was not.

THE SENSORS ARE CHOSEN BY A STATED RULE, NOT BY HAND: the TOP_N columns with
the largest absolute correlation with RUL over the training rows. Picking
them after seeing which ones worked would be the same error as tuning on the
test set, in a file written to test one clean claim.
"""

import random
import sys

from cmapss import (load_cmapss, by_unit, drop_dead_columns, forest_reg,
                    predict, nasa_score, CAP)
from conditions import LEAKS, regime_stats, normalise, names_for, featurise

SPIKE_LEAKS = (0.02, 0.10)      # one slow integral, one fast
# One likely-dense and one likely-sparse threshold. These were (1.5, 2.5)
# and on a trial run the 2.5 setting fired for 4 engines in 40 -- a constant
# column dressed as a feature. The liveness table says whether the grid
# suits the data; --mults moves it without editing anything.
MULTS = (0.5, 1.5)              # threshold, in spreads of that integral
TOP_N = 4                       # sensors, by |correlation with RUL|


# ------------------------------------------------------------- the neuron

def integrate(xs, k, theta, reset, raw=False):
    """Walk one sensor's history once, emitting (rate, gap) at every cycle.

    reset=True is the LIF neuron: crossing the threshold fires and sets the
    membrane back to zero, which is the step no weighted sum can express.
    reset=False leaves the membrane alone, so it is a plain EMA and the
    'spikes' are just the cycles it happened to sit above the threshold.

    Returns a list of (rate, gap), one per cycle, each using only the
    cycles up to and including that one.
    """
    v, count, last, out = 0.0, 0, None, []
    for t, x in enumerate(xs):
        # raw=True skips the integral entirely: the membrane IS the reading,
        # so this counts how many past cycles the sensor itself was extreme.
        # No leak, and a reset would be meaningless because v is overwritten
        # on the next step anyway.
        v = x if raw else v + k * (x - v)
        if v >= theta:
            count += 1
            last = t
            if reset and not raw:
                v = 0.0
        out.append((count / (t + 1.0),
                    float(t + 1 if last is None else t - last)))
    return out


def spike_stats(units, cols, leaks=SPIKE_LEAKS):
    """The spread of each (sensor, leak) leaky integral over the TRAINING
    fleet, so a threshold can be stated in units of it.

    Measured with the reset OFF on purpose. With the reset on, the spread
    depends on the threshold, and a threshold defined in units of a spread
    that depends on the threshold is circular. The no-reset integral is the
    plain EMA, whose spread is a property of the data alone.

    The textbook figure for white noise, sqrt(k/(2-k)), is not usable here:
    a sensor trace is strongly autocorrelated, so its EMA wanders far more
    than that formula predicts, and a threshold built on it would almost
    never fire.
    """
    out = {}
    for c in cols:
        for k in leaks:
            vals = []
            for rs in units.values():
                v = 0.0
                for r in rs:
                    v += k * (float(r[c]) - v)
                    vals.append(v)
            m = sum(vals) / len(vals)
            var = sum((x - m) ** 2 for x in vals) / len(vals)
            out[(c, k)] = max(var ** 0.5, 1e-9)
        # the raw column's own spread, for the no-integral arm. Close to 1
        # because the columns are z-scored per operating condition, but
        # measured rather than assumed.
        raw = [float(r[c]) for rs in units.values() for r in rs]
        mr = sum(raw) / len(raw)
        vr = sum((x - mr) ** 2 for x in raw) / len(raw)
        out[(c, None)] = max(vr ** 0.5, 1e-9)
    return out


def spike_cols(rs, names, cols, sig, reset=True, leaks=SPIKE_LEAKS,
               mults=MULTS, raw=False):
    """Spike features for ONE engine, replayed over its history in order.

    raw=True drops the leak loop, because there is no integral to leak --
    so this arm produces FEWER columns than the others (one per sensor per
    threshold, not per sensor per leak per threshold). That asymmetry is
    left in and reported rather than padded out, because padding would mean
    duplicating columns, and a duplicated column gets offered to the split
    search twice. It makes the reading asymmetric but not ambiguous: the raw
    arm pays a smaller widening penalty, so if it TIES the integral arms it
    is in fact slightly favoured and the integral is adding nothing, while
    if it LOSES the gap is a lower bound on what the integral is worth.
    """
    extra = [[] for _ in rs]
    enames = []
    tag = "rw" if raw else ("sp" if reset else "cr")
    for c in cols:
        xs = [float(r[c]) for r in rs]
        for k in ((None,) if raw else leaks):
            for m in mults:
                series = integrate(xs, k or 0.0, m * sig[(c, k)], reset,
                                   raw=raw)
                for row, (rate, gap) in zip(extra, series):
                    row.append(rate)
                    row.append(gap)
                enames.append(f"{names[c]}_{tag}{k}x{m}_rate")
                enames.append(f"{names[c]}_{tag}{k}x{m}_gap")
    return extra, enames


def fire_report(units, cols, sig, leaks=SPIKE_LEAKS, mults=MULTS):
    """How often each neuron actually fires, over the training fleet.

    THIS EXISTS BECAUSE A DEAD NEURON LOOKS EXACTLY LIKE A NULL RESULT. If
    the threshold is too high the rate is 0 for every engine, the columns
    are constant, and the arm reports "no difference" while having measured
    nothing -- which is the failure that already cost this project one run
    of the sliced-modules experiment. A saturated neuron, firing on nearly
    every cycle, is degenerate the same way from the other end.

    Returns {(col, leak, mult): (mean final rate, fraction of engines that
    ever fired)}.
    """
    out = {}
    for c in cols:
        series = {u: [float(r[c]) for r in rs] for u, rs in units.items()}
        for k in leaks:
            for m in mults:
                rates, ever = [], 0
                for xs in series.values():
                    last = integrate(xs, k, m * sig[(c, k)], True)[-1]
                    rates.append(last[0])
                    ever += 1 if last[0] > 0 else 0
                out[(c, k, m)] = (sum(rates) / len(rates),
                                  ever / len(series))
    return out


def noise_cols(rs, n, seed):
    """n columns of noise, to price the widening of the feature set on its
    own. Drawn per row from a seeded generator so the arm is reproducible."""
    rng = random.Random(seed + 9191)
    return ([[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in rs],
            [f"noise{i}" for i in range(n)])


# ------------------------------------------------------------ the pipeline

def pick_sensors(units, kept_names, top_n=TOP_N):
    """The top_n columns most correlated with RUL over the TRAINING rows.

    A stated rule, applied before anything is measured. Column 0 is the
    engine's serial number and column 1 is its cycle; neither is a sensor,
    and the serial number can only ever be memorised.
    """
    xs = {c: [] for c in range(2, len(kept_names))}
    ys = []
    for rs in units.values():
        last = rs[-1][1]
        for r in rs:
            ys.append(min(CAP, last - r[1]))
            for c in xs:
                xs[c].append(float(r[c]))
    my = sum(ys) / len(ys)
    dy = [y - my for y in ys]
    vy = sum(d * d for d in dy) ** 0.5
    score = {}
    for c, col in xs.items():
        mx = sum(col) / len(col)
        dx = [x - mx for x in col]
        vx = sum(d * d for d in dx) ** 0.5
        score[c] = 0.0 if vx * vy == 0 else abs(
            sum(a * b for a, b in zip(dx, dy)) / (vx * vy))
    best = sorted(score, key=lambda c: -score[c])[:top_n]
    return sorted(best), {c: score[c] for c in best}


def build_arm(units, kept_names, cols, sig, kind, seed, n_extra=0,
              mults=MULTS, leaks=LEAKS):
    """Training rows and labels for one arm.

    leaks=() drops the EMA memory columns from the base, which is the whole
    point of --nomemory: it is the control that separates the two
    explanations for why raw counting beat the integral.
    """
    R, Y = [], []
    for rs in units.values():
        last = rs[-1][1]
        base, _ = featurise(rs, kept_names, leaks, (), ())
        add = _extra(rs, kept_names, cols, sig, kind, seed, n_extra, mults)
        R += [list(b) + list(e) for b, e in zip(base, add)]
        Y += [min(CAP, last - r[1]) for r in rs]
    return R, Y


def _extra(rs, kept_names, cols, sig, kind, seed, n_extra, mults=MULTS):
    if kind == "floor":
        return [[] for _ in rs]
    if kind == "noise":
        return noise_cols(rs, n_extra, seed)[0]
    extra, _ = spike_cols(rs, kept_names, cols, sig,
                          reset=(kind == "reset"), mults=mults,
                          raw=(kind == "raw"))
    return extra


def test_matrix(path, keep, kept_names, cols, sig, kind, seed, stats,
                n_extra=0, mults=MULTS, leaks=LEAKS):
    """One row per test engine: its LAST cycle, with the whole run replayed
    through the spike features first -- the integral needs the history."""
    units = normalise(by_unit(load_cmapss(path)), stats)
    out = []
    for u in sorted(units):
        rs = [[r[i] for i in keep] for r in units[u]]
        base, _ = featurise(rs, kept_names, leaks, (), ())
        add = _extra(rs, kept_names, cols, sig, kind, seed, n_extra, mults)
        out.append(list(base[-1]) + list(add[-1]))
    return out


def rmse(pred, want):
    return (sum((a - y) ** 2 for a, y in zip(pred, want)) / len(want)) ** 0.5


def mae(pred, want):
    return sum(abs(a - y) for a, y in zip(pred, want)) / len(want)


ARMS = (("memory only (floor)", "floor"),
        ("+ LIF spikes, WITH reset", "reset"),
        ("+ crossings, no reset", "noreset"),
        ("+ raw exceedances, no integral", "raw"),
        ("+ noise columns", "noise"))


def run(train_path, test_path, rul_path, n_trees=9, maxd=8, seeds=3,
        only=None, mults=MULTS, memory=True):
    raw = load_cmapss(train_path)
    units_raw = by_unit(raw)
    names = names_for(len(raw[0]))
    stats, counts = regime_stats(units_raw)
    units = normalise(units_raw, stats)
    # drop_dead_columns already returns the units with the dead columns
    # gone; `keep` is kept only so the TEST engines can be sliced the same
    # way, from the same training-derived decision.
    units, kept_names, keep = drop_dead_columns(units, names)
    want = [min(CAP, float(x)) for x in open(rul_path).read().split()]

    leaks = LEAKS if memory else ()
    cols, score = pick_sensors(units, kept_names)
    sig = spike_stats(units, cols)
    n_extra = len(cols) * len(SPIKE_LEAKS) * len(mults) * 2

    print(f"\n  {len(counts)} operating conditions, normalised per condition")
    print(f"  {len(units)} training engines, {len(want)} test engines")
    print(f"  depth {maxd}, {n_trees} trees, {seeds} seed(s)")
    if memory:
        print(f"  the base carries the EMA memory columns at leaks {LEAKS}")
    else:
        print(f"  --nomemory: THE EMA MEMORY COLUMNS ARE GONE from every "
              f"arm's base. Raw sensors\n  only, so nothing in the base "
              f"already holds a smoothed version of a sensor.")
    print(f"  spike leaks {SPIKE_LEAKS}, thresholds {mults} x the training "
          f"spread of that integral")
    print(f"  sensors by |corr with RUL| on TRAINING rows: "
          f"{', '.join(f'{kept_names[c]} {score[c]:.2f}' for c in cols)}")
    print(f"  {n_extra} columns added by each non-floor arm "
          f"(2 features x {len(cols)} sensors x {len(SPIKE_LEAKS)} leaks "
          f"x {len(mults)} thresholds)")

    fire = fire_report(units, cols, sig, mults=mults)
    alive = [key for key, (r, ev) in fire.items() if ev >= 0.25 and r < 0.5]
    print(f"\n  do the neurons actually fire? (training fleet, with reset)")
    print(f"  {'sensor':>8}{'leak':>8}{'x sigma':>9}{'mean rate':>11}"
          f"{'engines fired':>16}")
    for (c, k, m), (r, ev) in sorted(fire.items()):
        mark = "" if (c, k, m) in alive else "  <- degenerate"
        print(f"  {kept_names[c]:>8}{k:>8}{m:>9}{r:>11.3f}{ev:>15.0%}{mark}")
    print(f"  {len(alive)} of {len(fire)} settings usable: a neuron that "
          f"never fires makes a CONSTANT\n  column and would report "
          f"'no difference' having measured nothing.\n")

    arms = [a for a in ARMS
            if only is None or a[1] in only or a[0] in only]
    got, gotn = {}, {}
    print(f"  {'':<30}{'RMSE':>14}{'MAE':>14}{'NASA':>10}{'cols':>7}")
    for tag, kind in arms:
        rs_, ms, ns, ncol = [], [], [], 0
        for s in range(seeds):
            R, Y = build_arm(units, kept_names, cols, sig, kind, s,
                             n_extra, mults, leaks)
            ncol = len(R[0])
            f = forest_reg(R, Y, n_trees=n_trees, maxd=maxd, seed=s)
            teR = test_matrix(test_path, keep, kept_names, cols, sig, kind,
                              s, stats, n_extra, mults, leaks)
            p = [predict(f, r) for r in teR]
            rs_.append(rmse(p, want))
            ms.append(mae(p, want))
            ns.append(nasa_score(p, want))
        got[kind] = rs_
        gotn[kind] = ns

        def show(xs):
            if len(xs) == 1:
                return f"{xs[0]:.1f}"
            return f"{sum(xs)/len(xs):.1f} +/-{(max(xs)-min(xs))/2:.1f}"

        print(f"  {tag:<30}{show(rs_):>14}{show(ms):>14}"
              f"{sum(ns)/len(ns):>10.0f}{ncol:>7}", flush=True)
        sys.stdout.flush()

    # -- what it says ----------------------------------------------------
    def paired(a, b):
        """Same seeds, same splits, so compare seed for seed rather than
        reading one mean against another's spread."""
        d = [x - y for x, y in zip(got[a], got[b])]
        return sum(1 for x in d if x < 0), len(d), sum(d) / len(d)

    print()
    if "floor" not in got:
        print(f"  NO FLOOR ARM, so nothing can be compared. Drop --only or "
              f"include 'floor'.\n")
        return

    if "noise" in got:
        nw, nn, navg = paired("noise", "floor")
        print(f"  THE PRICE OF WIDENING: {n_extra} columns of pure noise move "
              f"RMSE {navg:+.1f} on\n  {nw} of {nn} seeds. Every arm below "
              f"pays this before it earns anything, and it\n  is the "
              f"127 -> 235 selection-bias finding showing up again if it is "
              f"positive.")

    for kind, label in (("reset", "LIF SPIKES, WITH RESET"),
                        ("noreset", "CROSSINGS, NO RESET")):
        if kind in got:
            w, n, avg = paired(kind, "floor")
            print(f"\n  {label}: {avg:+.1f} RMSE against the floor on {w} of "
                  f"{n} seeds.")

    if "raw" in got and "noreset" in got:
        w, n, avg = paired("noreset", "raw")
        print(f"\n  THE LEAKY INTEGRAL ITSELF: {avg:+.1f} RMSE against "
              f"counting raw exceedances\n  with no integral at all, on "
              f"{w} of {n} seeds. The raw arm carries fewer columns\n  so "
              f"it pays a smaller widening penalty -- a tie therefore "
              f"favours it, and the\n  integral adding nothing.")
        if avg > 0.2:
            print(f"  SO THE INTEGRAL IS NOT INERT -- IT IS HARMFUL. "
                  f"Removing it is better by {avg:+.1f}\n  RMSE, and the "
                  f"raw arm gets there with FEWER columns, so it wins on "
                  f"content and\n  on width at the same time.")
            print(f"  WHY, IN TWO PARTS. v = (1-k)v + k*x gives the newest "
                  f"reading weight k, so at\n  k=0.02 a 5-sigma spike "
                  f"moves the integral by 0.1 while a SUSTAINED 2-sigma\n  "
                  f"shift reaches 1.4. The leak is a low-pass filter: blind "
                  f"to transients, open\n  to level shifts. And the floor "
                  f"arm already holds EMA columns at these very\n  leaks, "
                  f"so an exceedance count of a slow integral is close to "
                  f"collinear with a\n  column the tree already has, while "
                  f"a raw excursion from eighty cycles ago is\n  in no "
                  f"column at any leak. The EMA count was already "
                  f"REACHABLE; the raw count\n  was not -- which is this "
                  f"project's rule, not a vague appeal to extremes.")
        elif abs(avg) <= 0.2:
            print(f"  SO THE INTEGRAL IS INERT TOO. What the data rewards "
                  f"is the plainest possible\n  historical feature: how "
                  f"often this sensor was extreme, and how long ago. "
                  f"Neither\n  the leak nor the reset is carrying it.")
        else:
            print(f"  THE INTEGRAL EARNS ITS PLACE: {avg:+.1f} RMSE against "
                  f"raw counting, so smoothing\n  before thresholding is "
                  f"doing real work here.")

    if "reset" in got and "noreset" in got:
        w, n, avg = paired("reset", "noreset")
        print(f"\n  THE RESET ITSELF: {avg:+.1f} RMSE against the same "
              f"integral without it, on\n  {w} of {n} seeds. Both arms "
              f"carry a historical count the tree cannot\n  reach, so this "
              f"number is the nonlinearity alone.")
        if avg < -0.3 and w * 10 >= n * 8:
            print(f"\n  THE NEURON IS NOT INERT AFTER ALL. The reset is the "
                  f"one part of a LIF that\n  no algebra removes -- a leaky "
                  f"integral with reset is not a linear functional\n  of its "
                  f"input, so it cannot be written as a weighted sum of EMAs "
                  f"at any leak.\n  Here it earns {avg:+.1f} RMSE over the "
                  f"identical integral with the reset taken\n  out. The "
                  f"premise survives, as a TEMPORAL FEATURE EXTRACTOR and "
                  f"not as a\n  splitting rule: that version is still "
                  f"provably a threshold test.")
        elif avg > 0.3:
            print(f"\n  THE RESET HURTS. Removing it is better by "
                  f"{avg:+.1f} RMSE, so the nonlinearity\n  is adding "
                  f"variance and not signal. A spike count throws away HOW "
                  f"FAR above\n  the threshold the integral went, and the "
                  f"no-reset arm keeps that, which is the\n  likeliest "
                  f"reason.")
        elif not alive:
            print(f"\n  NO CONCLUSION: {avg:+.1f} RMSE, but NOT ONE "
                  f"THRESHOLD SETTING FIRED USEFULLY,\n  so every spike "
                  f"column was near-constant and the arms above compared "
                  f"the floor\n  against itself plus padding. Lower them "
                  f"(--mults 0.5,1.0) and run it again.\n  This says "
                  f"nothing about the reset.")
        else:
            nw, nn, navg = paired("noreset", "floor")
            tot = navg + avg
            share = 0.0 if abs(tot) < 1e-9 else avg / tot
            print(f"\n  THE RESET IS WORTH {avg:+.1f} RMSE on {w} of {n} "
                  f"seeds -- consistent in sign but\n  small. The same "
                  f"integral WITHOUT it already gets {navg:+.1f} against "
                  f"the floor, so the\n  reset accounts for about "
                  f"{abs(share):.0%} of the total and a plain historical "
                  f"count\n  accounts for the rest.")
            print(f"\n  SO THE NEURON IS NOT WHAT IS WORKING -- THE HISTORY "
                  f"IS. How often a sensor\n  crossed a threshold in the "
                  f"past is information the tree cannot reach from one\n  "
                  f"row, and that is the part being rewarded. The "
                  f"nonlinearity that makes a LIF\n  irreducible rides "
                  f"along for a fraction of the effect. That is a FOURTH "
                  f"null\n  about the neuron and a WIN for temporal "
                  f"features, and they are not the same\n  claim: the "
                  f"integrate-and-fire framing is doing almost none of the "
                  f"work.")

    if "reset" in got:
        w, n, avg = paired("reset", "floor")
        if avg >= -0.3 and "noise" in got:
            nw, nn, navg = paired("noise", "floor")
            if navg > 0.3:
                print(f"\n  AND READ THAT AGAINST THE NOISE ARM. Widening by "
                      f"{n_extra} columns costs {navg:+.1f}\n  RMSE by "
                      f"itself, so a spike arm at {avg:+.1f} is roughly "
                      f"breaking even on content\n  while paying for the "
                      f"width. The features are not worthless -- they are "
                      f"not\n  worth their room in a 127-column greedy "
                      f"search.")
    # RMSE and NASA can move in OPPOSITE directions, and if they do, saying
    # "the features help" without naming the metric is a half-truth.
    if "floor" in gotn:
        for kind, label in (("reset", "LIF spikes"),
                            ("noreset", "crossings, no reset")):
            if kind not in gotn:
                continue
            dr = (sum(got[kind]) / len(got[kind])
                  - sum(got["floor"]) / len(got["floor"]))
            dn = (sum(gotn[kind]) / len(gotn[kind])
                  - sum(gotn["floor"]) / len(gotn["floor"]))
            if dr < -0.1 and dn > 10:
                print(f"\n  AND {label.upper()} TRADES THE WRONG WAY ON "
                      f"NASA: {dr:+.1f} RMSE but {dn:+.0f} NASA.\n  Lower "
                      f"squared error with a HIGHER asymmetric cost means "
                      f"the predictions moved\n  UP on average -- more "
                      f"over-statement of remaining life, which is the "
                      f"expensive\n  direction. An RMSE win and a NASA "
                      f"loss, and it has to be reported as both.")

    # Which arm actually won? Say it plainly, because the three paired
    # comparisons above each answer one sub-question and none of them says
    # outright which model a person should use.
    live = {k: sum(v) / len(v) for k, v in got.items()}
    if len(live) > 1:
        best = min(live, key=live.get)
        label = dict((k, t) for t, k in ARMS)[best]
        if best not in ("floor", "noise"):
            gap = live["floor"] - live[best] if "floor" in live else 0.0
            print(f"\n  BEST ARM: {label.strip('+ ')}, at "
                  f"{live[best]:.1f} RMSE against the floor's "
                  f"{live.get('floor', float('nan')):.1f}\n  -- "
                  f"{gap:.1f} RMSE, and that is the model to carry forward.")
        elif best == "floor":
            print(f"\n  BEST ARM: the floor. Nothing added here beat doing "
                  f"nothing, so the widening\n  penalty ate whatever the "
                  f"features were worth.")

    if not memory and "raw" in got and "noreset" in got:
        w, n, avg = paired("noreset", "raw")
        print(f"\n  READ THIS AGAINST THE RUN WITH MEMORY COLUMNS. With "
              f"them, raw counting beat the\n  integral and there were two "
              f"explanations: either the leak destroys a signal\n  that is "
              f"transient, or the integral's count was close to collinear "
              f"with EMA\n  columns the base already held. Those agree on "
              f"FD002 and disagree here, because\n  here there is no EMA "
              f"column for anything to duplicate.")
        if avg < -0.2:
            print(f"\n  COLLINEARITY WAS THE CAUSE. Strip the memory "
                  f"columns and the integral WINS by\n  {-avg:.1f} RMSE. "
                  f"It was never the weaker feature -- it was the REDUNDANT "
                  f"one, and\n  'the leak destroys the extremes' is the "
                  f"wrong explanation for the earlier\n  result.")
        elif avg > 0.2:
            print(f"\n  THE SIGNAL IS GENUINELY TRANSIENT. Even with no EMA "
                  f"column to duplicate, raw\n  counting still beats the "
                  f"integral by {avg:.1f} RMSE. So the leak is not "
                  f"redundant\n  here, it is destructive: it low-passes "
                  f"away excursions that carry the signal.")
        else:
            print(f"\n  NEITHER EXPLANATION SURVIVES CLEANLY: {avg:+.1f} "
                  f"RMSE with no memory columns to\n  duplicate, so the "
                  f"two features are near-equivalent once the redundancy is "
                  f"gone.\n  That favours collinearity over destruction, "
                  f"but weakly -- raise --seeds.")

    print(f"\n  Every feature here is causal, computed from cycles 0..t "
          f"only, and every\n  threshold comes from the training fleet's own "
          f"spread. Nothing was chosen by\n  looking at the official test "
          f"engines.\n")


if __name__ == "__main__":
    f = sys.argv[1:]
    # --nomemory takes NO value, so the positional walk has to skip one token
    # for it and two for every other flag. Getting this wrong is not
    # cosmetic: in conditions.py the same slip read "--seeds 3" as a
    # positional maxd=3 and silently ran a whole experiment at depth 3.
    BARE = ("--nomemory",)
    a, i = [], 0
    while i < len(f):
        if f[i].startswith("--"):
            i += 1 if f[i] in BARE else 2
        else:
            a.append(f[i])
            i += 1
    if len(a) < 3:
        raise SystemExit(__doc__)

    def opt(name, cast, default):
        return cast(f[f.index(name) + 1]) if name in f else default

    sel = opt("--only", str, None)
    mu = opt("--mults", str, None)
    run(a[0], a[1], a[2], n_trees=opt("--trees", int, 9),
        maxd=opt("--maxd", int, 8), seeds=opt("--seeds", int, 3),
        only=None if sel is None else set(
            ["floor"] + [x.strip() for x in sel.split(",")]),
        mults=MULTS if mu is None else tuple(
            float(x) for x in mu.split(",")),
        memory="--nomemory" not in f)
