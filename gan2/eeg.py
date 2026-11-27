"""Does the reset beat three lines of sign-counting, on rhythm data?

    python3 eeg.py epileptic_seizure_recognition.csv      # the Kaggle CSV
    python3 eeg.py bonn_folder/                           # the original .txt sets
    python3 eeg.py data.csv --seeds 5 --maxd 6

Brain waves are named after their rhythms, and seizures and sleep stages are
defined by which rhythm dominates. So this is the first dataset in the project
where the thing the reset measures -- how often a signal crosses a threshold --
is the thing the labels are about. Requirement 2 from FINDINGS.md holds by
definition rather than by luck, which makes it the mechanism's best case and
therefore the fair place to retire it if it fails.

THE COMPARISON IS RIGGED AGAINST THE NEURON ON PURPOSE, because the obvious
comparisons all flatter it:

  loudness        mean, spread, extremes, mean absolute step. Seizure EEG is
                  much BIGGER, not only faster, so amplitude alone scores well
                  and any rhythm feature that cannot beat it has shown nothing.

  sign counting   how often the signal crosses each of nine levels. This is
                  the 1960s frequency estimator and it is three lines. It is
                  the real competitor: the neuron is a fancier version of the
                  same idea, so beating raw samples proves nothing and beating
                  THIS is the whole question.

  the reset       leaky-integrate, threshold, reset, count -- nine (leak,
                  threshold) pairs, the same nine features sign counting gets,
                  so neither arm wins on feature count.

Both rhythm arms run on the z-scored segment, so amplitude is removed from
them and lives only in the loudness arm. That keeps the arms measuring
separate things instead of three versions of "something changed".

PREDICTION, recorded before running: the reset comes out roughly equal to sign
counting, perhaps slightly ahead because the leak low-passes the signal and so
is less noise-sensitive. A weak positive, not a vindication.

TWO WAYS TO FOOL YOURSELF, both handled here:

  1. The Kaggle version chops each 4097-sample recording into 23 chunks of
     178. Chunks from one recording are nearly identical, so a random ROW
     split puts siblings in train and test and the score is meaningless. The
     row identifier encodes the source file, so the split here is BY
     RECORDING. If the identifier cannot be parsed the script says so loudly
     rather than quietly leaking.

  2. The two-class task (healthy vs seizure) is saturated -- almost anything
     scores near 100% and no method can be distinguished from another. This
     uses all five classes, where published results sit nearer 85-90%.
"""

import math
import os
import random
import statistics
import sys

from neuron_tree import (fit_scale, apply_scale, build, use, size,
                         build_forest, use_forest)

LEVELS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
PAIRS = ((0.05, 0.5), (0.05, 1.0), (0.05, 1.5),
         (0.15, 0.5), (0.15, 1.0), (0.15, 1.5),
         (0.40, 0.5), (0.40, 1.0), (0.40, 1.5))


# ------------------------------------------------------------------ loading

def _from_csv(path):
    """The Kaggle 'Epileptic Seizure Recognition' layout: an identifier, 178
    samples, then y in 1..5. Returns (signal, label, recording_id) triples."""
    out, bad = [], 0
    with open(path) as f:
        header = f.readline().strip().split(",")
        xcols = [i for i, h in enumerate(header) if h.strip().strip('"').upper()
                 .startswith("X") and h.strip().strip('"')[1:].isdigit()]
        ycol = next((i for i, h in enumerate(header)
                     if h.strip().strip('"').lower() == "y"), len(header) - 1)
        idcol = 0 if not xcols or xcols[0] > 0 else None

        for line in f:
            p = [c.strip().strip('"') for c in line.strip().split(",")]
            if len(p) < len(header):
                continue
            sig = [float(p[i]) for i in xcols] if xcols else \
                  [float(x) for x in p[1:ycol]]
            label = p[ycol]
            if idcol is None:
                rid, bad = f"row{len(out)}", bad + 1
            else:
                # "X21.V1.791" -> the recording is "791"; chunk 21 of it
                bits = p[idcol].split(".")
                rid = bits[-1] if len(bits) > 1 else p[idcol]
                if len(bits) < 2:
                    bad += 1
            out.append((sig, label, rid))
    return out, bad


def _from_folders(root):
    """The original Bonn layout: one folder per set, 100 .txt files each, one
    sample per line. The folder name is the class and the file is the
    recording."""
    out = []
    for d in sorted(os.listdir(root)):
        sub = os.path.join(root, d)
        if not os.path.isdir(sub):
            continue
        for fn in sorted(os.listdir(sub)):
            if not fn.lower().endswith(".txt"):
                continue
            with open(os.path.join(sub, fn)) as f:
                sig = [float(x) for x in f.read().split() if x.strip()]
            if sig:
                out.append((sig, d, f"{d}/{fn}"))
    return out, 0


def load(path):
    """Pick the layout from what is actually on disk, and if nothing is, say
    where to get the data rather than raising a traceback at the open()."""
    if os.path.isdir(path):
        return _from_folders(path)
    if not os.path.exists(path):
        here = [f for f in sorted(os.listdir("."))
                if f.lower().endswith(".csv")]
        dirs = [f for f in sorted(os.listdir("."))
                if os.path.isdir(f) and not f.startswith(".")]
        raise SystemExit(
            f"\n  {path!r} is not here, so the data has not been downloaded "
            f"yet.\n\n"
            f"  The preprocessed file is 11500 rows of 178 samples plus a "
            f"label 1-5.\n"
            f"  Search Kaggle for \"Epileptic Seizure Recognition\" -- free "
            f"account, and the\n"
            f"  file is widely mirrored there. The UCI copy was removed. The "
            f"original\n"
            f"  Bonn distribution (five folders of 100 .txt files) also works: "
            f"pass the\n"
            f"  folder that CONTAINS those five folders.\n\n"
            f"  Then pass whatever the file is really called, in quotes if it "
            f"has spaces.\n\n"
            f"  csv files in this directory: "
            f"{', '.join(here) if here else '(none)'}\n"
            f"  folders in this directory:   "
            f"{', '.join(dirs) if dirs else '(none)'}\n")
    return _from_csv(path)


# ----------------------------------------------------------------- features

def zscore(sig):
    m = sum(sig) / len(sig)
    v = sum((x - m) ** 2 for x in sig) / len(sig)
    s = math.sqrt(v) if v > 1e-12 else 1.0
    return [(x - m) / s for x in sig]


def loudness(sig):
    m = sum(sig) / len(sig)
    v = sum((x - m) ** 2 for x in sig) / len(sig)
    steps = [abs(b - a) for a, b in zip(sig, sig[1:])] or [0.0]
    return [m, math.sqrt(v), min(sig), max(sig), sum(steps) / len(steps)]


def sign_counts(z, levels=LEVELS):
    """How often the signal crosses each level. Three lines of 1960s signal
    processing, and the arm the neuron has to beat."""
    out = []
    for lv in levels:
        out.append(sum(1 for a, b in zip(z, z[1:])
                       if (a - lv) * (b - lv) < 0))
    return out


def reset_counts(z, pairs=PAIRS):
    """Leaky-integrate, threshold, reset, count. The neuron, on the z-scored
    signal so it cannot win on amplitude."""
    out = []
    for k, theta in pairs:
        v, n = 0.0, 0
        for x in z:
            v += k * (x - v)
            if v >= theta:
                n += 1
                v = 0.0
        out.append(n)
    return out


def featurise(sig, mode):
    z = zscore(sig)
    f = []
    if "loud" in mode:
        f += loudness(sig)
    if "zc" in mode:
        f += sign_counts(z)
    if "reset" in mode:
        f += reset_counts(z)
    return f


# --------------------------------------------------------------------- run

def split_by_recording(rows, frac=0.7, seed=0, by_row=False):
    """Chunks of one recording are near-identical, so they must not straddle
    the split. Group first, then divide the GROUPS.

    by_row=True deliberately does it WRONG, splitting individual chunks, so
    the cost of that mistake can be measured instead of assumed. Published
    five-class results on this data sit near 85-90% against roughly 65% here;
    some of that gap is wavelet features against these simple ones, and some
    of it may be this. The difference between the two splits says how much.
    """
    if by_row:
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        cut = int(len(order) * frac)
        return order[:cut], order[cut:], len(rows)
    groups = {}
    for i, (_, _, rid) in enumerate(rows):
        groups.setdefault(rid, []).append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    cut = int(len(keys) * frac)
    tr = [i for k in keys[:cut] for i in groups[k]]
    te = [i for k in keys[cut:] for i in groups[k]]
    return tr, te, len(keys)


def run(path, seeds=3, maxd=6, n_trees=9, frac=0.7, by_row=False):
    rows, bad = load(path)
    if not rows:
        raise SystemExit(f"  nothing loaded from {path}")

    tr_i, te_i, n_groups = split_by_recording(rows, frac, by_row=by_row)
    labels = [l for _, l, _ in rows]
    classes = sorted(set(labels))

    print(f"\n  {len(rows)} segments of {len(rows[0][0])} samples, "
          f"{len(classes)} classes {classes}")
    if by_row:
        print(f"  split BY ROW ON PURPOSE: {len(tr_i)} train / {len(te_i)} "
              f"test segments.\n  Chunks of one recording are on both sides, "
              f"so these scores are inflated\n  and exist only to measure by "
              f"how much.")
    else:
        print(f"  {n_groups} distinct recordings, split BY RECORDING: "
              f"{len(tr_i)} train / {len(te_i)} test segments")
    if bad:
        print(f"\n  WARNING: {bad} rows had no parseable recording id, so "
              f"their chunks may\n  straddle the split and inflate every "
              f"score below. Check the first\n  column of the file.")

    teL = [labels[i] for i in te_i]
    big = max(classes, key=teL.count)
    print(f"  always guessing {big!r}: {teL.count(big) / len(teL):.1%}\n")

    print(f"  {'':<34}{'cols':>6}{'tree':>14}{'forest':>16}")
    out = {}
    for mode, tag in (("loud", "loudness only"),
                      ("zc", "sign counting only"),
                      ("reset", "the reset only"),
                      ("loud+zc", "loudness + sign counting"),
                      ("loud+reset", "loudness + the reset"),
                      ("loud+zc+reset", "all three")):
        feats = [featurise(s, mode) for s, _, _ in rows]
        a = [feats[i] for i in tr_i]
        b = [feats[i] for i in te_i]
        trL = [labels[i] for i in tr_i]
        sc = fit_scale(a)
        aN, bN = apply_scale(a, sc), apply_scale(b, sc)

        t = build(aN, trL, maxd=maxd)
        t_acc = sum(1 for r, l in zip(bN, teL) if use(t, r) == l) / len(teL)

        f_acc = []
        for s in range(seeds):
            fo = build_forest(aN, trL, n_trees=n_trees, maxd=maxd, seed=s)
            f_acc.append(sum(1 for r, l in zip(bN, teL)
                             if use_forest(fo, r) == l) / len(teL))
        out[mode] = statistics.mean(f_acc)
        sp = (max(f_acc) - min(f_acc)) / 2
        print(f"  {tag:<34}{len(a[0]):>6}{t_acc:>13.1%}"
              f"{statistics.mean(f_acc):>11.1%} +/-{sp:.1%}")

    print()
    # Whether the reset beats LOUDNESS ALONE is not the question. A rhythm
    # feature need not out-score an amplitude feature by itself; what matters
    # is whether it adds something amplitude does not already carry, and
    # whether it does so better than three lines of sign counting. The first
    # version of this gate asked the wrong question and declared a clear win
    # "finished".
    on_loud = out["loud+reset"] - out["loud"]
    vs_zc_added = out["loud+reset"] - out["loud+zc"]
    on_both = out["loud+zc+reset"] - out["loud+zc"]
    alone = out["reset"] - out["zc"]

    print(f"  the reset added to loudness           {on_loud:+.1%}")
    print(f"  that, against sign counting added     {vs_zc_added:+.1%}")
    print(f"  the reset on top of loudness AND zc   {on_both:+.1%}")
    print(f"  the reset alone over sign counting    {alone:+.1%}")

    if on_loud <= 0.02:
        print(f"\n  FINISHED. The reset adds nothing to four numbers "
              f"describing how loud the\n  signal is, on the data built "
              f"around the rhythms it claims to measure.")
    elif vs_zc_added <= 0.02 and on_both <= 0.02:
        print(f"\n  REDUNDANT. The reset carries real rhythm information, but "
              f"plain sign\n  counting carries the same thing in three lines "
              f"and the reset adds\n  nothing on top of it. Real, and not "
              f"worth its complexity.")
    else:
        print(f"\n  THE RESET EARNS ITS PLACE. It adds to loudness, and it "
              f"adds more than\n  the three-line frequency estimator does, "
              f"which is the only comparison\n  that was ever in doubt.")
    print()


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    f = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)

    def opt(name, cast, default):
        return cast(f[f.index(name) + 1]) if name in f else default

    run(a[0], seeds=opt("--seeds", int, 3), maxd=opt("--maxd", int, 6),
        n_trees=opt("--trees", int, 9), by_row="--rowsplit" in f)
