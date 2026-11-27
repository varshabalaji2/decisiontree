"""Train the neuron tree and forest on a real CSV.

    python3 drilling.py xai_drilling.csv                 look at the columns
    python3 drilling.py xai_drilling.csv "Main Failure"   train on that label

Add columns to skip with --drop:

    python3 drilling.py xai_drilling.csv "Main Failure" --drop ID,Flank Wear Failure
"""

import csv
import random
import sys

from neuron_tree import (fit_scale, apply_scale, build, use,
                         build_forest, use_forest, size)


# ------------------------------------------------------------------- loading

def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw = [r for r in reader if any(c.strip() for c in r)]
    return header, raw


def inspect(header, raw):
    """Print every column so you can decide what to predict and what to drop."""
    print(f"\n  {len(raw)} rows, {len(header)} columns\n")
    print(f"  {'#':>3}  {'column':<28}{'kind':<9}{'distinct':>9}   examples")
    print("  " + "-" * 74)
    for i, name in enumerate(header):
        vals = [r[i] for r in raw]
        distinct = sorted(set(vals))
        try:
            [float(v) for v in vals]
            kind = "number"
        except ValueError:
            kind = "text"
        note = ""
        if len(distinct) == len(vals):
            note = "  <-- unique per row, an ID: DROP"
        elif len(distinct) == 1:
            note = "  <-- constant, useless"
        elif len(distinct) == 2:
            note = "  <-- could be the label"
        print(f"  {i:>3}  {name:<28}{kind:<9}{len(distinct):>9}   "
              f"{', '.join(distinct[:3])[:24]}{note}")
    print("\n  Anything that is a CONSEQUENCE of the label rather than a cause")
    print("  must be dropped, or the tree will just read the answer off it.\n")


def to_numbers(header, raw, label_col, drop=()):
    """Every kept column becomes a float; text columns get numbered."""
    if label_col not in header:
        raise SystemExit(f"  no column called {label_col!r}. "
                         f"Columns are: {', '.join(header)}")
    skip = set(drop) | {label_col}
    keep = [i for i, n in enumerate(header) if n not in skip]
    codes = {}
    for i in keep:
        vals = [r[i] for r in raw]
        try:
            [float(v) for v in vals]
        except ValueError:
            codes[i] = {v: k for k, v in enumerate(sorted(set(vals)))}
    names = [header[i] + (" (coded)" if i in codes else "") for i in keep]
    rows = [[float(codes[i][r[i]]) if i in codes else float(r[i]) for i in keep]
            for r in raw]
    labels = [r[header.index(label_col)] for r in raw]
    return rows, labels, names


# ------------------------------------------------------------------ scoring

def split(rows, labels, frac=0.7, seed=0):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    cut = int(len(order) * frac)
    pick = lambda idx: ([rows[i] for i in idx], [labels[i] for i in idx])
    return pick(order[:cut]), pick(order[cut:])


def report(name, predict, rows, labels, positive):
    right = sum(1 for r, l in zip(rows, labels) if predict(r) == l)
    tp = sum(1 for r, l in zip(rows, labels) if predict(r) == positive and l == positive)
    fp = sum(1 for r, l in zip(rows, labels) if predict(r) == positive and l != positive)
    fn = sum(1 for r, l in zip(rows, labels) if predict(r) != positive and l == positive)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"  {name:<16} accuracy {right/len(labels):>6.1%}   "
          f"precision {prec:>6.1%}   recall {rec:>6.1%}")


def train(path, label_col, drop, maxd=6, n_trees=15):
    header, raw = load_csv(path)
    rows, labels, names = to_numbers(header, raw, label_col, drop)

    counts = {v: labels.count(v) for v in sorted(set(labels))}
    biggest = max(counts, key=counts.get)
    positive = min(counts, key=counts.get)
    print(f"\n  predicting {label_col!r} from {len(names)} columns, "
          f"{len(rows)} rows")
    print(f"  label counts: {counts}")
    print(f"  always guessing {biggest!r} would score "
          f"{counts[biggest]/len(labels):.1%} accuracy and 0% recall\n")

    (trR, trL), (teR, teL) = split(rows, labels)
    scale = fit_scale(trR)
    trN, teN = apply_scale(trR, scale), apply_scale(teR, scale)

    t = build(trN, trL, maxd=maxd)
    f = build_forest(trN, trL, n_trees=n_trees, maxd=maxd)

    print(f"  one tree ({size(t)} nodes)")
    report("    training", lambda r: use(t, r), trN, trL, positive)
    report("    never seen", lambda r: use(t, r), teN, teL, positive)

    print(f"\n  forest of {n_trees}, on never-seen rows, by vote threshold")
    majority = n_trees // 2 + 1
    for need in sorted({majority, n_trees // 3, n_trees // 5, 2, 1}, reverse=True):
        if not 1 <= need <= n_trees:
            continue
        tag = f"    needs {need}/{n_trees}" + ("  (majority)" if need == majority else "")
        report(tag, lambda r, k=need: use_forest(f, r, positive, k),
               teN, teL, positive)

    solo = sorted(
        sum(1 for r, l in zip(teN, teL) if use(one, r) == positive == l)
        / max(1, teL.count(positive)) for one in f)
    print(f"\n  recall of each tree on its own, worst to best:")
    print("    " + "  ".join(f"{v:.0%}" for v in solo))
    print("    (spread out = the trees are specialists, so lower the threshold)\n")
    return t, f, names


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    if len(sys.argv) < 3:
        inspect(*load_csv(path))
        raise SystemExit(0)
    drop = ()
    if "--drop" in sys.argv:
        i = sys.argv.index("--drop")
        drop = tuple(x.strip() for x in " ".join(sys.argv[i + 1:]).split(","))
        del sys.argv[i:]
    train(path, sys.argv[2], drop)