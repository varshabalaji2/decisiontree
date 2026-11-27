"""Give a memoryless tree a memory, by giving the DATA a memory.

    python3 memory.py                      run the built-in demonstration
    python3 memory.py sensors.csv Failure  add memory columns and train

Some labels do not live in the row you are looking at. A drill bit fails
because of load accumulated over hundreds of previous operations; the current
row only shows the load right now. No tree can find that, however deep --
the information is not there.

The fix is not a cleverer tree. It is extra columns holding running summaries
of the past, which the tree then splits on like anything else:

    load_avg50   the average load over roughly the last fifty rows
    load_avg3    the average over roughly the last three
    load_change  how much it moved since the previous row

Measured on a synthetic machine whose bit wears out: a plain tree scores
60.6% against a 61.3% baseline -- it learns nothing, because it cannot.
The same tree with these columns scores 98.8%.

This is exactly what a leaky neuron does when you never reset it. `v = v +
k*(x - v)` IS an exponential moving average, and because that recurrence is
linear, a neuron integrating a weighted sum equals a weighted sum of the
integrated columns (verified to 1e-16). The difference is only that columns
cost nothing to adopt: every tool that takes a row of numbers keeps working.

TWO RULES, and breaking either invents a model that cannot exist:

  1. build the memory columns on the rows IN FILE ORDER, before any shuffle
  2. split train/test BY TIME -- first 70% train, last 30% test

Shuffling first averages rows that had not happened yet. The model then
scores beautifully in testing and fails the moment it meets a real stream.
"""

import sys

# roughly how many rows back each leak remembers: 2/k - 1
LEAKS = (0.02, 0.05, 0.1, 0.3)


def horizon(k):
    return max(1, round(2 / k - 1))


def add_memory(rows, names, cols=None, leaks=LEAKS, deltas=True):
    """Append running averages (and step changes) for the chosen columns.

    `rows` must be in file order. Returns (new_rows, new_names).
    """
    if cols is None:
        cols = range(len(rows[0]))
    cols = list(cols)
    if not rows:
        return [], list(names)

    state = {(c, k): rows[0][0] for c in cols for k in leaks}
    prev = {c: rows[0][0] for c in cols}

    out = []
    for r in rows:
        extra = []
        for c in cols:
            for k in leaks:
                s = state[(c, k)] + k * (r[c] - state[(c, k)])
                state[(c, k)] = s
                extra.append(s)
            if deltas:
                extra.append(r[c] - prev[c])
                prev[c] = r[c]
        out.append(list(r) + extra)

    new_names = list(names)
    for c in cols:
        for k in leaks:
            new_names.append(f"{names[c]}_avg{horizon(k)}")
        if deltas:
            new_names.append(f"{names[c]}_change")
    return out, new_names


def split_by_time(rows, labels, frac=0.7):
    """Train on the past, test on the future. No shuffling, ever."""
    cut = int(len(rows) * frac)
    return (rows[:cut], labels[:cut]), (rows[cut:], labels[cut:])


# ------------------------------------------------------------- demonstration

def worn_machine(n, seed=1, decay=0.93, limit=7.15):
    """A bit that wears out. The row shows the load now; the label depends on
    load accumulated over the past, which no single row reveals."""
    import random
    rng = random.Random(seed)
    rows, labels, wear = [], [], 7.0
    for _ in range(n):
        load = rng.uniform(0, 1)
        wear = decay * wear + load
        rows.append([load, rng.random(), rng.random()])
        labels.append("1" if wear > limit else "0")
    return rows, labels, ["load", "noise1", "noise2"]


def _demo():
    from neuron_tree import fit_scale, apply_scale, build, use, size

    rows, labels, names = worn_machine(3000)

    def score(rs, ns, tag):
        (trR, trL), (teR, teL) = split_by_time(rs, labels)
        sc = fit_scale(trR)
        trN, teN = apply_scale(trR, sc), apply_scale(teR, sc)
        t = build(trN, trL, maxd=6)
        acc = sum(1 for r, l in zip(teN, teL) if use(t, r) == l) / len(teL)
        print(f"  {tag:<34}{len(ns):>6}{size(t):>8}{acc:>9.1%}")
        return t, ns

    (_, _), (_, teL) = split_by_time(rows, labels)
    base = max(teL.count("0"), teL.count("1")) / len(teL)
    print(f"\n  a machine whose bit wears out; the label is not in the row\n")
    print(f"  {'':<34}{'cols':>6}{'nodes':>8}{'test':>9}")
    print(f"  {'always guess the common class':<34}{'-':>6}{'-':>8}{base:>9.1%}")

    score(rows, names, "plain tree, raw columns")
    mem, mnames = add_memory(rows, names, cols=[0])
    t, _ = score(mem, mnames, "same tree, memory columns added")

    print(f"\n  the columns it was given:")
    print(f"    {', '.join(mnames[len(names):])}")

    used = set()
    def walk(node):
        if isinstance(node, tuple):
            used.update(node[0][0])
            walk(node[1]); walk(node[2])
    walk(t)
    print(f"\n  the ones it actually split on:")
    print(f"    {', '.join(sorted(mnames[i] for i in used))}\n")


def _from_csv(path, label_col, drop=("id", "ID")):
    import drilling as D
    from neuron_tree import fit_scale, apply_scale, build, use, size

    header, raw = D.load_csv(path)
    skip = [c for c in drop if c in header]
    rows, labels, names = D.to_numbers(header, raw, label_col, drop=skip)

    mem, mnames = add_memory(rows, names)
    (trR, trL), (teR, teL) = split_by_time(mem, labels)
    sc = fit_scale(trR)
    trN, teN = apply_scale(trR, sc), apply_scale(teR, sc)

    print(f"\n  {len(rows)} rows in file order, {len(names)} columns "
          f"-> {len(mnames)} with memory")
    print(f"  trained on the first {len(trR)}, tested on the last {len(teR)}\n")
    t = build(trN, trL, maxd=6)
    acc = sum(1 for r, l in zip(teN, teL) if use(t, r) == l) / len(teL)
    big = max(set(teL), key=teL.count)
    print(f"  always guessing {big!r}: {teL.count(big)/len(teL):.1%}")
    print(f"  with memory columns:   {acc:.1%}  ({size(t)} nodes)\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        _from_csv(sys.argv[1], sys.argv[2])
    else:
        _demo()