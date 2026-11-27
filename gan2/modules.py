"""One module per class, and a controller that picks the loudest.

    python3 modules.py steel_plates_faults_original_dataset.csv \
        Pastry Z_Scratch K_Scatch Stains Dirtiness Bumps Other_Faults

hierarchy.py answers "is anything wrong?" -- an OR over failure modes, with
normal rows in the majority. This answers a different question: every row has
exactly one class, and the job is to say which.

So the controller cannot be an OR. Each module reports how many of its trees
raised a hand, and the class with the loudest module wins. When nobody speaks
the answer falls back to the largest class, which is exactly what a model
that has learned nothing would say anyway.

It also trains one flat forest over all the classes at once, because the
whole point is to find out whether splitting into specialists beats not
splitting.
"""

import random
import sys
from collections import Counter

from neuron_tree import fit_scale, apply_scale, build_forest, use, size
import drilling as D
from hierarchy import focus, hands


def label_of(row_raw, header, flags):
    """Which flag is set on this raw CSV row."""
    for f in flags:
        if row_raw[header.index(f)].strip() not in ("", "0"):
            return f
    return "None"


def loudest(modules, row, fallback):
    """The class whose module raised the most hands. Ties go to the bigger class."""
    best, best_n = fallback, 0
    for name, mod in modules:
        n = hands(mod, row, "1")
        if n > best_n:
            best, best_n = name, n
    return best


def run(path, flags, n_trees=9, maxd=5, drop=("id",)):
    header, raw = D.load_csv(path)
    missing = [f for f in flags if f not in header]
    if missing:
        raise SystemExit(f"  columns not in this file: {missing}")

    rows, _, names = D.to_numbers(header, raw, flags[0],
                                  drop=list(drop) + list(flags[1:]))
    truth = [label_of(r, header, flags) for r in raw]

    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    cut = int(len(order) * 0.7)
    tr_i, te_i = order[:cut], order[cut:]

    scale = fit_scale([rows[i] for i in tr_i])
    trR = apply_scale([rows[i] for i in tr_i], scale)
    teR = apply_scale([rows[i] for i in te_i], scale)
    trL = [truth[i] for i in tr_i]
    teL = [truth[i] for i in te_i]

    counts = Counter(trL)
    biggest = counts.most_common(1)[0][0]
    print(f"\n  {len(rows)} rows, {len(names)} feature columns, "
          f"{len(flags)} classes")
    print(f"  always guessing {biggest!r} scores "
          f"{teL.count(biggest)/len(teL):.1%} on the held-out rows\n")

    # ---- one flat forest over every class at once (the thing to beat) -------
    flat = build_forest(trR, trL, n_trees=n_trees, maxd=maxd, seed=0)
    flat_acc = sum(1 for r, l in zip(teR, teL)
                   if Counter(use(t, r) for t in flat).most_common(1)[0][0] == l)
    flat_acc /= len(teL)

    # ---- one module per class ----------------------------------------------
    print(f"  {'module':<16}{'train':>8}{'nodes':>8}{'precision':>11}{'recall':>9}")
    modules = []
    for i, f in enumerate(flags):
        binary = ["1" if t == f else "0" for t in trL]
        sub_r, sub_l = focus(trR, binary, "1", keep_ratio=20, seed=i)
        mod = build_forest(sub_r, sub_l, n_trees=n_trees, maxd=maxd, seed=i)
        modules.append((f, mod))

        te_bin = ["1" if t == f else "0" for t in teL]
        tp = sum(1 for r, l in zip(teR, te_bin) if hands(mod, r, "1") >= 5 and l == "1")
        fp = sum(1 for r, l in zip(teR, te_bin) if hands(mod, r, "1") >= 5 and l == "0")
        fn = sum(1 for r, l in zip(teR, te_bin) if hands(mod, r, "1") < 5 and l == "1")
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {f:<16}{counts[f]:>8}{sum(size(t) for t in mod):>8}"
              f"{prec:>11.1%}{rec:>9.1%}")

    # ---- the comparison that matters ---------------------------------------
    picked = [loudest(modules, r, biggest) for r in teR]
    hier_acc = sum(1 for p, l in zip(picked, teL) if p == l) / len(teL)

    print(f"\n  one flat forest over all classes   {flat_acc:>6.1%}")
    print(f"  {len(flags)} modules, loudest wins          {hier_acc:>6.1%}")
    print(f"  {'':34}{hier_acc - flat_acc:>+6.1%}")

    print(f"\n  per class, held-out rows")
    print(f"  {'class':<16}{'in test':>9}{'flat':>9}{'modules':>10}")
    for f in flags:
        idx = [j for j, l in enumerate(teL) if l == f]
        if not idx:
            continue
        fl = sum(1 for j in idx
                 if Counter(use(t, teR[j]) for t in flat).most_common(1)[0][0] == f)
        hi = sum(1 for j in idx if picked[j] == f)
        print(f"  {f:<16}{len(idx):>9}{fl/len(idx):>9.1%}{hi/len(idx):>10.1%}")
    print()
    return modules


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    run(sys.argv[1], sys.argv[2:])