"""Which features does the model actually use?

    python3 importance.py XAI_Drilling_Dataset.csv

Walks every split in every tree and counts. Unlike a per-prediction
explanation this says nothing about one row -- it describes the whole model,
and it is exact: these are the features, and there are no others.

The useful output is usually the bottom of the list. A feature used in zero
splits is one you can delete from the file.
"""

import random
import sys
from collections import Counter

from neuron_tree import fit_scale, apply_scale, build_forest
import drilling as D
import hierarchy as H


def count_splits(trees):
    """How often each feature is used, and how close to a root it gets.

    Depth matters: a split at the root sorts every row that reaches the tree,
    while one eight levels down sorts a handful. So alongside the raw count we
    keep the shallowest depth the feature ever reaches.
    """
    used = Counter()
    shallowest = {}

    def walk(t, depth=0):
        if not isinstance(t, tuple):
            return
        test, lo, hi = t
        for f in test[0]:
            used[f] += 1
            if f not in shallowest or depth < shallowest[f]:
                shallowest[f] = depth
        walk(lo, depth + 1)
        walk(hi, depth + 1)

    for t in trees:
        walk(t)
    return used, shallowest


def report(trees, names, title="the model"):
    used, shallowest = count_splits(trees)
    total = sum(used.values()) or 1
    print(f"\n  {title}: {len(trees)} trees, {total} splits\n")
    print(f"  {'feature':<30}{'splits':>8}{'share':>8}{'nearest root':>14}")
    print("  " + "-" * 60)
    order = sorted(range(len(names)), key=lambda f: (-used[f], names[f]))
    for f in order:
        depth = shallowest.get(f)
        flag = "   <-- never used, drop it" if used[f] == 0 else ""
        print(f"  {names[f]:<30}{used[f]:>8}{used[f]/total:>8.1%}"
              f"{('depth ' + str(depth)) if depth is not None else '-':>14}{flag}")
    dead = [names[f] for f in range(len(names)) if used[f] == 0]
    if dead:
        print(f"\n  {len(dead)} of {len(names)} columns are never used by any tree.")
    return used


def run(path, n_trees=9, maxd=5):
    header, raw = D.load_csv(path)
    have_modes = [m for m in H.MODES if m in header]

    rows, labels, names = D.to_numbers(header, raw, "Main Failure",
                                       drop=["ID"] + have_modes)
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    tr_i = order[:int(len(order) * 0.7)]

    scale = fit_scale([rows[i] for i in tr_i])
    trR = apply_scale([rows[i] for i in tr_i], scale)

    if have_modes:
        per_mode = {m: [r[header.index(m)] for r in raw] for m in have_modes}
        everything = []
        for m in have_modes:
            mode_tr = [per_mode[m][i] for i in tr_i]
            sub_r, sub_l = H.focus(trR, mode_tr, "1", seed=have_modes.index(m))
            mod = build_forest(sub_r, sub_l, n_trees=n_trees, maxd=maxd,
                               seed=have_modes.index(m))
            report(mod, names, f"module {m}")
            everything += mod
        report(everything, names, "all four modules together")
    else:
        trL = [labels[i] for i in tr_i]
        report(build_forest(trR, trL, n_trees=n_trees, maxd=maxd), names,
               "the flat forest")
    print()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "XAI_Drilling_Dataset.csv")