"""Feature groups: forests that each see a different part of the machine.

    python3 groups.py XAI_Drilling_Dataset.csv

Every group trains on ALL the rows and the SAME label. The only difference
between them is which columns they are allowed to look at.

That is what separates this from the per-mode modules in hierarchy.py. Those
split the data by answer, so 1000 failures became four small piles and the
rarest module had almost nothing to learn from. Groups split by question
instead: every group still sees every failure, so nothing is ever starved.

The cost is that each group is deliberately handicapped -- a group can never
learn a rule that needs a column it cannot see. So groups score worse alone
than a forest with every column, and the controller has to earn that back by
combining them.
"""

import random
import sys

from neuron_tree import fit_scale, apply_scale, build, use, size
import drilling as D
import hierarchy as H


# Columns are matched by substring, so this survives small header differences.
#
# Grouped by which features the RULES use together, read off the importance
# report, not by which ones sound related. The obvious physical grouping --
#
#     thermal    cooling, power
#     kinematic  cutting speed, spindle, feed
#     setup      material, drill bit, process time
#
# scores near-zero recall, because three of the four failure modes pair a
# thermal feature with a kinematic one and that grouping cuts every one of
# those rules in half. Power and cutting speed top both CCF and FWF, so they
# belong together.
GROUPS = {
    "load":     ["power", "cutting speed"],
    "rotation": ["spindle"],
    "advance":  ["feed"],                           # matches both feed columns
    "setup":    ["material", "drill bit", "cooling", "process time"],
}


def find_columns(names):
    """Turn the substring patterns into column numbers for this file."""
    out, claimed = {}, set()
    for group, patterns in GROUPS.items():
        cols = [i for i, n in enumerate(names)
                if any(p in n.lower() for p in patterns)]
        out[group] = cols
        claimed.update(cols)
    missed = [names[i] for i in range(len(names)) if i not in claimed]
    return out, missed


def build_group(rows, labels, cols, positive, n_trees=9, maxd=5, min_leaf=1,
                seed=0, keep_ratio=20):
    """A forest restricted to one group's columns.

    Note this does NOT use build_forest: that function picks its own random
    columns, which would undo the grouping. Here the columns are fixed and
    only the rows are bootstrapped, so every tree in the group looks at the
    same part of the machine.

    The rows ARE rebalanced first. With failures at 4% and only two or three
    columns to work with, Gini is happiest calling everything normal, and a
    group trained on the raw proportions returns 0% recall -- it never says
    failure at all.
    """
    sub_r, sub_l = H.focus(rows, labels, positive, keep_ratio=keep_ratio,
                           seed=seed)
    rng = random.Random(seed)
    n = len(sub_r)
    forest = []
    for _ in range(n_trees):
        pick = [rng.randrange(n) for _ in range(n)]
        forest.append(build([sub_r[i] for i in pick], [sub_l[i] for i in pick],
                            maxd=maxd, min_leaf=min_leaf, only=cols))
    return forest


def hands(forest, row, positive):
    return sum(1 for t in forest if use(t, row) == positive)


def verdict(groups, row, positive, need_hands=5, need_groups=1):
    spoke = sum(1 for g in groups.values() if hands(g, row, positive) >= need_hands)
    return positive if spoke >= need_groups else "other"


def score(predict, rows, labels, positive):
    tp = sum(1 for r, l in zip(rows, labels) if predict(r) == positive == l)
    fp = sum(1 for r, l in zip(rows, labels) if predict(r) == positive != l)
    fn = sum(1 for r, l in zip(rows, labels) if predict(r) != positive == l)
    acc = sum(1 for r, l in zip(rows, labels)
              if (predict(r) == positive) == (l == positive)) / len(labels)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return acc, prec, rec


def run(path, n_trees=9, maxd=5, label_col="Main Failure"):
    header, raw = D.load_csv(path)
    extra = [c for c in ("BEF", "CCF", "FWF", "WDF") if c in header]
    rows, labels, names = D.to_numbers(header, raw, label_col,
                                       drop=["ID"] + extra)

    cols, missed = find_columns(names)
    print(f"\n  {len(rows)} rows, {len(names)} columns\n")
    for g, c in cols.items():
        print(f"  {g:<12}{', '.join(names[i] for i in c)}")
    if missed:
        print(f"  {'UNGROUPED':<12}{', '.join(missed)}   <-- no group sees these")

    counts = {v: labels.count(v) for v in set(labels)}
    positive = min(counts, key=counts.get)
    print(f"\n  predicting {label_col!r} = {positive!r}, "
          f"{counts[positive]} of {len(labels)} rows "
          f"({counts[positive]/len(labels):.1%})")

    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    cut = int(len(order) * 0.7)
    tr_i, te_i = order[:cut], order[cut:]
    scale = fit_scale([rows[i] for i in tr_i])
    trR = apply_scale([rows[i] for i in tr_i], scale)
    teR = apply_scale([rows[i] for i in te_i], scale)
    trL = [labels[i] for i in tr_i]
    teL = [labels[i] for i in te_i]

    groups = {}
    for i, (g, c) in enumerate(cols.items()):
        if not c:
            continue
        groups[g] = build_group(trR, trL, c, positive,
                                n_trees=n_trees, maxd=maxd, seed=i)

    print(f"\n  each group alone, on never-seen rows (needs 5 of {n_trees})")
    print(f"  {'':<14}{'accuracy':>10}{'precision':>11}{'recall':>9}{'nodes':>8}")
    for g, f in groups.items():
        a, p, r = score(lambda x, f=f: positive if hands(f, x, positive) >= 5
                        else "other", teR, teL, positive)
        print(f"  {g:<14}{a:>10.1%}{p:>11.1%}{r:>9.1%}"
              f"{sum(size(t) for t in f):>8}")

    print(f"\n  all groups together")
    print(f"  {'':<14}{'accuracy':>10}{'precision':>11}{'recall':>9}")
    for nh in (7, 5, 3, 2):
        a, p, r = score(lambda x, k=nh: verdict(groups, x, positive, k),
                        teR, teL, positive)
        print(f"  {'needs ' + str(nh) + '/' + str(n_trees):<14}"
              f"{a:>10.1%}{p:>11.1%}{r:>9.1%}")

    for r, l in zip(teR, teL):
        if l == positive:
            told = {g: hands(f, r, positive) for g, f in groups.items()}
            print(f"\n  one failing row, group by group: {told}")
            break
    print()
    return groups


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "steel_plates_faults_original_datset.csv")