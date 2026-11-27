"""A hierarchical forest: one module per failure mode, one controller over them.

    python3 hierarchy.py XAI_Drilling_Dataset.csv

Four modules, each a small forest trained on ONE failure mode. Above them a
controller that hears every module's show of hands and decides.

The flat forest in neuron_tree.py had to discover the four modes by accident,
and each tree ended up knowing some of them and not others -- which is what
broke majority voting. Here each module is told which mode it is responsible
for, so its trees all study the same thing and the specialisation is designed
rather than emergent.
"""

import random
import sys

from neuron_tree import fit_scale, apply_scale, build_forest, use, size
import drilling as D


# ------------------------------------------------------------------ a module

def focus(rows, labels, positive, keep_ratio=100, seed=0):
    """Keep every positive row and `keep_ratio` negatives for each one.

    A failure mode fires on about 1% of rows, so most of a module's effort
    goes on rows that could never matter to it. Trimming the negatives makes
    each module faster -- the whole hierarchy trains in half the time of one
    flat forest, despite having more than twice as many trees.

    But trim too hard and the module learns the wrong world. At 3 negatives
    per positive it sees failures as 25% of life, then meets a test set where
    they are 1%, and cries wolf constantly:

        negatives per positive     accuracy   precision   recall
                             3        90.9%       32.3%   100.0%
                            20        92.3%       35.8%    96.9%
                           100        99.7%       99.2%    94.3%
                   all of them        99.8%       99.2%    96.2%

    A hundred keeps the calibration and still saves most of the time.
    """
    yes = [i for i, l in enumerate(labels) if l == positive]
    no = [i for i, l in enumerate(labels) if l != positive]
    rng = random.Random(seed)
    if len(no) > keep_ratio * len(yes):
        no = rng.sample(no, keep_ratio * len(yes))
    pick = yes + no
    rng.shuffle(pick)
    return [rows[i] for i in pick], [labels[i] for i in pick]


def build_module(rows, labels, positive, n_trees=9, maxd=5, seed=0):
    """One forest, responsible for one failure mode and nothing else."""
    sub_r, sub_l = focus(rows, labels, positive, seed=seed)
    return build_forest(sub_r, sub_l, n_trees=n_trees, maxd=maxd, seed=seed)


def hands(module, row, positive):
    """How many trees in this module raised a hand for `positive`."""
    return sum(1 for t in module if use(t, row) == positive)


# -------------------------------------------------------------- the controller

def controller(modules, row, need_hands=5, need_modules=1):
    """Hear every module, then decide.

    need_hands    how many trees inside a module must agree before that
                  module counts as having spoken up at all
    need_modules  how many modules must speak up before we call a failure

    need_modules=1 is an OR, which is what the data actually says: Main
    Failure is true when ANY mode fires. Raising it demands corroboration
    between modes, which trades recall for precision.
    """
    spoke = sum(1 for name, mod, pos in modules
                if hands(mod, row, pos) >= need_hands)
    return "1" if spoke >= need_modules else "0"


def explain(modules, row, need_hands=3):
    """Which modules spoke up, and how loudly. This is the XAI part."""
    return {name: hands(mod, row, pos) for name, mod, pos in modules}


# ------------------------------------------------------------------- training

 
MODES = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]


def run(path, n_trees=9, maxd=5):
    header, raw = D.load_csv(path)
    missing = [m for m in MODES if m not in header]
    if missing:
        raise SystemExit(f"  columns not in this file: {missing}\n"
                         f"  it has: {', '.join(header)}")

    drop = ("ID", "Main Failure") + tuple(MODES)
    rows, main_labels, names = D.to_numbers(header, raw, "Main Failure",
                                            drop=[d for d in drop
                                                  if d != "Main Failure"])
    per_mode = {m: [r[header.index(m)] for r in raw] for m in MODES}

    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    cut = int(len(order) * 0.7)
    tr_i, te_i = order[:cut], order[cut:]

    scale = fit_scale([rows[i] for i in tr_i])
    trR = apply_scale([rows[i] for i in tr_i], scale)
    teR = apply_scale([rows[i] for i in te_i], scale)
    trL = [main_labels[i] for i in tr_i]
    teL = [main_labels[i] for i in te_i]

    print(f"\n  {len(rows)} rows, {len(names)} feature columns")
    print(f"  Main Failure is true for {main_labels.count('1')} of them\n")

    print(f"  {'module':<8}{'positives':>11}{'rows used':>12}{'trees':>7}{'nodes':>8}")
    modules = []
    for m in MODES:
        mode_tr = [per_mode[m][i] for i in tr_i]
        sub_r, sub_l = focus(trR, mode_tr, "1", seed=MODES.index(m))
        mod = build_forest(sub_r, sub_l, n_trees=n_trees, maxd=maxd,
                           seed=MODES.index(m))
        modules.append((m, mod, "1"))
        print(f"  {m:<8}{mode_tr.count('1'):>11}{len(sub_r):>12}"
              f"{len(mod):>7}{sum(size(t) for t in mod):>8}")

    print(f"\n  each module alone, on never-seen rows (needs 5 of {n_trees} trees)")
    for m, mod, pos in modules:
        mode_te = [per_mode[m][i] for i in te_i]
        tp = sum(1 for r, l in zip(teR, mode_te) if hands(mod, r, pos) >= 5 and l == "1")
        fp = sum(1 for r, l in zip(teR, mode_te) if hands(mod, r, pos) >= 5 and l == "0")
        fn = sum(1 for r, l in zip(teR, mode_te) if hands(mod, r, pos) < 5 and l == "1")
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {m:<8} precision {prec:>6.1%}   recall {rec:>6.1%}   "
              f"({tp + fn} real cases)")

    print(f"\n  the whole hierarchy on Main Failure, never-seen rows")
    print(f"  {'':<26}{'accuracy':>10}{'precision':>11}{'recall':>9}")
    for nh in (5, 3, 2, 1):
        pred = [controller(modules, r, need_hands=nh) for r in teR]
        acc = sum(1 for p, l in zip(pred, teL) if p == l) / len(teL)
        tp = sum(1 for p, l in zip(pred, teL) if p == "1" and l == "1")
        fp = sum(1 for p, l in zip(pred, teL) if p == "1" and l == "0")
        fn = sum(1 for p, l in zip(pred, teL) if p == "0" and l == "1")
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {'  needs ' + str(nh) + f'/{n_trees} in any module':<26}"
              f"{acc:>10.1%}{prec:>11.1%}{rec:>9.1%}")

    # what a single prediction looks like from the outside
    for r, l in zip(teR, teL):
        if l == "1":
            print(f"\n  one failing row, module by module: {explain(modules, r)}")
            break
    print()
    return modules


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "steel_plates_faults_original_dataset.csv")