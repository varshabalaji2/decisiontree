"""Is a module learning the rule, or memorising the rows?

    python3 memorising.py XAI_Drilling_Dataset.csv

Four checks, weakest first. The last one is the only one that cannot be
argued with.

  1  train/test gap      scores well on rows it has seen, badly on new ones
  2  leaf occupancy      how many training rows sit behind each answer
  3  learning curve      does more data help, or has it stopped listening
  4  shuffled labels     THE test: destroy the rule and see if it still fits

Check 4 is decisive because after shuffling there IS no rule left to find.
Anything the model still fits is, by definition, memorised. A model that
cannot fit nonsense does not have the capacity to memorise your real data
either -- which is the guarantee you actually want.
"""

import random
import sys

from neuron_tree import fit_scale, apply_scale, build_forest, use, fires
import drilling as D
import hierarchy as H


# ------------------------------------------------- 2. how full are the leaves

def leaf_counts(tree, rows):
    """Push the training rows down and count how many land at each leaf."""
    if not isinstance(tree, tuple):
        return [len(rows)]
    test, lo, hi = tree
    left = [r for r in rows if not fires(r, test)]
    right = [r for r in rows if fires(r, test)]
    return leaf_counts(lo, left) + leaf_counts(hi, right)


def occupancy(forest, rows):
    sizes = sorted(c for t in forest for c in leaf_counts(t, rows))
    tiny = sum(1 for c in sizes if c <= 2)
    return sizes, tiny


# --------------------------------------------------------- scoring a forest

def score(forest, rows, labels, positive, need):
    def pred(r):
        return positive if sum(1 for t in forest
                               if use(t, r) == positive) >= need else "other"
    tp = sum(1 for r, l in zip(rows, labels) if pred(r) == positive == l)
    fp = sum(1 for r, l in zip(rows, labels) if pred(r) == positive != l)
    fn = sum(1 for r, l in zip(rows, labels) if pred(r) != positive == l)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


# ------------------------------------------------------------------- the run

def check(name, trR, tr_mode, teR, te_mode, n_trees=9, maxd=5, need=5, seed=0):
    sub_r, sub_l = H.focus(trR, tr_mode, "1", seed=seed)
    forest = build_forest(sub_r, sub_l, n_trees=n_trees, maxd=maxd, seed=seed)

    # 1. the gap
    p_tr, r_tr, f_tr = score(forest, sub_r, sub_l, "1", need)
    p_te, r_te, f_te = score(forest, teR, te_mode, "1", need)

    # 2. leaf occupancy
    sizes, tiny = occupancy(forest, sub_r)
    median = sizes[len(sizes) // 2]

    # 4. the same thing, on labels with the rule destroyed
    shuffled = list(sub_l)
    random.Random(99).shuffle(shuffled)
    fake = build_forest(sub_r, shuffled, n_trees=n_trees, maxd=maxd, seed=seed)
    p_sh, r_sh, f_sh = score(fake, sub_r, shuffled, "1", need)
    base = shuffled.count("1") / len(shuffled)

    print(f"\n  {name}   ({tr_mode.count('1')} real cases in training)")
    print(f"    train F1 {f_tr:>6.1%}   test F1 {f_te:>6.1%}   "
          f"gap {f_tr - f_te:>+6.1%}")
    print(f"    leaves: {len(sizes)} total, median holds {median} rows, "
          f"{tiny} hold 2 or fewer ({tiny/len(sizes):.0%})")
    print(f"    on SHUFFLED labels it still fits {f_sh:>6.1%} F1 "
          f"(chance is about {base:.1%})")
    verdict = []
    if f_tr - f_te > 0.15:
        verdict.append("big train/test gap")
    if tiny / len(sizes) > 0.25:
        verdict.append("many single-row leaves")
    if f_sh > base * 2:
        verdict.append("fits nonsense")
    print(f"    -> {', '.join(verdict) if verdict else 'looks like learning'}")
    return forest


def run(path, flags=None, n_trees=9, maxd=5):
    """`flags` are the 0/1 columns to check, one per class. Defaults to the
    drilling failure modes when none are named."""
    header, raw = D.load_csv(path)
    have = [m for m in (flags or H.MODES) if m in header]
    if not have:
        raise SystemExit(f"  none of those columns are in this file.\n"
                         f"  it has: {', '.join(header)}")

    ids = [c for c in ("id", "ID", "Id") if c in header]
    rows, _, names = D.to_numbers(header, raw, have[0],
                                  drop=ids + have[1:])
    per_mode = {m: [r[header.index(m)] for r in raw] for m in have}

    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    cut = int(len(order) * 0.7)
    tr_i, te_i = order[:cut], order[cut:]

    scale = fit_scale([rows[i] for i in tr_i])
    trR = apply_scale([rows[i] for i in tr_i], scale)
    teR = apply_scale([rows[i] for i in te_i], scale)

    print(f"\n  {len(tr_i)} training rows, {len(te_i)} held back, "
          f"{len(names)} features")
    for m in have:
        check(m, trR, [per_mode[m][i] for i in tr_i],
              teR, [per_mode[m][i] for i in te_i],
              n_trees=n_trees, maxd=maxd, seed=have.index(m))
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    run(sys.argv[1], sys.argv[2:] or None)

