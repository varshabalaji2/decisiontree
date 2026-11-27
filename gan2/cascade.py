"""A second model for the rows the first one cannot decide.

    python3 cascade.py --csv steel_single.csv --label Label --drop id
    python3 cascade.py --csv data.csv --label Class --seeds 5 --agree 7
    python3 cascade.py --csv steel_single.csv --label Label --rounds 4
    python3 cascade.py --csv steel_single.csv --label Label --slices 3 \
                       --no-cascade

--slices IS AN EXPERIMENT AND NOT A SETTING. Talking modules and a learned
referee were both flat against plain argmax, and the reason offered was that
every module already trained on every row, so a neighbour's confidence told
it nothing it could not compute itself. --slices tests that reason by taking
the global view away: the rows are cut into N disjoint class-stratified parts
and each module gets one. If the reason is right, coordination should start
earning something here, and the sliced arms should still lose overall --
which converts four null results into a boundary condition rather than four
failures. --no-cascade skips the slow A + B arms, whose question is settled.

Seven attempts in this project gave different modules different slices of the
data, and all seven lost. Each module saw less data, and each needed a router
that was as hard to build as the original problem.

A cascade avoids both. Stage A trains on EVERYTHING. The router is not learned
at all -- it is A's own disagreement, which is already computed:

    8 or 9 of 9 trees agree   ->  take A's answer
    fewer than that           ->  hand the row to B

--agree is a COUNT of trees, not a fraction, because a fraction interacts with
the tree count in a way that silently does nothing: with 5 trees the only
reachable margins are 60%, 80% and 100%, so a 78% threshold catches only the
60% case and B is never built. The count says what it means.

That signal is known to work here. Thresholding show-of-hands was the largest
single win in this project, taking drilling recall from 66.9% to 100%.

OUT OF FOLD, and this is not optional. A's votes on its own training rows are
optimistic, because it has seen them, so the rows it "disputes" there are not
the rows it will dispute in the field. B would train on the wrong set and
learn to fix problems that do not occur. Training engines -- here training
rows -- are split into folds, and each fold's votes come from a forest that
never saw it.

FOUR ARMS, and the second one is the control that matters:

    A alone                     the baseline
    A + B, B the same forest    does training ONLY on hard rows help at all?
    A + B, B another geometry   does the model TYPE add anything on top?

If the same-forest arm already gets the gain, then the win is the training
distribution and the geometry is a footnote -- the same lesson the Adaline
control taught, where fitting weights properly changed nothing and that is
what identified the real limitation.

THE COST THIS DOES NOT AVOID: B trains on disputed rows only, so it has less
data for a harder problem. That is the modules failure in miniature. What
differs is that B's job matches its training exactly, the same rule choosing
both, where a learned router could send a module rows unlike anything it saw.
Whether B's thinner data costs more than its specialisation buys is what the
table below is for.
"""

import random
import statistics
import sys

from shapes import (load_csv_task, shuffled_split, build, build_forest,
                    use, KINDS)
from neuron_tree import fit_scale, apply_scale


def votes(forest, row):
    """Every tree's answer, counted. The margin is the router."""
    out = {}
    for t in forest:
        v = use(t, row)
        out[v] = out.get(v, 0) + 1
    return out


def decide(forest, row):
    v = votes(forest, row)
    best = max(v, key=v.get)
    return best, v[best] / len(forest)


def disputed_rows(rows, labels, kind, folds, n_trees, maxd, seed, agree):
    """Which rows would A be unsure about, judged out of fold.

    A forest asked about its own training rows is optimistic, so the rows it
    disputes there are not the ones it will dispute in the field.
    """
    order = list(range(len(rows)))
    random.Random(seed + 77).shuffle(order)
    parts = [order[i::folds] for i in range(folds)]

    hard = []
    for f in range(folds):
        held = set(parts[f])
        fit = [i for i in order if i not in held]
        fo = build_forest([rows[i] for i in fit], [labels[i] for i in fit],
                          (kind,), n_trees=n_trees, maxd=maxd, seed=seed + f)
        for i in parts[f]:
            v = votes(fo, rows[i])
            if max(v.values()) < agree:
                hard.append(i)
    return hard


def rebalance(rows, labels, seed=0):
    """Oversample every class up to the largest, with replacement.

    Here as a CONTROL, not an improvement. The cascade's gain showed up in
    macro recall while accuracy barely moved, and more than half the rows went
    to B -- which points at B's training set simply being more class-balanced
    than A's, since rare classes get disputed more often. If plain rebalancing
    captures the same gain, the cascade is an expensive way to do one line.
    """
    rng = random.Random(seed + 404)
    by = {}
    for i, l in enumerate(labels):
        by.setdefault(l, []).append(i)
    biggest = max(len(v) for v in by.values())
    out = []
    for l, idx in by.items():
        out += idx
        out += [rng.choice(idx) for _ in range(biggest - len(idx))]
    rng.shuffle(out)
    return [rows[i] for i in out], [labels[i] for i in out]


def build_modules(rows, labels, classes, kind, n_trees, maxd, seed):
    """One binary forest per class, each trained on rebalanced data.

    This is modules.py's design, rebuilt here so it meets the same splits and
    the same metric as the other arms. modules.py calls focus() to rebalance
    each module's training data, which is the whole reason this arm exists:
    if per-class modules only matched a single rebalanced forest, their
    reported win was the rebalancing inside them rather than the modularity.

    NOTE WHAT THIS IS NOT. Every module trains on EVERY row, relabelled
    c-versus-rest, and loudest() asks all of them about every row. No row is
    diverted and no module sees less data, so the cost that sank hierarchy.py
    and inhibit.py is not paid here. What changes is the TARGET -- k yes/no
    questions instead of one k-way question -- which is a change to what the
    model is told, not to what it is.
    """
    out = []
    for i, c in enumerate(classes):
        binary = [c if l == c else "rest" for l in labels]
        if binary.count(c) < 10:
            continue
        fitR, fitL = rebalance(rows, binary, seed + i)
        out.append((c, build_forest(fitR, fitL, (kind,), n_trees=n_trees,
                                    maxd=maxd, seed=seed + i)))
    return out


def slice_rows(rows, labels, n_slices, seed):
    """Partition the rows into n disjoint, class-stratified slices.

    Stratified because a plain random cut starves the rare classes: with
    seven slices Dirtiness would land five rows in each and most modules
    would refuse to build. Dealing each class round-robin keeps every slice
    at roughly 1/n of every class.
    """
    by = {}
    for i, l in enumerate(labels):
        by.setdefault(l, []).append(i)
    parts = [[] for _ in range(n_slices)]
    rng = random.Random(seed + 606)
    for l, idx in by.items():
        order = list(idx)
        rng.shuffle(order)
        for j, i in enumerate(order):
            parts[j % n_slices].append(i)
    # sorted back into the original row order, which is not cosmetic: the
    # resampling in rebalance() and the bootstrap in build_forest both walk
    # the rows they are handed, so a shuffled slice would differ from the
    # unsliced set by ORDER as well as by content. With this, n_slices=1 is
    # build_modules to the prediction, and the only thing the sliced arms
    # change is how much data each module gets.
    return [sorted(p) for p in parts]


def slice_assign(rows, labels, classes, n_slices, seed):
    """Which row indices each class's module is allowed to train on.

    THE HANDICAP HAS TO APPLY TO EVERY STAGE OF A MODULE, not just round 0.
    The round-1 talking modules see the raw row, so if they were trained on
    all of it they would hand the global view straight back and any gain
    would be the handicap lifting rather than coordination working. They get
    the same slice their round-0 counterpart got.

    Keyed by CLASS and not by position, because a module whose slice held too
    few positives is skipped, so the module list and `classes` fall out of
    step. slice_rows is deterministic in (labels, n_slices, seed), so this
    reproduces exactly the partition build_modules_sliced used.
    """
    if n_slices <= 1:
        whole = list(range(len(rows)))
        return {c: whole for c in classes}
    parts = slice_rows(rows, labels, n_slices, seed)
    return {c: parts[i % n_slices] for i, c in enumerate(classes)}


def build_modules_sliced(rows, labels, classes, kind, n_trees, maxd, seed,
                         n_slices, min_pos=10):
    """One binary forest per class, each trained on a DISJOINT SLICE.

    This is a handicap, applied on purpose. build_modules gives every module
    every row, and that is exactly why modules beat hierarchy.py: no module
    sees less data. Here each module is cut off from most of the data, which
    manufactures the condition hierarchy.py suffered.

    THE POINT IS WHAT IT DOES TO COORDINATION. Talking and a learned referee
    were both flat with the global view intact, and the argument was that
    they had nothing to add because every module already saw every row.
    If that argument is right, coordination should start to EARN something
    once the modules genuinely hold different information -- and the sliced
    arms should still lose overall, because the slicing cost more than the
    coordination recovers.

    Which would turn four null results into a boundary condition: these
    mechanisms substitute for a global view, and are worth nothing when one
    is available.

    n_slices is separate from the number of classes so the handicap can be
    dialled. n_slices=1 is build_modules exactly, which is the self-check.
    """
    parts = slice_rows(rows, labels, n_slices, seed)
    out = []
    for i, c in enumerate(classes):
        idx = parts[i % n_slices]
        sr = [rows[j] for j in idx]
        sl = [c if labels[j] == c else "rest" for j in idx]
        if sl.count(c) < min_pos:
            continue
        fitR, fitL = rebalance(sr, sl, seed + i)
        out.append((c, build_forest(fitR, fitL, (kind,), n_trees=n_trees,
                                    maxd=maxd, seed=seed + i)))
    return out


def modules_maker(classes, kind, n_trees, maxd, n_slices=1):
    """A callable that builds the module set, so the coordination code can
    be handed either the full-view or the sliced version without caring."""
    def make(rows, labels, seed):
        if n_slices <= 1:
            return build_modules(rows, labels, classes, kind, n_trees, maxd,
                                 seed)
        return build_modules_sliced(rows, labels, classes, kind, n_trees,
                                    maxd, seed, n_slices)
    # the talking arm needs to reproduce the same partition for its round-1
    # modules, so the callable carries what slice_assign needs
    make.classes = classes
    make.n_slices = n_slices
    return make


def loudest(modules, row, fallback):
    """The module whose trees shout loudest wins, as in modules.py."""
    best, best_v = fallback, 0.0
    for c, fo in modules:
        v = votes(fo, row).get(c, 0) / len(fo)
        if v > best_v:
            best, best_v = c, v
    return best


def _conf_vector(modules, row):
    """Each module's confidence that this row belongs to its class."""
    return [votes(fo, row).get(c, 0) / len(fo) for c, fo in modules]


def _oof_confidences(rows, labels, make, seed, folds, names):
    """Every training row's module confidences, from modules that never saw
    it. Returns (confidences, modules missing from a fold), or (None, n) if
    a fold could not build two modules at all.

    Both the talking modules and the learned referee need this, and for the
    same reason: a module asked about its own training rows is optimistic, so
    anything trained on those confidences learns that the modules are always
    right and has nothing to fall back on when they are not.

    THE VECTOR IS KEYED BY NAME, NOT BY POSITION, and that is a correction.
    This returned None the moment a fold's module list differed from the full
    one, on the grounds that the vectors would be inconsistent. The effect of
    that under --slices 3 was silent and total: the handicaps compound, 2/3
    of the rows in a fold times 1/3 for the slice leaves a rare class below
    the ten-positive floor, every fold lost its Dirtiness module, this
    returned None, and the talking and referee arms fell back to plain
    argmax on the SAME modules -- so all three sliced arms printed the same
    number to the decimal and the run looked like a clean null result when
    nothing had been measured.

    So a fold that loses a module now contributes 0.0 in that module's slot
    instead of voiding the whole thing. The vectors stay the width of
    `names`, which is what _conf_vector produces at predict time, and a
    class whose module could not be built reads as "nobody claimed this
    row" -- which is true. The count comes back so the caller can say how
    degraded the run was, because a column that is always 0.0 is a module
    the downstream stage cannot learn anything from.
    """
    order = list(range(len(rows)))
    random.Random(seed + 303).shuffle(order)
    parts = [order[i::folds] for i in range(folds)]
    conf = [None] * len(rows)
    missing = 0
    for f in range(folds):
        held = set(parts[f])
        fit = [i for i in order if i not in held]
        mods = make([rows[i] for i in fit], [labels[i] for i in fit],
                    seed + 50 + f)
        at = {c: j for j, (c, _) in enumerate(mods)}
        if len(at) < 2:
            return None, missing + len(names)
        missing += sum(1 for c in names if c not in at)
        for i in parts[f]:
            v = _conf_vector(mods, rows[i])
            conf[i] = [v[at[c]] if c in at else 0.0 for c in names]
    return conf, missing


def build_stacked(rows, labels, make, kind, n_trees, maxd, seed, folds=3):
    """A referee that LEARNS how much to believe each module.

    loudest() is plain argmax: whichever module is loudest wins, and every
    module's confidence is taken at face value. That has a specific weakness.
    Each module was trained on REBALANCED data, so its confidence is
    calibrated to a world where its class is half of everything. In the real
    file Dirtiness is about 2%. A Dirtiness module saying 0.6 and an
    Other_Faults module saying 0.6 do not mean the same thing, and argmax
    treats them as if they do.

    So instead of argmax, train one forest whose inputs are the k module
    confidences and whose output is the true class. It can learn "believe
    Dirtiness above 0.55, but make Other_Faults clear 0.85".

    CONFIDENCES ONLY, not the raw columns. Handing the referee the features
    as well would let it ignore the modules and re-solve the original
    problem, and then the arm measures nothing about stacking.

    THE REFEREE IS REBALANCED TOO, and that is a choice worth stating. Its
    training labels are the true multi-class ones, which are badly skewed;
    left alone it would re-introduce exactly the majority bias the modules'
    rebalancing removed, and the arm would look bad for a reason that has
    nothing to do with stacking. The metric here is macro recall, which wants
    every class to count the same, so the referee is given data that says so.

    This is not modules talking to each other. It is a referee who has
    learned which of them to believe, which is the version that works in
    practice.

    MEASURED, steel, 5 seeds, and this is the row that closes the question:

        combiner                 accuracy   macro recall   vs baseline
        argmax (loudest)            75.5%          79.5%   +13.8% on 5/5
        iterate to agreement        76.1%          79.5%   +13.8% on 5/5
        learned referee             73.6%          79.6%   +13.8% on 5/5

    IDENTICAL. A fixed rule, a consensus loop and a trained model all reach
    the same number, so HOW THE MODULES ARE COMBINED CONTRIBUTES NOTHING. The
    +13.8% is entirely the decomposition -- seven yes/no questions instead of
    one seven-way question -- and the merging of the answers is free to be as
    naive as argmax.

    The prediction above was that the referee would do nothing because every
    module is distorted the same way by its rebalancing, leaving argmax
    comparing like with like. That held: +0.1% on 3 of 5 seeds.

    ONE DESIGN NOTE AGAINST MY OWN CHOICE. Rebalancing the referee cost 1.9
    points of ACCURACY (73.6 against 75.5) while matching macro recall. It
    shifted predictions toward the rare classes without catching more of
    them, so that choice paid and bought nothing. An unbalanced referee is
    worth a run before the rebalanced one is kept.
    """
    base = make(rows, labels, seed)
    if len(base) < 2:
        return base, None, 0
    names = [c for c, _ in base]
    conf, missing = _oof_confidences(rows, labels, make, seed, folds, names)
    if conf is None:
        return base, None, missing
    fitR, fitL = rebalance(conf, labels, seed + 707)
    return base, build_forest(fitR, fitL, (kind,), n_trees=n_trees,
                              maxd=maxd, seed=seed + 700), missing


def stacked_predict(base, ref, row, fallback):
    v = votes(ref, _conf_vector(base, row))
    return max(v, key=v.get) if v else fallback


def build_talking(rows, labels, make, kind, n_trees, maxd, seed, folds=3):
    """Modules that re-predict after reading each other's answers.

    Round 0 is the ordinary one-versus-rest set. Round 1 modules take the row
    PLUS the other modules' confidences, so a module can learn "when the
    Bumps module is loud, I am probably wrong".

    THE NEIGHBOUR CONFIDENCES MUST BE OUT OF FOLD, for exactly the reason
    disputed_rows is. A module asked about its own training rows is
    optimistic, so a round-1 module trained on those learns that its
    neighbours are always right -- and at test time, where they are not, it
    has nothing to fall back on.

    A MODULE NEVER SEES ITS OWN PREVIOUS CONFIDENCE. Letting it would give a
    trivial fixed point: copy the last answer, converge on round one, and
    learn nothing.

    WHY THIS IS EXPECTED TO LOSE, stated before running it. No new
    information enters the loop: after round 0 every module has seen the whole
    row, and a neighbour's confidence is a deterministic function of that same
    row. What iterating can do is let correlated models reinforce each other,
    and these seven are correlated -- same rows, same columns, same tree code.
    A confident shared error gets more confident each round. Convergence finds
    a fixed point, not a truth.

    It also throws away this file's own signal. The cascade uses DISAGREEMENT
    to find hard rows; iterating until the modules agree manufactures
    consensus and deletes that flag.

    THE KNOWN SHORTCUT. Round 1 modules are trained on ROUND 0 confidences but
    from round 2 are fed their own previous outputs, which is a different
    distribution. Training a separate model per round is what the literature
    does and would fix it. It is not done here because the question is whether
    iterating helps at all; if round 2 already drifts, more machinery will not
    rescue it.

    MEASURED, steel, 5 seeds, against the silent modules:

        modules, loudest wins    75.5% acc  79.5% macro  +13.8% on 5/5
        modules, talking         76.1% acc  79.5% macro  +13.8% on 5/5

        TALKING: 1.1 rounds on average, 0.3% of rows never settled

    FLAT, not a loss -- the prediction above said it would lose and that was
    wrong. Macro recall is identical to the decimal and every per-class change
    sits inside 1.5 points with no pattern.

    The 1.1 rounds is the result, and it is better than a loss would have
    been. The modules settle on round one nine times in ten, which means a
    round-1 module handed its six neighbours' confidences as extra features
    LEARNED TO IGNORE THEM and reproduce its round-0 answer. A loss could be
    blamed on this implementation. Converging on "ignore the neighbours" is
    the model reporting that the channel is empty -- given free access to what
    the others think, the best thing it found to do was nothing.

    That is the "no new information enters the loop" argument arriving as
    behaviour rather than as an argument, and it closes the question.
    """
    base = make(rows, labels, seed)
    if len(base) < 2:
        return base, None, [], 0
    names = [c for c, _ in base]
    conf, missing = _oof_confidences(rows, labels, make, seed, folds, names)
    if conf is None:
        return base, None, [], missing

    allow = slice_assign(rows, labels, getattr(make, "classes", names),
                         getattr(make, "n_slices", 1), seed)
    talk = []
    for k, c in enumerate(names):
        idx = allow.get(c, list(range(len(rows))))
        ext = [list(rows[i]) + [v for j, v in enumerate(conf[i]) if j != k]
               for i in idx]
        binary = [c if labels[i] == c else "rest" for i in idx]
        fitR, fitL = rebalance(ext, binary, seed + k)
        talk.append((c, build_forest(fitR, fitL, (kind,), n_trees=n_trees,
                                     maxd=maxd, seed=seed + 90 + k)))
    return base, talk, names, missing


def talk_predict(base, talk, names, row, fallback, rounds):
    """Pass answers round until nobody changes their mind, or rounds run out.

    Returns (label, rounds used, whether it settled).
    """
    conf = _conf_vector(base, row)
    if not conf:
        return fallback, 0, True
    last = max(range(len(conf)), key=lambda i: conf[i])
    for used in range(1, rounds + 1):
        new = []
        for k, (c, fo) in enumerate(talk):
            ext = list(row) + [v for j, v in enumerate(conf) if j != k]
            new.append(votes(fo, ext).get(c, 0) / len(fo))
        pick = max(range(len(new)), key=lambda i: new[i])
        conf = new
        if pick == last:
            return names[pick], used, True
        last = pick
    return names[last], rounds, False


def per_class_recall(pred, truth, classes):
    """Recall for every class separately -- where a reinforced error shows."""
    out = {}
    for c in classes:
        n = sum(1 for t in truth if t == c)
        if n:
            out[c] = sum(1 for p, t in zip(pred, truth)
                         if t == c and p == c) / n
    return out


def macro_recall(pred, truth, classes):
    """Mean of the per-class recalls: every class counts the same, however
    rare. Overall accuracy is dominated by the big classes and hides exactly
    the ones a cascade is meant to rescue."""
    out = []
    for c in classes:
        tp = sum(1 for p, t in zip(pred, truth) if t == c and p == c)
        n = sum(1 for t in truth if t == c)
        if n:
            out.append(tp / n)
    return statistics.mean(out) if out else 0.0


def run(csv, label, drop=(), seeds=5, maxd=6, n_trees=9, folds=3,
        agree=None, base="sum", rounds=3, slices=3, cascade=True):
    agree = n_trees - 1 if agree is None else agree
    rows, labels, names = load_csv_task(csv, label, drop)
    classes = sorted(set(labels))
    big = max(classes, key=labels.count)

    print(f"\n  {csv}: {len(rows)} rows, {len(names)} columns, "
          f"{len(classes)} classes")
    print(f"  always guessing {big!r}: {labels.count(big)/len(labels):.1%}")
    print(f"  stage A is a {base!r} forest of {n_trees} at depth {maxd}; a row "
          f"goes to B when\n  fewer than {agree} of its {n_trees} trees "
          f"agree, judged over {folds} out-of-fold folds")
    print(f"  talking modules iterate at most {rounds} rounds")
    if slices > 1:
        print(f"  the sliced arms cut the training rows into {slices} "
              f"disjoint class-stratified\n  parts and give each module one "
              f"of them, round-1 talking modules included")
    if not cascade:
        print(f"  the A + B cascade arms are skipped (--no-cascade)")
    print(f"  {seeds} seeds\n")

    print(f"  {'':<34}{'accuracy':>16}{'macro recall':>16}{'sent to B':>12}")

    results, never_built, per_seed = {}, [], {}
    recalls, talk_rounds, talk_stuck, talk_seen = {}, {}, {}, {}
    # how many of the k modules actually got built. Under slicing a rare
    # class can fall below min_pos in its slice and its module is skipped,
    # which costs macro recall because that class becomes unanswerable --
    # a different cost from "each module saw less data", and worth separating
    built = {}
    # an arm whose coordination stage could not be built falls back to plain
    # argmax on the same modules, which looks exactly like "coordination
    # changed nothing" and is not the same claim at all. Counted per arm so
    # a degraded run cannot be read as a null result.
    fellback, padded = {}, {}
    # Per-class modules train one forest PER CLASS, so with 7 classes they
    # hold 7x the trees of every other arm. More trees is the oldest free
    # lunch in ensembles and has nothing to do with splitting the problem,
    # so a single rebalanced forest gets the same tree budget as a control.
    # Without it, "modularity works" and "more trees works" are the same
    # number.
    nc = len(classes)
    nsl = max(1, slices)
    arms = ([("A alone (no cascade)", None, False, 1, 1),
             ("A alone, classes rebalanced", None, True, 1, 1),
             (f"A alone, rebalanced, {nc}x trees", None, True, nc, 1),
             ("per-class modules, loudest wins", "MODULES", False, 1, 1),
             ("per-class modules, talking", "TALK", False, 1, 1),
             ("per-class modules, learned referee", "STACK", False, 1, 1)]
            # the same three mechanisms with the global view taken away, so
            # the question stops being "does coordination help?" and becomes
            # "does coordination help ONLY when no module sees everything?"
            + ([(f"sliced /{nsl} modules, loudest wins", "MODULES", False,
                 1, nsl),
                (f"sliced /{nsl} modules, talking", "TALK", False, 1, nsl),
                (f"sliced /{nsl} modules, learned referee", "STACK", False,
                 1, nsl)] if nsl > 1 else [])
            + ([(f"A + B, B also {base!r}", base, False, 1, 1)]
               + [(f"A + B, B {k!r}", k, False, 1, 1)
                  for k in KINDS if k != base] if cascade else []))
    for tag, bkind, balance, mult, nparts in arms:
        acc, mac, frac = [], [], []
        for s in range(seeds):
            trR, trL, teR, teL = shuffled_split(rows, labels, seed=s)
            sc = fit_scale(trR)
            a, b = apply_scale(trR, sc), apply_scale(teR, sc)

            if bkind in ("MODULES", "TALK", "STACK"):
                make = modules_maker(classes, base, n_trees, maxd, nparts)
                if bkind == "MODULES":
                    mods = make(a, trL, s)
                    built.setdefault(tag, []).append(len(mods))
                    pred = [loudest(mods, r, big) for r in b]
                elif bkind == "STACK":
                    bs, ref, miss = build_stacked(a, trL, make, base, n_trees,
                                                  maxd, s, folds)
                    built.setdefault(tag, []).append(len(bs))
                    padded[tag] = padded.get(tag, 0) + miss
                    if ref is None:     # no referee could be built at all
                        fellback[tag] = fellback.get(tag, 0) + 1
                        pred = [loudest(bs, r, big) for r in b]
                    else:
                        pred = [stacked_predict(bs, ref, r, big) for r in b]
                else:
                    bs, tk, nm, miss = build_talking(a, trL, make, base,
                                                     n_trees, maxd, s, folds)
                    built.setdefault(tag, []).append(len(bs))
                    padded[tag] = padded.get(tag, 0) + miss
                    if tk is None:          # no round-1 modules at all
                        fellback[tag] = fellback.get(tag, 0) + 1
                        pred = [loudest(bs, r, big) for r in b]
                    else:
                        pred = []
                        for r in b:
                            lab, used, ok = talk_predict(bs, tk, nm, r, big,
                                                         rounds)
                            pred.append(lab)
                            talk_rounds.setdefault(tag, []).append(used)
                            talk_seen[tag] = talk_seen.get(tag, 0) + 1
                            if not ok:
                                talk_stuck[tag] = talk_stuck.get(tag, 0) + 1
                acc.append(sum(1 for p, l in zip(pred, teL)
                               if p == l) / len(teL))
                mac.append(macro_recall(pred, teL, classes))
                frac.append(0.0)
                recalls.setdefault(tag, []).append(
                    per_class_recall(pred, teL, classes))
                continue

            fitR, fitL = rebalance(a, trL, s) if balance else (a, trL)
            A = build_forest(fitR, fitL, (base,), n_trees=n_trees * mult,
                             maxd=maxd, seed=s)
            B, n_hard = None, 0
            if bkind is not None:
                hard = disputed_rows(a, trL, base, folds, n_trees, maxd, s,
                                     agree)
                n_hard = len(hard)
                # too few hard rows to fit anything on is a real outcome, not
                # an error: B simply never gets built and A answers alone
                if n_hard >= 30 and len(set(trL[i] for i in hard)) > 1:
                    B = build_forest([a[i] for i in hard],
                                     [trL[i] for i in hard], (bkind,),
                                     n_trees=n_trees, maxd=maxd, seed=s + 5)
                elif bkind == base:
                    never_built.append(n_hard)

            pred, sent = [], 0
            for r in b:
                v = votes(A, r)
                ans = max(v, key=v.get)
                if B is not None and max(v.values()) < agree * mult:
                    ans = max(votes(B, r), key=votes(B, r).get)
                    sent += 1
                pred.append(ans)

            acc.append(sum(1 for p, l in zip(pred, teL) if p == l) / len(teL))
            mac.append(macro_recall(pred, teL, classes))
            frac.append(sent / len(teL))

        results[tag] = statistics.mean(mac)
        per_seed[tag] = list(mac)

        def show(xs):
            return (f"{statistics.mean(xs):.1%} "
                    f"+/-{(max(xs) - min(xs)) / 2:.1%}")

        print(f"  {tag:<34}{show(acc):>16}{show(mac):>16}"
              f"{statistics.mean(frac):>12.1%}")

    print()
    if never_built:
        print(f"  B WAS NEVER BUILT: only {max(never_built)} training rows "
              f"were disputed, too few\n  to fit on. Raise --agree (closer "
              f"to {n_trees}) to send more rows to B, or\n  --trees to make "
              f"the vote finer. Every arm above is just A.\n")
    ref = "A alone (no cascade)"

    def paired(tag):
        """How often this arm beat the baseline on the SAME split, and by how
        much. Every arm sees identical splits seed for seed, so comparing a
        mean gap against a spread wastes that and is conservative in the
        wrong direction."""
        d = [x - y for x, y in zip(per_seed[tag], per_seed[ref])]
        return sum(1 for x in d if x > 0), len(d), statistics.mean(d)

    for tag in (x[0] for x in arms if x[0] != ref):
        w, n, avg = paired(tag)
        print(f"  {tag:<34}{avg:+8.1%}  on {w} of {n} seeds")

    cas = f"A + B, B also {base!r}"
    bal = "A alone, classes rebalanced"
    mod = "per-class modules, loudest wins"
    tlk = "per-class modules, talking"
    stk = "per-class modules, learned referee"
    cw, cn, cavg = paired(cas) if cas in per_seed else (0, 0, 0.0)
    bw, bn, bavg = paired(bal)
    mw, mn, mavg = paired(mod)
    solid = cas in per_seed and cw * 10 >= cn * 8 and cavg > 0.02

    # Two questions, in order. Do modules beat rebalancing one model? And if
    # so, is that the modularity or just the 7x tree budget they came with?
    matched = f"A alone, rebalanced, {len(classes)}x trees"
    xw, xn, xavg = paired(matched) if matched in per_seed else (0, 0, 0.0)

    if mavg <= bavg + 0.01:
        print(f"\n  MODULES ARE REBALANCING: per-class modules get "
              f"{mavg:+.1%} against {bavg:+.1%} for one\n  rebalanced forest. "
              f"Since modules.py rebalances each module via focus(), their\n  "
              f"reported win is the rebalancing, not the modularity.")
    elif matched not in per_seed:
        print(f"\n  MODULES BEAT REBALANCING: {mavg:+.1%} against "
              f"{bavg:+.1%}, but the tree-count\n  control did not run, so "
              f"capacity is not ruled out.")
    elif mavg <= xavg + 0.01:
        print(f"\n  IT IS THE TREE COUNT, NOT THE MODULARITY. Modules get "
              f"{mavg:+.1%}; one\n  rebalanced forest with the same "
              f"{len(classes)}x tree budget gets {xavg:+.1%}. Splitting\n  "
              f"the problem per class is not what bought the gain -- the "
              f"extra trees are.")
    else:
        gap = mavg - xavg
        print(f"\n  THE DECOMPOSITION IS REAL. Modules get {mavg:+.1%} "
              f"against {xavg:+.1%} for one\n  rebalanced forest holding the "
              f"SAME {len(classes)}x tree budget -- {gap:+.1%} that\n  "
              f"capacity does not explain, on {mw} of {mn} seeds.")
        print(f"\n  AND IT IS NOT ROUTING. build_modules trains every module "
              f"on EVERY row,\n  relabelled c-versus-rest, and loudest() asks "
              f"all of them about every row.\n  No row is diverted and no "
              f"module sees less data, so the cost that sank\n  hierarchy.py "
              f"and inhibit.py is not paid. What changes is the TARGET: "
              f"{len(classes)}\n  yes/no questions instead of one "
              f"{len(classes)}-way question. That is a change to what\n  the "
              f"model is TOLD, which is the side of the rule that keeps "
              f"winning.")

    # -- did talking help? ------------------------------------------------
    if tlk in per_seed:
        tw, tn, tavg = paired(tlk)
        d = [x - y for x, y in zip(per_seed[tlk], per_seed[mod])]
        wins = sum(1 for x in d if x > 0)
        gap = statistics.mean(d)
        rr = talk_rounds.get(tlk, [])
        avg_r = statistics.mean(rr) if rr else 0.0
        seen = talk_seen.get(tlk, 0)
        stuck = talk_stuck.get(tlk, 0) / seen if seen else 0.0
        print(f"\n  TALKING: {avg_r:.1f} rounds on average, {stuck:.1%} of "
              f"rows never settled")
        if gap > 0.01 and wins * 10 >= len(d) * 8:
            print(f"  IT HELPS: {gap:+.1%} over silent modules on {wins} of "
                  f"{len(d)} seeds. Letting the\n  modules read each other "
                  f"buys something, which is not what correlated\n  models "
                  f"reinforcing each other was predicted to do.")
        elif gap < -0.01:
            print(f"  IT HURTS: {gap:+.1%} against silent modules on {wins} "
                  f"of {len(d)} seeds. No new\n  information enters the loop "
                  f"-- a neighbour's confidence is a function of\n  the same "
                  f"row -- so all iterating can do is let correlated errors\n"
                  f"  reinforce each other, and it did.")
        else:
            print(f"  NO DIFFERENCE: {gap:+.1%} on {wins} of {len(d)} seeds. "
                  f"The modules reading each\n  other changes nothing, "
                  f"which is what no new information entering a loop\n  "
                  f"should look like.")
        if mod in recalls and tlk in recalls:
            print(f"\n  per-class recall, silent against talking "
                  f"(mean over seeds)")
            print(f"  {'class':<18}{'silent':>9}{'talking':>9}{'change':>9}")
            for c in classes:
                sv = [r.get(c) for r in recalls[mod] if c in r]
                tv = [r.get(c) for r in recalls[tlk] if c in r]
                if sv and tv:
                    a_, b_ = statistics.mean(sv), statistics.mean(tv)
                    print(f"  {str(c):<18}{a_:>9.1%}{b_:>9.1%}"
                          f"{b_ - a_:>+9.1%}")
            print(f"  Rare classes are where a reinforced error should show "
                  f"first: one module\n  wrong and six confident "
                  f"'not mine' votes agreeing with it.")

    # -- did a learned referee beat plain argmax? -------------------------
    if stk in per_seed and mod in per_seed:
        d = [x - y for x, y in zip(per_seed[stk], per_seed[mod])]
        wins = sum(1 for x in d if x > 0)
        gap = statistics.mean(d)
        if gap > 0.01 and wins * 10 >= len(d) * 8:
            print(f"\n  THE REFEREE HELPS: {gap:+.1%} over plain argmax on "
                  f"{wins} of {len(d)} seeds. The\n  modules' confidences "
                  f"were not comparable at face value, and learning how\n  "
                  f"to weigh them recovers what argmax threw away.")
        elif gap < -0.01:
            print(f"\n  THE REFEREE HURTS: {gap:+.1%} against plain argmax on "
                  f"{wins} of {len(d)} seeds. It\n  sees only k numbers where "
                  f"argmax sees the same k, so it cannot know more --\n  and "
                  f"it can overfit the fold confidences, which argmax has no "
                  f"way to do.")
        else:
            print(f"\n  THE REFEREE CHANGES NOTHING: {gap:+.1%} on {wins} of "
                  f"{len(d)} seeds. Every module was\n  rebalanced to 50/50, "
                  f"so they are all distorted the SAME way and argmax is\n  "
                  f"already comparing like with like. The miscalibration it "
                  f"was meant to fix\n  was largely symmetric.")

    # -- does coordination only pay when nobody has the whole picture? ----
    smod = f"sliced /{nsl} modules, loudest wins"
    stlk = f"sliced /{nsl} modules, talking"
    sstk = f"sliced /{nsl} modules, learned referee"
    if smod in per_seed:
        def gap_over(tag, ref_tag):
            d = [x - y for x, y in zip(per_seed[tag], per_seed[ref_tag])]
            return statistics.mean(d), sum(1 for x in d if x > 0), len(d)

        hcap, _, _ = gap_over(smod, mod)
        print(f"\n  THE HANDICAP: giving each module 1/{nsl} of the rows "
              f"instead of all of them\n  costs {hcap:+.1%} macro recall "
              f"before any coordination is added. That is the\n  cost "
              f"hierarchy.py and inhibit.py were paying, measured on its own.")
        bf = statistics.mean(built.get(mod, [0]))
        bc = statistics.mean(built.get(smod, [0]))
        if bc < bf - 0.01:
            print(f"\n  AND PART OF IT IS NOT THE DATA VOLUME: {bc:.1f} of "
                  f"{len(classes)} modules got built in the\n  sliced arms "
                  f"against {bf:.1f} with the global view, because a rare "
                  f"class can fall\n  below the {10}-positive floor inside "
                  f"a 1/{nsl} slice. A class with no module is\n  "
                  f"unanswerable, so that much of the {hcap:+.1%} is a "
                  f"missing module rather than a\n  thinner one. The "
                  f"coordination columns below are still clean -- they "
                  f"compare\n  sliced against sliced -- but the handicap "
                  f"figure itself is two effects.")

        print(f"\n  {'coordination gap over silent modules':<40}"
              f"{'global view':>14}{'sliced':>14}")
        pairs = []
        for name, full, cut in (("talking", tlk, stlk),
                                ("learned referee", stk, sstk)):
            if full not in per_seed or cut not in per_seed:
                continue
            gf, wf, nf = gap_over(full, mod)
            gc, wc, ncs = gap_over(cut, smod)
            pairs.append((name, gf, wf, nf, gc, wc, ncs))
            print(f"  {name:<40}{gf:>+13.1%} {gc:>+13.1%}")
        print(f"  Each column compares against the silent modules built the "
              f"SAME way, so the\n  handicap itself cancels and what is left "
              f"is what coordination added.")

        # -- did the sliced arms actually run? ---------------------------
        bad = [(t, fellback.get(t, 0)) for t in (stlk, sstk)
               if fellback.get(t, 0)]
        pad = [(t, padded.get(t, 0)) for t in (stlk, sstk)
               if padded.get(t, 0)]
        if bad:
            worst = max(n for _, n in bad)
            print(f"\n  THE SLICED COORDINATION ARMS DID NOT RUN. On "
                  f"{worst} of {seeds} seeds a fold could not\n  build two "
                  f"modules, so there was nothing to coordinate and the arm "
                  f"fell back\n  to plain argmax on the same modules. Those "
                  f"rows above are the silent arm\n  printed three times, "
                  f"NOT three measurements -- identical numbers across the\n"
                  f"  three sliced arms is the signature. Nothing here says "
                  f"anything about\n  coordination; the handicap figure "
                  f"above is still good.")
            print(f"\n  WHY: the handicaps compound. The out-of-fold folds "
                  f"hold back 1/{folds} of the\n  training rows and the "
                  f"slice keeps 1/{nsl} of what is left, so each module "
                  f"trains\n  on about {(folds - 1) / (folds * nsl):.0%} of "
                  f"the data and a rare class drops under the "
                  f"ten-positive\n  floor. Re-run with --slices 2, or "
                  f"--folds 2, so every module survives.")
        elif pad:
            worst = max(n for _, n in pad)
            print(f"\n  PARTIALLY DEGRADED: {worst} module-folds came back "
                  f"empty and were filled with\n  a 0.0 confidence, so the "
                  f"downstream stage saw a column that is always zero\n  "
                  f"for those classes and could not learn anything from it. "
                  f"The arms below DID\n  run, but read them knowing that. "
                  f"--slices 2 or --folds 2 avoids it.")

        if not bad:
            helps = [x for x in pairs if x[4] > 0.01 and x[5] * 10 >= x[6] * 8]
            flat_full = [x for x in pairs if abs(x[1]) <= 0.01]
            if helps and len(flat_full) == len(pairs):
                got = ", ".join(f"{x[0]} {x[4]:+.1%}" for x in helps)
                print(f"\n  BOUNDARY CONDITION FOUND. Coordination between "
                      f"modules pays only when no\n  module can see all the "
                      f"data: {got} once the rows are cut up, and\n  nothing "
                      f"when every module already has them. Four null results "
                      f"stop being\n  four failures and become one statement "
                      f"about WHEN these mechanisms apply.")
                if statistics.mean(per_seed[smod]) + max(x[4] for x in helps) \
                        < statistics.mean(per_seed[mod]) - 0.005:
                    print(f"\n  AND IT DOES NOT PAY FOR ITSELF. The best sliced "
                          f"arm is still below the\n  silent global-view "
                          f"modules, so coordination recovers part of what the\n"
                          f"  slicing destroyed and never all of it. If you can "
                          f"give every module\n  every row, do that instead of "
                          f"making them talk.")
                else:
                    print(f"\n  AND IT MORE THAN PAYS FOR ITSELF HERE, which was "
                          f"not predicted: a sliced\n  arm reaches the "
                          f"global-view modules. Worth a second look before it "
                          f"is\n  believed -- the slices are stratified, so each "
                          f"module still sees the same\n  class proportions, "
                          f"only fewer rows.")
            elif helps:
                got = ", ".join(f"{x[0]} {x[4]:+.1%}" for x in helps)
                print(f"\n  COORDINATION HELPS WHEN SLICED ({got}) BUT ALSO "
                      f"WITH THE GLOBAL VIEW, so\n  this is not a boundary "
                      f"condition -- it is a plain win that the earlier runs\n"
                      f"  were too quiet to show. Read the table above, not the "
                      f"earlier null result.")
            else:
                print(f"\n  NO BOUNDARY CONDITION. Taking the global view away "
                      f"does NOT make talking or\n  a referee start earning "
                      f"anything, so the four null results are not about\n  "
                      f"modules already seeing everything. The mechanisms add "
                      f"nothing here either\n  way, and the explanation that "
                      f"was offered for them -- no new information\n  enters "
                      f"the loop -- survives its own test, because cutting the "
                      f"data up DOES\n  put new information in each module and "
                      f"the loop still cannot use it.")

    if cas not in per_seed:
        print(f"\n  THE CASCADE ARMS DID NOT RUN (--no-cascade). Nothing "
              f"here speaks to A + B;\n  that question was settled in an "
              f"earlier run and the arms are skipped only\n  because they "
              f"are the slow ones.")
    elif solid and bavg >= cavg - 0.01:
        print(f"\n  THE CASCADE IS DOING CLASS REBALANCING. Simply oversampling "
              f"the rare classes\n  gets {bavg:+.1%} against the cascade's "
              f"{cavg:+.1%}, in one line and without a second\n  model. "
              f"Prefer the one line.")
    elif solid:
        print(f"\n  THE CASCADE HELPS beyond rebalancing: {cavg:+.1%} on "
              f"{cw} of {cn} seeds, against\n  {bavg:+.1%} for plain "
              f"rebalancing. Training a second model on the disputed\n  rows "
              f"is buying something the class balance alone does not.")
    else:
        print(f"\n  NOT SOLID. The cascade averages {cavg:+.1%} but wins only "
              f"{cw} of {cn} seeds, so it\n  is not consistent enough to "
              f"rely on. Raise --seeds before believing it.")
    print(f"\n  Macro recall averages the per-class recalls, so a rare class "
          f"counts as much\n  as a common one -- accuracy hides exactly the "
          f"classes a cascade should help.\n")


if __name__ == "__main__":
    f = sys.argv[1:]

    def opt(name, cast, default):
        return cast(f[f.index(name) + 1]) if name in f else default

    csv = f[f.index("--csv") + 1] if "--csv" in f else None
    label = f[f.index("--label") + 1] if "--label" in f else None
    if not csv or not label:
        raise SystemExit(__doc__)
    drop = (f[f.index("--drop") + 1].split(",") if "--drop" in f
            else ("id", "ID", "Id"))

    run(csv, label, drop, seeds=opt("--seeds", int, 5),
        maxd=opt("--maxd", int, 6), n_trees=opt("--trees", int, 9),
        folds=opt("--folds", int, 3), agree=opt("--agree", int, None),
        rounds=opt("--rounds", int, 3), slices=opt("--slices", int, 3),
        cascade="--no-cascade" not in f)
