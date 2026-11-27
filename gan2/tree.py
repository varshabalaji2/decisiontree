


def to_numbers(header, raw, label_col, drop=()):
    """Text columns become numbers. Returns rows, labels, feature names."""
    drop = set(drop) | {label_col}
    keep = [i for i, n in enumerate(header) if n not in drop]

    maps = {}
    for i in keep:
        vals = [r[i] for r in raw]
        try:
            [float(v) for v in vals]
        except ValueError:
            maps[i] = {v: n for n, v in enumerate(sorted(set(vals)))}

    names = [header[i] + (" (coded)" if i in maps else "") for i in keep]
    rows = [[maps[i][r[i]] if i in maps else float(r[i]) for i in keep] for r in raw]
    labels = [r[header.index(label_col)] for r in raw]
    return rows, labels, names


def asks(x, split, k=0.5):
    """One neuron. Returns 1 if x is bigger than split, else 0."""
    v = 0.0
    v = v + k * (x - v)          # your neuron, one step from empty
    return 1 if v >= k * split else 0
print(asks(7, 5))    # 7 > 5  -> 1
print(asks(2, 5))    # 2 > 5  -> 0
def tree(x, y):
    if asks(x, 5):               # neuron 1
        return "A" if asks(y, 8) else "B"    # neuron 2
    else:
        return "C" if asks(y, 7) else "D"    # neuron 3
for x, y in [(8, 4), (8, 1), (2, 9), (2, 3)]:
    print(x, y, "->", tree(x, y))
for y in range(10, -1, -1):
    print(f"y={y:>2} |" + "".join(tree(x, y) for x in range(11)))
def mixed(labels):
    """0.0 if all the same. Higher if jumbled."""
    if len(labels) == 0:
        return 0.0
    score = 1.0
    for kind in set(labels):
        p = labels.count(kind) / len(labels)
        score = score - p * p
    return score

def score(rows, labels, feature, split):
    left  = [l for r, l in zip(rows, labels) if not asks(r[feature], split)]
    right = [l for r, l in zip(rows, labels) if     asks(r[feature], split)]
    n = len(labels)
    return len(left)/n * mixed(left) + len(right)/n * mixed(right)
def best_split(rows, labels):
    best = None
    for feature in range(len(rows[0])):
        for split in sorted(set(r[feature] for r in rows)):
            s = score(rows, labels, feature, split)
            if best is None or s < best[0]:
                best = (s, feature, split)
    return best
import random
random.seed(1)

rows = [[random.randint(0, 10), random.randint(0, 10)] for _ in range(30)]

# a rule only YOU know
labels = ["YES" if r[1] > 3 else "NO" for r in rows]

impurity, feature, split = best_split(rows, labels)
print("it found: column", feature, "split at", split, " impurity", round(impurity, 3))
def majority(labels):
    return max(set(labels), key=labels.count)
def build(rows, labels, depth=0,maxd=8):
    if mixed(labels) == 0 or depth >= maxd or len(labels) <4:
        return majority(labels)                    # pure enough — this is a leaf

    s, f, sp = best_split(rows, labels)

    left_r  = [r for r in rows if not asks(r[f], sp)]
    left_l  = [l for r, l in zip(rows, labels) if not asks(r[f], sp)]
    right_r = [r for r in rows if     asks(r[f], sp)]
    right_l = [l for r, l in zip(rows, labels) if asks(r[f], sp)]
    if not left_l or not right_l:        # <-- ADD THIS
        return majority(labels)

    return (f, sp, build(left_r, left_l, depth+1,maxd),
                   build(right_r, right_l, depth+1,maxd))

def use(tree, row):
    while isinstance(tree, tuple):
        f, sp, lo, hi = tree
        tree = hi if asks(row[f], sp) else lo
    return tree
labels = ["YES" if r[1] > 3 else "NO" for r in rows]
t = build(rows, labels)
print(use(t, [0, 9]), use(t, [0, 1]))     # expect: YES NO

def make(kind, rng):
    img = [[0] * 8 for _ in range(8)]
    if kind == "H":
        r = rng.randint(1, 6)                  # <-- ADD: pick a random row
        for c in range(1, 7): img[r][c] = 1    # <-- CHANGE: img[3] becomes img[r]
    else:
        c = rng.randint(1, 6)                  # <-- ADD: pick a random column
        for r in range(1, 7): img[r][c] = 1    # <-- CHANGE: [3] becomes [c]
    for _ in range(4):
        img[rng.randrange(8)][rng.randrange(8)] ^= 1
    rows_sum = [sum(r) for r in img]
    cols_sum = [sum(img[r][c] for r in range(8)) for c in range(8)]
    return [p for row in img for p in row] + rows_sum + cols_sum
import random
rng = random.Random(0)
img = make("H", rng)
for r in range(8):
    print("".join("#" if img[r*8 + c] else "." for c in range(8)))
rng = random.Random(0)
data = [(make("H", rng), "H") for _ in range(150)] + \
       [(make("V", rng), "V") for _ in range(150)]
rng.shuffle(data)

rows   = [d[0] for d in data]
labels = [d[1] for d in data]

train_rows, train_labels = rows[:200], labels[:200]
test_rows,  test_labels  = rows[200:], labels[200:]

tree = build(train_rows, train_labels, maxd=8)

def acc(t, R, L):
    return sum(use(t, r) == l for r, l in zip(R, L)) / len(L)

print("trained on:", round(acc(tree, train_rows, train_labels) * 100), "%")
print("never seen:", round(acc(tree, test_rows,  test_labels)  * 100), "%")

tree = build(train_rows, train_labels, maxd=8)

print(majority(["H", "H", "V"]))    # H
print(majority(["V", "H", "V"]))    # V

def acc(t, R, L):
    return sum(use(t, r) == l for r, l in zip(R, L)) / len(L)

print("maxd  trained  unseen")
for maxd in (1, 2, 3, 4, 6, 8, 12):
    t = build(train_rows, train_labels, maxd=maxd)
    print(f"{maxd:>4}  {acc(t, train_rows, train_labels):>6.0%}  {acc(t, test_rows, test_labels):>6.0%}")

 # ================= HAPPY PATH TESTS =================
assert asks(7, 5) == 1,  "7 is above 5, should fire"
assert asks(2, 5) == 0,  "2 is below 5, should not fire"
assert asks(5, 5) == 1,  "exactly at the split counts as above"

assert mixed(["a","a","a"]) == 0.0
assert mixed(["a","b"]) == 0.5
assert mixed([]) == 0.0

assert majority(["H","H","V"]) == "H"

rows = [[1],[2],[8],[9]]; labels = ["lo","lo","hi","hi"]
assert score(rows, labels, 0, 5) == 0.0,  "a perfect cut leaves both sides pure"
assert score(rows, labels, 0, 1) == 0.5,  "a useless cut leaves the mess intact"

imp, f, sp = best_split(rows, labels)
assert imp == 0.0 and f == 0

rng = random.Random(1)
r2 = [[rng.randint(0,10), rng.randint(0,10)] for _ in range(40)]
lab_x = ["Y" if r[0] > 5 else "N" for r in r2]
lab_y = ["Y" if r[1] > 5 else "N" for r in r2]
assert best_split(r2, lab_x)[1] == 0, "rule is in column 0"
assert best_split(r2, lab_y)[1] == 1, "rule is in column 1"

t = build(r2, lab_x)
assert use(t, [9, 0]) == "Y"
assert use(t, [1, 0]) == "N"

noise = [rng.choice(["Y","N"]) for _ in r2]
assert best_split(r2, noise)[0] > 0.3, "must not invent patterns in noise"

def depth_of(t): return 0 if not isinstance(t,tuple) else 1+max(depth_of(t[2]),depth_of(t[3]))
for d in (1,2,3):
    assert depth_of(build(r2, noise, maxd=d)) <= d, f"maxd={d} not respected"

print("all tests passed")  
def drill_row(rng):
    speed     = rng.uniform(10, 50)
    feed      = rng.uniform(0.1, 0.6)
    sharpness = rng.uniform(0, 100)
    fail = 1 if (sharpness < 40 and feed > 0.35) else 0
    if rng.random() < 0.05: fail = 1 - fail
    return [speed, feed, sharpness], fail

def make_drill_data(n, seed):
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n):
        r, l = drill_row(rng)
        rows.append(r); labels.append(l)
    return rows, labels
train_rows, train_labels = make_drill_data(600, seed=1)
test_rows,  test_labels  = make_drill_data(600, seed=2)

print(f"\n  failures in training data: {sum(train_labels)} of {len(train_labels)}")
print(f"  always-guess-'no-failure' scores "
      f"{1 - sum(test_labels)/len(test_labels):.1%}\n")

for maxd in (1, 2, 3, 4, 6, 8, 12):
    t = build(train_rows, train_labels, maxd=maxd)
    tr = sum(1 for r, l in zip(train_rows, train_labels) if use(t, r) == l) / len(train_labels)
    te = sum(1 for r, l in zip(test_rows,  test_labels)  if use(t, r) == l) / len(test_labels)
    print(f"  maxd {maxd:>2}   train {tr:>6.1%}   test {te:>6.1%}")
def report(tree, rows, labels, name):
    right = sum(1 for r, l in zip(rows, labels) if use(tree, r) == l)
    tp = sum(1 for r, l in zip(rows, labels) if use(tree, r) == 1 and l == 1)
    fp = sum(1 for r, l in zip(rows, labels) if use(tree, r) == 1 and l == 0)
    fn = sum(1 for r, l in zip(rows, labels) if use(tree, r) == 0 and l == 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    print(f"  {name:<12} accuracy {right/len(labels):>6.1%}"
          f"   precision {prec:>6.1%}   recall {rec:>6.1%}")

print()
t = build(train_rows, train_labels, maxd=4)
report(t, train_rows, train_labels, "training")
report(t, test_rows,  test_labels,  "never seen")
NAMES = ["speed", "feed", "sharpness"]

def show(tree, depth=0):
    pad = "  " * (depth + 1)
    if not isinstance(tree, tuple):
        print(f"{pad}-> {'FAIL' if tree == 1 else 'ok'}")
        return
    f, sp, lo, hi = tree
    print(f"{pad}is {NAMES[f]} below {sp:.2f}?   (yes first, no second)")
    show(lo, depth + 1)
    show(hi, depth + 1)

print("\n  true rule: sharpness < 40 AND feed > 0.35\n")
show(build(train_rows, train_labels, maxd=3))