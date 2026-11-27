
"""Collapse several 0/1 columns into one label column.

    python3 onehot.py in.csv out.csv Pastry Z_Scratch K_Scatch Stains ...

Datasets often spread one question across several columns: seven fault types
as seven flags, exactly one of them set. That is fine for predicting ONE
fault, but to ask "which fault is this?" they have to become a single column.

Writes a new CSV with the named columns replaced by one called Label.
"""

import csv
import sys


def collapse(header, raw, flags, new_name="Label", none_name="None"):
    idx = []
    for f in flags:
        if f not in header:
            raise SystemExit(f"  no column {f!r}. file has: {', '.join(header)}")
        idx.append(header.index(f))

    keep = [i for i in range(len(header)) if i not in idx]
    out_header = [header[i] for i in keep] + [new_name]

    out_rows, tally, odd = [], {}, 0
    for r in raw:
        on = [f for f, i in zip(flags, idx) if r[i].strip() not in ("", "0")]
        if len(on) == 1:
            label = on[0]
        elif not on:
            label = none_name
        else:
            label = on[0]          # more than one set: take the first
            odd += 1
        tally[label] = tally.get(label, 0) + 1
        out_rows.append([r[i] for i in keep] + [label])

    return out_header, out_rows, tally, odd


def main(argv):
    if len(argv) < 4:
        raise SystemExit(__doc__)
    src, dst, flags = argv[1], argv[2], argv[3:]

    with open(src, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw = [r for r in reader if any(c.strip() for c in r)]

    out_header, out_rows, tally, odd = collapse(header, raw, flags)

    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(out_header)
        w.writerows(out_rows)

    print(f"\n  {len(raw)} rows: {len(header)} columns in, {len(out_header)} out")
    print(f"  {len(flags)} flag columns became one column called 'Label'\n")
    for k in sorted(tally, key=lambda k: -tally[k]):
        print(f"    {k:<20}{tally[k]:>6}{tally[k]/len(raw):>8.1%}")
    if odd:
        print(f"\n  {odd} rows had more than one flag set; took the first.")
    print(f"\n  wrote {dst}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))