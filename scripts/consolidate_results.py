"""Collect every finished LOSO fold into one table.

Usage:
    python consolidate_results.py <SCLDGN/output/bci42a> [out.csv]
"""
import csv
import os
import re
import sys

import numpy as np

PAPER_MEAN, PAPER_STD = 69.85, 9.16


def parse(results_csv):
    txt = open(results_csv).read()

    def acc(tag):
        m = re.search(rf"^{tag}.*?'acc': ([0-9.]+)", txt, re.M | re.S)
        return float(m.group(1)) * 100

    return (acc("train"), acc("val"), acc("test"),
            int(float(re.search(r"^bestEpoch,([0-9.]+)", txt, re.M).group(1))),
            float(re.search(r"^runTime,([0-9.]+)", txt, re.M).group(1)))


def main():
    root = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "RESULTS.csv"

    rows = []
    for dirpath, _, files in os.walk(root):
        if "results.csv" not in files:
            continue
        sub = re.search(r"sub(\d+)$", dirpath)
        if not sub:
            continue
        tr, va, te, be, rt = parse(os.path.join(dirpath, "results.csv"))
        rows.append((int(sub.group(1)), tr, va, te, be, rt))

    rows.sort()
    if not rows:
        raise SystemExit(f"no fold results found under {root}")

    test = np.array([r[3] for r in rows])
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "train_acc", "val_acc", "test_acc",
                    "best_epoch", "runtime_min"])
        for i, tr, va, te, be, rt in rows:
            w.writerow([f"A{i+1:02d}", f"{tr:.2f}", f"{va:.2f}",
                        f"{te:.2f}", be, f"{rt:.1f}"])
        w.writerow([])
        w.writerow(["mean", "", "", f"{test.mean():.2f}", "", ""])
        w.writerow(["std", "", "", f"{test.std(ddof=1):.2f}", "", ""])
        w.writerow(["paper_mean", "", "", f"{PAPER_MEAN}", "", ""])
        w.writerow(["paper_std", "", "", f"{PAPER_STD}", "", ""])
        w.writerow(["difference", "", "", f"{test.mean() - PAPER_MEAN:+.2f}", "", ""])

    print(f"{len(rows)} folds -> {out}")
    print(f"mean {test.mean():.2f} +/- {test.std(ddof=1):.2f}   "
          f"paper {PAPER_MEAN} +/- {PAPER_STD}   diff {test.mean() - PAPER_MEAN:+.2f}")


if __name__ == "__main__":
    main()
