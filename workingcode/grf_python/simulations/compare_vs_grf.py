#!/usr/bin/env python3
"""
Python NumbaCausalForest vs. R grf 2.6.1 on IDENTICAL data (issue #5 validation).

R grf is the ground truth this package ports.  For each replication we draw one
(train, test) pair from the non-null harness DGP (tau(x) = x0), write it to CSV,
fit BOTH estimators on exactly those rows, and compare the metrics that PR #4
reported:

    slope     : OLS slope of predicted CATE on the true CATE (1.0 = no attenuation)
    coverage  : fraction of held-out points whose 95% CI covers the true tau
    mean|err| : mean absolute error of the point estimate

The R side is simulations/grf_reference.R; this driver shells out to Rscript.
Set R_LIBS if grf lives in a non-default library (see simulations/README_grf.md).

Usage:
    python simulations/compare_vs_grf.py --reps 10 --n 800 --p 6 --num-trees 2000
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grf.forest_numba import NumbaCausalForest


def tau_fn(X, dgp):
    """Ground-truth CATE for the chosen DGP."""
    x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
    if dgp == "linear":            # original: linear, monotonic, univariate
        return x0
    if dgp == "sine":              # non-monotonic, univariate (one trough + one peak)
        return np.sin(np.pi * x0)
    if dgp == "hills":             # non-linear, non-monotonic, multivariate w/ interaction
        # sin(pi*x0): non-monotone in x0; x1*x2: saddle (non-monotone interaction);
        # bump in x0: a localized feature that fine leaves must resolve.
        return np.sin(np.pi * x0) + x1 * x2 + 0.5 * np.exp(-8.0 * x0 ** 2)
    raise ValueError(f"unknown dgp {dgp!r}")


def make_data(n, p, noise, seed, dgp="linear"):
    """W randomized; homoskedastic Gaussian noise; tau set by `dgp`."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, p))
    W = rng.binomial(1, 0.5, n).astype(float)
    tau = tau_fn(X, dgp)
    Y = tau * W + rng.normal(0, noise, n)
    return X, Y, W, tau


def metrics(pred, lo, hi, tau_true):
    slope = float(np.polyfit(tau_true, pred, 1)[0])
    coverage = float(np.mean((tau_true >= lo) & (tau_true <= hi)))
    mae = float(np.mean(np.abs(pred - tau_true)))
    return slope, coverage, mae


def run_r_grf(workdir, Xtr, Ytr, Wtr, Xte, num_trees, min_node, seed):
    """Write fixtures, invoke grf_reference.R, read back predictions."""
    p = Xtr.shape[1]
    cols = [f"x{j}" for j in range(p)]
    import csv

    with open(os.path.join(workdir, "train.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["Y", "W"])
        for i in range(len(Xtr)):
            w.writerow(list(Xtr[i]) + [Ytr[i], Wtr[i]])
    with open(os.path.join(workdir, "test.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(Xte)):
            w.writerow(list(Xte[i]))

    env = dict(os.environ,
               GRF_NUM_TREES=str(num_trees),
               GRF_MIN_NODE=str(min_node),
               GRF_SEED=str(seed))
    r_script = os.path.join(os.path.dirname(__file__), "grf_reference.R")
    subprocess.run(["Rscript", r_script, workdir], check=True, env=env)

    out = np.genfromtxt(os.path.join(workdir, "grf_pred.csv"),
                        delimiter=",", names=True)
    return out["pred"], out["lower"], out["upper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--p", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--num-trees", type=int, default=2000)
    ap.add_argument("--min-node", type=int, default=10)
    ap.add_argument("--dgp", choices=["linear", "sine", "hills"], default="linear",
                    help="ground-truth CATE: linear (default), sine (non-monotone), "
                         "hills (non-linear, non-monotone, multivariate interaction)")
    args = ap.parse_args()

    py = np.zeros((args.reps, 3))
    r = np.zeros((args.reps, 3))

    for rep in range(args.reps):
        seed_tr, seed_te = 100 + rep, 900 + rep
        Xtr, Ytr, Wtr, _ = make_data(args.n, args.p, args.noise, seed_tr, args.dgp)
        Xte, _, _, tau_te = make_data(args.n_test, args.p, args.noise, seed_te, args.dgp)

        # Python
        f = NumbaCausalForest(n_trees=args.num_trees, min_leaf_size=args.min_node,
                              max_depth=8, n_jobs=-1, random_state=seed_tr).fit(Xtr, Ytr, Wtr)
        pred, lo, hi = f.predict_interval(Xte)
        py[rep] = metrics(pred, lo, hi, tau_te)

        # R grf on the same rows
        with tempfile.TemporaryDirectory() as wd:
            rp, rlo, rhi = run_r_grf(wd, Xtr, Ytr, Wtr, Xte,
                                     args.num_trees, args.min_node, seed_tr)
        r[rep] = metrics(rp, rlo, rhi, tau_te)

        print(f"rep {rep}: py(slope={py[rep,0]:.3f} cov={py[rep,1]:.3f} "
              f"mae={py[rep,2]:.3f})  R(slope={r[rep,0]:.3f} cov={r[rep,1]:.3f} "
              f"mae={r[rep,2]:.3f})")

    print("\n=== mean over reps (dgp={}, n={}, p={}, num_trees={}, min_node={}) ==="
          .format(args.dgp, args.n, args.p, args.num_trees, args.min_node))
    hdr = f"{'estimator':<16}{'slope':>8}{'coverage':>10}{'mean|err|':>11}"
    print(hdr); print("-" * len(hdr))
    print(f"{'Python (this PR)':<16}{py[:,0].mean():>8.3f}{py[:,1].mean():>10.3f}{py[:,2].mean():>11.3f}")
    print(f"{'R grf 2.6.1':<16}{r[:,0].mean():>8.3f}{r[:,1].mean():>10.3f}{r[:,2].mean():>11.3f}")


if __name__ == "__main__":
    main()
