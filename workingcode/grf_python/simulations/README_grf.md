# Python ⇆ R grf comparison harness

R `grf` is the ground truth this package ports. `compare_vs_grf.py` fits
**both** `NumbaCausalForest` and `grf::causal_forest` on the **same** `(X, Y, W)`
rows and reports slope-vs-truth, 95% CI coverage, and mean|err| for each — the
apples-to-apples comparison PR #4 used and issue #5 asks us to keep using.

## Files
- `grf_reference.R` — fits `grf::causal_forest` on CSV fixtures, writes predictions + CIs.
- `compare_vs_grf.py` — driver: draws the DGP, runs Python, shells out to `Rscript`, tabulates.

## Running it

```bash
# from workingcode/grf_python
python simulations/compare_vs_grf.py --reps 10 --n 800 --p 6 --num-trees 1000
```

This needs **R with the `grf` package** on `PATH` (i.e. `Rscript` resolves and
`library(grf)` works). If `grf` is installed in a non-default library, point R
at it: `R_LIBS=/path/to/lib python simulations/compare_vs_grf.py ...`.

## Setting up R + grf

The only requirement is `grf` 2.6.1; everything else (`Matrix`, `methods`,
`stats`) ships with R. `grf`'s build-time deps are just `Rcpp` and `RcppEigen`.

**Normal environment (CRAN reachable):**
```r
install.packages("grf")        # pulls Rcpp, RcppEigen and compiles; a few minutes
```

**Restricted environment (CRAN blocked, e.g. Claude Code on the web):**
CRAN mirrors are not on the default network allowlist, so `install.packages`
fails at the proxy (HTTP 403). Two options:

1. **Allow CRAN in the environment's network policy** — add `cran.r-project.org`
   / `cloud.r-project.org` (and `p3m.dev` for precompiled binaries) when
   creating the environment, then `install.packages("grf")` just works. See
   https://code.claude.com/docs/en/claude-code-on-the-web (network policies).

2. **Build from the GitHub CRAN mirror** (no CRAN access needed; `github.com`
   and `codeload.github.com` are reachable). Each CRAN package is mirrored at
   `github.com/cran/<pkg>`:
   ```bash
   apt-get install -y r-base-core            # R itself, g++, gfortran, make
   mkdir rsrc && cd rsrc
   for p in Rcpp RcppEigen grf; do
     curl -sSL -o $p.tgz https://codeload.github.com/cran/$p/tar.gz/refs/heads/master
     tar xzf $p.tgz
   done
   # grf declares DiceKriging/lmtest/sandwich in Imports but only causal_forest
   # is needed here and those are not loaded by it; drop them so install does
   # not demand them (they pull a heavier dependency chain):
   sed -i -E 's/^Imports:.*/Imports: Matrix, methods, Rcpp (>= 0.12.15)/' grf-master/DESCRIPTION
   export R_LIBS=$PWD/rlib && mkdir -p "$R_LIBS"
   R CMD INSTALL --library="$R_LIBS" --no-docs Rcpp-master RcppEigen-master grf-master
   R_LIBS="$R_LIBS" Rscript -e 'library(grf); packageVersion("grf")'   # 2.6.1
   ```
   The `grf` C++ core takes a few minutes to compile. This is exactly how the
   reference numbers in PR #6 were produced.

## What "matches grf" means here
The comparison isolates the *implementation*: both sides use honest splitting,
`sample.fraction = 0.5`, `ci.group.size = 2`, the grf `mtry` default, and an OOB
regression-forest nuisance, with `num.trees` / `min.node.size` / `seed` pinned to
the Python run. A faithful port should land within Monte-Carlo noise of grf on
slope / coverage / mean|err| — **not** at slope 1.0 / coverage 0.95, because R
grf itself does not reach those on this finite-n DGP (its CATEs are attenuated
and its normal-CI coverage is below nominal too).
