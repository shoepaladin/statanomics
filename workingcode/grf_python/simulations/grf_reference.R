#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R grf reference for the Python<->R CATE comparison (issue #5 validation).
#
# Fits grf::causal_forest on the SAME (X, Y, W) the Python harness uses (read
# from CSV fixtures written by compare_vs_grf.py) and writes per-test-point
# predictions + variance estimates back to CSV.  The Python side then computes
# slope-vs-truth, coverage, and mean|err| for both estimators on identical data
# — the apples-to-apples comparison PR #4 reported and issue #5 asks us to keep
# using.
#
# Ground truth for this port is R grf; this script IS that reference.
#
# Usage (driven by compare_vs_grf.py, but runnable standalone):
#   Rscript grf_reference.R <fixtures_dir>
# where <fixtures_dir> contains train.csv (cols x0..x{p-1}, Y, W) and
# test.csv (cols x0..x{p-1}).  Writes grf_pred.csv (pred, lower, upper, se).
# ---------------------------------------------------------------------------

suppressMessages(library(grf))

args <- commandArgs(trailingOnly = TRUE)
fixtures <- if (length(args) >= 1) args[1] else "."

train <- read.csv(file.path(fixtures, "train.csv"))
test  <- read.csv(file.path(fixtures, "test.csv"))

xcols <- grep("^x[0-9]+$", names(train), value = TRUE)
Xtr <- as.matrix(train[, xcols])
Ytr <- train$Y
Wtr <- train$W
Xte <- as.matrix(test[, xcols])

# Match the Python forest's headline knobs so the comparison isolates the
# *implementation*, not hyperparameters: grf defaults already use
# honest splitting, sample.fraction=0.5, ci.group.size=2, mtry=min(p,ceil(sqrt(p)+20)),
# and the OOB regression-forest nuisance.  We only pin num.trees / min.node.size
# / seed to the Python run via environment variables set by the driver.
num_trees <- as.integer(Sys.getenv("GRF_NUM_TREES", "2000"))
min_node  <- as.integer(Sys.getenv("GRF_MIN_NODE",  "10"))
seed      <- as.integer(Sys.getenv("GRF_SEED",      "1"))

cf <- causal_forest(
  X = Xtr, Y = Ytr, W = Wtr,
  num.trees = num_trees,
  min.node.size = min_node,
  honesty = TRUE,
  seed = seed
)

pr <- predict(cf, Xte, estimate.variance = TRUE)
se <- sqrt(pmax(pr$variance.estimates, 0))
z  <- qnorm(0.975)

out <- data.frame(
  pred  = pr$predictions,
  lower = pr$predictions - z * se,
  upper = pr$predictions + z * se,
  se    = se
)
write.csv(out, file.path(fixtures, "grf_pred.csv"), row.names = FALSE)
cat(sprintf("grf %s: wrote %d predictions (num.trees=%d, min.node.size=%d, seed=%d)\n",
            as.character(packageVersion("grf")), nrow(out), num_trees, min_node, seed))
