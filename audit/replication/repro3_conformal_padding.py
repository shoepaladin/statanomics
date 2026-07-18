"""Repro 3: time_block_permutation pads the CWZ moving-block permutation set
with iid random permutations of time (panelib.py:1515-1522). Under serially
correlated residuals (the case the block scheme exists for), the padded set
over-rejects; the pure block set is approximately size-correct."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/statanomics/workingcode/panelmodels')
import numpy as np, pandas as pd
from panelib import conformal_inf

T_pre, T_post, rho, reps = 20, 5, 0.9, 800
T = T_pre + T_post
rng = np.random.default_rng(11)

# build the two permutation sets once (library's padded set is random; rebuild each rep is slow, reuse ok)
dummy = pd.DataFrame({'t': np.arange(T), 'post': (np.arange(T) >= T_pre).astype(int)})
padded, pre_pst = conformal_inf.time_block_permutation(data=dummy, time_unit='t', post='post')
blocks_only = padded[:T]          # the T cyclic shifts
print(f"permutation set sizes: padded={len(padded)}, blocks_only={len(blocks_only)}")

rej_pad = rej_blk = 0
for r in range(reps):
    # H0 true: observed - counterfactual = AR(1) noise, no treatment effect
    u = np.empty(T)
    u[0] = rng.normal(0, 1/np.sqrt(1-rho**2))
    for t in range(1, T):
        u[t] = rho*u[t-1] + rng.normal()
    actual = u
    cf = np.zeros(T)
    p_pad = conformal_inf.pvalue_calc(counterfactual=cf, actual=actual.copy(),
                                      permutation_list=padded, pre_pst_lengths=pre_pst, h0=0)
    p_blk = conformal_inf.pvalue_calc(counterfactual=cf, actual=actual.copy(),
                                      permutation_list=blocks_only, pre_pst_lengths=pre_pst, h0=0)
    rej_pad += int(p_pad <= 0.05)
    rej_blk += int(p_blk <= 0.05)

print(f"AR(1) rho={rho}, H0 true, {reps} reps, T_pre={T_pre}, T_post={T_post}")
print(f"rejection at 5% -- library padded permutations : {rej_pad/reps:.3f}")
print(f"rejection at 5% -- CWZ moving-block only       : {rej_blk/reps:.3f}")
