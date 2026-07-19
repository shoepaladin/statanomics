"""Repro 6: staggered adoption + dynamic heterogeneous effects.
did.twfe fits one constant post dummy per treated unit; with staggered timing
and event-time-varying effects the time FEs are contaminated (Goodman-Bacon
2021 / Sun-Abraham 2021 logic) and the late cohort's ATT is biased even with
many never-treated controls. Noise sd = 0.02, so bias is not sampling error."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/statanomics/workingcode/panelmodels')
import numpy as np, pandas as pd
from panelib import did

dd = {'treatment':'treated','date':'time','post':'post','unitid':'unit_id','outcome':'y'}
rng = np.random.default_rng(0)
T = 20
rows = []
def add_unit(uid, treated, g, effect_fn):
    a = rng.normal(0, 1)
    for t in range(T):
        post = int(treated and t >= g)
        eff = effect_fn(t - g) if post else 0.0
        rows.append(dict(unit_id=uid, time=t, treated=treated,
                         post=post, y=a + 0.1*t + eff + rng.normal(0, .02)))

for i in range(5):   # early cohort, treated t>=5, effect grows 0.4*(e+1)
    add_unit(f'E{i}', 1, 5, lambda e: 0.4*(e+1))
for i in range(5):   # late cohort, treated t>=15, constant effect 1.0
    add_unit(f'L{i}', 1, 15, lambda e: 1.0)
for i in range(20):  # never treated
    add_unit(f'C{i:02d}', 0, 99, lambda e: 0.0)

df = pd.DataFrame(rows)
res = did.twfe(data=df, data_dict=dd)
tw = res['twfe']
early = tw[tw['treated_unit'].str.startswith('E')]['coef_']
late  = tw[tw['treated_unit'].str.startswith('L')]['coef_']
true_early = np.mean([0.4*(e+1) for e in range(15)])   # avg over its 15 post periods
print(f"early cohort: mean coef = {early.mean():.3f}   true avg effect = {true_early:.3f}")
print(f"late  cohort: mean coef = {late.mean():.3f}   true effect     = 1.000")
print(f"late-cohort bias = {late.mean()-1.0:+.3f}  (noise sd 0.02)")
# pre-trend F test contamination
ev = res['event_study']
print("pre-trend joint F p-value (should NOT reject; DGP has exact parallel trends):",
      float(ev['FJointPValue'].iloc[0]))
