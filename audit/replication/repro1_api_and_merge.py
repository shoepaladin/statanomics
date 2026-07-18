"""Repro 1: (a) missing API symbols referenced by sibling scripts;
(b) event-study counterfactual merge keyed on positional index instead of date;
(c) string-vs-native unit-id merge failure."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/statanomics/workingcode/panelmodels')
import numpy as np, pandas as pd
import panelib
from panelib import did, sc, dgp

print("== (a) API symbols used by sibling scripts ==")
print("did.twfe_conformal exists:", hasattr(did, 'twfe_conformal'))
print("sc.sc_att exists:", hasattr(sc, 'sc_att'))

print("\n== (b) event-study counterfactual with calendar dates (2001..2012) ==")
rng = np.random.default_rng(0)
N, dates = 12, np.arange(2001, 2013)
rows = []
for i in range(N):
    a = rng.normal(2, 1)
    for t in dates:
        treated = int(i == 0)
        post = int(t >= 2010)
        y = a + 0.1*(t-2001) + rng.normal(0, .1) + 3.0*treated*post
        rows.append(dict(unit_id=f'u{i}', time=t, treated=treated, post=post, y=y))
df = pd.DataFrame(rows)
dd = {'treatment':'treated','date':'time','post':'post','unitid':'unit_id','outcome':'y'}
res = did.twfe(data=df, data_dict=dd)
ev_c = res['event_study_c']
tr = ev_c[ev_c['unit_id']=='u0'].sort_values('time')
post_rows = tr[tr['time']>=2010]
print("event_study coef estimates (post):")
print(res['event_study'].query('pre_event==0')[['time_period','treated_unit','coef_']])
print("\npost rows of event_study_c for treated unit:")
print(post_rows[['time','y','y_hats','y_hat_counterfactual','coef_']])
gap = (post_rows['y_hats'] - post_rows['y_hat_counterfactual']).abs().max()
print("max |y_hats - y_hat_counterfactual| in post =", gap,
      "-> counterfactual NOT adjusted (silently equals fitted value)" if gap < 1e-12 else "")

print("\n== (c) same panel but integer unit ids ==")
df2 = df.copy(); df2['unit_id'] = df2.groupby('unit_id').ngroup()  # ints
try:
    res2 = did.twfe(data=df2, data_dict=dd)
    tw_c = res2['twfe_c']
    tr2 = tw_c[tw_c['unit_id']==0]
    p2 = tr2[tr2['time']>=2010]
    print("twfe coef:", res2['twfe']['coef_'].values)
    print("max |y_hats - y_hat_counterfactual| in post (int ids) =",
          (p2['y_hats']-p2['y_hat_counterfactual']).abs().max())
except Exception as e:
    print("CRASH:", type(e).__name__, str(e)[:200])
