"""
Run the four stress tests from the review against Anran's pipeline.
Re-uses functions from _replicate.py (re-imported here to be self-contained).
"""
import sys, time, pickle
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('/Users/AnranSeverac/PriceImpact/data')
EXPL_HORIZON = 6
STAT_COLS = ['xy', 'xx', 'yy', 'x', 'y', 'count']
N_BINS = 15

# Load saved state
with open('/Users/AnranSeverac/PriceImpact/_replicate_state.pkl', 'rb') as f:
    st = pickle.load(f)
tv      = st['tv']
px      = st['px']
scaling = st['scaling']

print(f"Loaded state. Panel: {len(tv):,} stock-days, {tv.index.get_level_values('stock').nunique()} stocks.")

# -----------------------------------------------------------------------
# Pipeline functions (Anran's exact code)
# -----------------------------------------------------------------------
def impact_state(traded_volume_df, scaling_factor, half_life, model_type):
    space_kernels = {
        'linear': lambda x: x,
        'sqrt':   lambda x: np.sign(x) * np.sqrt(np.abs(x)),
    }
    beta = np.log(2) / (half_life / 10)
    decay = np.exp(-beta)
    pre = traded_volume_df.copy()
    pre = pre.divide(scaling_factor['volume'], axis='rows')
    pre = space_kernels[model_type](pre)
    pre = pre.multiply(scaling_factor['px_vol'], axis='rows')
    pre.iloc[:, 1:] /= (1 - decay)
    return pre.T.ewm(alpha=1 - decay, adjust=False).mean().T

def impact_regression_statistics(cum_impact, tau, px_df):
    impact_changes = cum_impact.diff(tau, axis='columns').T.unstack()
    df = (impact_changes.reset_index()
          .rename({'level_2':'time', 0:'x'}, axis='columns'))
    rets = (px_df.pct_change(tau, axis='columns').T.unstack()
            .reset_index().rename({'level_2':'time', 0:'y'}, axis='columns'))
    df = df.loc[df['time'] >= '10:00:00'].dropna(axis=0).copy()
    df['y'] = rets['y']
    df['xy'] = df['x'] * df['y']
    df['xx'] = df['x'] ** 2
    df['yy'] = df['y'] ** 2
    df['count'] = 1
    return df

def ols_from_sums(s):
    n = s['count']
    cov = s['xy'] - s['x']*s['y']/n
    var = s['xx'] - s['x']**2/n
    beta = cov/var
    alpha = s['y']/n - beta*s['x']/n
    return beta, alpha

def r2_from_sums(s, beta, alpha):
    n = s['count']
    ss_tot = s['yy'] - s['y']**2/n
    ss_res = (s['yy'] - 2*beta*s['xy'] - 2*alpha*s['y']
              + 2*alpha*beta*s['x'] + beta**2*s['xx'] + alpha**2*n)
    return 1 - ss_res / ss_tot


# Cache the obs_df at H_STAR for both models (used by 4,5,6)
H_STAR = 3150
H_OW   = 3600
H_AFS  = 2700

def build_obs(H, mt):
    """Observation-level df with month label."""
    ci = impact_state(tv, scaling, H, mt)
    rs = impact_regression_statistics(ci, EXPL_HORIZON, px)
    rs['date']  = pd.to_datetime(rs['date'])
    rs['month'] = rs['date'].dt.month
    return rs[['stock','date','month','x','y']].copy()


# =======================================================================
# STRESS 3 — half-life sensitivity at the per-model optimum vs midpoint
# =======================================================================
print("\n" + "="*72)
print("STRESS 3: Per-model H* vs adopted midpoint (52.5 min)")
print("="*72)

def baseline_oos_at_H(H, mt):
    """Mean OOS R^2 for one (H, model) — same protocol as Anran's cell 15."""
    ci = impact_state(tv, scaling, H, mt)
    rs = impact_regression_statistics(ci, EXPL_HORIZON, px)
    s = rs.groupby(['stock','date'])[STAT_COLS].sum().reset_index()
    s = s.loc[s['yy'] > 1e-12].copy()
    s['date'] = pd.to_datetime(s['date'])
    s['month'] = s['date'].dt.month
    rs_oos = []
    for tm in range(1,11):
        train = s.loc[s['month']==tm].groupby('stock')[STAT_COLS].sum()
        val   = s.loc[s['month']==tm+2].groupby('stock')[STAT_COLS].sum()
        common = train.index.intersection(val.index)
        train, val = train.loc[common], val.loc[common]
        b, a = ols_from_sums(train)
        rs_oos.extend(r2_from_sums(val, b, a).values)
    return np.mean(rs_oos)

print("\n  H (min) |  OW R^2  | AFS R^2 | OW loss vs opt | AFS loss vs opt")
print("  --------+----------+---------+----------------+----------------")
opt = {'linear': 0, 'sqrt': 0}
results_3 = {}
for H in [2700, 3150, 3600]:
    ow  = baseline_oos_at_H(H, 'linear')
    afs = baseline_oos_at_H(H, 'sqrt')
    results_3[H] = (ow, afs)
opt_ow  = max(r[0] for r in results_3.values())
opt_afs = max(r[1] for r in results_3.values())
for H, (ow, afs) in results_3.items():
    print(f"  {H/60:5.1f}   |  {ow:.4f}  | {afs:.4f}  |  {opt_ow-ow:+.4f}        | {opt_afs-afs:+.4f}")
print(f"\n  Conclusion: midpoint H*=52.5min costs {opt_ow-results_3[3150][0]:.4f} R^2 for OW, "
      f"{opt_afs-results_3[3150][1]:.4f} for AFS (per-stock-window mean).")


# =======================================================================
# STRESS 4 — bin edge diagnostics
# =======================================================================
print("\n" + "="*72)
print("STRESS 4: Bin edges from train month + +/- inf caps — fraction in tail bins")
print("="*72)

def build_bin_stats(odf, month, bin_edges=None):
    mdf = odf.loc[odf['month']==month].copy()
    if bin_edges is None:
        bin_edges = np.quantile(mdf['x'].dropna(), np.linspace(0,1,N_BINS+1))
        bin_edges[0]  = -np.inf
        bin_edges[-1] =  np.inf
    mdf['bin'] = pd.cut(mdf['x'], bins=bin_edges, labels=False, include_lowest=True)
    mdf = mdf.dropna(subset=['bin']); mdf['bin'] = mdf['bin'].astype(int)
    stats = (mdf.groupby(['stock','bin'])
             .agg(sy=('y','sum'), syy=('y', lambda v:(v**2).sum()), n=('y','count'))
             .reset_index())
    return stats, bin_edges

def universal_bin_means(train_stats):
    p = train_stats.groupby('bin')[['sy','n']].sum()
    p['g_bar'] = p['sy']/p['n']
    return p['g_bar']

def regularised_bin_means(train_stats, g_bar, gamma):
    m = train_stats.merge(g_bar.rename('g_bar'), on='bin', how='left')
    m['g_reg'] = (m['sy'] + gamma*m['g_bar']) / (m['n'] + gamma)
    return m

def predict_and_score(test_stats, g_lookup):
    m = test_stats.merge(g_lookup[['stock','bin','g_reg']], on=['stock','bin'], how='inner')
    ss_res = (m['syy'] - 2*m['g_reg']*m['sy'] + m['g_reg']**2*m['n'])
    tot_n  = m['n'].sum()
    tot_sy = m['sy'].sum()
    tot_syy= m['syy'].sum()
    y_bar = tot_sy/tot_n
    ss_tot = tot_syy - tot_n*y_bar**2
    r2 = 1 - ss_res.sum()/ss_tot if ss_tot > 0 else np.nan
    n_used = tot_n
    return r2, n_used

print("\nBuilding observation panel at H*...")
obs = {mt: build_obs(H_STAR, mt) for mt in ['linear','sqrt']}

print("\n--- Fraction of TEST/VAL observations in extreme bins (0 or 14) ---")
print(f"  Expected baseline if quantile bins were applied to same dist: "
      f"{2/N_BINS:.1%} = {2/N_BINS:.4f}\n")
print(f"  {'model':6} {'window':6} {'train':5} {'val':5} {'test_tail%':>11} "
      f"{'val_tail%':>11} {'test_>train':>13} {'val_>train':>11}")
records_4 = []
for mt in ['linear','sqrt']:
    odf = obs[mt]
    for tm in range(1,11):
        train_x = odf.loc[odf['month']==tm, 'x']
        edges_train = np.quantile(train_x.dropna(), np.linspace(0,1,N_BINS+1))
        finite_lo, finite_hi = edges_train[0], edges_train[-1]

        for which, m in [('test', tm+1), ('val', tm+2)]:
            xs = odf.loc[odf['month']==m, 'x']
            n  = len(xs)
            beyond = ((xs < finite_lo) | (xs > finite_hi)).sum()
            # tail-bin fraction WITH caps
            edges_capped = edges_train.copy()
            edges_capped[0] = -np.inf; edges_capped[-1] = np.inf
            bins = pd.cut(xs, bins=edges_capped, labels=False, include_lowest=True)
            tail_frac = ((bins==0) | (bins==N_BINS-1)).mean()
            records_4.append({'model':mt, 'tm':tm, 'which':which,
                              'n':n, 'tail_frac':tail_frac,
                              'beyond_frac': beyond/n})
        rec_t = records_4[-2]; rec_v = records_4[-1]
        print(f"  {mt:6} {tm:>4}    {tm:>3}  {tm+2:>3}     "
              f"{rec_t['tail_frac']:>9.3%}     {rec_v['tail_frac']:>9.3%}     "
              f"{rec_t['beyond_frac']:>11.3%}   {rec_v['beyond_frac']:>9.3%}")

r4 = pd.DataFrame(records_4)
print(f"\n  Mean tail bin frac across windows:")
print(r4.groupby(['model','which'])['tail_frac'].mean().round(4).to_string())
print(f"\n  Mean fraction *beyond* train support (would be NaN without inf caps):")
print(r4.groupby(['model','which'])['beyond_frac'].mean().round(4).to_string())

# Compare global edges vs train-only edges
print("\n--- OOS R^2 with TRAIN-only edges (Anran's) vs GLOBAL edges (full year) ---")
def np_eval(odf, edges_strategy):
    """edges_strategy in {'train','global'}"""
    out = []
    if edges_strategy == 'global':
        global_edges = np.quantile(odf['x'].dropna(), np.linspace(0,1,N_BINS+1))
        global_edges[0] = -np.inf; global_edges[-1] = np.inf
    for tm in range(1,11):
        if edges_strategy == 'train':
            train_stats, edges = build_bin_stats(odf, tm, bin_edges=None)
        else:
            train_stats, edges = build_bin_stats(odf, tm, bin_edges=global_edges)
        test_stats, _  = build_bin_stats(odf, tm+1, bin_edges=edges)
        val_stats, _   = build_bin_stats(odf, tm+2, bin_edges=edges)
        g_bar = universal_bin_means(train_stats)
        # tune gamma on test
        median_n = train_stats['n'].median()
        grid = median_n * np.logspace(-3,3,30)
        best_g, best_mse = None, np.inf
        for g in grid:
            reg = regularised_bin_means(train_stats, g_bar, g)
            m = test_stats.merge(reg[['stock','bin','g_reg']], on=['stock','bin'])
            ssr = (m['syy'] - 2*m['g_reg']*m['sy'] + m['g_reg']**2*m['n']).sum()
            mse = ssr / m['n'].sum()
            if mse < best_mse: best_mse = mse; best_g = g
        reg_best = regularised_bin_means(train_stats, g_bar, best_g)
        r2_reg, n_used  = predict_and_score(val_stats, reg_best)
        # Also compute raw (gamma=0)
        raw = regularised_bin_means(train_stats, g_bar, 0.0)
        r2_raw, _ = predict_and_score(val_stats, raw)
        out.append({'tm':tm, 'r2_reg':r2_reg, 'r2_raw':r2_raw, 'n_used':n_used})
    return pd.DataFrame(out)

print(f"\n  {'model':6}  {'edges':7}  {'mean R^2 raw':>13}  {'mean R^2 reg':>13}")
for mt in ['linear','sqrt']:
    for strat in ['train','global']:
        df = np_eval(obs[mt], strat)
        print(f"  {mt:6}  {strat:7}  {df['r2_raw'].mean():>13.5f}  {df['r2_reg'].mean():>13.5f}")


# =======================================================================
# STRESS 5 — apples-to-apples comparison: parametric R^2 on the NP-merged subset
# =======================================================================
print("\n" + "="*72)
print("STRESS 5: Parametric R^2 restricted to NP-merged subset (apples-to-apples)")
print("="*72)

def stress_5(mt):
    odf = obs[mt]
    rec = []
    for tm in range(1,11):
        # Build train sufficient stats AT OBSERVATION LEVEL → bin assignments
        train_stats, edges = build_bin_stats(odf, tm, bin_edges=None)
        val_stats, _ = build_bin_stats(odf, tm+2, bin_edges=edges)

        # NP regularised (matches Anran's pipeline at fixed best_gamma=median_n*logspace mid)
        g_bar = universal_bin_means(train_stats)
        # tune
        median_n = train_stats['n'].median()
        grid = median_n * np.logspace(-3,3,30)
        # tune on (tm+1)
        test_stats, _ = build_bin_stats(odf, tm+1, bin_edges=edges)
        best_g, best_mse = None, np.inf
        for g in grid:
            reg = regularised_bin_means(train_stats, g_bar, g)
            m = test_stats.merge(reg[['stock','bin','g_reg']], on=['stock','bin'])
            ssr = (m['syy'] - 2*m['g_reg']*m['sy'] + m['g_reg']**2*m['n']).sum()
            mse = ssr / m['n'].sum()
            if mse < best_mse: best_mse = mse; best_g = g
        reg_best = regularised_bin_means(train_stats, g_bar, best_g)

        # NP merged subset
        merged = val_stats.merge(reg_best[['stock','bin','g_reg']],
                                 on=['stock','bin'], how='inner')
        kept_keys = merged[['stock','bin']].drop_duplicates()
        # NP R^2
        ssr_np = (merged['syy'] - 2*merged['g_reg']*merged['sy']
                  + merged['g_reg']**2 * merged['n']).sum()
        n_np = merged['n'].sum()
        sy_np = merged['sy'].sum()
        syy_np = merged['syy'].sum()
        ybar = sy_np/n_np
        ss_tot = syy_np - n_np*ybar**2
        r2_np = 1 - ssr_np/ss_tot

        # Parametric R^2 on FULL val subset (Anran's reported)
        train_full = odf.loc[odf['month']==tm].groupby('stock').agg(
            xy=('x', lambda v: (v*odf.loc[v.index,'y']).sum()),
            xx=('x', lambda v: (v**2).sum()),
            yy=('y', lambda v: (v**2).sum()),
            x =('x','sum'),
            y =('y','sum'),
            count=('x','count'),
        )
        # easier: use the already-grouped sufficient stats
        full = odf.loc[odf['month']==tm].copy()
        full['xy'] = full['x']*full['y']; full['xx']=full['x']**2; full['yy']=full['y']**2; full['count']=1
        train_sums = full.groupby('stock')[STAT_COLS].sum()
        valf = odf.loc[odf['month']==tm+2].copy()
        valf['xy'] = valf['x']*valf['y']; valf['xx']=valf['x']**2; valf['yy']=valf['y']**2; valf['count']=1
        val_sums_full = valf.groupby('stock')[STAT_COLS].sum()
        common = train_sums.index.intersection(val_sums_full.index)
        train_sums, val_sums_full = train_sums.loc[common], val_sums_full.loc[common]
        b, a = ols_from_sums(train_sums)
        # full parametric R^2 (across stocks pooled — matching Anran's mean-of-per-stock)
        r2_param_full = r2_from_sums(val_sums_full, b, a).mean()

        # Parametric R^2 restricted to NP-merged (stock,bin)
        # Need val obs whose (stock, bin) is in kept_keys
        valf['bin'] = pd.cut(valf['x'], bins=edges, labels=False, include_lowest=True)
        valf = valf.dropna(subset=['bin']); valf['bin'] = valf['bin'].astype(int)
        valf_kept = valf.merge(kept_keys, on=['stock','bin'], how='inner')

        # Use per-stock fitted (b,a) on the kept observations, by stock
        valf_kept = valf_kept.merge(b.rename('beta'), left_on='stock', right_index=True)
        valf_kept = valf_kept.merge(a.rename('alpha'), left_on='stock', right_index=True)
        yhat = valf_kept['alpha'] + valf_kept['beta']*valf_kept['x']
        resid = valf_kept['y'] - yhat
        ss_res = (resid**2).sum()
        ybar2 = valf_kept['y'].mean()
        ss_tot_k = ((valf_kept['y']-ybar2)**2).sum()
        r2_param_kept = 1 - ss_res/ss_tot_k

        n_full = valf['count'].sum() if 'count' in valf.columns else len(valf)
        n_full = len(valf)  # raw obs count
        n_kept = len(valf_kept)

        rec.append({'model':mt, 'tm':tm,
                    'r2_param_full': r2_param_full,
                    'r2_param_kept': r2_param_kept,
                    'r2_np_subset':  r2_np,
                    'n_full': n_full, 'n_kept': n_kept,
                    'frac_kept': n_kept/n_full})
    return pd.DataFrame(rec)

print(f"\n  {'model':6}  {'mean R^2_param (full)':>20}  {'R^2_param (NP subset)':>22}  "
      f"{'R^2_NP':>8}  {'frac obs kept':>13}")
for mt in ['linear','sqrt']:
    df = stress_5(mt)
    print(f"  {mt:6}  {df['r2_param_full'].mean():>20.5f}  "
          f"{df['r2_param_kept'].mean():>22.5f}  "
          f"{df['r2_np_subset'].mean():>8.5f}  "
          f"{df['frac_kept'].mean():>13.3%}")
