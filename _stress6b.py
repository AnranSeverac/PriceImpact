"""
Stress 6 (corrected): Shrink the per-cell sample size by training on a
SHORT window (5 trading days) instead of a full month. This is the
right lever — n_per_cell scales with days_in_train, not stocks.

Also try MORE bins (B=40) which raises ~3000 → ~1100 obs/cell.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/Users/AnranSeverac/PriceImpact')
from _stress import (impact_state, impact_regression_statistics, build_obs,
                     universal_bin_means, regularised_bin_means,
                     predict_and_score, obs, H_STAR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_stats(df, edges):
    df = df.copy()
    df['bin'] = pd.cut(df['x'], bins=edges, labels=False, include_lowest=True)
    df = df.dropna(subset=['bin']); df['bin'] = df['bin'].astype(int)
    return (df.groupby(['stock','bin'])
            .agg(sy=('y','sum'), syy=('y', lambda v:(v**2).sum()), n=('y','count'))
            .reset_index())


def run_eval(odf, train_dates, test_dates, val_dates, n_bins=15, gamma_grid_n=60):
    train = odf.loc[odf['date'].isin(train_dates)]
    test  = odf.loc[odf['date'].isin(test_dates)]
    val   = odf.loc[odf['date'].isin(val_dates)]

    if len(train) == 0 or len(test) == 0 or len(val) == 0:
        return None

    edges = np.quantile(train['x'].dropna(), np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf

    train_s = make_stats(train, edges)
    test_s  = make_stats(test, edges)
    val_s   = make_stats(val, edges)
    if len(train_s) == 0 or len(test_s) == 0:
        return None

    g_bar = universal_bin_means(train_s)
    median_n = train_s['n'].median()
    grid = median_n * np.logspace(-3, 3, gamma_grid_n)

    best_g, best_mse = None, np.inf
    mses = []
    for g in grid:
        reg = regularised_bin_means(train_s, g_bar, g)
        m = test_s.merge(reg[['stock','bin','g_reg']], on=['stock','bin'])
        if len(m) == 0:
            mses.append(np.nan); continue
        ssr = (m['syy'] - 2*m['g_reg']*m['sy'] + m['g_reg']**2*m['n']).sum()
        mse = ssr / m['n'].sum()
        mses.append(mse)
        if mse < best_mse: best_mse = mse; best_g = g

    reg_best = regularised_bin_means(train_s, g_bar, best_g)
    r2_reg, _ = predict_and_score(val_s, reg_best)
    raw  = regularised_bin_means(train_s, g_bar, 0.0)
    r2_raw, _  = predict_and_score(val_s, raw)
    univ = regularised_bin_means(train_s, g_bar, 1e15)
    r2_univ, _ = predict_and_score(val_s, univ)

    return {'median_n': median_n, 'best_g': best_g,
            'r2_raw': r2_raw, 'r2_univ': r2_univ, 'r2_reg': r2_reg,
            'grid': grid, 'mses': np.array(mses)}


print("=== STRESS 6 (corrected): per-CELL N is the right lever ===\n")

# Build a per-stock list of trading days
all_dates = sorted(obs['linear']['date'].unique())
print(f"Total trading days in 2019: {len(all_dates)}")

TRAIN_DAYS, TEST_DAYS, VAL_DAYS = 5, 5, 5
print(f"Training window: {TRAIN_DAYS} days, tune: {TEST_DAYS}, val: {VAL_DAYS}\n")

# 10 rolling windows of consecutive trading days, well-spaced
window_starts = [10 + 20*i for i in range(10)]  # spaced ~1 month apart
windows = []
for s in window_starts:
    if s + TRAIN_DAYS + TEST_DAYS + VAL_DAYS > len(all_dates):
        break
    train_d = all_dates[s : s + TRAIN_DAYS]
    test_d  = all_dates[s + TRAIN_DAYS : s + TRAIN_DAYS + TEST_DAYS]
    val_d   = all_dates[s + TRAIN_DAYS + TEST_DAYS : s + TRAIN_DAYS + TEST_DAYS + VAL_DAYS]
    windows.append((train_d, test_d, val_d))

print(f"Built {len(windows)} short-window splits.\n")

print(f"{'model':6}  {'win':4}  {'median_n':>9}  {'best_g':>10}  {'g/n':>6}  "
      f"{'r2_raw':>7}  {'r2_univ':>7}  {'r2_reg':>7}  {'reg-raw':>9}")

results_short = {'linear':[], 'sqrt':[]}
tuning_save = {}
for mt in ['linear','sqrt']:
    for w_idx, (tr, te, vl) in enumerate(windows):
        r = run_eval(obs[mt], tr, te, vl)
        if r is None: continue
        results_short[mt].append(r)
        tuning_save[(mt, w_idx)] = r
        print(f"{mt:6}  {w_idx:>4}  {r['median_n']:>9.0f}  {r['best_g']:>10.2f}  "
              f"{r['best_g']/r['median_n']:>6.3f}  "
              f"{r['r2_raw']:>7.4f}  {r['r2_univ']:>7.4f}  {r['r2_reg']:>7.4f}  "
              f"{r['r2_reg']-r['r2_raw']:>+9.4f}")

print(f"\n{'model':6}  {'mean median_n':>13}  {'mean R^2 raw':>12}  {'mean R^2 univ':>13}  "
      f"{'mean R^2 reg':>12}  {'mean gain':>10}")
for mt in ['linear','sqrt']:
    df = pd.DataFrame(results_short[mt])
    print(f"{mt:6}  {df['median_n'].mean():>13.1f}  {df['r2_raw'].mean():>12.4f}  "
          f"{df['r2_univ'].mean():>13.4f}  {df['r2_reg'].mean():>12.4f}  "
          f"{(df['r2_reg']-df['r2_raw']).mean():>+10.4f}")


# ----------------------------------------------------------
# Even smaller: 2 days train. n drops to ~250 per cell.
# ----------------------------------------------------------
print("\n\n=== Push harder: 2-day train window ===\n")
TRAIN_DAYS, TEST_DAYS, VAL_DAYS = 2, 2, 2
window_starts = [10 + 20*i for i in range(10)]
windows = []
for s in window_starts:
    if s + TRAIN_DAYS + TEST_DAYS + VAL_DAYS > len(all_dates): break
    windows.append((all_dates[s:s+TRAIN_DAYS],
                    all_dates[s+TRAIN_DAYS:s+TRAIN_DAYS+TEST_DAYS],
                    all_dates[s+TRAIN_DAYS+TEST_DAYS:s+TRAIN_DAYS+TEST_DAYS+VAL_DAYS]))

print(f"{'model':6}  {'win':4}  {'median_n':>9}  {'best_g':>10}  {'g/n':>6}  "
      f"{'r2_raw':>7}  {'r2_univ':>7}  {'r2_reg':>7}  {'reg-raw':>9}")
results_tiny = {'linear':[], 'sqrt':[]}
tuning_tiny = {}
for mt in ['linear','sqrt']:
    for w_idx, (tr, te, vl) in enumerate(windows):
        r = run_eval(obs[mt], tr, te, vl)
        if r is None: continue
        results_tiny[mt].append(r)
        tuning_tiny[(mt, w_idx)] = r
        print(f"{mt:6}  {w_idx:>4}  {r['median_n']:>9.0f}  {r['best_g']:>10.2f}  "
              f"{r['best_g']/r['median_n']:>6.3f}  "
              f"{r['r2_raw']:>7.4f}  {r['r2_univ']:>7.4f}  {r['r2_reg']:>7.4f}  "
              f"{r['r2_reg']-r['r2_raw']:>+9.4f}")

print(f"\n{'model':6}  {'mean median_n':>13}  {'mean R^2 raw':>12}  {'mean R^2 univ':>13}  "
      f"{'mean R^2 reg':>12}  {'mean gain':>10}")
for mt in ['linear','sqrt']:
    df = pd.DataFrame(results_tiny[mt])
    print(f"{mt:6}  {df['median_n'].mean():>13.1f}  {df['r2_raw'].mean():>12.4f}  "
          f"{df['r2_univ'].mean():>13.4f}  {df['r2_reg'].mean():>12.4f}  "
          f"{(df['r2_reg']-df['r2_raw']).mean():>+10.4f}")


# Plot: tuning curves at 3 N levels (full month, 5 days, 2 days) for OW
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)

# Full-month curve (re-derive from cached `obs`)
def full_month_curve(mt, tm=3):
    odf = obs[mt]
    train = odf.loc[odf['month']==tm]
    test  = odf.loc[odf['month']==tm+1]
    edges = np.quantile(train['x'].dropna(), np.linspace(0,1,16))
    edges[0] = -np.inf; edges[-1] = np.inf
    ts = make_stats(train, edges); te = make_stats(test, edges)
    g_bar = universal_bin_means(ts)
    median_n = ts['n'].median()
    grid = median_n * np.logspace(-3,3,60)
    mses = []
    for g in grid:
        reg = regularised_bin_means(ts, g_bar, g)
        m = te.merge(reg[['stock','bin','g_reg']], on=['stock','bin'])
        mses.append((m['syy'] - 2*m['g_reg']*m['sy'] + m['g_reg']**2*m['n']).sum()/m['n'].sum())
    return grid, np.array(mses), median_n

# OW only
mt = 'linear'
g_full, m_full, n_full = full_month_curve(mt)
ax = axes[0]
ax.plot(g_full, m_full, lw=2)
ax.set_xscale('log'); ax.set_xlabel('γ'); ax.set_ylabel('Test MSE')
ax.set_title(f'Full month train (~{int(n_full)} obs/cell)')
ax.axvline(g_full[np.nanargmin(m_full)], ls='--', color='red', alpha=.6,
           label=f'argmin γ = {g_full[np.nanargmin(m_full)]:.0f}')
ax.legend()

# 5-day train
r5 = tuning_save[(mt, 0)]
ax = axes[1]
ax.plot(r5['grid'], r5['mses'], lw=2, color='C1')
ax.set_xscale('log'); ax.set_xlabel('γ'); ax.set_title(f'5-day train (~{int(r5["median_n"])} obs/cell)')
ax.axvline(r5['best_g'], ls='--', color='red', alpha=.6, label=f"argmin γ = {r5['best_g']:.0f}")
ax.legend()

# 2-day train
r2 = tuning_tiny[(mt, 0)]
ax = axes[2]
ax.plot(r2['grid'], r2['mses'], lw=2, color='C2')
ax.set_xscale('log'); ax.set_xlabel('γ'); ax.set_title(f'2-day train (~{int(r2["median_n"])} obs/cell)')
ax.axvline(r2['best_g'], ls='--', color='red', alpha=.6, label=f"argmin γ = {r2['best_g']:.0f}")
ax.legend()

plt.suptitle('OW (linear) — γ tuning curves at three sample sizes', fontsize=12)
plt.tight_layout()
plt.savefig('/Users/AnranSeverac/PriceImpact/figures/stress6b_tuning_progression.png', dpi=120)
print("\nSaved figure: figures/stress6b_tuning_progression.png")
