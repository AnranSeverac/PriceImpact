"""
Stress 6: Show that regularisation kicks in when N is small.

Strategy: re-run the NP estimator with K stocks per training month
(K=5) instead of 50, so that median bin count drops by ~10x and
shrinkage has bias-variance teeth.

Two designs:
  A) Both per-stock bins AND universal g_bar are computed on the same K-stock subset.
     This is the symmetric stress test.
  B) Per-stock bins on K stocks, but g_bar still pooled across ALL 50.
     This is the realistic shrinkage scenario: small per-stock data, strong universal prior.
"""
import sys, time, pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Reuse functions
sys.path.insert(0, '/Users/AnranSeverac/PriceImpact')
from _stress import (impact_state, impact_regression_statistics, build_obs,
                     build_bin_stats, universal_bin_means, regularised_bin_means,
                     predict_and_score, obs, H_STAR, N_BINS)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print(f"Running Stress 6 with K=5 stocks per training month.")
print(f"Models cached: {list(obs.keys())}, panel size each: {len(obs['linear']):,}\n")

ALL_STOCKS = sorted(obs['linear']['stock'].unique())
print(f"Total stocks in panel: {len(ALL_STOCKS)}")

K = 5
SEED = 42
rng = np.random.default_rng(SEED)


def small_n_eval(odf, K, design='B'):
    """
    For each rolling window, restrict per-stock training to K stocks.
    design B: g_bar uses ALL stocks (the realistic shrinkage scenario)
    design A: g_bar uses only the K subset (symmetric)
    Returns per-window: best_gamma, oos R^2 for raw / universal / regularised.
    """
    out = []
    tuning_curves = {}
    for tm in range(1, 11):
        # pick a fresh K stocks per window
        stocks_K = list(rng.choice(ALL_STOCKS, size=K, replace=False))

        odf_train = odf.loc[(odf['month']==tm) & (odf['stock'].isin(stocks_K))]
        odf_test  = odf.loc[(odf['month']==tm+1) & (odf['stock'].isin(stocks_K))]
        odf_val   = odf.loc[(odf['month']==tm+2) & (odf['stock'].isin(stocks_K))]

        # bin edges from K-stock pooled training
        bin_edges = np.quantile(odf_train['x'].dropna(), np.linspace(0, 1, N_BINS + 1))
        bin_edges[0] = -np.inf; bin_edges[-1] = np.inf

        def make_stats(df, edges):
            df = df.copy()
            df['bin'] = pd.cut(df['x'], bins=edges, labels=False, include_lowest=True)
            df = df.dropna(subset=['bin']); df['bin'] = df['bin'].astype(int)
            s = (df.groupby(['stock','bin'])
                 .agg(sy=('y','sum'), syy=('y', lambda v:(v**2).sum()), n=('y','count'))
                 .reset_index())
            return s

        train_stats = make_stats(odf_train, bin_edges)
        test_stats  = make_stats(odf_test,  bin_edges)
        val_stats   = make_stats(odf_val,   bin_edges)

        # universal bin means
        if design == 'A':
            g_bar = universal_bin_means(train_stats)
        else:  # B: pool over ALL 50 stocks
            full_train = odf.loc[odf['month']==tm].copy()
            full_train['bin'] = pd.cut(full_train['x'], bins=bin_edges, labels=False, include_lowest=True)
            full_train = full_train.dropna(subset=['bin']); full_train['bin'] = full_train['bin'].astype(int)
            full_pooled = (full_train.groupby('bin')
                           .agg(sy=('y','sum'), n=('y','count')))
            full_pooled['g_bar'] = full_pooled['sy']/full_pooled['n']
            g_bar = full_pooled['g_bar']

        median_n = train_stats['n'].median()
        gamma_grid = median_n * np.logspace(-3, 3, 60)

        gamma_mses = []
        best_gamma, best_mse = None, np.inf
        for g in gamma_grid:
            reg = regularised_bin_means(train_stats, g_bar, g)
            m = test_stats.merge(reg[['stock','bin','g_reg']], on=['stock','bin'])
            ssr = (m['syy'] - 2*m['g_reg']*m['sy'] + m['g_reg']**2*m['n']).sum()
            mse = ssr / m['n'].sum()
            gamma_mses.append(mse)
            if mse < best_mse: best_mse = mse; best_gamma = g
        tuning_curves[tm] = (gamma_grid, np.array(gamma_mses), median_n, best_gamma)

        reg_best = regularised_bin_means(train_stats, g_bar, best_gamma)
        r2_reg, _ = predict_and_score(val_stats, reg_best)

        raw = regularised_bin_means(train_stats, g_bar, 0.0)
        r2_raw, _ = predict_and_score(val_stats, raw)

        univ = regularised_bin_means(train_stats, g_bar, 1e15)
        r2_univ, _ = predict_and_score(val_stats, univ)

        out.append({'tm':tm, 'median_n':median_n, 'best_gamma':best_gamma,
                    'r2_raw':r2_raw, 'r2_univ':r2_univ, 'r2_reg':r2_reg,
                    'gamma_div_n': best_gamma/median_n})

    return pd.DataFrame(out), tuning_curves


print(f"\n=== Design A: K={K} stocks for both g_i,b AND g_bar (symmetric) ===")
results_A = {}
for mt in ['linear','sqrt']:
    df_A, tc_A = small_n_eval(obs[mt], K, design='A')
    results_A[mt] = (df_A, tc_A)
    print(f"\n  {mt}:")
    print(df_A[['tm','median_n','best_gamma','gamma_div_n','r2_raw','r2_univ','r2_reg']].round(4).to_string(index=False))
    print(f"    Mean R^2 raw = {df_A['r2_raw'].mean():.4f}  univ = {df_A['r2_univ'].mean():.4f}  reg = {df_A['r2_reg'].mean():.4f}")
    print(f"    Median best_gamma/median_n = {df_A['gamma_div_n'].median():.4f}")

print(f"\n\n=== Design B: K={K} per-stock, but g_bar pooled over ALL 50 (realistic) ===")
results_B = {}
for mt in ['linear','sqrt']:
    df_B, tc_B = small_n_eval(obs[mt], K, design='B')
    results_B[mt] = (df_B, tc_B)
    print(f"\n  {mt}:")
    print(df_B[['tm','median_n','best_gamma','gamma_div_n','r2_raw','r2_univ','r2_reg']].round(4).to_string(index=False))
    print(f"    Mean R^2 raw = {df_B['r2_raw'].mean():.4f}  univ = {df_B['r2_univ'].mean():.4f}  reg = {df_B['r2_reg'].mean():.4f}")
    print(f"    Median best_gamma/median_n = {df_B['gamma_div_n'].median():.4f}")


# Plot tuning curves: full N vs small N (design B), for OW and AFS, train month 3
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for col, mt in enumerate(['linear','sqrt']):
    # full N tuning curve (re-run quickly)
    odf_full = obs[mt]
    bin_edges_f = np.quantile(odf_full.loc[odf_full['month']==3,'x'].dropna(),
                               np.linspace(0,1,N_BINS+1))
    bin_edges_f[0] = -np.inf; bin_edges_f[-1] = np.inf
    def stat(df):
        df = df.copy(); df['bin'] = pd.cut(df['x'], bins=bin_edges_f, labels=False, include_lowest=True)
        df = df.dropna(subset=['bin']); df['bin'] = df['bin'].astype(int)
        return df.groupby(['stock','bin']).agg(sy=('y','sum'), syy=('y', lambda v:(v**2).sum()), n=('y','count')).reset_index()
    train_full = stat(odf_full.loc[odf_full['month']==3])
    test_full  = stat(odf_full.loc[odf_full['month']==4])
    g_bar_f = universal_bin_means(train_full)
    median_n_f = train_full['n'].median()
    grid_f = median_n_f * np.logspace(-3, 3, 60)
    mses_f = []
    for g in grid_f:
        reg = regularised_bin_means(train_full, g_bar_f, g)
        m = test_full.merge(reg[['stock','bin','g_reg']], on=['stock','bin'])
        mses_f.append((m['syy'] - 2*m['g_reg']*m['sy'] + m['g_reg']**2*m['n']).sum()/m['n'].sum())

    ax = axes[0, col]
    ax.plot(grid_f, mses_f, lw=2, label=f'Full N (~{int(median_n_f)} obs/cell)')
    ax.set_xscale('log')
    ax.set_xlabel('gamma'); ax.set_ylabel('Test MSE')
    ax.set_title(f'{mt} — full panel (50 stocks)')
    ax.legend()

    ax = axes[1, col]
    grid, mses, mn, bg = tc_B[3]
    ax.plot(grid, mses, lw=2, color='C1', label=f'K={K} stocks (~{int(mn)} obs/cell)')
    ax.axvline(bg, color='red', ls='--', alpha=0.7, label=f'argmin γ = {bg:.1f}')
    ax.set_xscale('log')
    ax.set_xlabel('gamma'); ax.set_ylabel('Test MSE')
    ax.set_title(f'{mt} — small N, g_bar pooled over 50')
    ax.legend()

plt.tight_layout()
plt.savefig('/Users/AnranSeverac/PriceImpact/figures/stress6_tuning_curves.png', dpi=120)
print("\nSaved figure: figures/stress6_tuning_curves.png")

# Compare full-panel results vs small-N results (table)
print("\n=== Headline comparison: full panel (50) vs small-N (K=5) ===")
# Run the full-panel NP one more time using the same fn for consistency
print(f"\n  {'model':6}  {'design':22}  {'mean R^2 raw':>12}  {'mean R^2 univ':>13}  {'mean R^2 reg':>12}  {'gain reg-raw':>14}")
def full_eval(odf):
    out = []
    for tm in range(1,11):
        train_stats, edges = build_bin_stats(odf, tm)
        test_stats, _ = build_bin_stats(odf, tm+1, bin_edges=edges)
        val_stats,  _ = build_bin_stats(odf, tm+2, bin_edges=edges)
        g_bar = universal_bin_means(train_stats)
        median_n = train_stats['n'].median()
        grid = median_n * np.logspace(-3,3,30)
        best_g, best_mse = None, np.inf
        for g in grid:
            reg = regularised_bin_means(train_stats, g_bar, g)
            m = test_stats.merge(reg[['stock','bin','g_reg']], on=['stock','bin'])
            mse = (m['syy'] - 2*m['g_reg']*m['sy'] + m['g_reg']**2*m['n']).sum()/m['n'].sum()
            if mse < best_mse: best_mse = mse; best_g = g
        reg_best = regularised_bin_means(train_stats, g_bar, best_g)
        r2_reg, _ = predict_and_score(val_stats, reg_best)
        raw = regularised_bin_means(train_stats, g_bar, 0.0)
        r2_raw, _ = predict_and_score(val_stats, raw)
        univ = regularised_bin_means(train_stats, g_bar, 1e15)
        r2_univ, _ = predict_and_score(val_stats, univ)
        out.append({'tm':tm, 'r2_raw':r2_raw, 'r2_univ':r2_univ, 'r2_reg':r2_reg})
    return pd.DataFrame(out)

for mt in ['linear','sqrt']:
    df_full = full_eval(obs[mt])
    print(f"  {mt:6}  {'full panel (50)':22}  {df_full['r2_raw'].mean():>12.5f}  "
          f"{df_full['r2_univ'].mean():>13.5f}  {df_full['r2_reg'].mean():>12.5f}  "
          f"{df_full['r2_reg'].mean()-df_full['r2_raw'].mean():>+14.5f}")
    df_A = results_A[mt][0]
    print(f"  {mt:6}  {'K=5, g_bar on K':22}  {df_A['r2_raw'].mean():>12.5f}  "
          f"{df_A['r2_univ'].mean():>13.5f}  {df_A['r2_reg'].mean():>12.5f}  "
          f"{df_A['r2_reg'].mean()-df_A['r2_raw'].mean():>+14.5f}")
    df_B = results_B[mt][0]
    print(f"  {mt:6}  {'K=5, g_bar on 50':22}  {df_B['r2_raw'].mean():>12.5f}  "
          f"{df_B['r2_univ'].mean():>13.5f}  {df_B['r2_reg'].mean():>12.5f}  "
          f"{df_B['r2_reg'].mean()-df_B['r2_raw'].mean():>+14.5f}")
