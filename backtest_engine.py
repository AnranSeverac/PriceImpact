"""
Backtest engine (§2.3): Waelbroeck-style simulator + multi-day carry expansion.

Baseline: daily reset of normalized impact state Ī and positions (per stock, per day).

Expansion: carry Ī and inventory across days; optional split detection / exclusion list.

Price construction follows `project.ipynb` / Part 4 slides:

    P_sim(t) = P_0 · ( 1 + cumret_t + λ · (Ī_sim(t) - Ī_ref(t)) )

with discrete OU recursion  Ī_{t+1} = (1-β)Ī_t + q̃_t  (β = log(2)/H_bins, 10-second bins).

Default impact state uses `impact_state_ou` (SDE-consistent, supports carry). For byte-for-byte
agreement with `project.ipynb`'s `scipy.signal.lfilter` call, use `impact_state_project_lfilter`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-not-found]

ModelType = Literal["linear", "sqrt"]


def _half_life_bins(half_life_minutes: float) -> float:
    """10-second bins: H_bins = half_life_minutes * 6 (course convention)."""
    return half_life_minutes * 6.0


def q_tilde_series(
    q: np.ndarray,
    sigma: float,
    adv: float,
    model_type: ModelType,
) -> np.ndarray:
    """Normalized order flow q̃_t used in the Ī recursion (linear or square-root)."""
    q = np.asarray(q, dtype=float)
    if model_type == "linear":
        return sigma * q / adv
    return sigma * np.sign(q) * np.sqrt(np.abs(q) / adv)


def _decay_from_half_life(half_life_minutes: float) -> float:
    h_bins = _half_life_bins(half_life_minutes)
    beta = np.log(2.0) / h_bins
    return 1.0 - beta


def impact_state_ou(
    q_tilde: np.ndarray,
    half_life_minutes: float,
    i_bar0: float = 0.0,
) -> np.ndarray:
    """
    OU discretization Ī_{t+1} = (1-β)Ī_t + q̃_t (numpy fallback when SciPy absent, should work with scipy tho).

    Returns impact state at each bin (same length as q_tilde).
    """
    q_tilde = np.asarray(q_tilde, dtype=float)
    decay = _decay_from_half_life(half_life_minutes)
    n = len(q_tilde)
    out = np.zeros(n)
    state = float(i_bar0)
    for t in range(n):
        state = decay * state + q_tilde[t]
        out[t] = state
    return out


def impact_state_project_lfilter(
    q_tilde: np.ndarray,
    half_life_minutes: float,
) -> np.ndarray:
    try:
        from scipy.signal import lfilter  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("impact_state_project_lfilter requires scipy") from e

    q_tilde = np.asarray(q_tilde, dtype=float)
    decay = _decay_from_half_life(half_life_minutes)
    return lfilter([0, 1], [1, -decay], q_tilde)


def decay_factor_overnight(
    half_life_minutes: float,
    overnight_minutes: float,
) -> float:
    """Decay applied to Ī over `overnight_minutes` with no new order flow (10-sec bins)."""
    if overnight_minutes <= 0:
        return 1.0
    decay = _decay_from_half_life(half_life_minutes)
    n_bins = overnight_minutes * 6.0
    return float(decay**n_bins)


def compute_normalized_impact_states_multi_day(
    data,
    daily_stats,
    half_life_minutes: float,
    model_type: ModelType = "linear",
    overnight_minutes: float = 16.0 * 60.0,
    stock_col: str = "stock",
    date_col: str = "date",
    time_col: str = "time",
    order_flow_col: str = "orderFlow",
):
    """
    Compute normalized impact state `I_bar` for each bin but carry the end-of-day
    state across days with overnight decay.

    This mirrors the recursion used elsewhere in this module and returns a
    DataFrame with columns `[stock, date, time, I_bar, q_tilde]` aligned to
    the input `data` rows (rows without matching `daily_stats` are dropped).
    """

    df = data[[stock_col, date_col, time_col, order_flow_col]].copy()
    df = df.merge(
        daily_stats[["sigma", "ADV"]].reset_index(),
        on=[stock_col, date_col],
        how="inner",
    )

    # prepare output columns
    df = df.sort_values([stock_col, date_col, time_col])
    df["q_tilde"] = 0.0
    df["I_bar"] = 0.0

    # iterate by stock then by date, carrying the last I_bar across days with overnight decay
    for stock, g_stock in df.groupby(stock_col, sort=False):
        i_bar = 0.0
        for date, g_day in g_stock.groupby(date_col, sort=False):
            gg = g_day.sort_values(time_col)
            sigma = float(gg["sigma"].iat[0])
            adv = float(gg["ADV"].iat[0])

            q = gg[order_flow_col].to_numpy(dtype=float)
            qt = q_tilde_series(q, sigma, adv, model_type)
            i_day = impact_state_ou(qt, half_life_minutes, i_bar0=i_bar)

            df.loc[gg.index, "q_tilde"] = qt
            df.loc[gg.index, "I_bar"] = i_day

            # carry to next session with overnight decay
            ovn = decay_factor_overnight(half_life_minutes, overnight_minutes)
            i_bar = float(i_day[-1]) * ovn if len(i_day) else i_bar * ovn

    return df[[stock_col, date_col, time_col, "I_bar", "q_tilde"]]


def waelbroeck_prices(
    mid: np.ndarray,
    q_reference: np.ndarray,
    q_simulated: np.ndarray,
    lam: float,
    half_life_minutes: float,
    sigma: float,
    adv: float,
    model_type: ModelType = "linear",
    i_bar_ref0: float = 0.0,
    i_bar_sim0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Waelbroeck backtest (Part 4, slides 20–24):

        P_t(Q) = P^r_t - I_t(Q^r) + I_t(Q)

    In return space (see `project.ipynb`):

        P_sim(t) = P_0 · ( 1 + cumret_t + λ · (Ī_sim(t) - Ī_ref(t)) )

    Parameters
    ----------
    mid
        Observed mids (includes reference-path impact).
    q_reference, q_simulated
        Signed trade paths in shares (e.g. tape orderFlow vs alternative strategy).
    i_bar_ref0, i_bar_sim0
        Initial impact states before the first bin (multi-day carry).
    """
    mid = np.asarray(mid, dtype=float)
    q_reference = np.asarray(q_reference, dtype=float)
    q_simulated = np.asarray(q_simulated, dtype=float)
    if not (len(mid) == len(q_reference) == len(q_simulated)):
        raise ValueError("mid, q_reference, q_simulated must have the same length")

    qt_ref = q_tilde_series(q_reference, sigma, adv, model_type)
    qt_sim = q_tilde_series(q_simulated, sigma, adv, model_type)

    i_ref = impact_state_ou(qt_ref, half_life_minutes, i_bar0=i_bar_ref0)
    i_sim = impact_state_ou(qt_sim, half_life_minutes, i_bar0=i_bar_sim0)

    cum_ret = (mid - mid[0]) / mid[0]
    p_sim = mid[0] * (1.0 + cum_ret + lam * (i_sim - i_ref))
    return p_sim, i_ref, i_sim


def mark_to_market_pnl(
    mid_sim: np.ndarray,
    position: np.ndarray,
) -> float:
    """
    Discrete P&L: sum_t position_{t-1} * (mid_t - mid_{t-1}) with position aligned
    to shares held entering each bin (length n, uses position[:-1] with diff).
    """
    mid_sim = np.asarray(mid_sim, dtype=float)
    position = np.asarray(position, dtype=float)
    if len(mid_sim) != len(position):
        raise ValueError("mid_sim and position must match")
    if len(mid_sim) < 2:
        return 0.0
    return float(np.sum(position[:-1] * np.diff(mid_sim)))


def daily_reset_backtest_path(
    mid: np.ndarray,
    q_reference: np.ndarray,
    q_simulated: np.ndarray,
    lam: float,
    half_life_minutes: float,
    sigma: float,
    adv: float,
    model_type: ModelType = "linear",
) -> dict[str, np.ndarray]:
    """
    Baseline §2.3: reset Ī at the start of each path (single trading day segment).

    Returns arrays: p_sim, i_ref, i_sim, position (cum trades), pnl components.
    """
    p_sim, i_ref, i_sim = waelbroeck_prices(
        mid,
        q_reference,
        q_simulated,
        lam,
        half_life_minutes,
        sigma,
        adv,
        model_type,
        i_bar_ref0=0.0,
        i_bar_sim0=0.0,
    )
    q = np.asarray(q_simulated, dtype=float)
    position = np.cumsum(q)
    pnl = mark_to_market_pnl(p_sim, position)
    return {
        "p_sim": p_sim,
        "i_ref": i_ref,
        "i_sim": i_sim,
        "position": position,
        "q": q,
        "pnl_mtm": np.array([pnl]),
    }


def detect_splits_from_daily(
    daily: "pd.DataFrame",
    *,
    price_col: str = "close",
    volume_col: str = "volume",
    ratio_threshold: float = 0.03,
    notional_tol: float = 0.15,
):
    """
    Heuristic split / reverse-split flag from daily price vs volume jumps.

    Flags rows where fractional change in price and volume are large, opposite-signed,
    and daily notional (price * volume) is approximately stable — as suggested in §2.3.

    Parameters
    ----------
    daily
        Must contain stock, date (sorted within stock), price_col, volume_col.
    """
    import pandas as pd  # type: ignore[import-not-found]

    df = daily.sort_values(["stock", "date"]).copy()
    parts: list[pd.DataFrame] = []

    for _, gg in df.groupby("stock", sort=False):
        pc = gg[price_col].astype(float)
        vc = gg[volume_col].astype(float).replace(0, np.nan)
        rp = pc.pct_change()
        rv = vc.pct_change()
        notional = pc * vc
        rn = notional.pct_change()
        opp = (rp * rv < 0) & rp.notna() & rv.notna()
        mag = rp.abs() > ratio_threshold
        stable = rn.abs() < notional_tol
        parts.append(
            gg.assign(
                split_flag=opp & mag & stable,
                price_chg=rp,
                vol_chg=rv,
                notional_chg=rn,
            )
        )

    return pd.concat(parts, ignore_index=True)


def build_daily_ohlcv_from_bins(bin_df: "pd.DataFrame"):
    """Last mid as close; sum |trade| as volume proxy for split detection."""

    df = bin_df.copy()
    if "trade" not in df.columns:
        raise ValueError("bin_df must contain 'trade'")
    agg = (
        df.groupby(["stock", "date"], sort=False)
        .agg(close=("mid", "last"), volume=("trade", lambda s: np.abs(s).sum()))
        .reset_index()
    )
    return agg


@dataclass
class MultiDayState:
    """End-of-session carry for one symbol."""

    i_bar_ref: float = 0.0
    i_bar_sim: float = 0.0
    position: float = 0.0


def multi_day_carry_paths(
    sessions: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]],
    lam: float,
    half_life_minutes: float,
    sigma: float,
    adv: float,
    model_type: ModelType,
    overnight_minutes: float = 16.0 * 60.0,
    initial: MultiDayState | None = None,
) -> dict[str, list[np.ndarray] | list[float]]:
    """
    Expansion §2.3: carry reference/sim impact states and inventory across ordered sessions.

    Each session is one trading day (or any contiguous segment): (mid, q_ref, q_sim, sigma, adv).

    Overnight: multiply both Ī_ref and Ī_sim by `decay_factor_overnight` (no flow).
    Position carries as cumulative sum of simulated fills across sessions.

    Parameters
    ----------
    sessions
        Ordered list of days for one stock.
    overnight_minutes
        Default 16 hours of decay between sessions.
    """
    if initial is None:
        initial = MultiDayState()
    ovn = decay_factor_overnight(half_life_minutes, overnight_minutes)

    out_p: list[np.ndarray] = []
    out_iref: list[np.ndarray] = []
    out_isim: list[np.ndarray] = []
    out_pos: list[np.ndarray] = []
    pnls: list[float] = []

    i_ref = initial.i_bar_ref
    i_sim = initial.i_bar_sim
    pos = initial.position

    for mid, q_ref, q_sim, sig, advv in sessions:
        mid = np.asarray(mid, dtype=float)
        q_ref = np.asarray(q_ref, dtype=float)
        q_sim = np.asarray(q_sim, dtype=float)

        p_sim, ir, is_ = waelbroeck_prices(
            mid,
            q_ref,
            q_sim,
            lam,
            half_life_minutes,
            sig,
            advv,
            model_type,
            i_bar_ref0=i_ref,
            i_bar_sim0=i_sim,
        )
        q_c = np.cumsum(q_sim) + pos
        pnl = mark_to_market_pnl(p_sim, q_c)

        out_p.append(p_sim)
        out_iref.append(ir)
        out_isim.append(is_)
        out_pos.append(q_c)
        pnls.append(pnl)

        i_ref = float(ir[-1]) * ovn
        i_sim = float(is_[-1]) * ovn
        pos = float(q_c[-1])

    return {
        "p_sim_days": out_p,
        "i_ref_days": out_iref,
        "i_sim_days": out_isim,
        "position_days": out_pos,
        "pnl_per_session": pnls,
        "final_state": MultiDayState(i_bar_ref=i_ref, i_bar_sim=i_sim, position=pos),
    }


def stocks_to_exclude_from_splits(
    split_flags: "pd.DataFrame",
    *,
    stock_col: str = "stock",
) -> set[str]:
    """Union of tickers with any split_flag True."""
    if "split_flag" not in split_flags.columns:
        return set()
    return set(split_flags.loc[split_flags["split_flag"], stock_col].unique())


def multi_day_carry_from_bins(
    bins: "pd.DataFrame",
    *,
    lam: float,
    half_life_minutes: float,
    sigma: float,
    model_type: ModelType = "linear",
    overnight_minutes: float = 16.0 * 60.0,
    stock_col: str = "stock",
    date_col: str = "date",
    time_col: str = "bin_sec",
    mid_col: str = "mid",
    q_reference_col: str = "orderFlow",
    q_simulated_col: str = "fill_trade",
    adv_source_col: str = "trade",
    exclude_stocks: set[str] | None = None,
    min_adv: float = 1e-12,
) -> "pd.DataFrame":
    """
    Convenience wrapper for §2.3 expansion: run multi-day carry backtests per stock
    directly from a bin-level DataFrame.

    Expected input is already aligned on the 10-second grid (one row per bin) and
    includes both reference order flow and simulated fills.

    Notes
    -----
    - ADV is estimated per (stock,date) as sum(abs(adv_source_col)) on that day.
    - Impact and position are carried across dates within each stock (chronological order).
    - Use `exclude_stocks` (e.g. from split screening) to drop symbols entirely.
    """
    import pandas as pd  # type: ignore[import-not-found]

    df = bins.copy()
    if exclude_stocks:
        df = df[~df[stock_col].isin(exclude_stocks)]

    need = {
        stock_col,
        date_col,
        time_col,
        mid_col,
        q_reference_col,
        q_simulated_col,
        adv_source_col,
    }
    missing = sorted(need.difference(df.columns))
    if missing:
        raise ValueError(f"bins is missing required columns: {missing}")

    df = df.sort_values([stock_col, date_col, time_col])

    rows: list[dict[str, object]] = []
    for stock, g_stock in df.groupby(stock_col, sort=False):
        state = MultiDayState()
        for date, g_day in g_stock.groupby(date_col, sort=False):
            gg = g_day.sort_values(time_col)
            mid = gg[mid_col].to_numpy(dtype=float)
            q_ref = gg[q_reference_col].to_numpy(dtype=float)
            q_sim = gg[q_simulated_col].to_numpy(dtype=float)

            adv_est = float(np.sum(np.abs(gg[adv_source_col].to_numpy(dtype=float))))
            if not np.isfinite(adv_est) or adv_est <= min_adv:
                continue

            p_sim, i_ref, i_sim = waelbroeck_prices(
                mid=mid,
                q_reference=q_ref,
                q_simulated=q_sim,
                lam=lam,
                half_life_minutes=half_life_minutes,
                sigma=sigma,
                adv=adv_est,
                model_type=model_type,
                i_bar_ref0=state.i_bar_ref,
                i_bar_sim0=state.i_bar_sim,
            )

            position = np.cumsum(q_sim) + state.position
            pnl = mark_to_market_pnl(p_sim, position)

            rows.append(
                {
                    "stock": stock,
                    "date": date,
                    "pnl": float(pnl),
                    "adv_est": float(adv_est),
                    "end_position": float(position[-1])
                    if len(position)
                    else float(state.position),
                    "avg_abs_position": float(np.mean(np.abs(position)))
                    if len(position)
                    else 0.0,
                    "i_bar_ref_end": float(i_ref[-1])
                    if len(i_ref)
                    else float(state.i_bar_ref),
                    "i_bar_sim_end": float(i_sim[-1])
                    if len(i_sim)
                    else float(state.i_bar_sim),
                    "max_abs_lam_i_ref": float(np.max(np.abs(lam * i_ref)))
                    if len(i_ref)
                    else 0.0,
                    "max_abs_lam_i_sim": float(np.max(np.abs(lam * i_sim)))
                    if len(i_sim)
                    else 0.0,
                    "sum_abs_qsim": float(np.sum(np.abs(q_sim))),
                    "gross_notional_proxy": float(np.sum(np.abs(q_sim) * mid)),
                }
            )

            # carry to next session
            ovn = decay_factor_overnight(half_life_minutes, overnight_minutes)
            state = MultiDayState(
                i_bar_ref=(float(i_ref[-1]) if len(i_ref) else state.i_bar_ref) * ovn,
                i_bar_sim=(float(i_sim[-1]) if len(i_sim) else state.i_bar_sim) * ovn,
                position=float(position[-1])
                if len(position)
                else float(state.position),
            )

    return pd.DataFrame(rows)


__all__ = [
    "ModelType",
    "MultiDayState",
    "build_daily_ohlcv_from_bins",
    "daily_reset_backtest_path",
    "decay_factor_overnight",
    "detect_splits_from_daily",
    "impact_state_ou",
    "impact_state_project_lfilter",
    "mark_to_market_pnl",
    "multi_day_carry_from_bins",
    "multi_day_carry_paths",
    "q_tilde_series",
    "stocks_to_exclude_from_splits",
    "waelbroeck_prices",
    "compute_normalized_impact_states_multi_day",
]
