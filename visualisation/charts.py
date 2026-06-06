"""
Publication-quality charts for the US Rates / SOFR research paper.

All charts follow a consistent Bloomberg-terminal-inspired style:
  - Dark background option or clean white (academic paper default)
  - Labeled axes with units
  - Recession shading via FRED USREC series
  - Exportable at 300dpi for publication

Chart inventory:
  1.  sofr_history()          — SOFR vs. EFFR vs. IORB since 2018
  2.  sofr_forward_curve()    — Bootstrapped SOFR curve on a date
  3.  forward_curve_evolution()— SOFR forward curve on multiple dates (snake chart)
  4.  treasury_yield_curve()  — Full Treasury curve at select dates
  5.  ns_factor_history()     — β₀, β₁, β₂ over time with regime shading
  6.  taylor_rule_vs_actual() — Fed Funds vs. Taylor Rule implied rate
  7.  fomc_dot_plot()         — FOMC probability distribution (FedWatch style)
  8.  backtest_pnl()          — Walk-forward cumulative P&L with drawdown
  9.  signal_heatmap()        — Sub-signal agreement over time
  10. regime_performance_bar()— Sharpe by market regime
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import date
from typing import Sequence

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

PAPER_COLORS = {
    "sofr":     "#1f77b4",
    "effr":     "#ff7f0e",
    "treasury": "#2ca02c",
    "taylor":   "#d62728",
    "signal":   "#9467bd",
    "pnl":      "#17becf",
    "drawdown": "#e377c2",
    "neutral":  "#7f7f7f",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shade_recessions(ax, recession_dates: pd.Series | None = None) -> None:
    """Add NBER recession shading. recession_dates: boolean Series (True = recession)."""
    if recession_dates is None:
        return
    in_recession = False
    rec_start = None
    for dt, val in recession_dates.items():
        if val and not in_recession:
            rec_start = dt
            in_recession = True
        elif not val and in_recession:
            ax.axvspan(rec_start, dt, alpha=0.12, color="gray", label="_nolegend_")
            in_recession = False


def _save_or_show(fig, path: str | None, title: str) -> None:
    if path:
        fig.savefig(path)
        print(f"Saved: {path}")
    else:
        plt.tight_layout()
        plt.show()


# ── Chart 1: SOFR History ─────────────────────────────────────────────────────

def sofr_history(
    df: pd.DataFrame,
    recession_series: pd.Series | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """SOFR overnight vs. EFFR vs. IORB since 2018 — shows Fed cycle phases."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for col, label, color in [
        ("sofr",     "SOFR (overnight)",    PAPER_COLORS["sofr"]),
        ("fed_funds", "EFFR",               PAPER_COLORS["effr"]),
        ("iorb",     "IORB (floor)",        PAPER_COLORS["neutral"]),
    ]:
        if col in df.columns:
            ax.plot(df.index, df[col], label=label, color=color, linewidth=1.5)

    _shade_recessions(ax, recession_series)

    # Annotate key events
    annotations = [
        ("2020-03-16", "COVID\ncuts", -0.8),
        ("2022-03-17", "First\nhike", 0.5),
        ("2023-07-27", "Peak\n5.50%", 0.3),
        ("2024-09-18", "First\ncut", 0.5),
    ]
    for dt_str, text, yoffset in annotations:
        dt = pd.Timestamp(dt_str)
        if dt in df.index or df.index.searchsorted(dt) < len(df):
            idx = df.index.searchsorted(dt)
            if idx < len(df):
                y = df.iloc[idx].get("sofr", 2.5)
                ax.annotate(text, xy=(df.index[idx], y),
                            xytext=(df.index[idx], y + yoffset),
                            fontsize=7.5, ha="center",
                            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel("Date")
    ax.set_ylabel("Rate (%)")
    ax.set_title("SOFR, EFFR, and IORB — 2018 to Present", fontweight="bold")
    ax.legend(loc="upper left")
    _save_or_show(fig, save_path, "sofr_history")
    return fig


# ── Chart 2: SOFR Forward Curve ───────────────────────────────────────────────

def sofr_forward_curve(
    curve,
    label: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot bootstrapped SOFR forward curve: zero rates and instantaneous forwards."""
    tenors = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 3, 5, 7, 10, 15, 20, 30])
    tenors = tenors[tenors <= curve._times[-1]]

    zeros    = [curve.zero_rate(t) * 100 for t in tenors]
    forwards = [curve.forward_rate(t, t + 0.25) * 100 for t in tenors if t + 0.25 <= curve._times[-1]]
    fwd_tenors = tenors[:len(forwards)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tenors,     zeros,    color=PAPER_COLORS["sofr"],     linewidth=2, label="Zero Rate")
    ax.plot(fwd_tenors, forwards, color=PAPER_COLORS["effr"],     linewidth=2, linestyle="--", label="3M Forward Rate")

    # Mark pillar points
    ax.scatter(curve._times[1:], [curve.zero_rate(t) * 100 for t in curve._times[1:]],
               s=30, color=PAPER_COLORS["sofr"], zorder=5, label="Pillar Points")

    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"SOFR OIS Forward Curve{' — ' + label if label else ''}", fontweight="bold")
    ax.legend()
    _save_or_show(fig, save_path, "sofr_forward_curve")
    return fig


# ── Chart 3: Forward Curve Evolution ─────────────────────────────────────────

def forward_curve_evolution(
    curves: list,
    labels: list[str],
    save_path: str | None = None,
) -> plt.Figure:
    """Overlay multiple SOFR forward curves (snake chart) to show evolution."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(curves)))

    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
    for curve, label, color in zip(curves, labels, colors):
        t_max = curve._times[-1]
        t = tenors[tenors <= t_max]
        zeros = [curve.zero_rate(tt) * 100 for tt in t]
        ax.plot(t, zeros, color=color, linewidth=1.5, label=label)

    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Zero Rate (%)")
    ax.set_title("SOFR OIS Curve Evolution", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    _save_or_show(fig, save_path, "sofr_evolution")
    return fig


# ── Chart 4: Treasury Yield Curve ────────────────────────────────────────────

def treasury_yield_curve(
    yield_df: pd.DataFrame,
    dates: list[str],
    maturity_cols: list[str],
    maturities: list[float],
    save_path: str | None = None,
) -> plt.Figure:
    """Treasury yield curve on multiple dates with NS fit overlay."""
    from models.nelson_siegel import fit_nelson_siegel, ns_yield

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(dates)))

    for dt_str, color in zip(dates, colors):
        dt = pd.Timestamp(dt_str)
        row = yield_df.loc[:dt].iloc[-1][maturity_cols].values.astype(float)
        valid = ~np.isnan(row)
        t_obs = np.array(maturities)[valid]
        y_obs = row[valid]

        ax.scatter(t_obs, y_obs, color=color, s=40, zorder=5)

        try:
            params = fit_nelson_siegel(t_obs, y_obs)
            t_fit  = np.linspace(0.1, t_obs.max(), 200)
            y_fit  = ns_yield(t_fit, params)
            ax.plot(t_fit, y_fit, color=color, linewidth=1.5, label=dt_str)
        except Exception:
            ax.plot(t_obs, y_obs, color=color, linewidth=1.5, label=dt_str)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (%)")
    ax.set_title("US Treasury Yield Curve — Nelson-Siegel Fit", fontweight="bold")
    ax.legend(fontsize=8)
    _save_or_show(fig, save_path, "treasury_yield_curve")
    return fig


# ── Chart 5: NS Factor History ────────────────────────────────────────────────

def ns_factor_history(
    ns_df: pd.DataFrame,
    recession_series: pd.Series | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot Nelson-Siegel β₀, β₁, β₂ over time with regime shading."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    titles = [
        ("beta0", "β₀  (Level — Long-Run Rate)", PAPER_COLORS["sofr"]),
        ("beta1", "β₁  (Slope — 2s10s Driver)",   PAPER_COLORS["effr"]),
        ("beta2", "β₂  (Curvature — Belly)",       PAPER_COLORS["treasury"]),
    ]
    for ax, (col, title, color) in zip(axes, titles):
        if col in ns_df.columns:
            ax.plot(ns_df.index, ns_df[col], color=color, linewidth=1.2)
            ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
            _shade_recessions(ax, recession_series)
            ax.set_ylabel(col)
            ax.set_title(title, fontsize=9, loc="left")

    axes[-1].set_xlabel("Date")
    fig.suptitle("Nelson-Siegel Factors — US Treasury Curve", fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path, "ns_factors")
    return fig


# ── Chart 6: Taylor Rule vs. Actual Rate ─────────────────────────────────────

def taylor_rule_vs_actual(
    actual_rate: pd.Series,
    taylor_df: pd.DataFrame,
    recession_series: pd.Series | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Fed Funds Rate vs. Taylor Rule — highlights over/under-tightening episodes."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(actual_rate.index, actual_rate, color=PAPER_COLORS["effr"],
             linewidth=2, label="Fed Funds Rate (actual)")
    if "taylor_rate" in taylor_df.columns:
        ax1.plot(taylor_df.index, taylor_df["taylor_rate"], color=PAPER_COLORS["taylor"],
                 linewidth=2, linestyle="--", label="Taylor Rule (standard)")

    _shade_recessions(ax1, recession_series)
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Federal Funds Rate vs. Taylor Rule Implied Rate", fontweight="bold")
    ax1.legend()

    # Policy gap in lower panel
    if "taylor_rate" in taylor_df.columns:
        gap = actual_rate.reindex(taylor_df.index, method="ffill") - taylor_df["taylor_rate"]
        ax2.fill_between(gap.index, gap, 0,
                         where=gap > 0, color=PAPER_COLORS["taylor"], alpha=0.5, label="Overtightening")
        ax2.fill_between(gap.index, gap, 0,
                         where=gap < 0, color=PAPER_COLORS["sofr"], alpha=0.5, label="Accommodative")
        ax2.axhline(0, color="black", linewidth=0.7)
        ax2.axhline(0.75, color="gray", linewidth=0.7, linestyle=":")
        ax2.axhline(-0.75, color="gray", linewidth=0.7, linestyle=":")
        ax2.set_ylabel("Gap (%)")
        ax2.set_xlabel("Date")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save_path, "taylor_rule")
    return fig


# ── Chart 7: FOMC Probability Bar Chart ──────────────────────────────────────

def fomc_dot_plot(
    fomc_probs: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """FedWatch-style bar chart of hike/hold/cut probabilities per FOMC meeting."""
    fig, ax = plt.subplots(figsize=(12, 4))

    x     = np.arange(len(fomc_probs))
    width = 0.25

    if "p_hike" in fomc_probs.columns:
        ax.bar(x - width, fomc_probs["p_hike"] * 100, width, label="Hike", color=PAPER_COLORS["taylor"])
    if "p_hold" in fomc_probs.columns:
        ax.bar(x,         fomc_probs["p_hold"] * 100, width, label="Hold", color=PAPER_COLORS["neutral"])
    if "p_cut" in fomc_probs.columns:
        ax.bar(x + width, fomc_probs["p_cut"]  * 100, width, label="Cut",  color=PAPER_COLORS["sofr"])

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in fomc_probs.index], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Market-Implied FOMC Meeting Outcome Probabilities", fontweight="bold")
    ax.legend()
    _save_or_show(fig, save_path, "fomc_probs")
    return fig


# ── Chart 8: Backtest P&L ─────────────────────────────────────────────────────

def backtest_pnl(
    result_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Walk-forward cumulative P&L with drawdown panel."""
    fig = plt.figure(figsize=(12, 8))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1], hspace=0.1)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Cumulative P&L
    ax1.plot(result_df.index, result_df["cumulative_pnl_bps"],
             color=PAPER_COLORS["pnl"], linewidth=2, label="Strategy")
    if benchmark_df is not None and "cumulative_pnl_bps" in benchmark_df.columns:
        ax1.plot(benchmark_df.index, benchmark_df["cumulative_pnl_bps"],
                 color=PAPER_COLORS["neutral"], linewidth=1.5, linestyle="--", label="Buy & Hold")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Cumulative P&L (bps)")
    ax1.set_title("Walk-Forward Backtest — US Rates Signal Strategy", fontweight="bold")
    ax1.legend()

    # Drawdown
    ax2.fill_between(result_df.index, result_df["drawdown_bps"], 0,
                     color=PAPER_COLORS["drawdown"], alpha=0.6)
    ax2.set_ylabel("Drawdown (bps)")

    # Daily P&L distribution
    ax3.bar(result_df.index, result_df["daily_pnl_bps"],
            color=np.where(result_df["daily_pnl_bps"] >= 0, PAPER_COLORS["pnl"], PAPER_COLORS["drawdown"]),
            width=1, alpha=0.7)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("Daily P&L (bps)")
    ax3.set_xlabel("Date")

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    _save_or_show(fig, save_path, "backtest_pnl")
    return fig


# ── Chart 9: Signal Heatmap ───────────────────────────────────────────────────

def signal_heatmap(
    signal_df: pd.DataFrame,
    signal_cols: list[str],
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of individual signal components over time."""
    import matplotlib.colors as mcolors

    data  = signal_df[signal_cols].resample("W").last().fillna(0)
    cmap  = mcolors.LinearSegmentedColormap.from_list("rg", ["#d62728", "#f5f5f5", "#2ca02c"])

    fig, ax = plt.subplots(figsize=(14, len(signal_cols) * 0.8 + 1.5))
    im = ax.imshow(data.T, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
                   extent=[0, len(data), 0, len(signal_cols)])

    ax.set_yticks(np.arange(len(signal_cols)) + 0.5)
    ax.set_yticklabels(signal_cols, fontsize=9)

    # X-axis: yearly labels
    years = pd.date_range(data.index[0], data.index[-1], freq="YS")
    x_pos = [data.index.searchsorted(y) for y in years]
    ax.set_xticks(x_pos)
    ax.set_xticklabels([y.year for y in years], fontsize=9)

    plt.colorbar(im, ax=ax, label="Signal (-1=Short, 0=Flat, +1=Long)", orientation="vertical", pad=0.01)
    ax.set_title("Signal Heatmap — Component Agreement Over Time", fontweight="bold")
    _save_or_show(fig, save_path, "signal_heatmap")
    return fig
