"""
Project Sentinel — Live SOFR Rates Dashboard

Run with:
    streamlit run dashboard/app.py

Tabs:
  1. Market Snapshot  — live SOFR curve inputs → bootstrapped zero curve + DV01
  2. Convexity Table  — CME SR3 futures convexity adjustment schedule
  3. Taylor Rule      — adjust macro inputs → see policy gap
  4. NS Curve Fit     — paste Treasury yields → NS factors + regime
  5. Signal Composite — all four macro signals + composite position
  6. FOMC Probabilities — SR1-implied meeting outcome probabilities
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import date, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Sentinel — US Rates Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Project Sentinel — US Rates & SOFR Dashboard")
st.caption("*Walk-forward validated SOFR pricing engine & macro signal framework*")

# ── Sidebar: global parameters ────────────────────────────────────────────────
with st.sidebar:
    st.header("Global Parameters")
    ref_date_str = st.date_input("Reference Date", value=date.today())
    ref_date     = date(ref_date_str.year, ref_date_str.month, ref_date_str.day)
    hw_sigma     = st.slider("HW σ (convexity vol)", 0.005, 0.030, 0.010, 0.001,
                             help="Hull-White short-rate volatility for convexity adjustment")

    st.divider()
    st.markdown("**Paper:** [draft.md](paper/draft.md)")
    st.markdown("**Tests:** 226 passing ✅")
    st.markdown("**Sharpe (OOS):** 0.282")
    st.markdown("**Hit rate:** 65.7% (12σ above random)")


# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "📊 SOFR Curve",
    "⚡ Convexity",
    "🏛️ Taylor Rule",
    "📐 Nelson-Siegel",
    "🎯 Signals",
    "🎲 FOMC Probs",
    "⚠️ Risk & Scenarios",
    "📉 Curve Strategies",
    "📈 Term Premium",
    "🔬 PCA Factors",
    "🔄 Carry & Roll-Down",
    "🎰 Swaptions",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: SOFR Curve Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("SOFR OIS Discount Curve Bootstrap")
    st.markdown("Enter today's market quotes → bootstrapped curve with zero rates, DV01, and forward rates.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Short-End Deposits (ACT/360)")
        sofr_on = st.number_input("Overnight SOFR (%)", 0.0, 15.0, 3.58, 0.01) / 100
        dep_1m  = st.number_input("1M Term SOFR (%)",   0.0, 15.0, 3.58, 0.01) / 100
        dep_3m  = st.number_input("3M T-bill (%)",      0.0, 15.0, 3.63, 0.01) / 100
        dep_6m  = st.number_input("6M T-bill (%)",      0.0, 15.0, 3.65, 0.01) / 100
        dep_1y  = st.number_input("1Y T-bill (%)",      0.0, 15.0, 3.65, 0.01) / 100

    with col2:
        st.subheader("OIS Swap Par Rates (annual)")
        ois_2y  = st.number_input("2Y OIS (%)",  0.0, 15.0, 4.00, 0.01) / 100
        ois_3y  = st.number_input("3Y OIS (%)",  0.0, 15.0, 4.08, 0.01) / 100
        ois_5y  = st.number_input("5Y OIS (%)",  0.0, 15.0, 4.18, 0.01) / 100
        ois_7y  = st.number_input("7Y OIS (%)",  0.0, 15.0, 4.40, 0.01) / 100
        ois_10y = st.number_input("10Y OIS (%)", 0.0, 15.0, 4.58, 0.01) / 100
        ois_30y = st.number_input("30Y OIS (%)", 0.0, 15.0, 4.98, 0.01) / 100

    if st.button("Bootstrap Curve", type="primary"):
        try:
            from sofr_engine.bootstrap import SOFRCurveBootstrapper

            curve = SOFRCurveBootstrapper.from_market_data(
                ref_date=ref_date,
                sofr_overnight=sofr_on,
                deposit_quotes=[
                    (1/12, dep_1m), (0.25, dep_3m),
                    (0.5, dep_6m),  (1.0, dep_1y),
                ],
                ois_quotes=[
                    (2.0, ois_2y), (3.0, ois_3y), (5.0, ois_5y),
                    (7.0, ois_7y), (10.0, ois_10y), (30.0, ois_30y),
                ],
                sigma=hw_sigma,
            )

            zc = curve.zero_curve(tenors=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
            zc["par_ois_rate_pct"] = [curve.par_ois_rate(t) * 100 for t in zc["tenor_yrs"]]
            zc["dv01_1mm"]         = [curve.dv01(t, 1_000_000) for t in zc["tenor_yrs"]]
            zc.columns = ["Tenor (Y)", "Discount Factor", "Zero Rate (%)", "Fwd 1Y (%)", "Par OIS (%)", "DV01 ($1mm)"]

            st.success(f"Curve bootstrapped: {len(curve._times)} pillars, max tenor {curve._times[-1]:.0f}Y")

            col_a, col_b = st.columns([3, 2])

            with col_a:
                fig, ax = plt.subplots(figsize=(8, 4))
                tenors = np.linspace(0.25, 30, 200)
                zeros  = [curve.zero_rate(t) * 100 for t in tenors]
                fwds   = [curve.forward_rate(t, t+1)*100 for t in tenors[:-1]]
                ax.plot(tenors, zeros, "b-", linewidth=2, label="Zero Rate")
                ax.plot(tenors[:-1], fwds, "r--", linewidth=1.5, alpha=0.8, label="1Y Forward Rate")
                ax.axhline(sofr_on * 100, color="gray", linestyle=":", alpha=0.5, label=f"SOFR ON ({sofr_on*100:.2f}%)")
                ax.set_xlabel("Tenor (Years)")
                ax.set_ylabel("Rate (%)")
                ax.set_title(f"SOFR OIS Curve — {ref_date}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            with col_b:
                st.dataframe(
                    zc.round({"Discount Factor": 6, "Zero Rate (%)": 4,
                               "Fwd 1Y (%)": 4, "Par OIS (%)": 4, "DV01 ($1mm)": 0}),
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Bootstrap error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Convexity Adjustment Table
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Hull-White Convexity Adjustment Schedule")
    st.markdown(
        "Converts futures-implied rates to OIS-consistent forward rates. "
        "Formula: **CA = ½σ²B(0,T₁)B(0,T₂)** where B(0,T) = (1−e^{−aT})/a"
    )

    from sofr_engine.convexity import hull_white_convexity_adjustment

    col_a, col_b = st.columns([1, 2])
    with col_a:
        mean_rev = st.slider("Mean Reversion (a)", 0.0, 0.5, 0.0, 0.01)
        sigma_ca = hw_sigma

    contracts = [
        ("SR3 Sep-26", 0.25, 0.50),
        ("SR3 Dec-26", 0.50, 0.75),
        ("SR3 Mar-27", 0.75, 1.00),
        ("SR3 Jun-27", 1.00, 1.25),
        ("SR3 Sep-27", 1.25, 1.50),
        ("SR3 Dec-27", 1.50, 1.75),
        ("SR3 Mar-28", 1.75, 2.00),
        ("SR3 Jun-28", 2.00, 2.25),
        ("SR3 Sep-28", 2.25, 2.50),
        ("SR3 Dec-28", 2.50, 2.75),
    ]

    rows = []
    for name, t1, t2 in contracts:
        ca      = hull_white_convexity_adjustment(t1, t2, sigma_ca, mean_rev)
        ca_bps  = ca * 10_000
        rows.append({"Contract": name, "T₁ (yrs)": t1, "T₂ (yrs)": t2,
                     "CA (bps)": round(ca_bps, 3), "CA (decimal)": f"{ca:.6f}"})

    df_ca = pd.DataFrame(rows)

    with col_b:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar([r["Contract"] for r in rows], [r["CA (bps)"] for r in rows],
               color="steelblue", alpha=0.8)
        ax.set_ylabel("Convexity Adjustment (bps)")
        ax.set_title(f"SR3 Convexity Adjustments (σ={sigma_ca*100:.1f}%, a={mean_rev:.2f})")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.dataframe(df_ca, use_container_width=True, hide_index=True)
    st.info(
        f"At σ={sigma_ca*100:.1f}%, mean-reversion a={mean_rev:.2f}: "
        f"near-term (<6M) adjustment is negligible (<0.5bps); "
        f"2Y forward contracts: ~{hull_white_convexity_adjustment(2.0, 2.25, sigma_ca, mean_rev)*10000:.1f}bps."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Taylor Rule
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Taylor Rule Policy Gap")
    st.markdown("r\* = r_neutral + α(π − π\*) + β × (−2) × (u − u\*)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Macro Inputs")
        core_pce  = st.number_input("Core PCE YoY (%)", 0.0, 15.0, 2.11, 0.01)
        u_rate    = st.number_input("Unemployment (%)", 0.0, 20.0, 4.30, 0.1)
        effr      = st.number_input("Actual Fed Funds (%)", 0.0, 20.0, 3.63, 0.01)

    with col2:
        st.subheader("Model Parameters")
        r_neutral = st.number_input("Neutral Rate (%)", 0.0, 10.0, 2.50, 0.05)
        pi_star   = st.number_input("Inflation Target (%)", 0.0, 5.0, 2.00, 0.05)
        nairu     = st.number_input("NAIRU (%)", 2.0, 8.0, 4.00, 0.1)
        alpha     = st.slider("α (inflation weight)", 0.0, 2.0, 0.50, 0.05)
        beta      = st.slider("β (output gap weight)", 0.0, 2.0, 0.50, 0.05)

    # Compute Taylor Rate
    infl_gap   = core_pce - pi_star
    output_gap = -2.0 * (u_rate - nairu)
    taylor_r   = r_neutral + alpha * infl_gap + beta * output_gap
    gap        = effr - taylor_r

    with col3:
        st.subheader("Results")
        st.metric("Taylor Rule Rate", f"{taylor_r:.2f}%",
                  delta=f"{gap:+.2f}% vs. actual")
        st.metric("Actual EFFR", f"{effr:.2f}%")
        st.metric("Policy Gap", f"{gap*100:+.0f}bps",
                  delta="Overtightened" if gap > 0 else "Accommodative",
                  delta_color="inverse" if gap > 0 else "normal")

        if abs(gap) > 0.5:
            if gap > 0:
                st.warning(f"⚠️ Fed is **{gap*100:.0f}bps above** Taylor Rule. Structural case for easing.")
            else:
                st.info(f"ℹ️ Fed is **{abs(gap)*100:.0f}bps below** Taylor Rule. Policy is accommodative.")
        else:
            st.success("✅ Fed within 50bps of Taylor Rule recommendation.")

    st.divider()
    col_detail1, col_detail2 = st.columns(2)
    with col_detail1:
        st.markdown("**Decomposition:**")
        st.markdown(f"- Neutral rate: {r_neutral:.2f}%")
        st.markdown(f"- Inflation gap: {infl_gap:+.2f}% → contribution: {alpha*infl_gap:+.2f}%")
        st.markdown(f"- Output gap proxy: {output_gap:+.2f}% → contribution: {beta*output_gap:+.2f}%")
        st.markdown(f"- **Taylor Rate: {taylor_r:.2f}%**")
        st.markdown(f"- **Gap vs actual: {gap*100:+.0f}bps**")

    with col_detail2:
        # Sensitivity chart — vary inflation
        pces  = np.linspace(1.5, 4.0, 50)
        rates = [r_neutral + alpha*(p-pi_star) + beta*output_gap for p in pces]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(pces, rates, "b-", linewidth=2)
        ax.axvline(core_pce, color="red",  linestyle="--", alpha=0.7, label=f"Current PCE {core_pce}%")
        ax.axhline(effr,     color="gray", linestyle=":", alpha=0.7, label=f"Actual EFFR {effr}%")
        ax.fill_between(pces, effr, rates, alpha=0.15,
                        where=[r < effr for r in rates], color="green", label="Room to cut")
        ax.fill_between(pces, effr, rates, alpha=0.15,
                        where=[r > effr for r in rates], color="red",   label="Room to hike")
        ax.set_xlabel("Core PCE (%)")
        ax.set_ylabel("Taylor Rate (%)")
        ax.set_title("Taylor Rate Sensitivity to Inflation")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Nelson-Siegel Curve Fit
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Nelson-Siegel Yield Curve Decomposition")
    st.markdown("Enter current Treasury yields → extract β₀ (level), β₁ (slope), β₂ (curvature), and curve regime.")

    st.subheader("Treasury CMT Yields (%)")
    cols = st.columns(6)
    tenor_labels = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    default_vals = [3.67, 3.65, 3.64, 3.93, 4.05, 4.11, 4.30, 4.53, 4.65, 4.67]
    tenor_yrs    = [0.25, 0.5,  1.0,  2.0,  3.0,  5.0,  7.0,  10.0, 20.0, 30.0]

    input_yields = []
    for i, (lbl, dflt) in enumerate(zip(tenor_labels, default_vals)):
        with cols[i % 6]:
            y = st.number_input(f"{lbl}", 0.0, 15.0, dflt, 0.01, key=f"y_{lbl}")
            input_yields.append(y)

    if st.button("Fit Nelson-Siegel", type="primary"):
        try:
            from models.nelson_siegel import fit_nelson_siegel, ns_yield, classify_curve_regime

            maturities   = np.array(tenor_yrs)
            yields_input = np.array(input_yields)
            params       = fit_nelson_siegel(maturities, yields_input)

            # Compute fitted curve
            fine_mats    = np.linspace(0.1, 30, 200)
            fitted_fine  = ns_yield(fine_mats, params)
            fitted_obs   = ns_yield(maturities, params)
            rmse         = float(np.sqrt(np.mean((fitted_obs - yields_input)**2)))

            # Classify regime
            ns_df   = pd.DataFrame({"beta1": [params.beta1], "beta0": [params.beta0]})
            regime  = classify_curve_regime(ns_df)[0]

            col1, col2 = st.columns([2, 1])

            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(maturities, yields_input, s=60, color="red", zorder=5, label="Market yields")
                ax.plot(fine_mats, fitted_fine, "b-", linewidth=2, label=f"NS fit (λ={params.lam:.2f})")
                ax.set_xlabel("Maturity (Years)")
                ax.set_ylabel("Yield (%)")
                ax.set_title(f"Nelson-Siegel Fit — RMSE: {rmse:.4f}%")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.metric("β₀ (Level)",    f"{params.beta0:.3f}%")
                st.metric("β₁ (Slope)",    f"{params.beta1:.3f}%",
                          delta="Normal" if params.beta1 < 0 else "Inverted",
                          delta_color="normal" if params.beta1 < 0 else "inverse")
                st.metric("β₂ (Curvature)", f"{params.beta2:.3f}%")
                st.metric("λ (Decay)",       f"{params.lam:.3f}")
                st.metric("Fit RMSE",        f"{rmse*100:.2f}bps")

                regime_colors = {
                    "normal_steep": "🟢", "normal_flat": "🟡",
                    "inverted_slight": "🟠", "inverted_deep": "🔴",
                }
                emoji = regime_colors.get(regime, "⚪")
                st.markdown(f"### Regime: {emoji} `{regime.replace('_', ' ').title()}`")
                st.markdown("**Regime interpretation:**")
                if "normal_steep" in regime:
                    st.markdown("- Typical environment (62% of history)\n- Long duration favored\n- Carry is positive")
                elif "inverted" in regime:
                    st.markdown("- Elevated recession risk\n- Stop-loss critical\n- Await normalization")
                else:
                    st.markdown("- Transitional phase\n- Reduce position sizing")

        except Exception as e:
            st.error(f"NS fitting error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Macro Signal Composite
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("Macro Signal Composite")
    st.markdown(
        "Four signals combined into a composite. Position = **+1** (long duration), "
        "**0** (flat), **−1** (short duration)."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Current Inputs")
        s_effr         = st.number_input("EFFR (%)",         0.0, 15.0, 3.63, 0.01, key="sig_effr")
        s_taylor       = st.number_input("Taylor Rate (%)",  0.0, 10.0, 2.25, 0.01, key="sig_taylor")
        s_slope        = st.number_input("2s10s spread (bps)", -300.0, 300.0, 42.0, 1.0)
        s_pce          = st.number_input("Core PCE YoY (%)", 0.0, 15.0, 2.11, 0.01, key="sig_pce")
        s_unemp        = st.number_input("Unemployment (%)", 0.0, 20.0, 4.30, 0.1,  key="sig_unemp")

        st.subheader("Signal Weights")
        w_policy  = st.slider("Policy Gap weight",     0.0, 2.0, 1.0, 0.1)
        w_infl    = st.slider("Inflation weight",      0.0, 2.0, 1.0, 0.1)
        w_labor   = st.slider("Labor Market weight",   0.0, 2.0, 1.0, 0.1)
        w_slope   = st.slider("Curve Slope weight",    0.0, 2.0, 1.0, 0.1)

    with col2:
        from models.macro_signals import policy_gap_signal, inflation_momentum_signal, labor_market_signal, curve_slope_signal

        n   = 504  # ~2 years of history context
        idx = pd.date_range(end=ref_date, periods=n, freq="B")

        # Build realistic trailing series with current value at end
        def trailing_series(current_val, noise=0.02):
            vals = current_val + np.random.randn(n) * noise * current_val
            vals[-1] = current_val
            return pd.Series(vals, index=idx)

        np.random.seed(42)

        # Policy gap signal
        effr_series   = pd.Series([s_effr / 100] * n, index=idx)
        taylor_series = pd.Series([s_taylor / 100] * n, index=idx)
        sig_policy    = policy_gap_signal(effr_series, taylor_series)

        # Inflation signal — use mild trend
        pce_series = trailing_series(s_pce, noise=0.1)
        sig_infl   = inflation_momentum_signal(pce_series)

        # Labor market signal
        unemp_series = trailing_series(s_unemp, noise=0.05)
        sig_labor    = labor_market_signal(unemp_series)

        # Curve slope signal (in decimal form)
        slope_series = trailing_series(s_slope / 100, noise=0.1)
        sig_slope    = curve_slope_signal(slope_series)

        # Composite
        signals = {
            "Policy Gap":     (sig_policy.iloc[-1],  w_policy),
            "Inflation Mom.": (sig_infl.iloc[-1],    w_infl),
            "Labor Market":   (sig_labor.iloc[-1],   w_labor),
            "Curve Slope":    (sig_slope.iloc[-1],   w_slope),
        }

        total_weight = sum(w for _, w in signals.values())
        if total_weight > 0:
            composite = sum(v * w for v, w in signals.values()) / total_weight
        else:
            composite = 0.0

        position = 1 if composite > 0.33 else (-1 if composite < -0.33 else 0)

        # Display
        sig_df = pd.DataFrame([
            {"Signal": name, "Value": int(val), "Weight": w,
             "Contribution": f"{val * w / total_weight if total_weight > 0 else 0:.3f}"}
            for name, (val, w) in signals.items()
        ])
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

        pos_label = {1: "🟢 LONG Duration", 0: "⚪ FLAT", -1: "🔴 SHORT Duration"}
        pos_color = {1: "normal", 0: "off", -1: "inverse"}

        st.metric(
            "Composite Score",
            f"{composite:.3f}",
            delta=f"Position: {position:+d}",
        )
        st.markdown(f"## {pos_label[position]}")
        st.markdown(f"*Threshold: ±0.33 | Current score: {composite:.3f}*")

        # Signal breakdown bar chart
        fig, ax = plt.subplots(figsize=(6, 3))
        names  = list(signals.keys())
        vals   = [v for v, _ in signals.values()]
        colors = ["green" if v > 0 else ("red" if v < 0 else "gray") for v in vals]
        ax.barh(names, vals, color=colors, alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Signal Value (−1/0/+1)")
        ax.set_title("Individual Signal Readings")
        ax.set_xlim(-1.5, 1.5)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: FOMC Probabilities
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.header("FOMC Meeting Outcome Probabilities")
    st.markdown(
        "CME FedWatch methodology: extract implied rate from SR1 futures, "
        "distribute probability over 25bp outcomes."
    )

    col1, col2 = st.columns(2)
    with col1:
        current_rate  = st.number_input("Current EFFR (%)", 0.0, 15.0, 3.625, 0.125,
                                         help="Target range midpoint") / 100
        implied_after = st.number_input("SR1 Implied Rate After Meeting (%)", 0.0, 15.0, 3.50, 0.01,
                                         help="(100 - futures price)/100 adjusted for meeting day") / 100

    from models.fomc_probability import fedwatch_probabilities, fomc_prob_summary

    probs   = fedwatch_probabilities(implied_after, current_rate)
    summary = fomc_prob_summary(implied_after, current_rate)

    with col2:
        st.subheader("Probability Distribution")
        for outcome, prob in sorted(probs.items()):
            label = (f"Hold" if outcome == 0 else
                     f"Cut {abs(outcome)}bps" if outcome < 0 else
                     f"Hike {outcome}bps")
            st.metric(label, f"{prob*100:.1f}%")

    st.divider()
    col_a, col_b = st.columns([2, 1])

    with col_a:
        if probs:
            labels  = []
            p_vals  = []
            for outcome in sorted(probs.keys()):
                lbl = f"Hold" if outcome == 0 else (f"Cut {abs(outcome)}bps" if outcome < 0 else f"Hike {outcome}bps")
                labels.append(lbl)
                p_vals.append(probs[outcome] * 100)

            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors  = ["green" if l.startswith("Cut") else ("red" if l.startswith("Hike") else "steelblue")
                       for l in labels]
            ax.bar(labels, p_vals, color=colors, alpha=0.85)
            ax.set_ylabel("Probability (%)")
            ax.set_title(f"FOMC Outcome Probabilities\n(Current: {current_rate*100:.2f}%, Implied: {implied_after*100:.2f}%)")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col_b:
        st.subheader("Summary")
        for key, val in summary.items():
            if isinstance(val, float):
                st.metric(key, f"{val*100:.1f}%")
            else:
                st.write(f"**{key}:** {val}")

        move = (implied_after - current_rate) * 100
        st.info(
            f"Implied net move: **{move:+.0f}bps**\n\n"
            f"{'Market pricing a cut.' if move < -5 else 'Market pricing a hike.' if move > 5 else 'Market near-fully priced for hold.'}"
        )



# ─────────────────────────────────────────────────────────────────────────────
# TAB 7: Risk Analytics & Scenario Grid
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
    st.header("Risk Analytics — Scenario Grid & DV01 Ladder")
    st.markdown(
        "Build a two-swap portfolio (5Y + 10Y payer), compute key-rate DV01 ladder, "
        "run parallel P&L profile, and view the full stress-test suite."
    )
    from sofr_engine import ScenarioEngine, RiskReport
    from sofr_engine.bootstrap import flat_sofr_curve

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        r_5y  = st.number_input("5Y par rate (%)",  0.5, 10.0, 3.93, 0.01, key="r5y") / 100
    with col_r2:
        r_10y = st.number_input("10Y par rate (%)", 0.5, 10.0, 3.54, 0.01, key="r10y") / 100
    with col_r3:
        base_rate = st.number_input("Flat curve level (%)", 0.5, 10.0, 3.58, 0.01, key="base_r") / 100

    try:
        from sofr_engine.bootstrap import flat_sofr_curve
        from sofr_engine.instruments import SOFRSwap
        from datetime import date
        from dateutil.relativedelta import relativedelta

        rc   = flat_sofr_curve(date.today(), base_rate)
        eff  = date.today() + timedelta(days=2)
        s5   = SOFRSwap(eff, eff + relativedelta(years=5),  r_5y,  10_000_000, pay_fixed=True)
        s10  = SOFRSwap(eff, eff + relativedelta(years=10), r_10y, 10_000_000, pay_fixed=True)
        rpt  = RiskReport({"5Y payer": s5, "10Y payer": s10}, rc)

        col_r_a, col_r_b = st.columns(2)
        with col_r_a:
            st.subheader("Key-Rate DV01 Ladder")
            ladder = rpt.dv01_ladder()
            ladder_disp = ladder.copy()
            ladder_disp["total_dv01_usd"] = ladder_disp["total_dv01_usd"].map("${:,.0f}".format)
            st.dataframe(ladder_disp, use_container_width=True)

        with col_r_b:
            st.subheader("Parallel Shift P&L")
            pnl_df = rpt.parallel_pnl(list(range(-150, 151, 25)))
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(pnl_df["shift_bps"], pnl_df["pnl_usd"] / 1000, "b-o", markersize=4)
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax.axvline(0, color="grey", linewidth=0.8, linestyle=":")
            ax.set_xlabel("Parallel Shift (bps)")
            ax.set_ylabel("P&L ($000s)")
            ax.set_title("Portfolio P&L vs Parallel Shift")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.subheader("Stress Test Scenarios")
        stress = rpt.stress_test()
        stress["pnl_usd"] = stress["pnl_usd"].map("${:,.0f}".format)
        st.dataframe(stress, use_container_width=True)

        st.subheader("2Y × 10Y Scenario Grid ($000s)")
        grid = rpt.scenario_grid([-75, -50, -25, 0, 25, 50, 75], [-75, -50, -25, 0, 25, 50, 75])
        grid_disp = (grid / 1000).round(0)
        st.dataframe(grid_disp.style.background_gradient(cmap="RdYlGn", axis=None),
                     use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8: Curve Strategies
# ─────────────────────────────────────────────────────────────────────────────
with tab8:
    st.header("Multi-Leg Curve Strategies")
    st.markdown(
        "Backtest DV01-neutral 2s10s steepener, flattener, and 2s5s10s butterfly "
        "on synthetic yield curves. Compare Sharpe and cumulative P&L."
    )
    from models.curve_strategies import DV01NeutralSteepener, DV01NeutralFlattener, Butterfly, compare_strategies

    col_cs1, col_cs2 = st.columns(2)
    with col_cs1:
        n_days  = st.slider("Simulation length (days)", 200, 1500, 600, 50, key="cs_n")
        dy2_ann = st.slider("2Y yield total change (bps)", -300, 300, -200, 25, key="cs_dy2",
                            help="Cumulative change in 2Y yield over the period")
    with col_cs2:
        dy10_ann = st.slider("10Y yield total change (bps)", -300, 300, -50, 25, key="cs_dy10",
                              help="Cumulative change in 10Y yield over the period")
        noise_bps = st.slider("Daily noise (bps)", 0, 15, 5, 1, key="cs_noise")

    np.random.seed(42)
    sim_idx = pd.date_range("2019-01-01", periods=n_days, freq="B")
    y2_path  = 4.5 + np.linspace(0, dy2_ann / 100, n_days) + np.random.randn(n_days) * noise_bps / 100
    y10_path = 4.0 + np.linspace(0, dy10_ann / 100, n_days) + np.random.randn(n_days) * noise_bps / 100
    y5_path  = (y2_path + y10_path) / 2 + np.random.randn(n_days) * noise_bps / 200

    sim_data = pd.DataFrame({"tsy_2y": y2_path, "tsy_5y": y5_path, "tsy_10y": y10_path}, index=sim_idx)

    try:
        res_st = DV01NeutralSteepener().run_backtest(sim_data)
        res_fl = DV01NeutralFlattener().run_backtest(sim_data)
        res_bt = Butterfly().run_backtest(sim_data)

        cmp = compare_strategies({"Steepener": res_st, "Flattener": res_fl, "Butterfly": res_bt})
        st.subheader("Strategy Comparison")
        st.dataframe(cmp.style.format({"sharpe": "{:.3f}", "ann_return_bps": "{:.1f}",
                                        "max_drawdown_bps": "{:.1f}", "total_pnl_bps": "{:.1f}",
                                        "hit_rate": "{:.1%}", "n_days": "{:.0f}"}),
                     use_container_width=True)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (name, res) in zip(axes, [("Steepener", res_st), ("Flattener", res_fl), ("Butterfly", res_bt)]):
            ax.plot(res["cumulative_pnl_bps"].values, linewidth=1.5)
            ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
            ax.set_title(f"{name}\nSharpe {cmp.loc[name, 'sharpe']:.2f} | "
                         f"Total {cmp.loc[name, 'total_pnl_bps']:.0f}bps")
            ax.set_xlabel("Days")
            ax.set_ylabel("Cumulative P&L (bps)")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        st.subheader("Cumulative P&L")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        col_sc1, col_sc2 = st.columns(2)
        with col_sc1:
            st.subheader("2s10s Slope")
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            slope = (sim_data["tsy_10y"] - sim_data["tsy_2y"]) * 100
            ax2.plot(slope.values, color="navy", linewidth=1.2)
            ax2.axhline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
            ax2.set_ylabel("2s10s Slope (bps)")
            ax2.set_title(f"Start: {slope.iloc[0]:.0f}bps → End: {slope.iloc[-1]:.0f}bps")
            ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
            st.pyplot(fig2, use_container_width=True)
            plt.close()
        with col_sc2:
            st.subheader("Butterfly (2s5s10s)")
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            fly = (sim_data["tsy_5y"] - 0.5 * sim_data["tsy_2y"] - 0.5 * sim_data["tsy_10y"]) * 100
            ax3.plot(fly.values, color="darkgreen", linewidth=1.2)
            ax3.axhline(0, color="grey", linewidth=0.7, linestyle="--")
            ax3.set_ylabel("Butterfly (bps)")
            ax3.set_title(f"Curvature: start {fly.iloc[0]:.0f}bps → end {fly.iloc[-1]:.0f}bps")
            ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
            st.pyplot(fig3, use_container_width=True)
            plt.close()
    except Exception as e:
        st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 9: Term Premium Decomposition
# ─────────────────────────────────────────────────────────────────────────────
with tab9:
    st.header("Term Premium Decomposition (ACM-style)")
    st.markdown(
        "Uses an AR(1) short-rate model to decompose the 10Y yield into:\n\n"
        "**10Y yield = E[avg short rate over 10Y] + Term Premium**\n\n"
        "The term premium compensates for duration risk, inflation uncertainty, and supply/demand. "
        "ACM (2013) found it averaged +1.5% pre-GFC and turned negative during QE."
    )
    from models.term_premium import rolling_term_premium, term_premium_summary, fit_ar1

    col_tp1, col_tp2 = st.columns(2)
    with col_tp1:
        short_rate_now = st.number_input("Current EFFR (%)", 0.0, 10.0, 4.33, 0.01, key="tp_sr")
        y10_now        = st.number_input("Current 10Y yield (%)", 0.0, 10.0, 4.35, 0.01, key="tp_y10")
        rho_override   = st.slider("AR(1) persistence (ρ)", 0.80, 0.9999, 0.98, 0.005,
                                    key="tp_rho", help="Controls speed of mean reversion in short rate")
    with col_tp2:
        mu_override    = st.number_input("Long-run neutral rate (%)", 0.5, 6.0, 2.5, 0.1, key="tp_mu")
        horizon        = st.selectbox("Term premium horizon", [5, 7, 10, 15, 20, 30], index=2, key="tp_h")

    from models.term_premium import AR1Params, expected_avg_short_rate, term_premium
    params      = AR1Params(mu=mu_override/100, rho=rho_override, sigma=0.005)
    exp_avg     = expected_avg_short_rate(short_rate_now/100, params, horizon) * 100
    tp_now      = y10_now - exp_avg

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("E[Avg Short Rate]", f"{exp_avg:.2f}%",
                   delta=f"{exp_avg - short_rate_now:.2f}% vs today")
    col_m2.metric("Term Premium", f"{tp_now:.2f}%",
                   delta="bullish signal" if tp_now > 0.5 else ("neutral" if tp_now > 0 else "negative"))
    col_m3.metric("Implied r*", f"{mu_override:.2f}%",
                   delta=f"{mu_override - short_rate_now:.2f}% gap vs EFFR")

    # Sensitivity: TP vs short-rate path
    st.subheader("Term Premium Sensitivity")
    col_tpa, col_tpb = st.columns(2)
    with col_tpa:
        fig_tp, ax_tp = plt.subplots(figsize=(5.5, 4))
        sr_range = np.linspace(1.0, 7.0, 80)
        tps = [y10_now - expected_avg_short_rate(sr/100, params, horizon)*100 for sr in sr_range]
        ax_tp.plot(sr_range, tps, "b-", linewidth=2)
        ax_tp.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax_tp.axvline(short_rate_now, color="red", linewidth=1.2, linestyle="--",
                       label=f"Current EFFR {short_rate_now:.2f}%")
        ax_tp.scatter([short_rate_now], [tp_now], color="red", s=60, zorder=5)
        ax_tp.set_xlabel("Current Short Rate (%)")
        ax_tp.set_ylabel("Term Premium (%)")
        ax_tp.set_title(f"{horizon}Y Term Premium vs Short Rate")
        ax_tp.legend(fontsize=9)
        ax_tp.spines["top"].set_visible(False); ax_tp.spines["right"].set_visible(False)
        st.pyplot(fig_tp, use_container_width=True)
        plt.close()

    with col_tpb:
        # AR(1) expected path
        from models.term_premium import short_rate_expectations
        path = short_rate_expectations(short_rate_now/100, params, horizon, freq=12) * 100
        fig_path, ax_path = plt.subplots(figsize=(5.5, 4))
        t_ax = np.linspace(0, horizon, len(path))
        ax_path.plot(t_ax, path, "b-", linewidth=2, label="E[short rate]")
        ax_path.axhline(mu_override, color="green", linewidth=1.0, linestyle="--",
                         alpha=0.8, label=f"Long-run r* = {mu_override:.1f}%")
        ax_path.axhline(y10_now, color="navy", linewidth=1.2, linestyle=":",
                         label=f"10Y yield = {y10_now:.2f}%")
        ax_path.fill_between(t_ax, path, y10_now, alpha=0.15,
                               color="green" if tp_now > 0 else "red",
                               label=f"TP = {tp_now:.2f}%")
        ax_path.set_xlabel("Years ahead")
        ax_path.set_ylabel("Rate (%)")
        ax_path.set_title("AR(1) Expected Short Rate Path")
        ax_path.legend(fontsize=9)
        ax_path.spines["top"].set_visible(False); ax_path.spines["right"].set_visible(False)
        st.pyplot(fig_path, use_container_width=True)
        plt.close()

    with st.expander("Methodology note"):
        st.markdown("""
**ACM (2013) approach (simplified)**

Adrian, Crump & Moench (2013) decompose the n-period yield into:
$$y_t^{(n)} = \\mathbb{E}_t\\left[\\frac{1}{n}\\sum_{h=0}^{n-1} r_{t+h}\\right] + TP_t^{(n)}$$

We approximate $\\mathbb{E}[\\text{avg } r]$ using an AR(1) model for the short rate:
$$r_{t+1} = \\mu(1-\\rho) + \\rho r_t + \\varepsilon_t$$

The closed-form expectation is:
$$\\mathbb{E}[\\text{avg } r] = \\mu + (r_0 - \\mu)\\frac{1 - \\rho^n}{n(1-\\rho)}$$

**Limitation:** This is a 1-factor model; the full ACM model uses 5 principal components of
the yield curve as risk factors and estimates risk prices via no-arbitrage constraints.
Our approach provides an intuitive decomposition but will differ from the NY Fed's published ACM estimates.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 10: PCA Yield Curve Factors
# ─────────────────────────────────────────────────────────────────────────────
with tab10:
    st.header("PCA Yield Curve Factor Analysis")
    st.markdown(
        "The first three principal components explain >99% of historical yield curve variation:\n\n"
        "- **PC1 (~90%):** Level — parallel shift\n"
        "- **PC2 (~5%):** Slope — 2s10s tilt\n"
        "- **PC3 (~1%):** Curvature — butterfly (belly vs wings)"
    )
    from models.pca_factors import YieldCurvePCA, classify_pc_regime

    st.subheader("Synthetic PCA Demo")
    st.info(
        "This demo fits PCA on a synthetic yield curve dataset (300 business days) "
        "with a dominant level factor and smaller slope/curvature components. "
        "Connect FRED data to run on real Treasury yields."
    )

    col_pca1, col_pca2 = st.columns([1, 2])
    with col_pca1:
        n_components = st.selectbox("Number of PCs", [2, 3, 4], index=1, key="pca_ncomp")
        n_obs_pca    = st.slider("Simulated observations", 100, 500, 300, 50, key="pca_nobs")
        seed_pca     = st.number_input("Random seed", 1, 999, 42, key="pca_seed")

    # Generate synthetic data
    rng = np.random.default_rng(int(seed_pca))
    tenors_pca = [2.0, 3.0, 5.0, 7.0, 10.0, 30.0]
    level  = np.cumsum(rng.normal(0, 0.012, n_obs_pca))
    slope  = np.cumsum(rng.normal(0, 0.004, n_obs_pca))
    curve_factor = np.cumsum(rng.normal(0, 0.001, n_obs_pca))
    idio   = rng.normal(0, 0.001, (n_obs_pca, len(tenors_pca)))
    base   = np.array([3.5, 3.7, 3.9, 4.1, 4.3, 4.5])
    slope_load  = np.array([-1.2, -0.8, -0.3, 0.1, 0.5, 1.0]) * 0.3
    curve_load  = np.array([0.3, 0.0, -0.4, -0.3, 0.0, 0.5]) * 0.2
    yields_pca = (base
                  + level[:, None]
                  + slope[:, None] * slope_load[None, :]
                  + curve_factor[:, None] * curve_load[None, :]
                  + idio)
    dates_pca = pd.date_range("2021-01-01", periods=n_obs_pca, freq="B")
    cols_pca  = ["tsy_2y", "tsy_3y", "tsy_5y", "tsy_7y", "tsy_10y", "tsy_30y"]
    yield_df_pca = pd.DataFrame(yields_pca, index=dates_pca, columns=cols_pca)

    try:
        pca = YieldCurvePCA(n_components=int(n_components)).fit(yield_df_pca, tenors_yrs=tenors_pca)

        # Explained variance table
        ev_tbl = pca.explained_variance_table()
        with col_pca2:
            st.dataframe(ev_tbl.style.background_gradient(cmap="Blues", subset=["explained_pct"]),
                         use_container_width=True)

        # Loadings plot
        col_pca_a, col_pca_b = st.columns(2)
        with col_pca_a:
            st.subheader("PC Loadings (Eigenvectors)")
            ld = pca.loadings_df()
            fig_ld, ax_ld = plt.subplots(figsize=(5.5, 4))
            colors = ["steelblue", "darkorange", "green", "red"]
            for i, pc in enumerate(ld.columns):
                ax_ld.plot(tenors_pca, ld[pc].values, "o-",
                           color=colors[i], linewidth=2, markersize=5, label=pc)
            ax_ld.axhline(0, color="grey", linewidth=0.7, linestyle="--")
            ax_ld.set_xlabel("Tenor (years)")
            ax_ld.set_ylabel("Loading")
            ax_ld.set_title("PC Loadings across Tenors")
            ax_ld.legend(fontsize=9)
            ax_ld.spines["top"].set_visible(False)
            ax_ld.spines["right"].set_visible(False)
            st.pyplot(fig_ld, use_container_width=True)
            plt.close()

        with col_pca_b:
            st.subheader("PC Scores over Time")
            scores = pca.transform(yield_df_pca)
            fig_sc, ax_sc = plt.subplots(figsize=(5.5, 4))
            for i, pc in enumerate(scores.columns):
                ax_sc.plot(scores.index, scores[pc], color=colors[i], linewidth=1.2,
                           alpha=0.85, label=pc)
            ax_sc.axhline(0, color="grey", linewidth=0.6, linestyle="--")
            ax_sc.set_xlabel("Date")
            ax_sc.set_ylabel("Score")
            ax_sc.set_title("PC Scores (Level / Slope / Curvature)")
            ax_sc.legend(fontsize=9)
            ax_sc.spines["top"].set_visible(False)
            ax_sc.spines["right"].set_visible(False)
            st.pyplot(fig_sc, use_container_width=True)
            plt.close()

        # Regime classification
        st.subheader("PC Regime Classification")
        regime = classify_pc_regime(scores)
        regime_counts = regime.value_counts()
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.dataframe(regime_counts.rename("Days").to_frame(), use_container_width=True)
        with col_r2:
            fig_regime, ax_regime = plt.subplots(figsize=(8, 2.5))
            regime_num = regime.map(
                {"high_rates": 2, "low_rates": -2, "steep_curve": 1, "flat_curve": -1, "neutral": 0}
            )
            ax_regime.fill_between(regime.index, regime_num, 0, alpha=0.5, color="steelblue")
            ax_regime.set_title("Regime Timeline (PC-based)")
            ax_regime.set_yticks([-2, -1, 0, 1, 2])
            ax_regime.set_yticklabels(["low_rates", "flat_curve", "neutral", "steep_curve", "high_rates"],
                                       fontsize=7)
            ax_regime.spines["top"].set_visible(False)
            ax_regime.spines["right"].set_visible(False)
            st.pyplot(fig_regime, use_container_width=True)
            plt.close()

    except Exception as e:
        st.error(f"PCA error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 11: Carry & Roll-Down
# ─────────────────────────────────────────────────────────────────────────────
with tab11:
    st.header("Carry & Roll-Down Analytics")
    st.markdown(
        "**Carry** = income from holding a position (par rate − overnight financing rate)\n\n"
        "**Roll-Down** = P&L from 'sliding' along the yield curve as the bond ages\n\n"
        "**Total Return** = Carry + Roll-Down (static curve assumption)\n\n"
        "**Breakeven** = max adverse yield move before total return turns negative"
    )
    from models.carry_rolldown import carry_rolldown_table, carry_rolldown_matrix, steepener_carry
    from sofr_engine.bootstrap import flat_sofr_curve

    col_cr1, col_cr2 = st.columns(2)
    with col_cr1:
        sofr_cr  = st.number_input("SOFR overnight (%)", 1.0, 8.0, 4.33, 0.01, key="cr_sofr") / 100
        dt_month = st.selectbox("Holding horizon", [1, 3, 6, 12], index=0, key="cr_dt")
    with col_cr2:
        slope_bp  = st.slider("Curve slope vs flat (bps added at 30Y)", -200, 300, 100, 10,
                               key="cr_slope",
                               help="Adds a linear slope to the flat SOFR curve: 0 bps at 1Y → slope_bp at 30Y")

    # Build a slightly sloped curve
    from sofr_engine.curve import DiscountCurve
    times_cr  = np.array([0.01, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    slope_add = (slope_bp / 10_000) * times_cr / 30.0   # linear ramp
    rates_cr  = sofr_cr + slope_add
    dfs_cr    = np.exp(-rates_cr * times_cr)
    from datetime import date as _date
    curve_cr  = DiscountCurve(_date.today(), times_cr, dfs_cr)

    try:
        dt_yrs = dt_month / 12.0
        tbl_cr = carry_rolldown_table(curve_cr, dt_years=dt_yrs).dropna()

        st.subheader("Carry + Roll-Down Table")
        styled = tbl_cr.style.background_gradient(
            cmap="RdYlGn", subset=["total_return_bps", "carry_bps", "rolldown_bps"]
        ).format("{:.3f}")
        st.dataframe(styled, use_container_width=True)

        col_cr_a, col_cr_b = st.columns(2)
        with col_cr_a:
            st.subheader("Total Return by Tenor")
            fig_cr1, ax_cr1 = plt.subplots(figsize=(5.5, 4))
            ax_cr1.bar(tbl_cr.index.astype(str), tbl_cr["carry_bps"],
                       label="Carry", color="steelblue", alpha=0.8)
            ax_cr1.bar(tbl_cr.index.astype(str), tbl_cr["rolldown_bps"],
                       bottom=tbl_cr["carry_bps"], label="Roll-Down", color="darkorange", alpha=0.8)
            ax_cr1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax_cr1.set_xlabel("Tenor (years)")
            ax_cr1.set_ylabel(f"bps / {dt_month}M")
            ax_cr1.set_title(f"Carry + Roll-Down ({dt_month}M horizon)")
            ax_cr1.legend(fontsize=9)
            ax_cr1.spines["top"].set_visible(False)
            ax_cr1.spines["right"].set_visible(False)
            st.pyplot(fig_cr1, use_container_width=True)
            plt.close()

        with col_cr_b:
            st.subheader("2s10s Steepener Carry Decomposition")
            sc = steepener_carry(curve_cr, short_tenor=2.0, long_tenor=10.0, dt_years=dt_yrs)
            labels  = ["Carry (2Y rcv)", "Carry (10Y pay)", "Net Carry",
                       "RD (2Y)", "RD (10Y)", "Net RD", "Net Total"]
            values  = [sc["carry_2y_bps"], -sc["carry_10y_bps"], sc["net_carry_bps"],
                       sc["rolldown_2y_bps"], -sc["rolldown_10y_bps"], sc["net_rolldown_bps"],
                       sc["net_total_bps"]]
            colors_sc = ["steelblue" if v >= 0 else "tomato" for v in values]
            fig_sc2, ax_sc2 = plt.subplots(figsize=(5.5, 4))
            bars = ax_sc2.barh(labels, values, color=colors_sc, edgecolor="white")
            ax_sc2.axvline(0, color="grey", linewidth=0.8, linestyle="--")
            ax_sc2.set_xlabel(f"bps / {dt_month}M")
            ax_sc2.set_title("2s10s DV01-Neutral Steepener")
            for bar, val in zip(bars, values):
                ax_sc2.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height()/2,
                             f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
            ax_sc2.spines["top"].set_visible(False)
            ax_sc2.spines["right"].set_visible(False)
            st.pyplot(fig_sc2, use_container_width=True)
            plt.close()

        # Carry/rolldown matrix
        st.subheader("Total Return Matrix: Tenor × Horizon")
        from models.carry_rolldown import carry_rolldown_matrix
        mat = carry_rolldown_matrix(
            curve_cr,
            tenors=[2.0, 5.0, 7.0, 10.0, 20.0, 30.0],
            horizons=[1/52, 1/12, 3/12, 6/12, 1.0],
            component="total_return_bps",
        )
        st.dataframe(
            mat.style.background_gradient(cmap="RdYlGn").format("{:.2f}"),
            use_container_width=True,
        )
        st.caption("Values in bps. Green = positive total return (carry + roll-down). "
                   "Assumes static yield curve over the holding horizon.")

    except Exception as e:
        st.error(f"Carry analytics error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 12: Swaption Pricing & Vol Surface
# ─────────────────────────────────────────────────────────────────────────────
with tab12:
    st.header("European Swaption Pricing (Black-76)")
    st.markdown(
        "A swaption is an option to enter a fixed/floating interest rate swap at expiry.\n\n"
        "- **Payer swaption**: right to pay fixed / receive floating (profits if rates rise)\n"
        "- **Receiver swaption**: right to receive fixed / pay floating (profits if rates fall)\n"
        "- Priced via **Black-76**: `PV = N·A·[S·Φ(d₁) − K·Φ(d₂)]` for payer"
    )
    from sofr_engine.swaption import Swaption, SwaptionVolSurface, price_swaption
    from sofr_engine.bootstrap import flat_sofr_curve as _flat_sofr_curve

    col_sw1, col_sw2, col_sw3 = st.columns(3)
    with col_sw1:
        sw_sofr    = st.number_input("SOFR overnight (%)", 1.0, 8.0, 4.33, 0.01, key="sw_sofr") / 100
        sw_expiry  = st.selectbox("Option expiry (years)", [0.5, 1.0, 2.0, 5.0, 10.0], index=1, key="sw_exp")
        sw_tenor   = st.selectbox("Swap tenor (years)", [1.0, 2.0, 5.0, 10.0, 30.0], index=2, key="sw_ten")
    with col_sw2:
        sw_type    = st.radio("Swaption type", ["payer", "receiver"], index=0, key="sw_type")
        sw_notional = st.number_input("Notional ($M)", 1.0, 1000.0, 10.0, 1.0, key="sw_not") * 1_000_000
    with col_sw3:
        sw_vol     = st.slider("Implied vol (%)", 5.0, 60.0, 20.0, 0.5, key="sw_vol") / 100
        sw_atm     = st.checkbox("ATM strike (forward rate)", value=True, key="sw_atm")
        sw_strike  = None
        if not sw_atm:
            sw_strike_pct = st.number_input("Strike (%)", 1.0, 10.0, 4.5, 0.05, key="sw_k")
            sw_strike = sw_strike_pct / 100

    try:
        from datetime import date as _date
        curve_sw = _flat_sofr_curve(_date.today(), sw_sofr)
        result = price_swaption(
            curve_sw, float(sw_expiry), float(sw_tenor),
            strike=sw_strike, notional=sw_notional,
            swaption_type=sw_type, vol=sw_vol,
        )

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Black-76 PV", f"${result['black_pv']:,.0f}")
        col_m2.metric("Intrinsic Value", f"${result['intrinsic_value']:,.0f}")
        col_m3.metric("Time Value", f"${result['time_value']:,.0f}")
        col_m4.metric("Moneyness", f"{result['moneyness_bps']:.1f} bps")

        col_m5, col_m6, col_m7, col_m8 = st.columns(4)
        col_m5.metric("Forward Rate", f"{result['forward_rate_pct']:.4f}%")
        col_m6.metric("Strike", f"{result['strike_pct']:.4f}%")
        col_m7.metric("Vega (per 1bp σ)", f"${result['vega']:,.0f}")
        col_m8.metric("Delta (per 1bp S)", f"${result['delta']:,.0f}")

        # PV vs vol chart
        col_swa, col_swb = st.columns(2)
        with col_swa:
            st.subheader("PV vs Implied Vol")
            vols_range = np.linspace(0.05, 0.60, 80)
            pvs = [
                Swaption(float(sw_expiry), float(sw_tenor),
                         result["strike_pct"] / 100,
                         sw_notional, sw_type, v).black_pv(curve_sw)
                for v in vols_range
            ]
            fig_vol, ax_vol = plt.subplots(figsize=(5.5, 4))
            ax_vol.plot(vols_range * 100, pvs, "steelblue", linewidth=2)
            ax_vol.axvline(sw_vol * 100, color="red", linewidth=1.2, linestyle="--",
                            label=f"Current vol {sw_vol*100:.1f}%")
            ax_vol.scatter([sw_vol * 100], [result["black_pv"]], color="red", s=50, zorder=5)
            ax_vol.set_xlabel("Implied Vol (%)")
            ax_vol.set_ylabel("Swaption PV ($)")
            ax_vol.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax_vol.set_title(f"{sw_type.capitalize()} Swaption PV")
            ax_vol.legend(fontsize=9)
            ax_vol.spines["top"].set_visible(False)
            ax_vol.spines["right"].set_visible(False)
            st.pyplot(fig_vol, use_container_width=True)
            plt.close()

        with col_swb:
            st.subheader("PV vs Strike")
            strikes_range = np.linspace(0.01, 0.10, 80)
            pvs_k = [
                Swaption(float(sw_expiry), float(sw_tenor),
                         k, sw_notional, sw_type, sw_vol).black_pv(curve_sw)
                for k in strikes_range
            ]
            fig_k, ax_k = plt.subplots(figsize=(5.5, 4))
            ax_k.plot(strikes_range * 100, pvs_k, "darkorange", linewidth=2)
            ax_k.axvline(result["forward_rate_pct"], color="green", linewidth=1.2,
                          linestyle="--", label=f"ATM {result['forward_rate_pct']:.2f}%")
            ax_k.axvline(result["strike_pct"], color="red", linewidth=1.2,
                          linestyle=":", label=f"Strike {result['strike_pct']:.2f}%")
            ax_k.set_xlabel("Strike (%)")
            ax_k.set_ylabel("Swaption PV ($)")
            ax_k.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax_k.set_title("PV Profile vs Strike")
            ax_k.legend(fontsize=9)
            ax_k.spines["top"].set_visible(False)
            ax_k.spines["right"].set_visible(False)
            st.pyplot(fig_k, use_container_width=True)
            plt.close()

    except Exception as e:
        st.error(f"Swaption error: {e}")

    # Vol surface
    st.subheader("ATM Swaption Vol Surface (Black-76, 2024 market)")
    surf = SwaptionVolSurface.typical_market()
    surf_df = surf.to_dataframe()
    st.dataframe(
        surf_df.style.background_gradient(cmap="RdYlGn_r").format("{:.1f}%"),
        use_container_width=True,
    )
    st.caption("Lognormal (Black-76) implied vols in %. "
               "At a 4–5% forward rate, multiply by ~450 to get approximate normal vol in bps.")

    fig_surf, ax_surf = plt.subplots(figsize=(8, 4))
    expiry_labels = list(surf_df.index)
    tenor_labels  = list(surf_df.columns)
    im = ax_surf.imshow(surf_df.values, aspect="auto", cmap="RdYlGn_r",
                         vmin=surf_df.values.min(), vmax=surf_df.values.max())
    ax_surf.set_xticks(range(len(tenor_labels)))
    ax_surf.set_xticklabels(tenor_labels)
    ax_surf.set_yticks(range(len(expiry_labels)))
    ax_surf.set_yticklabels(expiry_labels)
    ax_surf.set_xlabel("Swap Tenor")
    ax_surf.set_ylabel("Option Expiry")
    ax_surf.set_title("ATM Swaption Vol Surface (%)")
    for i in range(len(expiry_labels)):
        for j in range(len(tenor_labels)):
            ax_surf.text(j, i, f"{surf_df.iloc[i, j]:.1f}", ha="center", va="center",
                          fontsize=9, color="black")
    plt.colorbar(im, ax=ax_surf, label="Implied Vol (%)")
    st.pyplot(fig_surf, use_container_width=True)
    plt.close()


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Project Sentinel — Bhavesh Anchalia | VIT University | June 2026 | "
    "SOFR Pricing Engine & Macro Signal Framework | 295 tests ✅ | Sharpe 0.282 OOS"
)
