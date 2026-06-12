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
    st.markdown("**Tests:** 885 passing ✅")
    st.markdown("**Sharpe (OOS):** 0.282")
    st.markdown("**Hit rate:** 65.7% (12σ above random)")


# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19, tab20, tab21 = st.tabs([
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
    "📐 SABR Smile",
    "🏦 Return Attribution",
    "🔔 Caps & Floors",
    "🎲 Hull-White MC & VaR",
    "🏛️ Bermudan Swaption",
    "📐 CMS Pricing",
    "🔵 G2++ Two-Factor",
    "🛡️ CDS Pricing",
    "📈 LMM / BGM",
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




# ─────────────────────────────────────────────────────────────────────────────
# TAB 13: SABR Vol Smile
# ─────────────────────────────────────────────────────────────────────────────
with tab13:
    st.header("SABR Volatility Smile")
    st.markdown(
        "The **SABR model** (Hagan et al. 2002) is the industry standard for swaption "
        "vol smile/skew. It models the forward rate and its vol as correlated processes, "
        "generating a realistic skewed vol surface across strikes."
    )

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.subheader("SABR Parameters")
        sabr_F   = st.number_input("Forward Rate (%)", min_value=0.5, max_value=15.0,
                                    value=4.53, step=0.05) / 100.0
        sabr_T   = st.selectbox("Expiry", [0.5, 1.0, 2.0, 5.0, 10.0], index=1)
        sabr_alp = st.slider("Alpha (α) — vol level", 0.01, 0.20, 0.05, 0.005)
        sabr_bet = st.slider("Beta (β) — backbone", 0.0, 1.0, 0.5, 0.1)
        sabr_rho = st.slider("Rho (ρ) — skew", -0.90, 0.90, -0.25, 0.05)
        sabr_nu  = st.slider("Nu (ν) — vol of vol", 0.01, 1.50, 0.40, 0.05)
        strike_range = st.slider("Strike range (±bps)", 50, 300, 150, 25)

    with col_r:
        try:
            from sofr_engine.sabr import SABRParams, sabr_vol_smile

            params = SABRParams(alpha=sabr_alp, beta=sabr_bet, rho=sabr_rho, nu=sabr_nu)
            smile_df = sabr_vol_smile(
                F=sabr_F, T=sabr_T, params=params,
                n_strikes=41, strike_range_bps=float(strike_range),
            )

            fig_sm, ax_sm = plt.subplots(figsize=(7, 4))
            ax_sm.plot(smile_df["moneyness_bps"], smile_df["sabr_vol_pct"],
                       color=BLUE, linewidth=2.5, label="SABR Black-76 vol")
            ax_sm.axvline(0, color="gray", linestyle="--", linewidth=1, label="ATM")
            atm_vol = float(smile_df.loc[smile_df["moneyness_bps"].abs().idxmin(), "sabr_vol_pct"])
            ax_sm.axhline(atm_vol, color="gray", linestyle=":", linewidth=0.8)
            ax_sm.set_xlabel("Moneyness (bps from ATM)")
            ax_sm.set_ylabel("Implied Vol (%)")
            ax_sm.set_title(f"SABR Vol Smile — {sabr_T}Y Expiry, F = {sabr_F*100:.2f}%")
            ax_sm.legend(fontsize=9)
            ax_sm.spines["top"].set_visible(False)
            ax_sm.spines["right"].set_visible(False)
            st.pyplot(fig_sm, use_container_width=True)
            plt.close()

            # Normal vol panel
            st.subheader("Normal (Bachelier) Vol in bps")
            fig_n, ax_n = plt.subplots(figsize=(7, 3))
            ax_n.fill_between(smile_df["moneyness_bps"], smile_df["normal_vol_bps"],
                               alpha=0.35, color=GREEN)
            ax_n.plot(smile_df["moneyness_bps"], smile_df["normal_vol_bps"],
                      color=GREEN, linewidth=2)
            ax_n.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax_n.set_xlabel("Moneyness (bps from ATM)")
            ax_n.set_ylabel("Normal vol (bps)")
            ax_n.set_title("Bachelier (Normal) Vol — σ_N ≈ σ_Black × √(FK)")
            ax_n.spines["top"].set_visible(False)
            ax_n.spines["right"].set_visible(False)
            st.pyplot(fig_n, use_container_width=True)
            plt.close()

        except Exception as e:
            st.error(f"SABR error: {e}")

    # SABR surface calibrated to ATM grid
    st.subheader("SABR Surface — Calibrated to ATM Quotes")
    st.markdown(
        "Fixing **β = 0.5, ρ = −0.25, ν = 0.40** (USD swaption convention), "
        "α is calibrated at each grid node to match the observed ATM vol exactly."
    )
    try:
        from sofr_engine.sabr import SABRSurface, sabr_implied_vol
        from datetime import date as _date

        _sabr_curve = flat_sofr_curve(_date.today(), 0.0453)
        _atm_surf   = SwaptionVolSurface.typical_market()
        _sabr_surf  = SABRSurface.calibrate_from_atm_surface(_atm_surf, _sabr_curve)

        alpha_grid = [[_sabr_surf._params[i][j].alpha * 100
                       for j in range(len(_sabr_surf._tenors))]
                      for i in range(len(_sabr_surf._expiries))]
        import pandas as _pd
        expiry_labels_s = [SwaptionVolSurface._label_years(e) for e in _sabr_surf._expiries]
        tenor_labels_s  = [SwaptionVolSurface._label_years(t) for t in _sabr_surf._tenors]
        alpha_df = _pd.DataFrame(alpha_grid, index=expiry_labels_s, columns=tenor_labels_s)
        alpha_df.index.name   = "expiry"
        alpha_df.columns.name = "tenor"

        st.dataframe(
            alpha_df.style.background_gradient(cmap="Blues").format("{:.2f}%"),
            use_container_width=True,
        )
        st.caption("Calibrated α (%) per grid node. Higher α → higher overall vol level.")

    except Exception as e:
        st.error(f"SABR surface error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 14: Return Attribution (Campisi Framework)
# ─────────────────────────────────────────────────────────────────────────────
with tab14:
    st.header("Fixed-Income Return Attribution (Campisi Framework)")
    st.markdown(
        "Decomposes bond/swap P&L into five orthogonal components:\n\n"
        "**Total Return = Carry + Roll-Down + Duration + Convexity + Residual**\n\n"
        "- **Carry**: net coupon income minus overnight financing cost\n"
        "- **Roll-Down**: price appreciation as the bond ages along an upward-sloping curve\n"
        "- **Duration**: first-order sensitivity to parallel yield shift (−D × Δy)\n"
        "- **Convexity**: second-order Taylor correction (½ × C × Δy²)\n"
        "- **Residual**: unexplained P&L (twist, basis, model error)\n\n"
        "*Reference: Campisi (2000), GRAP Fixed-Income Attribution Standard.*"
    )

    from sofr_engine.bootstrap import flat_sofr_curve as _attr_flat_curve
    from sofr_engine.curve import DiscountCurve as _AttrCurve
    from models.return_attribution import (
        attribute_single_period as _attr_single_period,
        steepener_attribution as _attr_steepener_fn,
    )

    col_at1, col_at2, col_at3 = st.columns(3)
    with col_at1:
        at_sofr_start = st.number_input("SOFR start (%)", 0.5, 10.0, 4.33, 0.01, key="at_s") / 100
        at_sofr_end   = st.number_input("SOFR end (%)",   0.5, 10.0, 4.08, 0.01, key="at_e") / 100
    with col_at2:
        at_tenor  = st.selectbox("Position tenor (years)", [2.0, 5.0, 7.0, 10.0, 20.0, 30.0],
                                  index=3, key="at_ten")
        at_dt_mo  = st.selectbox("Holding period", [1, 3, 6, 12], index=0, key="at_dt",
                                  format_func=lambda x: f"{x} month{'s' if x > 1 else ''}")
    with col_at3:
        at_short = st.selectbox("Steepener short leg",  [1.0, 2.0, 3.0, 5.0], index=1, key="at_short")
        at_long  = st.selectbox("Steepener long leg",   [5.0, 7.0, 10.0, 30.0], index=2, key="at_long")

    try:
        from datetime import date as _attrdate
        _c_s = _attr_flat_curve(_attrdate.today(), at_sofr_start)
        _c_e = _attr_flat_curve(_attrdate.today(), at_sofr_end)
        _dt  = float(at_dt_mo) / 12.0

        r = _attr_single_period(_c_s, _c_e, float(at_tenor), _dt)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total (actual)", f"{r.total_actual_bps:.2f} bps")
        col_m2.metric("Yield Δ", f"{r.delta_y_bps:.1f} bps",
                      delta=f"start {r.yield_start_pct:.3f}% → end {r.yield_end_pct:.3f}%",
                      delta_color="inverse")
        col_m3.metric("Modified Duration", f"{r.modified_duration:.2f}y")
        col_m4.metric("Residual", f"{r.residual_bps:.3f} bps")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.subheader("Attribution Waterfall")
            components = {
                "Carry":     r.carry_bps,
                "Roll-Down": r.rolldown_bps,
                "Duration":  r.duration_bps,
                "Convexity": r.convexity_bps,
                "Residual":  r.residual_bps,
            }
            labels = list(components.keys())
            values = list(components.values())
            bar_colors = ["steelblue" if v >= 0 else "tomato" for v in values]

            import matplotlib.pyplot as _attrplt
            fig_at, ax_at = _attrplt.subplots(figsize=(6, 4))
            bars_at = ax_at.barh(labels, values, color=bar_colors, edgecolor="white", height=0.6)
            ax_at.axvline(0, color="grey", linewidth=0.8, linestyle="--")
            for bar, val in zip(bars_at, values):
                offset = 0.05 if val >= 0 else -0.05
                ha = "left" if val >= 0 else "right"
                ax_at.text(val + offset, bar.get_y() + bar.get_height() / 2,
                            f"{val:.2f}", va="center", ha=ha, fontsize=9)
            ax_at.set_xlabel(f"bps / {at_dt_mo}M")
            ax_at.set_title(
                f"{int(at_tenor)}Y position — SOFR {at_sofr_start*100:.2f}% → {at_sofr_end*100:.2f}%"
            )
            ax_at.spines["top"].set_visible(False)
            ax_at.spines["right"].set_visible(False)
            st.pyplot(fig_at, use_container_width=True)
            _attrplt.close()

        with col_b:
            st.subheader("Attribution Summary Table")
            attr_data = {
                "Component": ["Carry", "Roll-Down", "Duration", "Convexity",
                               "Total (approx)", "Total (actual)", "Residual"],
                "bps": [
                    r.carry_bps, r.rolldown_bps, r.duration_bps, r.convexity_bps,
                    r.total_approx_bps, r.total_actual_bps, r.residual_bps,
                ],
                "% of Total": [
                    v / r.total_actual_bps * 100 if abs(r.total_actual_bps) > 1e-8 else 0.0
                    for v in [r.carry_bps, r.rolldown_bps, r.duration_bps, r.convexity_bps,
                               r.total_approx_bps, r.total_actual_bps, r.residual_bps]
                ],
            }
            import pandas as _pd_at
            at_df = _pd_at.DataFrame(attr_data)
            st.dataframe(
                at_df.style.format({"bps": "{:.3f}", "% of Total": "{:.1f}%"})
                           .background_gradient(cmap="RdYlGn", subset=["bps"]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"Modified Duration: {r.modified_duration:.3f}y | "
                f"Convexity: {r.convexity_years2:.2f}y²"
            )

        # ── Steepener attribution ──────────────────────────────────────────────
        st.subheader(f"DV01-Neutral {int(at_short)}s{int(at_long)}s Steepener Attribution")
        if at_short >= at_long:
            st.warning("Short tenor must be less than long tenor.")
        else:
            steep = _attr_steepener_fn(_c_s, _c_e, float(at_short), float(at_long), _dt)

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Net Total", f"{steep['net_total_bps']:.2f} bps")
            col_s2.metric("Net Carry", f"{steep['net_carry_bps']:.2f} bps")
            col_s3.metric("Net Roll-Down", f"{steep['net_rolldown_bps']:.2f} bps")
            col_s4.metric("DV01 Ratio", f"{steep['dv01_ratio']:.3f}×",
                           help=f"recv {int(at_short)}Y notional = {steep['dv01_ratio']:.3f} × pay {int(at_long)}Y notional")

            steep_labels = [
                f"Carry (rcv {int(at_short)}Y)",
                f"Carry (pay {int(at_long)}Y)",
                "Net Carry",
                f"Roll-Down (rcv {int(at_short)}Y)",
                f"Roll-Down (pay {int(at_long)}Y)",
                "Net Roll-Down",
                f"Duration (net)",
                "Net Total",
            ]
            short_r = steep["short_leg"]
            long_r  = steep["long_leg"]
            dv01r   = steep["dv01_ratio"]
            steep_values = [
                short_r.carry_bps * dv01r,
                -long_r.carry_bps,
                steep["net_carry_bps"],
                short_r.rolldown_bps * dv01r,
                -long_r.rolldown_bps,
                steep["net_rolldown_bps"],
                steep["net_duration_bps"],
                steep["net_total_bps"],
            ]
            s_colors = ["steelblue" if v >= 0 else "tomato" for v in steep_values]
            fig_st, ax_st = _attrplt.subplots(figsize=(7, 4.5))
            bars_st = ax_st.barh(steep_labels, steep_values, color=s_colors,
                                  edgecolor="white", height=0.6)
            ax_st.axvline(0, color="grey", linewidth=0.8, linestyle="--")
            for bar, val in zip(bars_st, steep_values):
                offset = 0.03 if val >= 0 else -0.03
                ha = "left" if val >= 0 else "right"
                ax_st.text(val + offset, bar.get_y() + bar.get_height() / 2,
                            f"{val:.2f}", va="center", ha=ha, fontsize=8)
            ax_st.set_xlabel(f"bps / {at_dt_mo}M (DV01-scaled)")
            ax_st.set_title(
                f"Steepener: rcv {int(at_short)}Y × {dv01r:.2f} / pay {int(at_long)}Y"
            )
            ax_st.spines["top"].set_visible(False)
            ax_st.spines["right"].set_visible(False)
            st.pyplot(fig_st, use_container_width=True)
            _attrplt.close()

    except Exception as e:
        st.error(f"Attribution error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 15: Caps & Floors
# ─────────────────────────────────────────────────────────────────────────────
with tab15:
    st.header("SOFR Cap / Floor Pricing (Black-76)")
    st.markdown(
        "A **cap** is a strip of caplets — call options on quarterly SOFR compounding rates.\n\n"
        "- **Caplet**: pays max(SOFR_i − K, 0) × τ_i at the end of each reset period\n"
        "- **Floor**: symmetric put (receives when SOFR falls below K)\n"
        "- **Put-call parity**: Cap − Floor = PV(Float leg) − PV(Fixed leg at K)\n\n"
        "Model: **Black-76** (log-normal forward rates) — industry standard for USD rate options."
    )
    from sofr_engine.cap_floor import Cap as _Cap, Floor as _Floor
    from sofr_engine.cap_floor import (
        cap_floor_parity_pv as _cf_parity, CapFloorVolSurface as _CFVolSurf,
        strip_caplet_vols as _strip_vols,
    )
    from sofr_engine.bootstrap import flat_sofr_curve as _cf_flat

    col_cf1, col_cf2, col_cf3 = st.columns(3)
    with col_cf1:
        cf_sofr    = st.number_input("SOFR overnight (%)", 1.0, 8.0, 4.33, 0.01, key="cf_sofr") / 100
        cf_mat     = st.selectbox("Maturity (years)", [1, 2, 3, 5, 7, 10], index=3, key="cf_mat")
    with col_cf2:
        cf_atm_chk = st.checkbox("Use ATM strike", value=True, key="cf_atm")
        cf_strike_pct = st.number_input("Strike (%)", 1.0, 10.0, 4.33, 0.05, key="cf_k",
                                         disabled=cf_atm_chk)
        cf_vol     = st.slider("Black-76 flat vol (%)", 5, 80, 30, 1, key="cf_vol") / 100
    with col_cf3:
        cf_notional = st.number_input("Notional ($M)", 1.0, 500.0, 10.0, 1.0, key="cf_not") * 1_000_000
        cf_freq     = st.radio("Reset freq", [2, 4, 12], index=1, key="cf_freq",
                                format_func=lambda x: {2: "Semi-annual", 4: "Quarterly", 12: "Monthly"}[x])

    try:
        from datetime import date as _cfdate
        _cf_curve = _cf_flat(_cfdate.today(), cf_sofr)

        if cf_atm_chk:
            _tmp_cap = _Cap(float(cf_mat), 0.04, cf_notional, int(cf_freq))
            cf_strike = _tmp_cap.atm_forward(_cf_curve)
        else:
            cf_strike = cf_strike_pct / 100

        cap_obj = _Cap(float(cf_mat), cf_strike, cf_notional, int(cf_freq))
        flo_obj = _Floor(float(cf_mat), cf_strike, cf_notional, int(cf_freq))

        cap_pv  = cap_obj.pv(_cf_curve, cf_vol)
        flo_pv  = flo_obj.pv(_cf_curve, cf_vol)
        parity  = _cf_parity(_cf_curve, float(cf_mat), cf_strike, cf_notional, int(cf_freq))
        F_atm   = cap_obj.atm_forward(_cf_curve)

        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("Cap PV", f"${cap_pv:,.0f}")
        col_c2.metric("Floor PV", f"${flo_pv:,.0f}")
        col_c3.metric("Cap − Floor", f"${cap_pv - flo_pv:,.0f}",
                      delta=f"Parity: ${parity:,.0f}",
                      delta_color="off")
        col_c4.metric("ATM Forward", f"{F_atm * 100:.3f}%",
                      delta=f"Strike: {cf_strike * 100:.3f}%",
                      delta_color="off")

        col_cfa, col_cfb = st.columns(2)
        with col_cfa:
            st.subheader("PV vs Strike")
            import matplotlib.pyplot as _cfplt
            _strikes_r = np.linspace(max(0.005, F_atm - 0.025), F_atm + 0.025, 60)
            _cap_pvs   = [_Cap(float(cf_mat), k, cf_notional, int(cf_freq)).pv(_cf_curve, cf_vol)
                          for k in _strikes_r]
            _flo_pvs   = [_Floor(float(cf_mat), k, cf_notional, int(cf_freq)).pv(_cf_curve, cf_vol)
                          for k in _strikes_r]

            fig_cf, ax_cf = _cfplt.subplots(figsize=(6, 4))
            ax_cf.plot(_strikes_r * 100, _cap_pvs, "steelblue", linewidth=2, label="Cap PV")
            ax_cf.plot(_strikes_r * 100, _flo_pvs, "darkorange", linewidth=2, label="Floor PV")
            ax_cf.axvline(F_atm * 100, color="green", linestyle="--", linewidth=1.2,
                           label=f"ATM {F_atm*100:.2f}%")
            ax_cf.axvline(cf_strike * 100, color="red", linestyle=":", linewidth=1.2,
                           label=f"Strike {cf_strike*100:.2f}%")
            ax_cf.yaxis.set_major_formatter(
                __import__("matplotlib.ticker", fromlist=["FuncFormatter"]).FuncFormatter(
                    lambda x, _: f"${x/1e3:.0f}k"
                )
            )
            ax_cf.set_xlabel("Strike (%)")
            ax_cf.set_ylabel("PV ($)")
            ax_cf.set_title(f"{cf_mat}Y Cap/Floor (vol {cf_vol*100:.0f}%)")
            ax_cf.legend(fontsize=9)
            ax_cf.spines["top"].set_visible(False)
            ax_cf.spines["right"].set_visible(False)
            st.pyplot(fig_cf, use_container_width=True)
            _cfplt.close()

        with col_cfb:
            st.subheader("PV vs Vol")
            _vols_r = np.linspace(0.05, 0.80, 60)
            _cap_pv_v = [_Cap(float(cf_mat), cf_strike, cf_notional, int(cf_freq)).pv(_cf_curve, v)
                         for v in _vols_r]
            _flo_pv_v = [_Floor(float(cf_mat), cf_strike, cf_notional, int(cf_freq)).pv(_cf_curve, v)
                         for v in _vols_r]

            fig_cfv, ax_cfv = _cfplt.subplots(figsize=(6, 4))
            ax_cfv.plot(_vols_r * 100, _cap_pv_v, "steelblue", linewidth=2, label="Cap PV")
            ax_cfv.plot(_vols_r * 100, _flo_pv_v, "darkorange", linewidth=2, label="Floor PV")
            ax_cfv.axvline(cf_vol * 100, color="red", linestyle="--", linewidth=1.2,
                            label=f"Current vol {cf_vol*100:.0f}%")
            ax_cfv.yaxis.set_major_formatter(
                __import__("matplotlib.ticker", fromlist=["FuncFormatter"]).FuncFormatter(
                    lambda x, _: f"${x/1e3:.0f}k"
                )
            )
            ax_cfv.set_xlabel("Implied Vol (%)")
            ax_cfv.set_ylabel("PV ($)")
            ax_cfv.set_title("Cap/Floor PV sensitivity to vol")
            ax_cfv.legend(fontsize=9)
            ax_cfv.spines["top"].set_visible(False)
            ax_cfv.spines["right"].set_visible(False)
            st.pyplot(fig_cfv, use_container_width=True)
            _cfplt.close()

        # Greeks table
        dv01_cap  = cap_obj.dv01(_cf_curve, cf_vol)
        dv01_flo  = flo_obj.dv01(_cf_curve, cf_vol)
        vega_cap  = cap_obj.vega(_cf_curve, cf_vol)
        vega_flo  = flo_obj.vega(_cf_curve, cf_vol)
        theta_cap = cap_obj.theta(_cf_curve, cf_vol)

        st.subheader("Greeks Summary")
        import pandas as _cfpd
        greeks_df = _cfpd.DataFrame({
            "Instrument": ["Cap", "Floor"],
            "PV ($)":     [f"${cap_pv:,.0f}", f"${flo_pv:,.0f}"],
            "DV01 ($)":   [f"${dv01_cap:,.0f}", f"${dv01_flo:,.0f}"],
            "Vega ($/bp vol)": [f"${vega_cap:,.0f}", f"${vega_flo:,.0f}"],
            "Theta ($/day)":   [f"${theta_cap:,.0f}", "—"],
            "# Caplets":  [cap_obj.n_caplets(), flo_obj.n_floorlets()],
        })
        st.dataframe(greeks_df, use_container_width=True, hide_index=True)

        # Vol surface
        st.subheader("USD Cap Vol Surface (Black-76 term vols, 2024 market)")
        cf_surf = _CFVolSurf.typical_market(cf_sofr)
        import pandas as _cfpd2
        surf_rows = {}
        for K in cf_surf._strikes:
            surf_rows[f"{K*100:.1f}%"] = {
                f"{T}Y": round(cf_surf.vol(T, K) * 100, 1)
                for T in cf_surf._tenors
            }
        surf_df = _cfpd2.DataFrame(surf_rows).T
        surf_df.index.name   = "strike"
        surf_df.columns.name = "tenor"
        st.dataframe(
            surf_df.style.background_gradient(cmap="RdYlGn_r").format("{:.1f}%"),
            use_container_width=True,
        )

        # Caplet vol strip
        st.subheader("Caplet Vol Bootstrap (term → forward vols)")
        _term_vols = {T: cf_surf.vol(T, cf_strike) for T in cf_surf._tenors}
        _stripped  = _strip_vols(_term_vols, _cf_curve, strike=cf_strike)
        if _stripped:
            strip_df = _cfpd2.DataFrame(
                {"Expiry (y)": [round(e, 3) for e, _ in _stripped],
                 "Term vol (%)": [round(_term_vols.get(round(e + 0.25, 2), cf_vol) * 100, 2)
                                  for e, _ in _stripped],
                 "Fwd caplet vol (%)": [round(v * 100, 2) for _, v in _stripped]}
            )
            st.dataframe(strip_df, use_container_width=True, hide_index=True)
            st.caption("Forward caplet vol stripped from sequential cap differences. "
                       "Reflects the marginal option cost for each quarterly SOFR reset.")

    except Exception as e:
        st.error(f"Cap/Floor error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 16: Hull-White Monte Carlo & VaR
# ─────────────────────────────────────────────────────────────────────────────
with tab16:
    st.header("Hull-White 1F Monte Carlo Simulation & VaR")
    st.markdown(
        "The **Hull-White model** for the short rate:\n\n"
        "dr_t = [θ(t) − a·r_t] dt + σ·dW_t\n\n"
        "θ(t) is calibrated to fit the initial SOFR curve exactly. "
        "Simulation uses the **exact Ornstein-Uhlenbeck transition** — no Euler discretization error.\n\n"
        "- **VaR**: worst-case loss at given confidence over the horizon\n"
        "- **CVaR (Expected Shortfall)**: mean loss beyond VaR — more conservative risk measure"
    )
    from sofr_engine.monte_carlo import (
        HullWhiteParams as _HWP, simulate_hw as _sim_hw,
        portfolio_var_hw as _pvar_hw, parametric_var as _pvar_param,
        convergence_diagnostics as _conv_diag, price_zcb_mc as _zcb_mc,
    )
    from sofr_engine.bootstrap import flat_sofr_curve as _mc_flat

    col_hw1, col_hw2, col_hw3 = st.columns(3)
    with col_hw1:
        hw_sofr   = st.number_input("SOFR overnight (%)", 1.0, 8.0, 4.33, 0.01, key="hw_sofr") / 100
        hw_a      = st.slider("Mean reversion a", 0.00, 0.30, 0.05, 0.01, key="hw_a")
        hw_sigma  = st.slider("Vol σ (% /√yr)", 0.20, 3.00, 1.00, 0.05, key="hw_sig") / 100
    with col_hw2:
        hw_dv01   = st.number_input("Portfolio DV01 ($/bp)", -50_000, 50_000, 10_000, 1_000, key="hw_dv")
        hw_conf   = st.selectbox("VaR confidence", [0.90, 0.95, 0.99], index=2, key="hw_conf")
        hw_horiz  = st.selectbox("Horizon (days)", [1, 5, 10, 21], index=0, key="hw_hor")
    with col_hw3:
        hw_npaths = st.select_slider("n_paths", [1_000, 5_000, 10_000, 20_000], value=10_000, key="hw_n")
        hw_mat    = st.selectbox("Simulation horizon (y)", [0.5, 1.0, 2.0, 5.0], index=1, key="hw_mat")

    try:
        from datetime import date as _hwdate
        _hw_curve  = _mc_flat(_hwdate.today(), hw_sofr)
        _hw_params = _HWP(a=hw_a, sigma=hw_sigma)

        # VaR computation
        var_result = _pvar_hw(
            _hw_curve, _hw_params,
            portfolio_dv01 = hw_dv01,
            horizon        = hw_horiz / 252.0,
            confidence     = hw_conf,
            n_paths        = hw_npaths,
            seed           = 42,
        )
        par_result = _pvar_param(hw_dv01, hw_sigma * 10_000, hw_horiz / 252.0, hw_conf)

        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        col_v1.metric(f"MC VaR ({int(hw_conf*100)}%)",
                      f"${var_result['var_usd']:,.0f}")
        col_v2.metric("MC CVaR (ES)",
                      f"${var_result['cvar_usd']:,.0f}")
        col_v3.metric("Parametric VaR",
                      f"${-par_result['var_usd']:,.0f}",
                      delta="delta-normal")
        col_v4.metric("VaR in bps",
                      f"{abs(var_result['var_bps']):.1f} bps")

        # P&L distribution
        col_hwa, col_hwb = st.columns(2)
        with col_hwa:
            st.subheader("P&L Distribution (MC)")
            import matplotlib.pyplot as _hwplt

            sim = _sim_hw(_hw_curve, _hw_params, horizon=hw_horiz/252.0,
                          n_steps=5, n_paths=hw_npaths, seed=42)
            r_h = sim.r_paths[:, -1]
            from sofr_engine.monte_carlo import zcb_price_hw as _zcb_hw
            p_new = _zcb_hw(r_h, hw_horiz/252.0, hw_horiz/252.0 + 10.0, _hw_curve, _hw_params)
            from sofr_engine.monte_carlo import _inst_forward as _hwf
            y_old = _hw_curve.zero_rate(10.0)
            y_new = -np.log(np.maximum(p_new, 1e-10)) / 10.0
            pnl   = hw_dv01 * (-(y_new - y_old) * 10_000)

            fig_hw, ax_hw = _hwplt.subplots(figsize=(6, 4))
            ax_hw.hist(pnl, bins=60, color="steelblue", alpha=0.7, density=True,
                       edgecolor="white", linewidth=0.3)
            ax_hw.axvline(var_result["var_usd"], color="red", linewidth=2,
                           label=f"VaR {int(hw_conf*100)}%: ${var_result['var_usd']:,.0f}")
            ax_hw.axvline(var_result["cvar_usd"], color="darkred", linewidth=1.5,
                           linestyle="--", label=f"CVaR: ${var_result['cvar_usd']:,.0f}")
            ax_hw.axvline(0, color="grey", linewidth=0.8, linestyle=":")
            ax_hw.set_xlabel("P&L ($)")
            ax_hw.set_ylabel("Density")
            ax_hw.set_title(f"{hw_horiz}-day P&L (DV01={hw_dv01:+,d})")
            ax_hw.legend(fontsize=9)
            ax_hw.spines["top"].set_visible(False)
            ax_hw.spines["right"].set_visible(False)
            st.pyplot(fig_hw, use_container_width=True)
            _hwplt.close()

        with col_hwb:
            st.subheader("Short-Rate Paths (HW)")
            sim_long = _sim_hw(_hw_curve, _hw_params, horizon=float(hw_mat),
                               n_steps=100, n_paths=min(200, hw_npaths), seed=42)
            times_l, mean_l, std_l = sim_long.expected_path()

            fig_hw2, ax_hw2 = _hwplt.subplots(figsize=(6, 4))
            # Plot a sample of paths
            n_show = min(30, sim_long.n_paths)
            for i in range(n_show):
                ax_hw2.plot(times_l, sim_long.r_paths[i] * 100, alpha=0.15,
                             color="steelblue", linewidth=0.6)
            ax_hw2.plot(times_l, mean_l * 100, "steelblue", linewidth=2, label="Mean")
            ax_hw2.fill_between(times_l,
                                 (mean_l - std_l) * 100,
                                 (mean_l + std_l) * 100,
                                 alpha=0.25, color="steelblue", label="±1σ")
            ax_hw2.axhline(hw_sofr * 100, color="grey", linestyle=":", linewidth=1,
                            label=f"SOFR ON {hw_sofr*100:.2f}%")
            ax_hw2.set_xlabel("Time (years)")
            ax_hw2.set_ylabel("Short rate (%)")
            ax_hw2.set_title(f"HW Short-Rate Simulation (a={hw_a:.2f}, σ={hw_sigma*100:.2f}%)")
            ax_hw2.legend(fontsize=9)
            ax_hw2.spines["top"].set_visible(False)
            ax_hw2.spines["right"].set_visible(False)
            st.pyplot(fig_hw2, use_container_width=True)
            _hwplt.close()

        # Convergence diagnostics
        st.subheader("MC Convergence: ZCB Pricing Error vs n_paths")
        import pandas as _hwpd
        conv = _conv_diag(_hw_curve, _hw_params, test_maturity=5.0,
                          path_counts=[100, 500, 1000, 2000, 5000], seed=42)
        conv_df = _hwpd.DataFrame(conv)
        col_cva, col_cvb = st.columns(2)
        with col_cva:
            fig_conv, ax_conv = _hwplt.subplots(figsize=(5.5, 3.5))
            ax_conv.plot(conv_df["n_paths"], conv_df["error_bps"],
                         "o-", color="darkorange", linewidth=2, markersize=5)
            ax_conv.set_xscale("log")
            ax_conv.set_xlabel("n_paths (log scale)")
            ax_conv.set_ylabel("Error (bps)")
            ax_conv.set_title("MC vs Analytical ZCB Price (5Y)")
            ax_conv.spines["top"].set_visible(False)
            ax_conv.spines["right"].set_visible(False)
            st.pyplot(fig_conv, use_container_width=True)
            _hwplt.close()
        with col_cvb:
            st.dataframe(
                conv_df.style.format({"mc_price": "{:.6f}", "error_bps": "{:.3f}",
                                       "mc_stderr": "{:.8f}"}),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                f"Analytical 5Y ZCB = {_hw_curve.df(5.0):.6f}. "
                "MC error ∝ 1/√n — halving the error requires 4× the paths."
            )

        # P&L percentile table
        st.subheader("P&L Percentile Table")
        pcts = var_result["pnl_percentiles"]
        pct_df = _hwpd.DataFrame({
            "Percentile": list(pcts.keys()),
            "P&L ($)": [f"${v:,.0f}" for v in pcts.values()],
        })
        st.dataframe(pct_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"MC/VaR error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 17: Bermudan Swaption (LSM)
# ─────────────────────────────────────────────────────────────────────────────
with tab17:
    st.header("Bermudan Swaption — Longstaff-Schwartz MC (LSM)")
    st.markdown(
        "A **Bermudan swaption** allows exercise at any of a discrete set of dates "
        "(typically every 6 months). It is more valuable than a European swaption "
        "because of the early-exercise optionality.\n\n"
        "**Algorithm (Longstaff-Schwartz 2001):**\n"
        "1. Simulate N Hull-White short-rate paths forward\n"
        "2. Backward induction: at each exercise date, regress continuation value "
        "on Laguerre polynomial basis functions of the short rate\n"
        "3. Exercise if immediate value > fitted continuation value\n"
        "4. Price = discounted average optimal exercise payoff\n\n"
        "The **early exercise premium** is the additional value over the best European swaption."
    )
    from sofr_engine.bermudan import (
        price_bermudan_swaption as _berm_price,
        price_european_swaption_hw as _euro_hw,
    )
    from sofr_engine.monte_carlo import HullWhiteParams as _HWPB
    from sofr_engine.bootstrap import flat_sofr_curve as _berm_flat

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        b_sofr    = st.number_input("SOFR overnight (%)", 1.0, 8.0, 4.33, 0.01, key="b_sofr") / 100
        b_first   = st.number_input("First exercise (y)", 0.5, 5.0, 1.0, 0.5, key="b_first")
        b_mat     = st.number_input("Swap maturity (y)",  2.0, 20.0, 6.0, 0.5, key="b_mat")
    with col_b2:
        b_freq    = st.radio("Exercise freq", [1, 2, 4], index=1, key="b_freq",
                              format_func=lambda x: {1: "Annual", 2: "Semi-annual", 4: "Quarterly"}[x])
        b_type    = st.radio("Type", ["payer", "receiver"], index=0, key="b_type")
        b_atm     = st.checkbox("ATM strike", value=True, key="b_atm_chk")
        b_strike_pct = st.number_input("Strike (%)", 1.0, 10.0, 4.33, 0.05, key="b_k",
                                        disabled=b_atm)
    with col_b3:
        b_hw_a    = st.slider("HW mean reversion a", 0.0, 0.30, 0.05, 0.01, key="b_a")
        b_hw_sig  = st.slider("HW σ (% /√yr)", 0.20, 3.00, 1.00, 0.05, key="b_sig") / 100
        b_npaths  = st.select_slider("n_paths", [1000, 2000, 5000, 10000], value=3000, key="b_n")

    if b_first >= b_mat:
        st.warning("First exercise must be before swap maturity.")
    else:
        try:
            from datetime import date as _bdate
            _b_curve  = _berm_flat(_bdate.today(), b_sofr)
            _b_params = _HWPB(a=b_hw_a, sigma=b_hw_sig)
            _b_strike = None if b_atm else (b_strike_pct / 100)

            with st.spinner("Running LSM backward induction…"):
                berm = _berm_price(
                    _b_curve, _b_params,
                    first_exercise = float(b_first),
                    swap_maturity  = float(b_mat),
                    strike         = _b_strike,
                    pay_receive    = b_type,
                    exercise_freq  = int(b_freq),
                    n_paths        = b_npaths,
                    seed           = 42,
                )

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Bermudan PV", f"${berm.price:,.0f}")
            col_m2.metric("European Lower Bound", f"${berm.european_lower:,.0f}")
            col_m3.metric("Early Exercise Premium",
                          f"${berm.early_exercise_premium:,.0f}",
                          delta=f"{berm.early_exercise_premium/max(berm.european_lower,1)*100:.1f}%"
                                if berm.european_lower > 0 else None)
            col_m4.metric("Strike", f"{berm.strike * 100:.3f}%")

            col_ba, col_bb = st.columns(2)
            with col_ba:
                st.subheader("Exercise Probability by Date")
                import matplotlib.pyplot as _bplt
                import pandas as _bpd

                fig_b, ax_b = _bplt.subplots(figsize=(6, 4))
                ax_b.bar(
                    [f"{d:.1f}Y" for d in berm.exercise_dates],
                    [p * 100 for p in berm.exercise_probs],
                    color="steelblue", alpha=0.85, edgecolor="white",
                )
                ax_b.set_xlabel("Exercise Date")
                ax_b.set_ylabel("% of Paths In-the-Money")
                ax_b.set_title(f"{b_type.capitalize()} Bermudan — ITM Rate by Date")
                ax_b.spines["top"].set_visible(False)
                ax_b.spines["right"].set_visible(False)
                st.pyplot(fig_b, use_container_width=True)
                _bplt.close()

            with col_bb:
                st.subheader("European vs Bermudan Breakdown")
                labels = ["European\n(Best Date)", "Early Exercise\nPremium", "Bermudan\nTotal"]
                values = [berm.european_lower,
                          max(berm.early_exercise_premium, 0),
                          berm.price]
                bar_colors = ["steelblue", "darkorange", "green"]
                fig_b2, ax_b2 = _bplt.subplots(figsize=(6, 4))
                ax_b2.bar(labels, values, color=bar_colors, alpha=0.85, edgecolor="white")
                for rect, val in zip(ax_b2.patches, values):
                    ax_b2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 100,
                                f"${val:,.0f}", ha="center", fontsize=9)
                ax_b2.set_ylabel("PV ($)")
                ax_b2.set_title("Bermudan = European + Early Exercise Premium")
                ax_b2.spines["top"].set_visible(False)
                ax_b2.spines["right"].set_visible(False)
                st.pyplot(fig_b2, use_container_width=True)
                _bplt.close()

            # Exercise schedule table
            st.subheader("Exercise Schedule")
            sched_df = _bpd.DataFrame({
                "Exercise Date (y)": [round(d, 3) for d in berm.exercise_dates],
                "ITM Probability":   [f"{p*100:.1f}%" for p in berm.exercise_probs],
            })
            st.dataframe(sched_df, use_container_width=True, hide_index=True)
            st.caption(
                f"Bermudan: {len(berm.exercise_dates)} exercise dates × "
                f"{b_freq}× per year | HW: a={b_hw_a:.2f}, σ={b_hw_sig*100:.2f}% | "
                f"{berm.n_paths:,} paths"
            )

        except Exception as e:
            st.error(f"Bermudan error: {e}")


# TAB 18: CMS Pricing
# ─────────────────────────────────────────────────────────────────────────────
with tab18:
    st.header("CMS Pricing — Constant Maturity Swap")
    st.markdown(
        "**CMS (Constant Maturity Swap)** instruments pay a floating rate tied to a long-tenor "
        "swap rate (e.g. the 10-year swap rate) rather than an overnight or 3-month rate. "
        "Because a receiver of CMS is long the convexity of the yield curve, the CMS rate must "
        "be adjusted upward relative to the forward swap rate — the **convexity adjustment**.\n\n"
        "- **Linear TSR** (Terminal Swap Rate): closed-form Hagan (2003) adjustment\n"
        "- **CMS Caplet/Floorlet**: Black-76 with convexity-adjusted forward\n"
        "- **CMS Spread Option**: Kirk's bivariate-normal approximation (steepener/flattener)\n"
    )

    from sofr_engine.cms import (
        cms_convexity_adj as _cms_adj,
        CMSCaplet as _CMSCapletDash,
        cms_caplet_pv as _cms_cap_pv,
        CMSSpreadOption as _CMSSpreadDash,
        cms_spread_option_pv as _cms_spr_pv,
    )
    from sofr_engine.bootstrap import flat_sofr_curve as _cms_flat

    cms_sub = st.radio("Section", ["Convexity Schedule", "CMS Caplet/Floorlet", "Spread Option"],
                       horizontal=True, key="cms_sub")

    # ── Convexity Schedule ──────────────────────────────────────────────────
    if cms_sub == "Convexity Schedule":
        st.subheader("CMS Convexity Adjustment Schedule")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            cms_sofr    = st.number_input("SOFR overnight (%)", 1.0, 8.0, 4.33, 0.01, key="cms_sofr") / 100
        with col_c2:
            cms_tenor   = st.number_input("CMS swap tenor (y)", 1.0, 30.0, 10.0, 1.0, key="cms_tenor")
        with col_c3:
            cms_vol     = st.slider("Swaption vol (%)", 5, 80, 30, 1, key="cms_vol") / 100
        cms_model = st.radio("Model", ["linear_tsr", "replication"], horizontal=True, key="cms_model")

        try:
            import matplotlib.pyplot as _cplt
            import pandas as _cpd

            _cms_crv  = _cms_flat(__import__("datetime").date.today(), cms_sofr)
            expiries  = [1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
            rows = []
            for T in expiries:
                r = _cms_adj(_cms_crv, T, cms_tenor, cms_vol, model=cms_model)
                rows.append({
                    "Expiry (y)": round(T, 4),
                    "Forward Swap Rate (%)": round(r.forward_swap_rate * 100, 4),
                    "Conv. Adj. (bps)": round(r.convexity_adj_bps, 3),
                    "CMS Rate (%)": round(r.cms_rate * 100, 4),
                })
            df_sched = _cpd.DataFrame(rows)
            st.dataframe(df_sched, use_container_width=True, hide_index=True)

            fig_c, ax_c = _cplt.subplots(figsize=(9, 4))
            ax_c.bar(df_sched["Expiry (y)"].astype(str), df_sched["Conv. Adj. (bps)"],
                     color="royalblue", alpha=0.85, edgecolor="white")
            ax_c.set_xlabel("Expiry (years)")
            ax_c.set_ylabel("Convexity Adjustment (bps)")
            ax_c.set_title(f"CMS {cms_tenor:.0f}Y Convexity Adjustment — {cms_model.upper().replace('_', ' ')} Model")
            ax_c.spines["top"].set_visible(False)
            ax_c.spines["right"].set_visible(False)
            st.pyplot(fig_c, use_container_width=True)
            _cplt.close()

            n0 = rows[0]["Conv. Adj. (bps)"]
            n_last = rows[-1]["Conv. Adj. (bps)"]
            st.caption(
                f"CMS {cms_tenor:.0f}Y convexity adj grows from {n0:.1f}bps (1M expiry) "
                f"to {n_last:.1f}bps (10Y expiry) at vol={cms_vol*100:.0f}%"
            )
        except Exception as e:
            st.error(f"CMS convexity error: {e}")

    # ── CMS Caplet / Floorlet ───────────────────────────────────────────────
    elif cms_sub == "CMS Caplet/Floorlet":
        st.subheader("CMS Caplet / Floorlet Pricing")
        col_c4, col_c5, col_c6 = st.columns(3)
        with col_c4:
            cl_sofr   = st.number_input("SOFR (%)", 1.0, 8.0, 4.33, 0.01, key="cl_sofr") / 100
            cl_t_fix  = st.number_input("Fixing date (y)", 0.25, 10.0, 1.0, 0.25, key="cl_tfix")
            cl_t_pay  = st.number_input("Payment date (y)", 0.25, 10.5, 1.25, 0.25, key="cl_tpay")
        with col_c5:
            cl_tenor  = st.number_input("CMS swap tenor (y)", 1.0, 30.0, 10.0, 1.0, key="cl_tenor")
            cl_vol    = st.slider("Swaption vol (%)", 5, 80, 30, 1, key="cl_vol") / 100
            cl_model  = st.radio("Model", ["linear_tsr", "replication"], horizontal=True, key="cl_model")
        with col_c6:
            cl_strike_pct = st.number_input("Strike (%)", 0.5, 15.0, 4.33, 0.05, key="cl_k")
            cl_notional   = st.number_input("Notional ($M)", 0.1, 100.0, 10.0, 0.5, key="cl_not") * 1e6
            cl_cf         = st.radio("Cap / Floor", ["cap", "floor"], horizontal=True, key="cl_cf")

        try:
            _cl_crv = _cms_flat(__import__("datetime").date.today(), cl_sofr)
            cl_caplet = _CMSCapletDash(
                t_fix=float(cl_t_fix), t_pay=float(cl_t_pay), swap_tenor=float(cl_tenor),
                strike=cl_strike_pct / 100, notional=cl_notional, cap_floor=cl_cf,
            )
            cl_pv = _cms_cap_pv(cl_caplet, _cl_crv, cl_vol, model=cl_model)
            cl_res = _cms_adj(_cl_crv, cl_t_fix, cl_tenor, cl_vol, model=cl_model)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CMS Caplet PV", f"${cl_pv:,.0f}")
            c2.metric("CMS Rate", f"{cl_res.cms_rate * 100:.4f}%")
            c3.metric("Fwd Swap Rate", f"{cl_res.forward_swap_rate * 100:.4f}%")
            c4.metric("Conv. Adj.", f"{cl_res.convexity_adj_bps:.2f} bps")

            # Strike sensitivity
            import numpy as np
            import matplotlib.pyplot as _cl_plt

            strikes_pct = np.linspace(max(0.5, cl_strike_pct - 2), cl_strike_pct + 2, 60)
            pvs = []
            for K in strikes_pct:
                cl_tmp = _CMSCapletDash(t_fix=float(cl_t_fix), t_pay=float(cl_t_pay),
                                        swap_tenor=float(cl_tenor), strike=K / 100,
                                        notional=cl_notional, cap_floor=cl_cf)
                pvs.append(_cms_cap_pv(cl_tmp, _cl_crv, cl_vol, model=cl_model))

            fig_cl, ax_cl = _cl_plt.subplots(figsize=(9, 4))
            ax_cl.plot(strikes_pct, pvs, color="steelblue", linewidth=2)
            ax_cl.axvline(cl_strike_pct, color="red", linestyle="--", alpha=0.7, label=f"Strike={cl_strike_pct:.2f}%")
            ax_cl.axvline(cl_res.cms_rate * 100, color="green", linestyle="--", alpha=0.7,
                          label=f"CMS Rate={cl_res.cms_rate * 100:.3f}%")
            ax_cl.set_xlabel("Strike (%)")
            ax_cl.set_ylabel("PV ($)")
            ax_cl.set_title(f"CMS {cl_cf.capitalize()} PV vs Strike | {cl_tenor:.0f}Y CMS, T={cl_t_fix:.2f}Y")
            ax_cl.legend()
            ax_cl.spines["top"].set_visible(False)
            ax_cl.spines["right"].set_visible(False)
            st.pyplot(fig_cl, use_container_width=True)
            _cl_plt.close()

        except Exception as e:
            st.error(f"CMS caplet error: {e}")

    # ── CMS Spread Option ───────────────────────────────────────────────────
    else:
        st.subheader("CMS Spread Option — Steepener / Flattener")
        st.markdown(
            "A **steepener call** profits when the yield curve steepens "
            "(10Y−2Y spread widens above the strike). A **flattener put** profits "
            "when the curve flattens. Priced via Kirk's (1995) bivariate-normal approximation."
        )
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            spr_sofr   = st.number_input("SOFR (%)", 1.0, 8.0, 4.33, 0.01, key="spr_sofr") / 100
            spr_t_fix  = st.number_input("Expiry (y)", 0.25, 10.0, 1.0, 0.25, key="spr_tfx")
            spr_t_pay  = st.number_input("Payment (y)", 0.25, 10.5, 1.25, 0.25, key="spr_tpy")
        with col_s2:
            spr_l_ten  = st.number_input("Long tenor (y)", 2.0, 30.0, 10.0, 1.0, key="spr_lt")
            spr_s_ten  = st.number_input("Short tenor (y)", 1.0, 15.0, 2.0, 1.0, key="spr_st")
            spr_k_bps  = st.number_input("Strike (bps)", -100, 300, 50, 5, key="spr_k")
        with col_s3:
            spr_vl     = st.slider("Vol Long (%)", 5, 80, 30, 1, key="spr_vl") / 100
            spr_vs     = st.slider("Vol Short (%)", 5, 80, 30, 1, key="spr_vs") / 100
            spr_rho    = st.slider("Correlation ρ", -0.99, 0.99, 0.70, 0.01, key="spr_rho")
            spr_not    = st.number_input("Notional ($M)", 0.1, 500.0, 10.0, 1.0, key="spr_not") * 1e6
            spr_cp     = st.radio("Call / Put", ["call", "put"], horizontal=True, key="spr_cp")

        try:
            _spr_crv = _cms_flat(__import__("datetime").date.today(), spr_sofr)
            spr_opt = _CMSSpreadDash(
                t_fix=float(spr_t_fix), t_pay=float(spr_t_pay),
                long_tenor=float(spr_l_ten), short_tenor=float(spr_s_ten),
                spread_strike=spr_k_bps / 10_000, notional=spr_not, call_put=spr_cp,
            )
            spr_pv = _cms_spr_pv(spr_opt, _spr_crv, spr_vl, spr_vs, spr_rho)
            spr_rl = _cms_adj(_spr_crv, spr_t_fix, spr_l_ten, spr_vl)
            spr_rs = _cms_adj(_spr_crv, spr_t_fix, spr_s_ten, spr_vs)

            cv1, cv2, cv3, cv4 = st.columns(4)
            cv1.metric("Spread Option PV", f"${spr_pv:,.0f}")
            cv2.metric(f"CMS {spr_l_ten:.0f}Y", f"{spr_rl.cms_rate * 100:.4f}%",
                       delta=f"+{spr_rl.convexity_adj_bps:.2f} bps conv adj")
            cv3.metric(f"CMS {spr_s_ten:.0f}Y", f"{spr_rs.cms_rate * 100:.4f}%",
                       delta=f"+{spr_rs.convexity_adj_bps:.2f} bps conv adj")
            cv4.metric("CMS Spread", f"{(spr_rl.cms_rate - spr_rs.cms_rate) * 10000:.1f} bps")

            # Rho sensitivity
            import numpy as np
            import matplotlib.pyplot as _spl

            rhos     = np.linspace(-0.95, 0.95, 50)
            pv_rhos  = []
            for r in rhos:
                pv_rhos.append(_cms_spr_pv(spr_opt, _spr_crv, spr_vl, spr_vs, float(r)))

            fig_spr, ax_spr = _spl.subplots(figsize=(9, 4))
            ax_spr.plot(rhos, [v / 1000 for v in pv_rhos], color="darkorange", linewidth=2)
            ax_spr.axvline(spr_rho, color="red", linestyle="--", alpha=0.7,
                           label=f"ρ={spr_rho:.2f}")
            ax_spr.set_xlabel("Correlation ρ")
            ax_spr.set_ylabel("Option PV ($K)")
            title_type = "Steepener Call" if spr_cp == "call" else "Flattener Put"
            ax_spr.set_title(f"CMS Spread Option PV vs Correlation | {title_type} {spr_l_ten:.0f}Y−{spr_s_ten:.0f}Y")
            ax_spr.legend()
            ax_spr.spines["top"].set_visible(False)
            ax_spr.spines["right"].set_visible(False)
            st.pyplot(fig_spr, use_container_width=True)
            _spl.close()

            st.caption(
                f"Kirk's approximation | Correlation={spr_rho:.2f} | "
                f"Strike={spr_k_bps}bps | Notional=${spr_not/1e6:.0f}M"
            )
        except Exception as e:
            st.error(f"CMS spread option error: {e}")


# TAB 19: G2++ Two-Factor Model
# ─────────────────────────────────────────────────────────────────────────────
with tab19:
    st.header("G2++ Two-Factor Gaussian Interest Rate Model")
    st.markdown(
        "The **G2++** model (Brigo & Mercurio, 2006) adds a second mean-reverting factor "
        "to Hull-White 1F, achieving a richer term structure fit:\n\n"
        "$$r(t) = x(t) + y(t) + \\varphi(t)$$\n\n"
        "$$dx = -a\\,x\\,dt + \\sigma\\,dW_1, \\quad "
        "dy = -b\\,y\\,dt + \\eta\\,dW_2, \\quad dW_1 dW_2 = \\rho\\,dt$$\n\n"
        "The correlated two-factor structure generates humped yield curves and better fits "
        "the swaption vol surface than 1F models."
    )

    from sofr_engine.g2pp import (
        G2ppParams as _G2PP_P,
        simulate_g2pp as _sim_g2,
        g2pp_swaption_mc as _g2sw,
        g2pp_portfolio_var as _g2var,
        g2pp_zcb as _g2zcb,
    )
    from sofr_engine.bootstrap import flat_sofr_curve as _g2_flat

    g2_sub = st.radio("Section", ["ZCB & Yield Curve", "Swaption Pricing", "Portfolio VaR"],
                      horizontal=True, key="g2_sub")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        g2_sofr  = st.number_input("SOFR overnight (%)", 1.0, 8.0, 4.33, 0.01, key="g2_sofr") / 100
        g2_a     = st.slider("a (x-factor speed)", 0.01, 0.50, 0.05, 0.01, key="g2a")
        g2_b     = st.slider("b (y-factor speed)", 0.01, 0.50, 0.10, 0.01, key="g2b")
    with col_g2:
        g2_sigma = st.slider("σ (x-factor vol %)", 0.10, 3.00, 1.00, 0.05, key="g2s") / 100
        g2_eta   = st.slider("η (y-factor vol %)", 0.10, 3.00, 0.80, 0.05, key="g2e") / 100
        g2_rho   = st.slider("ρ (W1-W2 correlation)", -0.99, 0.99, -0.30, 0.01, key="g2rho")

    try:
        _g2_crv = _g2_flat(__import__("datetime").date.today(), g2_sofr)
        _g2_p   = _G2PP_P(a=g2_a, b=g2_b, sigma=g2_sigma, eta=g2_eta, rho=g2_rho)

        if g2_sub == "ZCB & Yield Curve":
            import numpy as np
            import matplotlib.pyplot as _g2plt

            # Simulate 3 paths and show short-rate evolution
            sim = _sim_g2(_g2_crv, _g2_p, horizon=5.0, n_steps=250, n_paths=6, seed=42)

            fig_g, axes = _g2plt.subplots(1, 2, figsize=(12, 4))
            ax1, ax2 = axes

            # Short rate paths
            for i in range(6):
                ax1.plot(sim.t_grid, sim.r_paths[i] * 100, alpha=0.7, linewidth=1)
            ax1.set_xlabel("Time (years)")
            ax1.set_ylabel("Short Rate (%)")
            ax1.set_title("G2++ Short Rate Paths")
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)

            # G2++ yield curve vs initial curve
            mats = np.linspace(0.25, 20.0, 80)
            yields_g2 = np.array([-np.log(_g2zcb(_g2_crv, _g2_p, 0.0, float(T), 0.0, 0.0)) / T
                                   for T in mats]) * 100
            yields_mkt = np.array([-np.log(float(_g2_crv.df(float(T)))) / T for T in mats]) * 100
            ax2.plot(mats, yields_mkt, "k--", linewidth=2, label="Initial curve")
            ax2.plot(mats, yields_g2, "royalblue", linewidth=2, label="G2++ repriced")
            ax2.set_xlabel("Maturity (years)")
            ax2.set_ylabel("Zero Yield (%)")
            ax2.set_title("G2++ vs Market Yield Curve")
            ax2.legend()
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)

            _g2plt.tight_layout()
            st.pyplot(fig_g, use_container_width=True)
            _g2plt.close()
            st.caption(f"G2++: a={g2_a:.2f}, b={g2_b:.2f}, σ={g2_sigma*100:.2f}%, η={g2_eta*100:.2f}%, ρ={g2_rho:.2f}")

        elif g2_sub == "Swaption Pricing":
            col_sw1, col_sw2 = st.columns(2)
            with col_sw1:
                sw_exp  = st.number_input("Expiry (y)", 0.25, 10.0, 1.0, 0.25, key="sw_exp")
                sw_ten  = st.number_input("Swap tenor (y)", 1.0, 20.0, 5.0, 1.0, key="sw_ten")
                sw_type = st.radio("Type", ["payer", "receiver"], horizontal=True, key="sw_type")
            with col_sw2:
                sw_np   = st.select_slider("n_paths", [2000, 5000, 10000], value=5000, key="sw_np")
                sw_atm  = st.checkbox("ATM strike", value=True, key="sw_atm")
                sw_k    = st.number_input("Strike (%)", 1.0, 10.0, 4.33, 0.05, key="sw_k",
                                          disabled=sw_atm)

            with st.spinner("Running G2++ MC swaption..."):
                sw_res = _g2sw(_g2_crv, _g2_p, expiry=float(sw_exp), swap_tenor=float(sw_ten),
                                strike=None if sw_atm else sw_k/100, notional=1_000_000.0,
                                pay_receive=sw_type, n_paths=sw_np, seed=42)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Swaption PV", f"${sw_res['pv']:,.0f}")
            c2.metric("MC Std Error", f"${sw_res['mc_stderr']:,.0f}")
            c3.metric("Fwd Swap Rate", f"{sw_res['forward_swap_rate_pct']:.3f}%")
            c4.metric("Annuity", f"{sw_res['annuity']:.4f}")

            # Vol surface: vary expiry and tenor
            import numpy as np
            import matplotlib.pyplot as _swplt
            from scipy.stats import norm as _snorm
            from scipy.optimize import brentq as _sbr

            expiries_sw = [0.5, 1.0, 2.0, 3.0, 5.0]
            tenors_sw   = [1.0, 2.0, 5.0, 10.0]

            def _iv(F, K, T, pv, ann):
                if pv <= 0 or F <= 0 or K <= 0 or ann <= 0:
                    return 0.0
                p_per = pv / ann
                def f(v):
                    d1 = (np.log(F/K) + 0.5*v**2*T) / (v*np.sqrt(T))
                    return F*_snorm.cdf(d1) - K*_snorm.cdf(d1 - v*np.sqrt(T)) - p_per
                try:
                    return _sbr(f, 1e-4, 5.0)
                except Exception:
                    return 0.0

            vol_grid = np.zeros((len(expiries_sw), len(tenors_sw)))
            for i, Te in enumerate(expiries_sw):
                for j, Ts in enumerate(tenors_sw):
                    r = _g2sw(_g2_crv, _g2_p, Te, Ts, None, 1.0, "payer", 2, 2000, 42)
                    F = r["forward_swap_rate_pct"]/100
                    vol_grid[i, j] = _iv(F, F, Te, r["pv"], r["annuity"]) * 100

            fig_vs, ax_vs = _swplt.subplots(figsize=(9, 4))
            for j, Ts in enumerate(tenors_sw):
                ax_vs.plot(expiries_sw, vol_grid[:, j], marker="o", linewidth=2,
                           label=f"{Ts:.0f}Y tenor")
            ax_vs.set_xlabel("Expiry (years)")
            ax_vs.set_ylabel("Implied Vol (%)")
            ax_vs.set_title(f"G2++ ATM Swaption Vol Term Structure (a={g2_a:.2f}, b={g2_b:.2f})")
            ax_vs.legend()
            ax_vs.spines["top"].set_visible(False)
            ax_vs.spines["right"].set_visible(False)
            st.pyplot(fig_vs, use_container_width=True)
            _swplt.close()

        else:  # VaR
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                v_dv01 = st.number_input("Portfolio DV01 ($)", -100_000, 100_000, 10_000, 1_000, key="v_dv01")
                v_hor  = st.number_input("Horizon (days)", 1, 20, 1, 1, key="v_hor")
            with col_v2:
                v_conf = st.selectbox("Confidence", [0.95, 0.99, 0.999], index=1, key="v_conf")
                v_np   = st.select_slider("n_paths", [5000, 10000, 20000], value=10000, key="v_np")

            with st.spinner("Running G2++ VaR simulation..."):
                v_res = _g2var(_g2_crv, _g2_p, portfolio_dv01=float(v_dv01),
                                horizon=v_hor/250.0, confidence=float(v_conf),
                                n_paths=v_np, seed=42)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VaR", f"${v_res['var_usd']:,.0f}")
            c2.metric("CVaR (ES)", f"${v_res['cvar_usd']:,.0f}")
            c3.metric("VaR (bps)", f"{v_res['var_bps']:.2f}")
            c4.metric("PnL Std Dev", f"${v_res['pnl_std']:,.0f}")
            st.caption(f"G2++ 2-factor VaR | {int(v_conf*100)}% confidence | {v_hor}d horizon | {v_np:,} paths")

    except Exception as e:
        st.error(f"G2++ error: {e}")


# TAB 20: CDS Pricing
# ─────────────────────────────────────────────────────────────────────────────
with tab20:
    st.header("Credit Default Swap (CDS) Pricing")
    st.markdown(
        "A **CDS** transfers default risk: the protection buyer pays a running spread "
        "(coupon) and receives par minus recovery on default. "
        "The standard ISDA model uses a **piecewise-constant hazard rate** curve bootstrapped "
        "from market par spreads.\n\n"
        "- **Fee leg**: periodic spread payments on surviving notional\n"
        "- **Protection leg**: (1-R) × notional contingent on default\n"
        "- **CS01**: sensitivity to 1bp shift in all hazard rates\n"
        "- **Hazard bootstrap**: strip survival probabilities from CDS term structure\n"
    )

    from sofr_engine.credit import (
        HazardRateCurve as _HC,
        CDSContract as _CDSC,
        bootstrap_hazard_curve as _bhaz,
        cds_pv as _cds_pv_dash,
        cds_par_spread as _cds_par_dash,
    )
    from sofr_engine.bootstrap import flat_sofr_curve as _cds_flat_crv
    import numpy as np
    import matplotlib.pyplot as _cdsplt

    cds_sub = st.radio("Section", ["Single CDS Pricer", "Hazard Rate Bootstrap"],
                       horizontal=True, key="cds_sub")

    if cds_sub == "Single CDS Pricer":
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            d_sofr   = st.number_input("SOFR (%)", 1.0, 8.0, 4.33, 0.01, key="d_sofr") / 100
            d_mat    = st.number_input("Maturity (y)", 0.5, 20.0, 5.0, 0.5, key="d_mat")
            d_coupon = st.number_input("Coupon (bps)", 0, 500, 100, 5, key="d_cpn") / 10_000
        with col_d2:
            d_haz    = st.number_input("Hazard rate (bps)", 1, 500, 200, 5, key="d_haz") / 10_000
            d_rec    = st.slider("Recovery (%)", 0, 80, 40, 5, key="d_rec") / 100
            d_not    = st.number_input("Notional ($M)", 0.1, 100.0, 10.0, 1.0, key="d_not") * 1e6
        with col_d3:
            d_side   = st.radio("Side", ["Buy Protection", "Sell Protection"], key="d_side")

        try:
            _d_crv  = _cds_flat_crv(__import__("datetime").date.today(), d_sofr)
            _d_hc   = _HC.flat(d_haz, [d_mat], recovery=d_rec)
            _d_cds  = _CDSC(maturity_years=d_mat, coupon=d_coupon, notional=d_not,
                             recovery=d_rec, buy_protection=(d_side == "Buy Protection"))
            _d_res  = _cds_pv_dash(_d_crv, _d_hc, _d_cds)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CDS PV", f"${_d_res.pv:,.0f}")
            c2.metric("Par Spread", f"{_d_res.par_spread_bps:.1f} bps")
            c3.metric("CS01", f"${_d_res.cs01:,.0f}")
            c4.metric("DV01", f"${_d_res.dv01:,.0f}")

            col_da, col_db = st.columns(2)
            with col_da:
                # Par spread vs maturity
                mats_v  = np.linspace(0.5, 10.0, 40)
                par_spr = [_cds_par_dash(_d_crv, _HC.flat(d_haz, [T], recovery=d_rec), T) * 10_000
                           for T in mats_v]
                fig_d1, ax_d1 = _cdsplt.subplots(figsize=(5, 3.5))
                ax_d1.plot(mats_v, par_spr, color="firebrick", linewidth=2)
                ax_d1.axvline(d_mat, color="k", linestyle="--", alpha=0.5, label=f"T={d_mat}y")
                ax_d1.set_xlabel("Maturity (years)")
                ax_d1.set_ylabel("Par Spread (bps)")
                ax_d1.set_title("CDS Term Structure")
                ax_d1.legend()
                ax_d1.spines["top"].set_visible(False)
                ax_d1.spines["right"].set_visible(False)
                st.pyplot(fig_d1, use_container_width=True)
                _cdsplt.close()

            with col_db:
                # Survival probability curve
                t_surv = np.linspace(0, d_mat * 1.5, 100)
                surv   = [_d_hc.survival(t) for t in t_surv]
                fig_d2, ax_d2 = _cdsplt.subplots(figsize=(5, 3.5))
                ax_d2.plot(t_surv, [s * 100 for s in surv], color="steelblue", linewidth=2)
                ax_d2.set_xlabel("Time (years)")
                ax_d2.set_ylabel("Survival Probability (%)")
                ax_d2.set_title(f"Survival Curve  λ={d_haz*10000:.0f}bps, R={d_rec*100:.0f}%")
                ax_d2.spines["top"].set_visible(False)
                ax_d2.spines["right"].set_visible(False)
                st.pyplot(fig_d2, use_container_width=True)
                _cdsplt.close()

        except Exception as e:
            st.error(f"CDS error: {e}")

    else:  # Bootstrap
        st.subheader("Hazard Rate Bootstrap from Par Spreads")
        st.markdown("Enter market CDS par spreads to bootstrap the survival probability curve.")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            b_sofr = st.number_input("SOFR (%)", 1.0, 8.0, 4.33, 0.01, key="b_sofr2") / 100
            b_rec  = st.slider("Recovery (%)", 0, 80, 40, 5, key="b_rec") / 100
        with col_b2:
            b_mats   = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
            b_defaults = [60, 90, 120, 160, 180, 200]
            b_spreads_bps = []
            for m, d in zip(b_mats, b_defaults):
                b_spreads_bps.append(
                    st.number_input(f"{m:.0f}Y par spread (bps)", 1, 1000, d, 5, key=f"bs_{m}"))

        try:
            _b_crv  = _cds_flat_crv(__import__("datetime").date.today(), b_sofr)
            _b_spr  = [s / 10_000 for s in b_spreads_bps]
            _b_hc   = _bhaz(_b_crv, b_mats, _b_spr, recovery=b_rec)

            import pandas as _bpd
            rows = []
            for T, s in zip(b_mats, _b_spr):
                rows.append({
                    "Maturity (y)":       T,
                    "Market Spread (bps)": round(s * 10_000, 1),
                    "Hazard Rate (bps)":   round(_b_hc.hazard_at(T) * 10_000, 2),
                    "Survival Prob (%)":   round(_b_hc.survival(T) * 100, 3),
                    "Default Prob (%)":    round((1 - _b_hc.survival(T)) * 100, 3),
                })
            st.dataframe(_bpd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Survival curve chart
            t_plot = np.linspace(0, 10.5, 200)
            surv_b = [_b_hc.survival(t) * 100 for t in t_plot]
            fig_b, ax_b = _cdsplt.subplots(figsize=(9, 4))
            ax_b.plot(t_plot, surv_b, color="steelblue", linewidth=2)
            ax_b.fill_between(t_plot, surv_b, 0, alpha=0.15, color="steelblue")
            ax_b.set_xlabel("Time (years)")
            ax_b.set_ylabel("Survival Probability (%)")
            ax_b.set_title("Bootstrapped Survival Probability Curve")
            ax_b.spines["top"].set_visible(False)
            ax_b.spines["right"].set_visible(False)
            st.pyplot(fig_b, use_container_width=True)
            _cdsplt.close()

        except Exception as e:
            st.error(f"CDS bootstrap error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 21: LMM / BGM (Libor Market Model)
# ─────────────────────────────────────────────────────────────────────────────
with tab21:
    st.header("📈 SOFR Libor Market Model (BGM)")
    st.caption(
        "Log-normal BGM on a discrete SOFR tenor grid. "
        "Black-76 cap pricing, Monte-Carlo swaptions under Q^{T_N}, "
        "Rebonato approximate vol surface, and caplet vol bootstrap."
    )

    from sofr_engine.lmm import (
        LMMParams as _LMMParams_db,
        initial_forwards as _lmm_fwds_db,
        simulate_lmm as _sim_lmm_db,
        caplet_black76 as _caplet_b76_db,
        cap_black76 as _cap_b76_db,
        cap_implied_vol as _cap_iv_db,
        swaption_lmm_mc as _sw_lmm_mc_db,
        rebonato_swaption_vol as _rebonato_db,
        calibrate_caplet_vols as _cal_cap_vols_db,
    )

    lmm_col1, lmm_col2 = st.columns([1, 2])

    with lmm_col1:
        st.subheader("Model Parameters")
        lmm_sofr = st.number_input("SOFR ON Rate", 0.010, 0.120, 0.0433, 0.001,
                                    format="%.4f", key="lmm_sofr")
        lmm_n    = st.slider("Number of Periods", 4, 20, 10, 1, key="lmm_n")
        lmm_mat  = st.slider("Tenor (years)", 1.0, 10.0, 5.0, 0.5, key="lmm_mat")
        lmm_vol  = st.slider("Flat vol (%)", 5.0, 60.0, 25.0, 1.0, key="lmm_vol") / 100.0
        lmm_lam  = st.slider("Corr decay λ", 0.0, 2.0, 0.10, 0.01, key="lmm_lam")
        lmm_K    = st.number_input("Cap/Swaption strike (%)", 0.5, 15.0, 4.0, 0.1,
                                    format="%.2f", key="lmm_K") / 100.0

    with lmm_col2:
        try:
            from sofr_engine.bootstrap import flat_sofr_curve as _fsc_lmm
            _lmm_curve  = _fsc_lmm(date.today(), lmm_sofr)
            _lmm_tenors = np.linspace(0.0, lmm_mat, lmm_n + 1)
            _lmm_vols   = np.full(lmm_n, lmm_vol)
            _lmm_params = _LMMParams_db(tenors=_lmm_tenors, vols=_lmm_vols, corr_decay=lmm_lam)
            _lmm_F0     = _lmm_fwds_db(_lmm_curve, _lmm_tenors)

            # ── Forward rate term structure ───────────────────────────────────
            st.subheader("Initial Forward Rate Term Structure")
            fig_fwd, ax_fwd = plt.subplots(figsize=(7, 3))
            ax_fwd.bar(
                _lmm_tenors[:-1],
                _lmm_F0 * 100,
                width=np.diff(_lmm_tenors) * 0.8,
                align="edge",
                color="#1f77b4",
                alpha=0.75,
                label="F_k(0)",
            )
            ax_fwd.set_xlabel("Period Start (years)")
            ax_fwd.set_ylabel("Forward Rate (%)")
            ax_fwd.set_title("LMM Initial Forward Rates")
            ax_fwd.legend()
            ax_fwd.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
            st.pyplot(fig_fwd, use_container_width=True)
            plt.close(fig_fwd)

            # ── Black-76 cap strip ────────────────────────────────────────────
            st.subheader("Black-76 Cap Strip")
            cap_pvs  = [
                _caplet_b76_db(_lmm_curve, _lmm_params, k, lmm_K, 1_000_000, True)
                for k in range(lmm_n)
            ]
            cum_cap  = np.cumsum(cap_pvs)
            fig_cap, ax_cap = plt.subplots(figsize=(7, 3))
            ax_cap.bar(
                range(1, lmm_n + 1),
                cap_pvs,
                color="#ff7f0e",
                alpha=0.75,
                label="Caplet PV",
            )
            ax2_cap = ax_cap.twinx()
            ax2_cap.plot(range(1, lmm_n + 1), cum_cap, "k--o", label="Cumulative Cap PV", ms=4)
            ax_cap.set_xlabel("Period k")
            ax_cap.set_ylabel("Caplet PV ($)")
            ax2_cap.set_ylabel("Cumulative Cap PV ($)")
            ax_cap.set_title(f"Cap Strip  |  K={lmm_K*100:.2f}%  |  Flat vol={lmm_vol*100:.0f}%")
            lines1, lab1 = ax_cap.get_legend_handles_labels()
            lines2, lab2 = ax2_cap.get_legend_handles_labels()
            ax_cap.legend(lines1 + lines2, lab1 + lab2, loc="upper left", fontsize=8)
            st.pyplot(fig_cap, use_container_width=True)
            plt.close(fig_cap)

        except Exception as e:
            st.error(f"LMM curve error: {e}")

    # ── Rebonato swaption vol surface ─────────────────────────────────────────
    st.subheader("Rebonato Approximate Swaption Vol Surface")
    try:
        vol_grid = np.zeros((lmm_n, lmm_n))
        for k_s in range(lmm_n - 1):
            for k_e in range(k_s + 2, lmm_n + 1):
                v = _rebonato_db(_lmm_curve, _lmm_params, k_s, k_e)
                vol_grid[k_s, k_e - 1] = v * 100.0

        expiry_labels = [f"{_lmm_tenors[k]:.1f}Y" for k in range(lmm_n)]
        tenor_labels  = [f"{_lmm_tenors[k]:.1f}Y" for k in range(1, lmm_n + 1)]

        fig_vol, ax_vol = plt.subplots(figsize=(9, 4))
        im = ax_vol.imshow(
            vol_grid, aspect="auto", cmap="RdYlGn_r",
            vmin=max(lmm_vol * 100 * 0.5, 1),
            vmax=lmm_vol * 100 * 1.2,
        )
        ax_vol.set_xticks(range(lmm_n))
        ax_vol.set_xticklabels(tenor_labels, rotation=45, fontsize=7)
        ax_vol.set_yticks(range(lmm_n))
        ax_vol.set_yticklabels(expiry_labels, fontsize=7)
        ax_vol.set_xlabel("Swap End Tenor")
        ax_vol.set_ylabel("Swaption Expiry")
        ax_vol.set_title("Rebonato Approx. Swaption Implied Vol (%)")
        plt.colorbar(im, ax=ax_vol, label="Vol (%)")
        st.pyplot(fig_vol, use_container_width=True)
        plt.close(fig_vol)

    except Exception as e:
        st.warning(f"Rebonato surface error: {e}")

    # ── MC Swaption Pricer ────────────────────────────────────────────────────
    st.subheader("Monte-Carlo Swaption Pricer")
    sw_col1, sw_col2 = st.columns(2)
    with sw_col1:
        sw_kstart = st.slider("Expiry period k_start", 0, max(lmm_n - 2, 1), min(2, lmm_n - 2), key="sw_ks")
        sw_kend   = st.slider("Swap end period k_end", sw_kstart + 1, lmm_n, min(sw_kstart + 4, lmm_n), key="sw_ke")
        sw_npaths = st.select_slider("MC Paths", [500, 1000, 2000, 4000], 2000, key="sw_np")
        sw_payer  = st.checkbox("Payer swaption", value=True, key="sw_pay")

    with sw_col2:
        if st.button("Price Swaption (MC)", key="sw_btn"):
            try:
                _sim = _sim_lmm_db(
                    _lmm_curve, _lmm_params,
                    n_steps=40, n_paths=sw_npaths, seed=42,
                )
                # ATM strike from simulation mean
                _S0  = float(_sim.swap_rate(_lmm_tenors[sw_kstart], sw_kstart, sw_kend).mean())
                _K   = _S0 if lmm_K <= 0 else lmm_K
                _res = _sw_lmm_mc_db(_sim, _lmm_curve, sw_kstart, sw_kend, _K,
                                      1_000_000, sw_payer)
                _reb = _rebonato_db(_lmm_curve, _lmm_params, sw_kstart, sw_kend)
                st.metric("Swaption PV ($)", f"{_res['pv']:,.2f}")
                st.metric("±1σ Std Error", f"{_res['std_err']:,.2f}")
                st.metric("Swap rate (mean %)", f"{_res['swap_rate_mean']*100:.4f}")
                st.metric("Rebonato σ (%)", f"{_reb*100:.3f}")
                st.metric("Strike (%)", f"{_K*100:.4f}")
            except Exception as ex:
                st.error(str(ex))

    # ── Caplet vol bootstrap ──────────────────────────────────────────────────
    st.subheader("Caplet Vol Bootstrap")
    st.caption(
        "Enter a flat cap implied vol term structure. "
        "The bootstrapper recovers per-period caplet vols."
    )
    default_cap_vols = ", ".join(["25.0"] * lmm_n)
    cap_vol_str = st.text_area(
        "Flat cap implied vols (%, comma-separated — one per period)",
        value=default_cap_vols, key="cap_vol_ts",
    )
    if st.button("Bootstrap Caplet Vols", key="boot_btn"):
        try:
            flat_vols = [float(x.strip()) / 100.0 for x in cap_vol_str.split(",")]
            if len(flat_vols) != lmm_n:
                st.error(f"Need exactly {lmm_n} values, got {len(flat_vols)}.")
            else:
                p_cal = _cal_cap_vols_db(_lmm_curve, _lmm_params, flat_vols)
                df_boot = pd.DataFrame({
                    "Period k"    : list(range(lmm_n)),
                    "T_fix (yrs)" : [round(float(_lmm_tenors[k]), 3) for k in range(lmm_n)],
                    "Input cap vol (%)": [round(v * 100, 3) for v in flat_vols],
                    "Caplet vol (%)"   : [round(v * 100, 4) for v in p_cal.vols.tolist()],
                    "Fwd rate (%)"     : [round(float(f) * 100, 4) for f in _lmm_F0.tolist()],
                })
                st.dataframe(df_boot, use_container_width=True)

                fig_b, ax_b = plt.subplots(figsize=(7, 3))
                ax_b.plot(range(lmm_n), [v * 100 for v in flat_vols], "o--",
                          color="grey", label="Input flat cap vol", alpha=0.7)
                ax_b.plot(range(lmm_n), p_cal.vols * 100, "s-",
                          color="#d62728", label="Bootstrapped caplet vol")
                ax_b.set_xlabel("Period k")
                ax_b.set_ylabel("Vol (%)")
                ax_b.set_title("Cap → Caplet Vol Bootstrap")
                ax_b.legend()
                st.pyplot(fig_b, use_container_width=True)
                plt.close(fig_b)
        except Exception as ex:
            st.error(f"Bootstrap error: {ex}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Project Sentinel — Bhavesh Anchalia | VIT University | June 2026 | "
    "SOFR Pricing Engine & Macro Signal Framework | 885 tests ✅ | Sharpe 0.282 OOS"
)
