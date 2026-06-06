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
    st.markdown("**Tests:** 114 passing ✅")
    st.markdown("**Sharpe (OOS):** 0.282")
    st.markdown("**Hit rate:** 65.7% (12σ above random)")


# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 SOFR Curve",
    "⚡ Convexity",
    "🏛️ Taylor Rule",
    "📐 Nelson-Siegel",
    "🎯 Signals",
    "🎲 FOMC Probs",
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


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Project Sentinel — Bhavesh Anchalia | VIT University | June 2026 | "
    "SOFR Pricing Engine & Macro Signal Framework | 114 tests ✅ | Sharpe 0.282 OOS"
)
