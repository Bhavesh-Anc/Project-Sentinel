# BRAIN.md — US Rates & SOFR Pricing Engine
**Living research document. Updated as we build. This is both our memory and the raw material for the paper.**

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Paper Draft](#2-paper-draft)
3. [Model Decisions & Rationale](#3-model-decisions--rationale)
4. [Data Notes](#4-data-notes)
5. [Key Findings Log](#5-key-findings-log)
6. [Quantitative Framework Reference](#6-quantitative-framework-reference)
7. [TODO / Next Steps](#7-todo--next-steps)
8. [Pitching Notes](#8-pitching-notes)

---

## 1. Project Overview

### What We Are Building
A dual-deliverable project targeting quant trading firms and investment banks:

**Deliverable A — Python Codebase (Quant Audience)**
A production-grade SOFR pricing engine with:
- SOFR forward curve bootstrapper (CME SR1/SR3 futures + OIS swaps)
- Instrument pricers: SOFR swaps, FRAs, term SOFR
- Convexity-adjusted futures pricing
- Fed policy forecasting model (Taylor Rule + FOMC probability)
- US Treasury Nelson-Siegel curve fitting
- Walk-forward validated trading signals on the US rates curve

**Deliverable B — Research Paper (IB/Macro Audience)**
"US Rate Cycle 2025–26: SOFR Curve Dynamics and Fed Policy Forecasting"
- Institutional-quality research note (10-12 pages)
- Target publication: SSRN + arXiv (q-fin.PR)
- Covers SOFR transition, curve construction, macro model, market implications

### Target Audience
| Audience | Hook | Key Metrics |
|----------|------|-------------|
| Quant Trading (Graviton, Quadeye, Goldman Strats, JP Morgan QR) | Live SOFR pricer, walk-forward Sharpe, curve analytics | Sharpe ratio, hit rate, drawdown |
| Investment Banking (JP Morgan, Deutsche, Goldman IBD/Research) | Fed forecasting, SOFR curve analysis, rates market implications | Macro accuracy, research quality |

### Data Sources
- **FRED API** (Free): SOFR overnight, Fed Funds, CPI, PCE, GDP, unemployment, Treasury yields
- **CME Group** (Free public settlements): SR3 3-month SOFR futures, SR1 1-month SOFR futures
- **FRBNY** (Free): Term SOFR rates (1M, 3M, 6M, 12M), SOFR compounded averages

---

## 2. Paper Draft

### Title
**"US Rate Cycle 2025–26: SOFR Curve Dynamics, Fed Policy Forecasting, and Tradeable Signals in the Post-LIBOR Regime"**

*Bhavesh Anchalia*
*[Institution/Affiliation]*
*Submitted to SSRN: [Date]*

---

### Abstract
*(Draft — to be refined with actual results)*

We present a quantitative framework for analyzing US interest rate dynamics in the post-LIBOR era, centered on the Secured Overnight Financing Rate (SOFR) as the new benchmark. We construct a SOFR forward curve via bootstrapping from CME 3-month SOFR futures (SR3) and OIS swap quotes, applying Hull-White convexity adjustments to reconcile futures-implied and forward rates. Separately, we estimate a Taylor Rule model for Fed Funds Rate forecasting and a logistic regression model for FOMC meeting probabilities. Combining macro regime signals with curve-level/slope dynamics, we construct a walk-forward validated trading strategy on SOFR swap spreads (2s10s), achieving a Sharpe ratio of [X] over the 2022–2025 out-of-sample period. Our results have direct applications to rate swap pricing, duration management, and relative value identification in US fixed income markets.

**Keywords:** SOFR, curve bootstrapping, Fed policy, Taylor Rule, Nelson-Siegel, OIS discounting, LIBOR transition, rate trading signals

---

### Section 1: Introduction

*(Key points to develop)*
- LIBOR transition completed June 30, 2023; SOFR now the global USD benchmark
- SOFR fundamentally different from LIBOR: overnight, nearly risk-free (secured by Treasuries), backward-looking
- This creates unique challenges: compounding in arrears, no natural term rate (until CME Term SOFR introduced)
- SOFR futures market now highly liquid: ~$1T+ daily volume on CME SR3
- Research question: How do we build a consistent SOFR forward curve, and what macro signals drive it?

**Key data point to open with:** Fed Funds target rate went from 0.25% (March 2022) to 5.50% (July 2023) — fastest tightening cycle in 40 years. SOFR moved in lockstep. This creates rich variation for our models.

---

### Section 2: Institutional Background — The SOFR Ecosystem

*(Key points to develop)*

**2.1 SOFR vs. LIBOR — What Changed**
- LIBOR: unsecured, survey-based, term structure (1W, 1M, 3M, 6M, 12M)
- SOFR: secured (repo), transaction-based, overnight
- ARRC recommended SOFR in 2017; final LIBOR cessation June 2023
- Legacy contracts converted via ISDA fallback protocol (SOFR + CAS spread adjustment)

**2.2 The CME SOFR Futures Complex**
- SR1: 1-month SOFR futures — settles to arithmetic average SOFR in delivery month
- SR3: 3-month SOFR futures — settles to compounded SOFR over IMM quarter
- Active contracts extend ~3 years forward; used to build the short-to-medium curve
- IMM dates: 3rd Wednesday of March, June, September, December

**2.3 Term SOFR (CME)**
- Published by CME Group under ARRC endorsement since 2021
- Forward-looking: 1M, 3M, 6M, 12M term rates derived from SOFR futures
- Used in lending markets where borrowers need to know the rate in advance
- Note: Term SOFR ≠ compounded SOFR in arrears; there is a basis

**2.4 SOFR OIS Swaps**
- Floating leg pays compounded SOFR daily (in arrears, with lag)
- Fixed leg pays fixed coupon annually (30/360) or semi-annually
- Used for tenors 1Y and beyond; most liquid: 2Y, 5Y, 10Y, 30Y
- OIS discount curve is the standard for CSA-collateralized derivatives

---

### Section 3: SOFR Curve Construction

*(Key points to develop)*

**3.1 Bootstrapping Algorithm Overview**
Standard approach: hybrid curve using:
1. Overnight SOFR for T=0 to T=1 business day
2. SR1/SR3 futures for T=1W to T=2Y (with convexity adjustment)
3. OIS swap quotes for T=2Y to T=30Y

**3.2 Convexity Adjustment**
The fundamental issue: futures prices are linear instruments with daily margin; forward rate agreements are not marked-to-market. Under Hull-White (1-factor):

$$\text{Convexity Adj} = \frac{1}{2}\sigma^2 T_1 T_2$$

where $T_1$ = futures expiry, $T_2$ = end of accrual period, $\sigma$ = short-rate volatility.

For SR3 contracts near expiry, this is negligible (<1bp). For far-dated contracts (2Y+), it becomes material (~5–8bps).

**3.3 Discount Factor Construction**
From the adjusted forward rates, build continuous discount factors:

$$DF(T) = \exp\left(-\int_0^T f(t)\,dt\right)$$

Implemented as piecewise-constant forward rates with log-linear interpolation on discount factors between pillar dates.

**3.4 Curve Outputs**
- Discount factors DF(T) for any T
- Zero rates z(T) = -ln(DF(T)) / T
- Forward rates f(T1, T2) = -ln(DF(T2)/DF(T1)) / (T2-T1)
- Par swap rates K(T) for OIS swaps to any tenor

*[Insert chart: SOFR forward curve as of a representative date, showing overnight, futures-strip, and swap-interpolated segments]*

---

### Section 4: Fed Policy Forecasting

*(Key points to develop)*

**4.1 Taylor Rule Estimation**

The classic Taylor Rule:
$$r^* = r_n + \alpha(\pi - \pi^*) + \beta(y - y^*)$$

Where:
- $r^*$ = recommended Fed Funds Rate
- $r_n$ = neutral rate (approximately 2.5% nominal; Fed's "longer run" dot)
- $\pi$ = PCE inflation (12-month trailing, Fed's preferred measure)
- $\pi^*$ = 2% target
- $y - y^*$ = output gap (real GDP vs. CBO potential GDP, or Okun's law via unemployment gap)
- $\alpha = \beta = 0.5$ (standard Rudebusch calibration)

**4.2 Variants to Test**
- Balanced approach rule (Bernanke et al. 2019)
- Inertial Taylor Rule (with lagged rate)
- Risk-management rule (asymmetric weights post-ZLB)

**4.3 FOMC Meeting Probability Model**
Extract market-implied probability of each outcome from SOFR futures:
- Next-meeting contract implies expected rate at next FOMC
- P(hike) = (implied_rate - current_rate) / hike_size (simplified)
- More sophisticated: extract full probability distribution over possible outcomes

**4.4 Model vs. Market**
Key insight: When Taylor Rule suggests a rate significantly different from where FOMC is priced, this is a potential signal.

*[Insert chart: Taylor Rule implied rate vs. actual Fed Funds Rate, 2015–2025]*

---

### Section 5: US Treasury Curve — Nelson-Siegel Analysis

*(Key points to develop)*

**5.1 Nelson-Siegel Parametrization**
$$y(\tau) = \beta_0 + \beta_1 \frac{1-e^{-\lambda\tau}}{\lambda\tau} + \beta_2\left(\frac{1-e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\right)$$

Interpretation:
- $\beta_0$: Level — long-run yield (parallel shift)
- $\beta_1$: Slope — 2s10s spread driver (negative = normal curve)
- $\beta_2$: Curvature — belly richness/cheapness
- $\lambda$: Decay factor (controls where curvature peaks; typically ~0.6 for US Treasuries)

**5.2 Svensson Extension**
Adds a second hump term — better fit for current kinked curves (post-hike flattening):
$$y(\tau) = \beta_0 + \beta_1 L_1 + \beta_2 C_1 + \beta_3 C_2$$

**5.3 Regime Analysis**
Classify historical curve regimes:
- Normal (steep positive slope): β₁ < 0, β₀ > r_sofr
- Flat: β₁ ≈ 0
- Inverted (2022-2024): β₁ > 0 (short end > long end)
- Humped: β₂ dominant

*[Insert chart: NS factors over time, annotated with Fed cycle phases]*

---

### Section 6: Trading Signals and Backtesting

*(Key points to develop)*

**6.1 Signal Construction**

Signal 1 — Macro Regime: 
- Long 10Y Treasuries when: Taylor gap > 75bps AND core PCE falling AND unemployment rising

Signal 2 — Curve Slope:
- Long 2s10s steepener when: NS slope factor (β₁) below 5th historical percentile AND Fed is near end of cycle

Signal 3 — SOFR Basis:
- Term SOFR vs. compounded SOFR basis trades on FOMC meeting timing

**6.2 Walk-Forward Validation**
- Train: 2015–2021 (one full rate cycle: zero bound, normalization, COVID cuts)
- Test: 2022–2025 (out-of-sample: fastest hike cycle, peak rates, potential cuts)
- Re-estimate model parameters quarterly in walk-forward manner
- No look-ahead bias: use only data available at each signal date

**6.3 Performance Metrics**
- Annualized Sharpe Ratio (target: >0.7 out-of-sample)
- Maximum Drawdown
- Calmar Ratio
- Hit Rate by regime
- Transaction cost sensitivity

*[Insert chart: Cumulative P&L walk-forward, 2022–2025]*

---

### Section 7: Market Implications

*(For IB audience — corporate issuance / rates strategy angle)*

**7.1 For Corporate Issuers**
- Current SOFR curve shape and forward rate expectations determine optimal fixed vs. floating issuance
- Using our par swap rate calculator: implied hedging cost for new issuance at each tenor
- Scenario analysis: if Fed cuts 150bps by end-2026, floating rate issuers save [X]bps vs. fixed

**7.2 For Rate Strategists**
- Relative value: identify richness/cheapness in the futures curve vs. OIS swaps (the SR3/OIS basis)
- Fed model deviation: when Taylor gap is large, historical mean-reversion provides signal

---

### Section 8: Conclusion

*(To be written after results are in)*

---

## 3. Model Decisions & Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| Curve interpolation | Log-linear on discount factors | Ensures positive forward rates; standard market practice |
| Convexity adj. method | Hull-White 1-factor analytical | Closed-form, calibratable to cap vol; sufficient for SR3 |
| Day count (SOFR) | ACT/360 | ISDA standard for SOFR; same as Fed Funds |
| Day count (Treasuries) | ACT/ACT | US Treasury market convention |
| Taylor Rule calibration | Standard (α=β=0.5) + estimated | Start with Rudebusch; then OLS estimate on 2015-2019 pre-ZLB data |
| Output gap proxy | Unemployment gap (Okun's law) | Real-time GDP data revised; unemployment is clean and timely |
| Nelson-Siegel λ | Estimated via grid search | Fix τ* (maturity of max curvature) at ~2.5Y; λ = -ln(0)/2.5 |
| Walk-forward window | Annual re-estimation | Quarterly introduces noise; annual preserves enough data |

---

## 4. Data Notes

### FRED API Series
| Name | FRED ID | Frequency | Notes |
|------|---------|-----------|-------|
| SOFR overnight | SOFR | Daily | From April 2018 (SOFR inception) |
| Effective Fed Funds | FEDFUNDS | Daily | Daily EFFR |
| IORB | IORB | Daily | Interest on Reserves; effective floor |
| 30-day SOFR avg | SOFR30DAYAVG | Daily | FRBNY compounded avg |
| 90-day SOFR avg | SOFR90DAYAVG | Daily | Key for SR3 comparison |
| Core PCE | PCEPILFE | Monthly | Fed's preferred inflation measure |
| Unemployment | UNRATE | Monthly | U-3; use as output gap proxy |
| Real GDP | GDPC1 | Quarterly | Revised; use HP filter |
| 10Y Treasury | DGS10 | Daily | Constant maturity |
| 2Y Treasury | DGS2 | Daily | Constant maturity |
| 5Y Treasury | DGS5 | Daily | Constant maturity |

### CME SOFR Futures
| Contract | Code | Settlement | Notes |
|----------|------|------------|-------|
| 3-month SOFR | SR3 | Compounded SOFR over IMM quarter | ~$1T+ daily volume |
| 1-month SOFR | SR1 | Arithmetic avg SOFR over calendar month | Shorter end of curve |

- CME publishes daily settlements at: [CME Group Settlements]
- IMM dates: 3rd Wednesday of March (H), June (M), September (U), December (Z)
- SR3 contract months: H, M, U, Z + serial months
- Settlement price = 100 - annualized rate

### Data Quality Issues to Watch
- SOFR only starts April 2018 — pre-2018 analysis uses Fed Funds as proxy
- Monthly data (CPI, GDP) needs forward-fill for daily signal construction
- CME settlements sometimes have small revisions next day; use T+1 for finality
- Treasury CMT rates are interpolated by Fed; use for curve fitting, not derivatives pricing

---

## 5. Key Findings Log

### 2026-06-06 — Project Kickoff + First Live Data Run

**Setup:**
- Pivoted from India rates (RBI/G-Sec) to US rates (Fed/SOFR)
- FRED API connected. Pulled 2,721 rows of macro/SOFR data (2018–present) and 2,982 rows of Treasury CMT yields (2015–present).
- All models running end-to-end on real data.

**Live Market Snapshot (as of June 2026):**
- IORB: **3.65%** (Fed has cut ~185bps from the 5.50% peak)
- Core PCE YoY: **2.11%** (essentially at the 2% target)
- Unemployment: **4.3%** (above NAIRU of 4.0% → 30bps of slack)
- 2Y Treasury: **4.05%** | 10Y: **4.47%** | 30Y: **4.97%**
- 2s10s slope: **+42bps** (back to normal after historic inversion)

**Nelson-Siegel Results:**
- Latest NS fit: β₀=5.37%, β₁=-1.64%, β₂=-0.47%, λ=0.20, RMSE=0.058%
- Current regime: **normal_steep** (β₁ < -1%)
- Regime history (597 weeks): normal_steep 62%, normal_flat 20%, inverted 18%
- **Inversion lasted 105 weeks** (Nov 2022 → Dec 2024) — deepest at β₁=+1.74% on 2023-05-14
- The 2s10s hit -108bps on 2023-07-03 — deepest inversion in 40 years

**Taylor Rule Results:**
- Standard rule (α=β=0.5): Taylor implied rate = **2.25%** vs. actual **3.63%**
- **Policy gap = +1.38%** (Fed is 138bps ABOVE Taylor Rule)
- Interpretation: With inflation at target and small unemployment gap, standard rule says rate should be ~2.25%. Fed is running significantly tighter than the rule recommends.
- Historical extremes: Most overtightened +8.90% (April 2020, COVID shock distortion); most accommodative -3.83% (Dec 2021, pre-hike era)
- OLS estimated coefficients (2018-2019 pre-ZLB): α̂=0.737, β̂=-0.429, R²=0.30
  - Note: β is negative in estimation — likely reflects Fed's *forward guidance* rather than contemporaneous reaction; limited sample size (only 2 years)
  - Decision: Use standard (0.5, 0.5) for the paper; report estimated as robustness check

**Walk-Forward Backtest:**
- Full history (2018–2026): Sharpe **-0.29**, Hit rate 61.8%
- OOS period (2022–2026): Sharpe **+0.11**, Hit rate 65.5%
- The win/loss ratio (0.52) is the problem — the strategy is right directionally 62% of the time but losses are ~2× wins. Classic trend-following problem in rates.
- Root cause: Signal is slow (annual re-estimation) and yield moves are violent during hike cycles — the return approximation `-duration × Δy` gets hammered during 500bp rate moves
- **Action items for signal improvement (see TODO)**:
  1. Add signal decay / position sizing (reduce size when vol is elevated)
  2. Consider a stop-loss rule (exit long duration if yield rises >50bps)
  3. The strategy is better at identifying *regime* than *timing* — focus paper framing on regime detection not tactical trading

**Key Paper Findings (to develop in Sections 3–5):**
1. The 2022–2023 inversion period is now fully resolved. The NS slope factor (β₁) provides a clean continuous measure of the inversion depth and duration.
2. The Taylor Rule currently signals 138bps of excess tightening. If the Fed follows the rule mechanically, it implies additional cuts to ~2.5% by end-2026.
3. The SOFR curve (from futures) is pricing only partial convergence to Taylor Rule levels — presenting a potential relative value opportunity.

*(Add new findings below as we discover them)*

---

## 6. Quantitative Framework Reference

### SOFR Compounding
$$\text{Compounded Rate} = \left(\prod_{i=1}^{n}\left(1 + r_i \frac{d_i}{360}\right) - 1\right) \times \frac{360}{\sum d_i}$$

where $r_i$ is the SOFR fixing for business day $i$ and $d_i$ is its day count weight (1 for Mon-Thu, 3 for Fri).

### Discount Factor to Zero Rate
$$z(T) = -\frac{\ln DF(T)}{T}$$

### Par Swap Rate (SOFR OIS)
$$K = \frac{DF(T_{start}) - DF(T_{end})}{\sum_{i} \alpha_i \cdot DF(T_i)}$$

where $\alpha_i$ is the year fraction for the $i$-th fixed leg payment period.

### DV01 of a SOFR Swap
$$DV01 = \frac{\partial PV}{\partial r} \approx \frac{PV(r+1\text{bp}) - PV(r-1\text{bp})}{2}$$

### Taylor Rule
$$r^*_t = r_n + 0.5(\pi_t - 2\%) + 0.5(u^*_t - u_t) \times \frac{-2}{1}$$
*(Okun's law: 1% unemployment gap ≈ -2% output gap)*

### Nelson-Siegel
$$y(\tau) = \beta_0 + \beta_1 \frac{1-e^{-\lambda\tau}}{\lambda\tau} + \beta_2\left(\frac{1-e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\right)$$

### Hull-White Convexity Adjustment
$$f^{fwd}(T_1, T_2) = f^{fut}(T_1, T_2) - \frac{1}{2}\sigma^2 T_1 T_2$$

---

## 7. TODO / Next Steps

### Completed ✓
- [x] FRED API connected, all data fetched (2015–2026)
- [x] Full project scaffold: sofr_engine, models, backtesting, visualisation
- [x] SOFR day count, curve, bootstrap, convexity, instruments
- [x] Nelson-Siegel rolling factors — 597 weeks of history extracted
- [x] Taylor Rule model — live results computed
- [x] Walk-forward backtest framework running end-to-end
- [x] BRAIN.md with live findings and paper draft

### Next (Signal Improvement)
- [ ] Add vol-scaling to position sizing (halve size when VIX > 25)
- [ ] Add stop-loss rule: exit long duration if 10Y yield rises >50bps from entry
- [ ] Test signal on individual components — which contributes most Sharpe?
- [ ] Re-run backtest with improved signal and document delta vs. baseline

### SOFR Curve (Real Data)
- [ ] Fetch actual CME SR3 settlement data for a historical date range
- [ ] Bootstrap real curve from SR3 strip + OIS quotes (use SOFR swap quotes from Bloomberg/ICAP proxied via FRED term SOFR)
- [ ] Validate: SOFR90DAYAVG vs. bootstrapped 90-day forward rate
- [ ] Build curve snapshots for 3 key dates: pre-hike (Jan 2022), peak (Jul 2023), post-cut (Jun 2026)
- [ ] Convexity adjustment table — show bps impact at each contract expiry

### Paper Writing
- [ ] Section 2 draft: SOFR ecosystem (SR3, OIS, term SOFR)
- [ ] Section 3 draft: Curve construction methodology + convexity adjustment
- [ ] Section 4 draft: Taylor Rule analysis — "Is the Fed still overtightened?"
- [ ] Section 5 draft: NS factor regimes — inversion, recovery, current state
- [ ] Section 6 draft: Backtest methodology and caveats (honest about limitations)
- [ ] Charts: generate all 10 charts from visualisation/charts.py with real data
- [ ] Abstract: write once all numbers are finalized

### Polish & Publish
- [ ] Jupyter notebook: clean, annotated, end-to-end run with real data
- [ ] GitHub README: add key result numbers and embed 2-3 charts
- [ ] SSRN upload
- [ ] LinkedIn post: key finding + chart + links

---

## 8. Pitching Notes

### For Quant Trading Desks
- Lead with: "I built a SOFR forward curve bootstrapper from CME SR3 futures with convexity adjustment, then layered a macro signal on top to identify relative value in the 2s10s curve"
- Emphasize: Convexity adjustment is the subtle part most students miss; shows real rates market understanding
- Key number: Walk-forward out-of-sample Sharpe ratio (report honestly, even if 0.5-0.7)
- Code quality: Type-annotated Python, no look-ahead, reproducible results

### For Investment Banking (Rates / DCM / Research)
- Lead with: "I published an institutional research note on US rate cycle dynamics — same format as your team publishes for clients"
- Emphasize: Fed policy forecasting model and market implications section
- Show: The charts — Bloomberg-style, clean, labeled
- SSRN link: Demonstrates ability to communicate quantitative ideas in writing

### Elevator Pitch (30 seconds)
"I built a US rates research platform — a SOFR pricing engine that bootstraps the forward curve from CME futures, and a macro model forecasting Fed policy via Taylor Rule and FOMC probabilities. I combined them into a walk-forward validated trading strategy on the rates curve, and wrote it up as an institutional research note. The code is open-source on GitHub and the paper is on SSRN."
