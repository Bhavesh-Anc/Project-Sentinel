# US Rates & SOFR Pricing Engine

A from-scratch quantitative framework for US interest rate analysis in the post-LIBOR era.

**Bhavesh Anchalia**

---

## What This Is

Two deliverables built from a single codebase:

1. **SOFR Pricing Engine** — bootstraps the SOFR forward curve from CME futures and OIS swaps, prices SOFR swaps/FRAs/caps, and outputs DV01/par rates
2. **Research Paper** — "US Rate Cycle 2025–26: SOFR Curve Dynamics and Fed Policy Forecasting" (forthcoming on SSRN)

The `BRAIN.md` file is a living document — research notes, paper draft sections, and model decisions updated as we build.

---

## Project Structure

```
us-rates-sofr/
├── BRAIN.md                     ← Living research doc / paper draft
├── README.md
├── requirements.txt
│
├── data/
│   ├── fetch_sofr_data.py       ← FRED: SOFR, Fed Funds, CPI, PCE, GDP
│   ├── fetch_cme_futures.py     ← CME: SR3 (3M) and SR1 (1M) SOFR futures
│   └── fetch_treasury_data.py  ← FRED: Treasury CMT yields (1M–30Y)
│
├── sofr_engine/                 ← Core pricing engine (from scratch, no QuantLib)
│   ├── day_count.py             ← ACT/360, ACT/ACT, 30/360, SOFR compounding
│   ├── curve.py                 ← DiscountCurve: log-linear DF interpolation
│   ├── bootstrap.py             ← SOFRCurveBootstrapper: futures + OIS swaps
│   ├── convexity.py             ← Hull-White convexity adjustment (futures → fwd)
│   └── instruments.py          ← SOFRSwap, FRA, SOFRFutures, Caplet pricers
│
├── models/
│   ├── taylor_rule.py           ← Fed Taylor Rule (standard + inertial + estimated)
│   ├── nelson_siegel.py         ← NS + Svensson fitting, rolling factor extraction
│   ├── fomc_probability.py      ← FedWatch-style FOMC meeting probabilities
│   └── macro_signals.py        ← Inflation, labor, policy gap, curve slope signals
│
├── backtesting/
│   ├── signal_backtest.py       ← Walk-forward engine (no look-ahead bias)
│   └── performance.py          ← Sharpe, drawdown, Calmar, hit rate, tearsheet
│
├── visualisation/
│   └── charts.py               ← 10 publication-quality charts for paper
│
└── notebooks/
    └── main_analysis.ipynb     ← End-to-end analysis walkthrough
```

---

## Key Features

### SOFR Pricing Engine
- Bootstraps SOFR forward curve from CME SR3/SR1 futures + OIS swap quotes
- **Hull-White convexity adjustment** converts futures rates to OIS-consistent forward rates
- Log-linear discount factor interpolation (guarantees positive forward rates)
- Prices: SOFR OIS swaps, FRAs, SR3 futures, SOFR caplets (Bachelier model)
- Computes: DV01, par swap rates, zero rates, forward rates for any tenor

### Fed Policy Model
- Taylor Rule (standard α=β=0.5, balanced approach, inertial variant)
- OLS estimation of coefficients on pre-ZLB data
- FOMC meeting probability extraction from SOFR futures (replicates CME FedWatch)
- Statistical FOMC prediction model (logistic regression on macro features)

### US Treasury Curve
- Nelson-Siegel fitting (grid search on λ, linear OLS on betas)
- Svensson extension for kinked curves
- Rolling factor extraction: β₀ (level), β₁ (slope), β₂ (curvature)
- Curve regime classification (normal / flat / inverted)

### Trading Strategy
- 4 signal components: inflation momentum, labor market, policy gap, curve slope
- Composite score with configurable weights
- Walk-forward validation: train 2015–2021, test 2022–2025
- Full tearsheet: Sharpe, drawdown, Calmar, hit rate, regime-conditional performance

---

## Setup

```bash
pip install -r requirements.txt
```

Set environment variable:
```bash
export FRED_API_KEY=your_fred_api_key_here
```

Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html

---

## Quick Start

```python
from datetime import date
import pandas as pd
from sofr_engine import SOFRCurveBootstrapper, SOFRSwap

# Build a SOFR curve from overnight rate + futures + OIS quotes
ref = date(2024, 6, 5)
sofr_on = 0.0530  # 5.30% overnight

futures = pd.DataFrame([
    {"expiry": date(2024, 9, 18), "accrual_end": date(2024, 12, 18), "implied_rate": 0.0520},
    {"expiry": date(2024, 12, 18), "accrual_end": date(2025, 3, 19), "implied_rate": 0.0490},
    {"expiry": date(2025, 3, 19),  "accrual_end": date(2025, 6, 18), "implied_rate": 0.0460},
])
ois_quotes = [(2.0, 0.0448), (5.0, 0.0415), (10.0, 0.0400)]

curve = SOFRCurveBootstrapper.from_market_data(ref, sofr_on, futures, ois_quotes)
print(curve.zero_curve())

# Price a 5Y SOFR payer swap
from dateutil.relativedelta import relativedelta
swap = SOFRSwap(
    effective_date=date(2024, 6, 7),
    maturity_date=date(2029, 6, 7),
    fixed_rate=curve.par_ois_rate(5.0),
    notional=10_000_000,
    pay_fixed=True,
)
print(f"Par rate: {swap.par_rate(curve)*100:.4f}%")
print(f"PV at par: ${swap.pv(curve):,.2f}")
print(f"DV01: ${swap.dv01(curve):,.0f}")
```

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| FRED (St. Louis Fed) | SOFR, Fed Funds, CPI, PCE, GDP, Treasuries | Free API key |
| CME Group | SR3/SR1 SOFR futures settlements | Free public settlements |
| FRBNY | 30/90/180-day SOFR averages | Via FRED (SOFR30DAYAVG, etc.) |

---

## Research Paper

**Title:** "US Rate Cycle 2025–26: SOFR Curve Dynamics, Fed Policy Forecasting, and Tradeable Signals in the Post-LIBOR Regime"

Forthcoming on SSRN. See `BRAIN.md` for the working draft, model decisions, and key findings.

**Topics covered:**
- SOFR ecosystem: futures, OIS, term rates
- Curve bootstrapping methodology and convexity adjustment
- Taylor Rule analysis: was the Fed's 2022–2023 hiking cycle overdone?
- Nelson-Siegel curve regimes across the rate cycle
- Walk-forward validated 2s10s steepener signal

---

## License

MIT
