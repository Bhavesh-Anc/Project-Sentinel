"""
Fetch US rates and macro data from FRED (Federal Reserve Economic Data).
Requires a FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set via environment variable: FRED_API_KEY=your_key
"""
import os
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime

# All FRED series used in this project
FRED_SERIES = {
    # ── Overnight / benchmark rates ──────────────────────────────────────────
    "sofr":           "SOFR",           # SOFR overnight (from Apr 2018)
    "fed_funds":      "FEDFUNDS",       # Effective Fed Funds Rate (daily)
    "iorb":           "IORB",           # Interest on Reserve Balances (floor)

    # ── SOFR compounded averages (FRBNY) ─────────────────────────────────────
    "sofr_30d_avg":   "SOFR30DAYAVG",
    "sofr_90d_avg":   "SOFR90DAYAVG",
    "sofr_180d_avg":  "SOFR180DAYAVG",

    # ── US Treasury constant maturity yields ─────────────────────────────────
    "tsy_1m":         "DGS1MO",
    "tsy_3m":         "DGS3MO",
    "tsy_6m":         "DGS6MO",
    "tsy_1y":         "DGS1",
    "tsy_2y":         "DGS2",
    "tsy_5y":         "DGS5",
    "tsy_7y":         "DGS7",
    "tsy_10y":        "DGS10",
    "tsy_20y":        "DGS20",
    "tsy_30y":        "DGS30",

    # ── Inflation ────────────────────────────────────────────────────────────
    "cpi":            "CPIAUCSL",       # Headline CPI (all urban, NSA)
    "core_cpi":       "CPILFESL",       # Core CPI (ex food & energy)
    "pce":            "PCEPI",          # PCE price index
    "core_pce":       "PCEPILFE",       # Core PCE (Fed's preferred target)
    "breakeven_5y":   "T5YIE",          # 5Y breakeven inflation
    "breakeven_10y":  "T10YIE",         # 10Y breakeven inflation

    # ── Real economy / labor ─────────────────────────────────────────────────
    "real_gdp":       "GDPC1",          # Real GDP (quarterly, seasonally adj.)
    "unemployment":   "UNRATE",         # U-3 unemployment rate
    "lfpr":           "CIVPART",        # Labor force participation rate
    "nfp":            "PAYEMS",         # Non-farm payrolls (level)
    "jolts_openings": "JTSJOL",         # JOLTS job openings

    # ── Fed balance sheet ────────────────────────────────────────────────────
    "fed_balance":    "WALCL",          # Fed total assets (weekly)
    "reserves":       "TOTRESNS",       # Bank reserves at the Fed

    # ── Financial conditions ─────────────────────────────────────────────────
    "vix":            "VIXCLS",
    "credit_hy":      "BAMLH0A0HYM2",   # HY credit spread (OAS)
    "credit_ig":      "BAMLC0A0CM",     # IG credit spread (OAS)
    "ted_spread":     "TEDRATE",        # TED spread (3M LIBOR - 3M T-bill)
}

# Series that are monthly/quarterly — forward-filled for daily alignment
SPARSE_SERIES = {
    "monthly": ["cpi", "core_cpi", "pce", "core_pce", "unemployment", "lfpr", "nfp", "jolts_openings"],
    "quarterly": ["real_gdp"],
    "weekly": ["fed_balance", "reserves"],
}


def fetch_fred(api_key: str, start_date: str = "2018-01-01") -> pd.DataFrame:
    """
    Download all FRED series and return a single daily-indexed DataFrame.
    Monthly/quarterly series are forward-filled to align with daily rates data.
    """
    fred = Fred(api_key=api_key)
    raw: dict[str, pd.Series] = {}

    print("Fetching from FRED...")
    for name, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start_date)
            raw[name] = s
            print(f"  ✓ {name:20s} ({series_id}): {len(s):>5} obs")
        except Exception as e:
            print(f"  ✗ {name:20s} ({series_id}): {e}")

    df = pd.DataFrame(raw)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Forward-fill sparse series so daily models have a value every business day
    for freq, cols in SPARSE_SERIES.items():
        for col in cols:
            if col in df.columns:
                df[col] = df[col].ffill()

    return df


def add_derived_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived series:
    - Year-over-year % changes for inflation / GDP
    - Unemployment gap (actual minus natural rate, approximated at 4.0%)
    - 2s10s slope (nominal term spread)
    - Real Fed Funds Rate
    """
    # YoY for level series (using 252 business day lag as proxy for 1 year)
    for col in ["cpi", "core_cpi", "pce", "core_pce"]:
        if col in df.columns:
            df[f"{col}_yoy"] = df[col].pct_change(252) * 100

    # Monthly pct change annualized for monthly series
    for col in ["cpi", "core_pce"]:
        if col in df.columns:
            df[f"{col}_mom_ann"] = df[col].pct_change(21) * 12 * 100  # ~21 bday/month

    # Unemployment gap (U-3 minus NAIRU; CBO estimates NAIRU ≈ 4.0%)
    NAIRU = 4.0
    if "unemployment" in df.columns:
        df["unemployment_gap"] = df["unemployment"] - NAIRU

    # 2s10s slope
    if "tsy_2y" in df.columns and "tsy_10y" in df.columns:
        df["slope_2s10s"] = df["tsy_10y"] - df["tsy_2y"]

    # 5s30s
    if "tsy_5y" in df.columns and "tsy_30y" in df.columns:
        df["slope_5s30s"] = df["tsy_30y"] - df["tsy_5y"]

    # Real Fed Funds (nominal EFFR minus core PCE yoy)
    if "fed_funds" in df.columns and "core_pce_yoy" in df.columns:
        df["real_fed_funds"] = df["fed_funds"] - df["core_pce_yoy"]

    # SOFR / Fed Funds spread (should be ~0 post-LIBOR; validates data quality)
    if "sofr" in df.columns and "fed_funds" in df.columns:
        df["sofr_effr_spread"] = df["sofr"] - df["fed_funds"]

    return df


def save(df: pd.DataFrame, path: str = "data/sofr_fred_data.csv") -> None:
    df.to_csv(path)
    print(f"\nSaved {len(df)} rows × {len(df.columns)} columns → {path}")


if __name__ == "__main__":
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("Set FRED_API_KEY environment variable")

    df = fetch_fred(api_key)
    df = add_derived_series(df)
    save(df)
