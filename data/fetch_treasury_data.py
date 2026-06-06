"""
Fetch US Treasury yield curve data from FRED.
Returns constant-maturity Treasury (CMT) yields at standard tenors.
Also fetches TIPS yields and computes breakeven inflation for real rate analysis.
"""
import os
import pandas as pd
import numpy as np
from fredapi import Fred

# Standard CMT tenors and their FRED series IDs
TREASURY_SERIES = {
    "1m":  "DGS1MO",
    "3m":  "DGS3MO",
    "6m":  "DGS6MO",
    "1y":  "DGS1",
    "2y":  "DGS2",
    "3y":  "DGS3",
    "5y":  "DGS5",
    "7y":  "DGS7",
    "10y": "DGS10",
    "20y": "DGS20",
    "30y": "DGS30",
}

# TIPS (inflation-linked) yields at select tenors
TIPS_SERIES = {
    "tips_5y":  "DFII5",
    "tips_7y":  "DFII7",
    "tips_10y": "DFII10",
    "tips_20y": "DFII20",
    "tips_30y": "DFII30",
}

# Breakeven inflation (nominal - TIPS) from FRED
BREAKEVEN_SERIES = {
    "be_5y":  "T5YIE",
    "be_10y": "T10YIE",
}

# Standard maturities in years (for Nelson-Siegel fitting)
STANDARD_MATURITIES = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
MATURITY_LABELS     = ["1m", "3m", "6m", "1y", "2y", "3y", "5y", "7y", "10y", "20y", "30y"]


def fetch_treasury_yields(api_key: str, start_date: str = "2015-01-01") -> pd.DataFrame:
    """
    Download CMT nominal yields and TIPS yields from FRED.
    Returns a daily DataFrame with columns named by tenor (e.g., 'tsy_10y', 'tips_10y').
    """
    fred = Fred(api_key=api_key)
    frames = {}

    print("Fetching Treasury nominal yields...")
    for tenor, series_id in TREASURY_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start_date)
            frames[f"tsy_{tenor}"] = s
            print(f"  ✓ tsy_{tenor:5s} ({series_id})")
        except Exception as e:
            print(f"  ✗ tsy_{tenor} ({series_id}): {e}")

    print("Fetching TIPS yields...")
    for name, series_id in TIPS_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start_date)
            frames[name] = s
            print(f"  ✓ {name} ({series_id})")
        except Exception as e:
            print(f"  ✗ {name} ({series_id}): {e}")

    print("Fetching breakeven inflation...")
    for name, series_id in BREAKEVEN_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start_date)
            frames[name] = s
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def get_curve_snapshot(df: pd.DataFrame, as_of: str) -> pd.Series:
    """
    Return the full yield curve for a single date, as a Series indexed by tenor label.
    Uses the nearest available prior date if the exact date has no data (e.g., holiday).
    """
    as_of_dt = pd.Timestamp(as_of)
    available = df.loc[:as_of_dt]
    if available.empty:
        raise ValueError(f"No data available on or before {as_of}")
    row = available.iloc[-1]

    tsy_cols = [c for c in df.columns if c.startswith("tsy_")]
    return row[tsy_cols].rename(lambda x: x.replace("tsy_", ""))


def yield_curve_to_arrays(snapshot: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a yield curve snapshot to (maturities, yields) arrays for NS fitting.
    Filters out NaN tenors.
    """
    maturity_map = {
        "1m": 1/12, "3m": 3/12, "6m": 6/12,
        "1y": 1.0, "2y": 2.0, "3y": 3.0,
        "5y": 5.0, "7y": 7.0, "10y": 10.0,
        "20y": 20.0, "30y": 30.0,
    }
    maturities, yields = [], []
    for label, mat in maturity_map.items():
        if label in snapshot.index and not pd.isna(snapshot[label]):
            maturities.append(mat)
            yields.append(float(snapshot[label]))

    return np.array(maturities), np.array(yields)


def compute_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standard Treasury spreads used in rates analysis."""
    spread_pairs = [
        ("slope_2s10s", "tsy_2y",  "tsy_10y"),
        ("slope_2s30s", "tsy_2y",  "tsy_30y"),
        ("slope_5s30s", "tsy_5y",  "tsy_30y"),
        ("slope_3m10y", "tsy_3m",  "tsy_10y"),  # classic recession predictor
        ("belly_2s5s10s", None,    None),         # 2*5Y - 2Y - 10Y curvature
    ]
    for name, short, long in spread_pairs:
        if name == "belly_2s5s10s":
            if all(c in df.columns for c in ["tsy_2y", "tsy_5y", "tsy_10y"]):
                df["belly_2s5s10s"] = 2 * df["tsy_5y"] - df["tsy_2y"] - df["tsy_10y"]
        elif short and long and short in df.columns and long in df.columns:
            df[name] = df[long] - df[short]
    return df


if __name__ == "__main__":
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("Set FRED_API_KEY environment variable")

    df = fetch_treasury_yields(api_key)
    df = compute_spreads(df)
    df.to_csv("data/treasury_yields.csv")
    print(f"\nSaved {len(df)} rows → data/treasury_yields.csv")

    # Quick sanity check
    snap = get_curve_snapshot(df, df.index[-1].strftime("%Y-%m-%d"))
    print("\nLatest yield curve:")
    print(snap.dropna().to_string())
