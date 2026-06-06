"""
Fetch CME SOFR futures settlement data.

CME publishes free end-of-day settlements for SOFR futures:
  SR3  — 3-month SOFR futures (quarterly IMM dates; settles to compounded SOFR)
  SR1  — 1-month SOFR futures (monthly; settles to arithmetic avg SOFR)

Free data endpoint (CME Group public API):
  https://www.cmegroup.com/CmeWS/mvc/Settlements/futures/settlements/{product}/download?tradeDate={YYYYMMDD}&ext=csv

For historical backtesting, CME DataMine (free tier) provides up to 5 years.
This module provides:
  1. Live settlement fetcher for current/recent dates
  2. A parser for CSV exports from CME DataMine
  3. Contract metadata utilities (IMM dates, expiry, tenor)
"""
import io
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

# CME product codes
CME_CODES = {"SR3": "sofr", "SR1": "1-month-sofr"}

# IMM month codes: H=Mar, M=Jun, U=Sep, Z=Dec
IMM_MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}
IMM_MONTHS = {v: k for k, v in IMM_MONTH_CODES.items()}

CME_SETTLEMENTS_URL = (
    "https://www.cmegroup.com/CmeWS/mvc/Settlements/futures/settlements"
    "/{product}/download?tradeDate={date}&ext=csv"
)


# ── IMM date utilities ────────────────────────────────────────────────────────

def third_wednesday(year: int, month: int) -> date:
    """Return the 3rd Wednesday of a given month (CME IMM date)."""
    first = date(year, month, 1)
    # weekday(): Monday=0, Wednesday=2
    days_to_wed = (2 - first.weekday()) % 7
    first_wed = first + timedelta(days=days_to_wed)
    return first_wed + timedelta(weeks=2)


def imm_dates(start: date, n: int = 16) -> list[date]:
    """Return the next n quarterly IMM dates (Mar/Jun/Sep/Dec)."""
    dates = []
    year, month = start.year, start.month
    # Round up to next IMM month
    imm_months_sorted = sorted(IMM_MONTHS.values())  # [3, 6, 9, 12]
    for _ in range(n * 2):
        for m in imm_months_sorted:
            d = third_wednesday(year, m)
            if d >= start:
                dates.append(d)
                if len(dates) == n:
                    return dates
        year += 1
    return dates


def contract_label(expiry: date, contract_type: str = "SR3") -> str:
    """
    Generate CME-style contract label, e.g. 'SR3H4' for March 2024 SR3.
    Year uses last 2 digits; decade prefix not needed for near-term contracts.
    """
    if expiry.month in IMM_MONTH_CODES:
        code = IMM_MONTH_CODES[expiry.month]
    else:
        # Serial (non-IMM) month: use numeric representation
        code = str(expiry.month).zfill(2)
    year_suffix = str(expiry.year)[-1]  # last digit of year
    return f"{contract_type}{code}{year_suffix}"


# ── CME settlement fetcher ────────────────────────────────────────────────────

def fetch_cme_settlement(trade_date: date, product: str = "sofr", retries: int = 3) -> pd.DataFrame:
    """
    Download CME SOFR futures settlements for a single trade date.
    Returns a DataFrame with columns: [contract, settlement_price, volume, open_interest].
    """
    url = CME_SETTLEMENTS_URL.format(
        product=product,
        date=trade_date.strftime("%Y%m%d"),
    )
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), skiprows=2)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            df["trade_date"] = trade_date
            return _clean_settlement_df(df)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Failed to fetch CME settlement for {trade_date}: {e}")


def _clean_settlement_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and parse settlement prices."""
    rename_map = {
        "month": "contract",
        "settle": "settlement_price",
        "vol": "volume",
        "prev_vol": "prev_volume",
        "open_int": "open_interest",
        "prev_int": "prev_open_interest",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Keep only rows with valid settlement prices
    if "settlement_price" in df.columns:
        df["settlement_price"] = pd.to_numeric(df["settlement_price"], errors="coerce")
        df = df.dropna(subset=["settlement_price"])

    # Derive implied rate: price = 100 - rate
    if "settlement_price" in df.columns:
        df["implied_rate"] = (100.0 - df["settlement_price"]) / 100.0

    return df[["trade_date", "contract", "settlement_price", "implied_rate",
               "volume", "open_interest"] if all(c in df.columns for c in
               ["volume", "open_interest"]) else
               [c for c in ["trade_date", "contract", "settlement_price", "implied_rate"]
                if c in df.columns]]


def fetch_cme_range(
    start: date,
    end: date,
    product: str = "sofr",
    sleep_sec: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch CME settlements for every business day in [start, end].
    sleep_sec: polite delay between requests to avoid rate-limiting.
    """
    bdays = pd.bdate_range(start, end)
    frames = []
    for d in bdays:
        try:
            df = fetch_cme_settlement(d.date(), product=product)
            frames.append(df)
            time.sleep(sleep_sec)
        except Exception as e:
            print(f"  ✗ {d.date()}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── CME DataMine CSV parser ───────────────────────────────────────────────────

def parse_datamine_csv(path: str) -> pd.DataFrame:
    """
    Parse a CME DataMine export CSV for SOFR futures.
    DataMine exports have variable headers depending on export settings;
    this parser handles the most common formats.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Normalise date column
    date_cols = [c for c in df.columns if "date" in c or "time" in c]
    if date_cols:
        df["trade_date"] = pd.to_datetime(df[date_cols[0]])

    # Normalise price column
    price_cols = [c for c in df.columns if "settle" in c or "close" in c or "price" in c]
    if price_cols:
        df["settlement_price"] = pd.to_numeric(df[price_cols[0]], errors="coerce")
        df["implied_rate"] = (100.0 - df["settlement_price"]) / 100.0

    return df


# ── Contract term structure builder ──────────────────────────────────────────

class SOFRFuturesCurve:
    """
    Represents the SOFR futures term structure for a single trade date.
    Maps each contract to its expiry, accrual period, and implied forward rate.
    """

    def __init__(self, trade_date: date, settlements: pd.DataFrame, contract_type: str = "SR3"):
        self.trade_date = trade_date
        self.contract_type = contract_type
        self._contracts = self._parse(settlements)

    def _parse(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach expiry date and accrual period to each contract row.
        SR3: accrual = [expiry, expiry + 91 days] approx (exact = next IMM date)
        SR1: accrual = [first day of delivery month, last day]
        """
        rows = []
        for _, row in df.iterrows():
            label = str(row.get("contract", "")).strip().upper()
            expiry = self._parse_expiry(label)
            if expiry is None:
                continue

            if self.contract_type == "SR3":
                # Accrual period: from expiry (3rd Wed) to next quarterly IMM date
                next_imm_dates = imm_dates(expiry + timedelta(days=1), n=1)
                accrual_end = next_imm_dates[0] if next_imm_dates else expiry + timedelta(days=91)
            else:
                # SR1: calendar month delivery
                accrual_end = (expiry.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)

            rows.append({
                "contract": label,
                "expiry": expiry,
                "accrual_start": expiry,
                "accrual_end": accrual_end,
                "settlement_price": row.get("settlement_price"),
                "implied_rate": row.get("implied_rate"),
                "tenor_days": (accrual_end - expiry).days,
            })

        return pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)

    def _parse_expiry(self, label: str) -> date | None:
        """Parse contract label like 'SR3H5' → 3rd Wednesday March 2025."""
        try:
            # Strip product prefix if present
            for prefix in ("SR3", "SR1", "SFR"):
                if label.startswith(prefix):
                    label = label[len(prefix):]
                    break
            if len(label) < 2:
                return None
            month_code = label[0]
            year_digit = int(label[1])
            if month_code not in IMM_MONTHS:
                return None
            month = IMM_MONTHS[month_code]
            # Year: disambiguate decade using trade date
            base_decade = (self.trade_date.year // 10) * 10
            year = base_decade + year_digit
            if year < self.trade_date.year:
                year += 10
            return third_wednesday(year, month)
        except Exception:
            return None

    @property
    def contracts(self) -> pd.DataFrame:
        return self._contracts

    def implied_forward_rate(self, t1: date, t2: date) -> float | None:
        """Find the futures contract whose accrual period best covers [t1, t2]."""
        for _, row in self._contracts.iterrows():
            if abs((row["accrual_start"] - t1).days) <= 3 and abs((row["accrual_end"] - t2).days) <= 3:
                return float(row["implied_rate"])
        return None


if __name__ == "__main__":
    today = date.today()
    print(f"Next 8 quarterly IMM dates from {today}:")
    for d in imm_dates(today, n=8):
        print(f"  {d}  {contract_label(d)}")
