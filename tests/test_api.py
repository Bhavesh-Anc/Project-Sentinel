"""
Tests for api/main.py FastAPI endpoints.
Uses httpx.AsyncClient via TestClient for synchronous testing.
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app, raise_server_exceptions=True)


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_fields(self):
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "engine" in body


# ── Curve endpoints ───────────────────────────────────────────────────────────

class TestCurveZero:

    def test_zero_curve_200(self):
        r = client.get("/curve/zero", params={"rate": 0.04})
        assert r.status_code == 200

    def test_zero_curve_structure(self):
        r = client.get("/curve/zero", params={"rate": 0.04, "tenors": "1,2,5,10"})
        body = r.json()
        assert "curve" in body
        assert len(body["curve"]) == 4

    def test_zero_curve_keys(self):
        r = client.get("/curve/zero", params={"rate": 0.04, "tenors": "5"})
        row = r.json()["curve"][0]
        assert "tenor_yrs" in row
        assert "zero_rate_pct" in row
        assert "df" in row

    def test_zero_curve_discount_factors_in_01(self):
        r = client.get("/curve/zero", params={"rate": 0.04, "tenors": "1,5,10,30"})
        for row in r.json()["curve"]:
            assert 0 < row["df"] < 1


class TestCurveBootstrap:

    def _payload(self):
        return {
            "ref_date_iso": "2024-01-02",
            "sofr_on": 0.0533,
            "futures": [],
            "ois_quotes": [
                {"tenor_years": 1.0, "rate": 0.052},
                {"tenor_years": 2.0, "rate": 0.048},
                {"tenor_years": 5.0, "rate": 0.042},
                {"tenor_years": 10.0, "rate": 0.040},
            ],
        }

    def test_bootstrap_200(self):
        r = client.post("/curve/bootstrap", json=self._payload())
        assert r.status_code == 200

    def test_bootstrap_has_curve(self):
        r = client.post("/curve/bootstrap", json=self._payload())
        body = r.json()
        assert "zero_curve" in body
        assert len(body["zero_curve"]) > 0


# ── Swap pricing ──────────────────────────────────────────────────────────────

class TestSwapPricing:

    def _swap_payload(self, pay_fixed=True, fixed_rate=0.04):
        return {
            "effective_date_iso": "2024-01-02",
            "maturity_date_iso":  "2029-01-02",
            "fixed_rate":  fixed_rate,
            "notional":    10_000_000,
            "pay_fixed":   pay_fixed,
            "sofr_on":     0.0400,
        }

    def test_swap_price_200(self):
        r = client.post("/price/swap", json=self._swap_payload())
        assert r.status_code == 200

    def test_swap_price_keys(self):
        r = client.post("/price/swap", json=self._swap_payload())
        body = r.json()
        assert "net_pv" in body
        assert "dv01" in body

    def test_payer_loses_when_rate_too_high(self):
        # Payer fixing at 10% on a 4% curve → very negative PV (overpaying fixed)
        r = client.post("/price/swap", json=self._swap_payload(fixed_rate=0.10))
        pv = r.json()["net_pv"]
        assert pv < 0

    def test_payer_receiver_pv_opposite(self):
        r_pay = client.post("/price/swap", json=self._swap_payload(pay_fixed=True))
        r_rec = client.post("/price/swap", json=self._swap_payload(pay_fixed=False))
        pv_pay = r_pay.json()["net_pv"]
        pv_rec = r_rec.json()["net_pv"]
        # Payer and receiver should have opposite PVs
        assert abs(pv_pay + pv_rec) < 1.0  # sum ≈ 0


# ── FRA pricing ───────────────────────────────────────────────────────────────

class TestFRAPricing:

    def test_fra_200(self):
        r = client.post("/price/fra", json={
            "start_date_iso": "2024-07-01",
            "end_date_iso":   "2025-01-01",
            "fixed_rate":     0.040,
            "notional":       10_000_000,
            "pay_receive":    "receive",
            "sofr_on":        0.0400,
        })
        assert r.status_code == 200

    def test_fra_has_pv(self):
        r = client.post("/price/fra", json={
            "start_date_iso": "2024-07-01",
            "end_date_iso":   "2025-01-01",
            "fixed_rate":     0.040,
            "notional":       10_000_000,
            "pay_receive":    "receive",
            "sofr_on":        0.0400,
        })
        assert "net_pv_usd" in r.json()


# ── Risk endpoints ────────────────────────────────────────────────────────────

class TestRiskScenario:

    def test_scenario_parallel_200(self):
        r = client.post("/risk/scenario", json={
            "sofr_on": 0.04,
            "swap_tenor_yrs": 5.0,
            "swap_notional":  1_000_000,
            "shift_type": "parallel",
            "shift1_bps": 25.0,
        })
        assert r.status_code == 200

    def test_scenario_has_pnl(self):
        r = client.post("/risk/scenario", json={
            "sofr_on": 0.04,
            "swap_tenor_yrs": 5.0,
            "shift_type": "parallel",
            "shift1_bps": 25.0,
        })
        body = r.json()
        assert "pnl_usd" in body

    def test_scenario_twist_200(self):
        r = client.post("/risk/scenario", json={
            "sofr_on": 0.04,
            "swap_tenor_yrs": 10.0,
            "shift_type": "twist",
            "shift1_bps": 10.0,
            "shift2_bps": -10.0,
        })
        assert r.status_code == 200


class TestDV01:

    def test_dv01_200(self):
        r = client.post("/risk/dv01", json={
            "sofr_on": 0.04,
            "swaps": [{"tenor_y": 5.0, "notional": 10_000_000, "pay_fixed": True}],
        })
        assert r.status_code == 200

    def test_dv01_ladder_has_rows(self):
        r = client.post("/risk/dv01", json={
            "sofr_on": 0.04,
            "swaps": [{"tenor_y": 10.0, "notional": 10_000_000, "pay_fixed": True}],
        })
        body = r.json()
        assert "ladder" in body
        assert len(body["ladder"]) > 0


# ── Carry table ───────────────────────────────────────────────────────────────

class TestCarryTable:

    def test_carry_200(self):
        r = client.get("/carry/table", params={"sofr_on": 0.04})
        assert r.status_code == 200

    def test_carry_has_rows(self):
        r = client.get("/carry/table", params={"sofr_on": 0.04})
        body = r.json()
        assert "table" in body
        assert len(body["table"]) > 0

    def test_carry_row_keys(self):
        r = client.get("/carry/table", params={"sofr_on": 0.04})
        row = r.json()["table"][0]
        assert "carry_bps" in row
        assert "rolldown_bps" in row
        assert "total_return_bps" in row


# ── Signals ───────────────────────────────────────────────────────────────────

class TestSignals:

    def _payload(self):
        return {
            "core_pce_yoy":  2.11,
            "unemployment":  4.30,
            "tsy_2y":        4.50,
            "tsy_10y":       4.35,
            "effr":          4.33,
            "taylor_rate":   2.26,
        }

    def test_signals_200(self):
        r = client.post("/signals/composite", json=self._payload())
        assert r.status_code == 200

    def test_signals_direction(self):
        r = client.post("/signals/composite", json=self._payload())
        body = r.json()
        assert "position" in body
        assert body["position"] in (-1, 0, 1)


# ── Taylor Rule ───────────────────────────────────────────────────────────────

class TestTaylor:

    def test_taylor_200(self):
        r = client.post("/taylor/rate", json={
            "inflation":   2.5,
            "unemployment": 4.0,
            "neutral_rate": 2.5,
        })
        assert r.status_code == 200

    def test_taylor_has_recommended_rate(self):
        r = client.post("/taylor/rate", json={
            "inflation":   2.5,
            "unemployment": 4.0,
            "neutral_rate": 2.5,
        })
        assert "taylor_rate_pct" in r.json()


# ── FOMC Probabilities ────────────────────────────────────────────────────────

class TestFOMC:

    def _payload(self):
        return {"current_rate": 0.0433, "implied_rate": 0.0408}

    def test_fomc_200(self):
        r = client.post("/fomc/probabilities", json=self._payload())
        assert r.status_code == 200

    def test_fomc_keys(self):
        r = client.post("/fomc/probabilities", json=self._payload())
        body = r.json()
        assert "probabilities" in body or "summary" in body
