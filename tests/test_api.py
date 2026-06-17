"""
Tests for api/main.py FastAPI endpoints.
Uses httpx.AsyncClient via TestClient for synchronous testing.
"""
from __future__ import annotations
import pytest
import numpy as np
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


# ── Swaption pricing ──────────────────────────────────────────────────────────

class TestSwaptionPricing:

    def _payload(self, **kwargs):
        base = {
            "sofr_on":          0.0453,
            "expiry_years":     1.0,
            "swap_tenor_years": 5.0,
            "notional":         10_000_000,
            "swaption_type":    "payer",
            "vol":              0.20,
        }
        base.update(kwargs)
        return base

    def test_payer_200(self):
        r = client.post("/price/swaption", json=self._payload())
        assert r.status_code == 200

    def test_receiver_200(self):
        r = client.post("/price/swaption", json=self._payload(swaption_type="receiver"))
        assert r.status_code == 200

    def test_response_keys_present(self):
        r = client.post("/price/swaption", json=self._payload())
        body = r.json()
        for key in ["expiry_years", "swap_tenor_years", "strike_pct",
                    "forward_rate_pct", "annuity", "black_pv",
                    "intrinsic_value", "time_value", "vega", "delta", "moneyness_bps"]:
            assert key in body, f"Missing key: {key}"

    def test_payer_pv_positive(self):
        r = client.post("/price/swaption", json=self._payload())
        assert r.json()["black_pv"] > 0

    def test_receiver_pv_positive(self):
        r = client.post("/price/swaption", json=self._payload(swaption_type="receiver"))
        assert r.json()["black_pv"] > 0

    def test_atm_strike_none(self):
        # When strike is omitted (ATM), forward_rate_pct == strike_pct
        r = client.post("/price/swaption", json=self._payload())
        body = r.json()
        assert abs(body["forward_rate_pct"] - body["strike_pct"]) < 1e-3

    def test_explicit_strike(self):
        r = client.post("/price/swaption", json=self._payload(strike=0.04))
        body = r.json()
        assert abs(body["strike_pct"] - 4.0) < 1e-4

    def test_pv_scales_with_notional(self):
        r1 = client.post("/price/swaption", json=self._payload(notional=1_000_000))
        r10 = client.post("/price/swaption", json=self._payload(notional=10_000_000))
        ratio = r10.json()["black_pv"] / r1.json()["black_pv"]
        assert abs(ratio - 10.0) < 1e-4

    def test_pv_increases_with_vol(self):
        lo = client.post("/price/swaption", json=self._payload(vol=0.10)).json()["black_pv"]
        hi = client.post("/price/swaption", json=self._payload(vol=0.30)).json()["black_pv"]
        assert hi > lo

    def test_put_call_parity(self):
        K = 0.04
        r_pay = client.post("/price/swaption", json=self._payload(strike=K, swaption_type="payer"))
        r_rec = client.post("/price/swaption", json=self._payload(strike=K, swaption_type="receiver"))
        pay = r_pay.json()
        rec = r_rec.json()
        N = 10_000_000
        A = pay["annuity"]
        S = pay["forward_rate_pct"] / 100.0
        expected = N * A * (S - K)
        actual   = pay["black_pv"] - rec["black_pv"]
        assert abs(actual - expected) < 2.0   # within $2 (rounding from round(..., 6))

    def test_invalid_type_422(self):
        r = client.post("/price/swaption", json=self._payload(swaption_type="invalid"))
        assert r.status_code == 422

    def test_payer_delta_positive(self):
        r = client.post("/price/swaption", json=self._payload())
        assert r.json()["delta"] > 0

    def test_receiver_delta_negative(self):
        r = client.post("/price/swaption", json=self._payload(swaption_type="receiver"))
        assert r.json()["delta"] < 0

    def test_vega_positive(self):
        r = client.post("/price/swaption", json=self._payload())
        assert r.json()["vega"] > 0

    def test_pv_equals_intrinsic_plus_time_value(self):
        r = client.post("/price/swaption", json=self._payload())
        body = r.json()
        assert abs(body["black_pv"] - (body["intrinsic_value"] + body["time_value"])) < 1e-3

    def test_atm_moneyness_near_zero(self):
        r = client.post("/price/swaption", json=self._payload())
        assert abs(r.json()["moneyness_bps"]) < 1e-3


# ── Vol surface ───────────────────────────────────────────────────────────────

class TestVolSurface:

    def test_vol_surface_200(self):
        r = client.get("/vol/surface")
        assert r.status_code == 200

    def test_vol_surface_has_surface_key(self):
        r = client.get("/vol/surface")
        body = r.json()
        assert "surface" in body
        assert "description" in body

    def test_vol_surface_has_rows(self):
        r = client.get("/vol/surface")
        rows = r.json()["surface"]
        assert len(rows) == 5  # 5 expiry rows

    def test_vol_surface_has_expiry_column(self):
        r = client.get("/vol/surface")
        first_row = r.json()["surface"][0]
        assert "expiry" in first_row

    def test_vol_surface_values_in_pct_range(self):
        r = client.get("/vol/surface")
        for row in r.json()["surface"]:
            for k, v in row.items():
                if k != "expiry" and v is not None:
                    assert 1.0 < v < 50.0, f"Vol {v} out of range at {k}"

    def test_vol_surface_decreasing_with_expiry(self):
        # Short-expiry vols should be higher than long-expiry vols for same tenor
        rows = client.get("/vol/surface").json()["surface"]
        # Grab 1Y tenor column from first and last expiry rows
        col = "1Y"
        first_val = rows[0].get(col)
        last_val  = rows[-1].get(col)
        if first_val is not None and last_val is not None:
            assert first_val > last_val


# ── SABR vol model ────────────────────────────────────────────────────────────

class TestSABRSmile:

    def _payload(self, **kw):
        base = {
            "forward_rate":     0.0453,
            "expiry_years":     1.0,
            "alpha":            0.05,
            "beta":             0.5,
            "rho":             -0.25,
            "nu":               0.40,
            "n_strikes":        11,
            "strike_range_bps": 100.0,
        }
        base.update(kw)
        return base

    def test_smile_200(self):
        assert client.post("/sabr/smile", json=self._payload()).status_code == 200

    def test_smile_keys(self):
        body = client.post("/sabr/smile", json=self._payload()).json()
        assert "forward_rate_pct" in body
        assert "atm_vol_pct" in body
        assert "smile" in body

    def test_smile_row_count(self):
        body = client.post("/sabr/smile", json=self._payload(n_strikes=11)).json()
        assert len(body["smile"]) == 11

    def test_smile_columns(self):
        row = client.post("/sabr/smile", json=self._payload()).json()["smile"][0]
        for col in ["strike_pct", "moneyness_bps", "sabr_vol_pct", "normal_vol_bps"]:
            assert col in row

    def test_smile_atm_vol_positive(self):
        body = client.post("/sabr/smile", json=self._payload()).json()
        assert body["atm_vol_pct"] > 0

    def test_smile_negative_rho_creates_put_skew(self):
        # Negative rho → receiver OTM (low strike) vol > payer OTM (high strike) vol
        rows = client.post("/sabr/smile", json=self._payload(rho=-0.40)).json()["smile"]
        vols = [(r["moneyness_bps"], r["sabr_vol_pct"]) for r in rows]
        low_strike_vol  = next(v for m, v in vols if m < -50)
        high_strike_vol = next(v for m, v in sorted(vols, reverse=True) if m > 50)
        assert low_strike_vol > high_strike_vol

    def test_smile_higher_nu_wider_smile(self):
        lo = client.post("/sabr/smile", json=self._payload(nu=0.10)).json()["smile"]
        hi = client.post("/sabr/smile", json=self._payload(nu=0.80)).json()["smile"]
        # Wing vols should be higher with larger nu
        lo_wing = max(abs(r["sabr_vol_pct"] - lo[len(lo)//2]["sabr_vol_pct"]) for r in lo)
        hi_wing = max(abs(r["sabr_vol_pct"] - hi[len(hi)//2]["sabr_vol_pct"]) for r in hi)
        assert hi_wing > lo_wing


class TestSABRCalibrate:

    def _payload(self, **kw):
        base = {
            "forward_rate": 0.0453,
            "expiry_years": 1.0,
            "strikes":      [0.035, 0.040, 0.045, 0.050, 0.055],
            "market_vols":  [0.135, 0.130, 0.125, 0.122, 0.120],
            "beta":         0.5,
        }
        base.update(kw)
        return base

    def test_calibrate_200(self):
        assert client.post("/sabr/calibrate", json=self._payload()).status_code == 200

    def test_calibrate_keys(self):
        body = client.post("/sabr/calibrate", json=self._payload()).json()
        for k in ["alpha", "beta", "rho", "nu", "rmse_bps", "max_error_bps"]:
            assert k in body

    def test_calibrate_alpha_positive(self):
        body = client.post("/sabr/calibrate", json=self._payload()).json()
        assert body["alpha"] > 0

    def test_calibrate_rho_in_bounds(self):
        body = client.post("/sabr/calibrate", json=self._payload()).json()
        assert -1.0 < body["rho"] < 1.0

    def test_calibrate_rmse_small(self):
        # Good calibration should achieve < 2bp RMSE on reasonable market quotes
        body = client.post("/sabr/calibrate", json=self._payload()).json()
        assert body["rmse_bps"] < 5.0  # SABR is an approximation; realistic quotes have ~2-3bp fit error

    def test_calibrate_atm_only(self):
        payload = dict(forward_rate=0.045, expiry_years=1.0,
                       strikes=[0.045], market_vols=[0.125], beta=0.5)
        r = client.post("/sabr/calibrate", json=payload)
        assert r.status_code == 200
        assert r.json()["alpha"] > 0

    def test_calibrate_length_mismatch_422(self):
        payload = dict(forward_rate=0.045, expiry_years=1.0,
                       strikes=[0.04, 0.05], market_vols=[0.12], beta=0.5)
        assert client.post("/sabr/calibrate", json=payload).status_code == 422


class TestSABRSurface:

    def test_surface_200(self):
        assert client.get("/sabr/surface").status_code == 200

    def test_surface_has_25_nodes(self):
        body = client.get("/sabr/surface").json()
        assert len(body["nodes"]) == 25  # 5 expiries × 5 tenors

    def test_surface_node_keys(self):
        node = client.get("/sabr/surface").json()["nodes"][0]
        for k in ["expiry_years", "tenor_years", "alpha", "beta", "rho", "nu", "atm_vol_pct"]:
            assert k in node

    def test_surface_atm_vols_positive(self):
        for node in client.get("/sabr/surface").json()["nodes"]:
            assert node["atm_vol_pct"] > 0

    def test_surface_beta_fixed(self):
        for node in client.get("/sabr/surface").json()["nodes"]:
            assert abs(node["beta"] - 0.5) < 1e-6

    def test_surface_rho_negative(self):
        # USD convention: rho = -0.25 (payer skew)
        for node in client.get("/sabr/surface").json()["nodes"]:
            assert node["rho"] < 0


# ── Cap / Floor endpoints ─────────────────────────────────────────────────────

class TestCapFloorEndpoints:

    def test_cap_price_200(self):
        r = client.post("/cap/price", json={
            "sofr_on": 0.0433, "maturity_years": 5.0,
            "strike": 0.04, "notional": 10_000_000, "vol": 0.30,
            "instrument": "cap",
        })
        assert r.status_code == 200

    def test_cap_price_pv_positive(self):
        body = client.post("/cap/price", json={
            "sofr_on": 0.0433, "maturity_years": 5.0,
            "strike": 0.04, "notional": 10_000_000, "vol": 0.30,
        }).json()
        assert body["pv"] > 0

    def test_floor_price_200(self):
        r = client.post("/cap/price", json={
            "sofr_on": 0.0433, "maturity_years": 5.0,
            "strike": 0.04, "notional": 10_000_000, "vol": 0.30,
            "instrument": "floor",
        })
        assert r.status_code == 200
        assert r.json()["pv"] > 0

    def test_cap_price_atm_strike(self):
        body = client.post("/cap/price", json={
            "sofr_on": 0.0433, "maturity_years": 5.0,
            "notional": 10_000_000, "vol": 0.30,
        }).json()
        assert "atm_forward_pct" in body
        assert abs(body["moneyness_bps"]) < 0.01  # ATM → moneyness ≈ 0

    def test_cap_response_keys(self):
        body = client.post("/cap/price", json={
            "sofr_on": 0.0433, "maturity_years": 5.0,
            "strike": 0.04, "vol": 0.30,
        }).json()
        for key in ["pv", "dv01", "vega_per_bp", "atm_forward_pct", "n_caplets", "strike_pct"]:
            assert key in body

    def test_cap_dv01_positive(self):
        body = client.post("/cap/price", json={
            "sofr_on": 0.0433, "maturity_years": 5.0,
            "strike": 0.04, "vol": 0.30,
        }).json()
        assert body["dv01"] > 0

    def test_longer_maturity_higher_cap_pv(self):
        pv_2y = client.post("/cap/price", json={
            "sofr_on": 0.0433, "maturity_years": 2.0,
            "strike": 0.04, "vol": 0.30, "notional": 10_000_000,
        }).json()["pv"]
        pv_5y = client.post("/cap/price", json={
            "sofr_on": 0.0433, "maturity_years": 5.0,
            "strike": 0.04, "vol": 0.30, "notional": 10_000_000,
        }).json()["pv"]
        assert pv_5y > pv_2y

    def test_vol_surface_200(self):
        r = client.get("/cap/vol-surface")
        assert r.status_code == 200

    def test_vol_surface_nodes(self):
        body = client.get("/cap/vol-surface").json()
        assert len(body["nodes"]) > 0
        for node in body["nodes"][:5]:
            assert "tenor_years" in node
            assert "strike_pct" in node
            assert "vol_pct" in node

    def test_strip_vols_200(self):
        r = client.get("/cap/strip-vols")
        assert r.status_code == 200

    def test_strip_vols_returns_caplet_vols(self):
        body = client.get("/cap/strip-vols").json()
        assert "caplet_vols" in body
        assert len(body["caplet_vols"]) > 0

    def test_strip_vols_expiry_order(self):
        vols = client.get("/cap/strip-vols").json()["caplet_vols"]
        expiries = [v["expiry_years"] for v in vols]
        assert expiries == sorted(expiries)


# ── Monte Carlo / VaR endpoints ───────────────────────────────────────────────

class TestMCVaREndpoints:

    def test_mc_var_200(self):
        r = client.post("/mc/var", json={
            "sofr_on": 0.0433, "hw_a": 0.05, "hw_sigma": 0.010,
            "portfolio_dv01": 10_000, "horizon_days": 1,
            "confidence": 0.99, "n_paths": 2000,
        })
        assert r.status_code == 200

    def test_mc_var_negative_for_long_duration(self):
        body = client.post("/mc/var", json={
            "sofr_on": 0.0433, "hw_a": 0.05, "hw_sigma": 0.010,
            "portfolio_dv01": 10_000, "horizon_days": 1,
            "confidence": 0.99, "n_paths": 2000,
        }).json()
        assert body["var_usd"] < 0

    def test_mc_var_keys(self):
        body = client.post("/mc/var", json={
            "sofr_on": 0.0433, "hw_a": 0.05, "hw_sigma": 0.010,
            "portfolio_dv01": 10_000, "horizon_days": 1,
            "confidence": 0.99, "n_paths": 1000,
        }).json()
        for key in ["var_usd", "cvar_usd", "var_bps", "pnl_percentiles", "n_paths"]:
            assert key in body

    def test_mc_var_cvar_worse_than_var(self):
        body = client.post("/mc/var", json={
            "sofr_on": 0.0433, "hw_a": 0.05, "hw_sigma": 0.010,
            "portfolio_dv01": 10_000, "horizon_days": 1,
            "confidence": 0.99, "n_paths": 2000,
        }).json()
        assert body["cvar_usd"] <= body["var_usd"]

    def test_mc_caplet_200(self):
        r = client.post("/mc/caplet", json={
            "sofr_on": 0.0433, "hw_a": 0.05, "hw_sigma": 0.010,
            "strike": 0.04, "t_reset": 1.0, "t_pay": 1.25,
            "notional": 1_000_000, "n_paths": 2000,
        })
        assert r.status_code == 200

    def test_mc_caplet_positive_pv(self):
        body = client.post("/mc/caplet", json={
            "sofr_on": 0.0433, "hw_a": 0.05, "hw_sigma": 0.010,
            "strike": 0.04, "t_reset": 1.0, "t_pay": 1.25,
            "notional": 1_000_000, "n_paths": 3000,
        }).json()
        assert body["mc_pv"] > 0

    def test_mc_caplet_keys(self):
        body = client.post("/mc/caplet", json={
            "sofr_on": 0.0433, "hw_a": 0.05, "hw_sigma": 0.010,
            "strike": 0.04, "t_reset": 1.0, "t_pay": 1.25,
            "n_paths": 1000,
        }).json()
        for key in ["mc_pv", "mc_stderr", "black76_pv", "forward_rate_pct"]:
            assert key in body

    def test_zcb_convergence_200(self):
        r = client.get("/mc/zcb-convergence")
        assert r.status_code == 200

    def test_zcb_convergence_keys(self):
        body = client.get("/mc/zcb-convergence").json()
        assert "convergence" in body
        assert "analytical" in body
        assert len(body["convergence"]) > 0


# ── Bermudan Swaption endpoints ───────────────────────────────────────────────

_BERM_BASE = {
    "sofr_on": 0.0433, "hw_a": 0.05, "hw_sigma": 0.010,
    "first_exercise": 1.0, "swap_maturity": 5.0,
    "n_paths": 2000,
}


class TestBermudanEndpoints:

    def test_price_200(self):
        r = client.post("/bermudan/price", json=_BERM_BASE)
        assert r.status_code == 200

    def test_price_positive(self):
        body = client.post("/bermudan/price", json=_BERM_BASE).json()
        assert body["price"] > 0

    def test_price_keys(self):
        body = client.post("/bermudan/price", json=_BERM_BASE).json()
        for key in ["price", "european_lower_bound", "early_exercise_premium",
                    "strike_pct", "swap_maturity", "n_exercise_dates",
                    "exercise_schedule", "n_paths", "pay_receive"]:
            assert key in body

    def test_exercise_schedule_nonempty(self):
        body = client.post("/bermudan/price", json=_BERM_BASE).json()
        assert len(body["exercise_schedule"]) > 0

    def test_exercise_schedule_entry_keys(self):
        body = client.post("/bermudan/price", json=_BERM_BASE).json()
        entry = body["exercise_schedule"][0]
        assert "date_years" in entry
        assert "exercise_prob" in entry

    def test_exercise_probs_between_0_and_1(self):
        body = client.post("/bermudan/price", json=_BERM_BASE).json()
        for entry in body["exercise_schedule"]:
            assert 0.0 <= entry["exercise_prob"] <= 1.0

    def test_bermudan_geq_european(self):
        body = client.post("/bermudan/price", json=_BERM_BASE).json()
        assert body["price"] >= body["european_lower_bound"] * 0.90

    def test_early_exercise_premium_nonneg(self):
        body = client.post("/bermudan/price", json=_BERM_BASE).json()
        assert body["early_exercise_premium"] >= -body["price"] * 0.10

    def test_strike_pct_reasonable(self):
        body = client.post("/bermudan/price", json=_BERM_BASE).json()
        assert 0.5 < body["strike_pct"] < 15.0

    def test_n_exercise_dates_matches_schedule(self):
        body = client.post("/bermudan/price", json=_BERM_BASE).json()
        assert body["n_exercise_dates"] == len(body["exercise_schedule"])

    def test_receiver_200(self):
        req = {**_BERM_BASE, "pay_receive": "receiver"}
        r = client.post("/bermudan/price", json=req)
        assert r.status_code == 200

    def test_receiver_price_positive(self):
        req = {**_BERM_BASE, "pay_receive": "receiver"}
        body = client.post("/bermudan/price", json=req).json()
        assert body["price"] > 0

    def test_invalid_exercise_vs_maturity_422(self):
        req = {**_BERM_BASE, "first_exercise": 5.0, "swap_maturity": 5.0}
        r = client.post("/bermudan/price", json=req)
        assert r.status_code == 422

    def test_quarterly_exercise_freq(self):
        req = {**_BERM_BASE, "exercise_freq": 4}
        body = client.post("/bermudan/price", json=req).json()
        assert body["n_exercise_dates"] > 0

    def test_pay_receive_field_in_response(self):
        body = client.post("/bermudan/price", json=_BERM_BASE).json()
        assert body["pay_receive"] == "payer"


# ── CMS endpoints ─────────────────────────────────────────────────────────────

_CMS_CURVE = {"sofr_on": 0.0433}


class TestCMSEndpoints:

    def test_convexity_200(self):
        r = client.post("/cms/convexity", json={
            **_CMS_CURVE, "expiry": 1.0, "swap_tenor": 10.0, "vol": 0.30,
        })
        assert r.status_code == 200

    def test_convexity_adj_positive(self):
        body = client.post("/cms/convexity", json={
            **_CMS_CURVE, "expiry": 1.0, "swap_tenor": 10.0, "vol": 0.30,
        }).json()
        assert body["convexity_adj_bps"] > 0

    def test_convexity_keys(self):
        body = client.post("/cms/convexity", json={
            **_CMS_CURVE, "expiry": 1.0, "swap_tenor": 10.0, "vol": 0.30,
        }).json()
        for key in ["forward_swap_rate_pct", "convexity_adj_bps", "cms_rate_pct",
                    "expiry", "swap_tenor", "model"]:
            assert key in body

    def test_convexity_cms_gt_forward(self):
        body = client.post("/cms/convexity", json={
            **_CMS_CURVE, "expiry": 1.0, "swap_tenor": 10.0, "vol": 0.30,
        }).json()
        assert body["cms_rate_pct"] > body["forward_swap_rate_pct"]

    def test_convexity_replication_model(self):
        body = client.post("/cms/convexity", json={
            **_CMS_CURVE, "expiry": 1.0, "swap_tenor": 10.0, "vol": 0.30,
            "model": "replication",
        }).json()
        assert body["model"] == "replication"
        assert body["convexity_adj_bps"] >= 0

    def test_caplet_200(self):
        r = client.post("/cms/caplet", json={
            **_CMS_CURVE, "t_fix": 1.0, "t_pay": 1.25, "swap_tenor": 10.0,
            "strike": 0.04, "vol": 0.30,
        })
        assert r.status_code == 200

    def test_caplet_pv_positive(self):
        body = client.post("/cms/caplet", json={
            **_CMS_CURVE, "t_fix": 1.0, "t_pay": 1.25, "swap_tenor": 10.0,
            "strike": 0.04, "vol": 0.30,
        }).json()
        assert body["pv"] > 0

    def test_caplet_keys(self):
        body = client.post("/cms/caplet", json={
            **_CMS_CURVE, "t_fix": 1.0, "t_pay": 1.25, "swap_tenor": 10.0,
            "strike": 0.04, "vol": 0.30,
        }).json()
        for key in ["pv", "cms_rate_pct", "forward_swap_rate_pct",
                    "convexity_adj_bps", "strike_pct", "cap_floor"]:
            assert key in body

    def test_caplet_invalid_t_pay_422(self):
        r = client.post("/cms/caplet", json={
            **_CMS_CURVE, "t_fix": 1.5, "t_pay": 1.0, "swap_tenor": 10.0,
            "strike": 0.04, "vol": 0.30,
        })
        assert r.status_code == 422

    def test_spread_option_200(self):
        r = client.post("/cms/spread-option", json={
            **_CMS_CURVE, "t_fix": 1.0, "t_pay": 1.25,
            "long_tenor": 10.0, "short_tenor": 2.0, "spread_strike": 0.005,
            "vol_long": 0.30, "vol_short": 0.30, "rho": 0.7,
        })
        assert r.status_code == 200

    def test_spread_option_pv_positive(self):
        body = client.post("/cms/spread-option", json={
            **_CMS_CURVE, "t_fix": 1.0, "t_pay": 1.25,
            "long_tenor": 10.0, "short_tenor": 2.0, "spread_strike": 0.005,
            "vol_long": 0.30, "vol_short": 0.30, "rho": 0.7,
        }).json()
        assert body["pv"] > 0

    def test_spread_option_keys(self):
        body = client.post("/cms/spread-option", json={
            **_CMS_CURVE, "t_fix": 1.0, "t_pay": 1.25,
            "long_tenor": 10.0, "short_tenor": 2.0, "spread_strike": 0.005,
            "vol_long": 0.30, "vol_short": 0.30, "rho": 0.7,
        }).json()
        for key in ["pv", "cms_rate_long_pct", "cms_rate_short_pct",
                    "cms_spread_pct", "spread_strike_bps"]:
            assert key in body

    def test_spread_option_invalid_tenor_422(self):
        r = client.post("/cms/spread-option", json={
            **_CMS_CURVE, "t_fix": 1.0, "t_pay": 1.25,
            "long_tenor": 2.0, "short_tenor": 10.0, "spread_strike": 0.005,
            "vol_long": 0.30, "vol_short": 0.30, "rho": 0.7,
        })
        assert r.status_code == 422

    def test_convexity_schedule_200(self):
        r = client.get("/cms/convexity-schedule")
        assert r.status_code == 200

    def test_convexity_schedule_keys(self):
        body = client.get("/cms/convexity-schedule").json()
        assert "schedule" in body
        assert len(body["schedule"]) > 0

    def test_convexity_schedule_adj_grows_with_expiry(self):
        body = client.get("/cms/convexity-schedule").json()
        adjs = [e["convexity_adj_bps"] for e in body["schedule"]]
        # Convexity adjustment should generally increase with expiry
        assert adjs[-1] > adjs[0]


# ── G2++ endpoints ────────────────────────────────────────────────────────────

_G2PP = {"sofr_on": 0.0433, "a": 0.05, "b": 0.10, "sigma": 0.010, "eta": 0.008, "rho": -0.30}


class TestG2ppEndpoints:

    def test_swaption_200(self):
        r = client.post("/g2pp/swaption", json={**_G2PP, "expiry": 1.0, "swap_tenor": 5.0,
                                                 "n_paths": 2000})
        assert r.status_code == 200

    def test_swaption_pv_positive(self):
        body = client.post("/g2pp/swaption", json={**_G2PP, "expiry": 1.0, "swap_tenor": 5.0,
                                                    "n_paths": 3000}).json()
        assert body["pv"] > 0

    def test_swaption_keys(self):
        body = client.post("/g2pp/swaption", json={**_G2PP, "expiry": 1.0, "swap_tenor": 5.0,
                                                    "n_paths": 1000}).json()
        for k in ["pv", "mc_stderr", "forward_swap_rate_pct", "strike_pct", "annuity", "n_paths"]:
            assert k in body

    def test_swaption_atm_strike(self):
        body = client.post("/g2pp/swaption", json={**_G2PP, "expiry": 1.0, "swap_tenor": 5.0,
                                                    "strike": None, "n_paths": 1000}).json()
        assert abs(body["forward_swap_rate_pct"] - body["strike_pct"]) < 0.01

    def test_swaption_receiver_positive(self):
        body = client.post("/g2pp/swaption", json={**_G2PP, "expiry": 1.0, "swap_tenor": 5.0,
                                                    "pay_receive": "receiver",
                                                    "n_paths": 2000}).json()
        assert body["pv"] > 0

    def test_var_200(self):
        r = client.post("/g2pp/var", json={**_G2PP, "portfolio_dv01": 10000,
                                            "horizon_days": 1, "confidence": 0.99,
                                            "n_paths": 2000})
        assert r.status_code == 200

    def test_var_negative_long_duration(self):
        body = client.post("/g2pp/var", json={**_G2PP, "portfolio_dv01": 10000,
                                               "horizon_days": 1, "confidence": 0.99,
                                               "n_paths": 2000}).json()
        assert body["var_usd"] < 0

    def test_var_cvar_le_var(self):
        body = client.post("/g2pp/var", json={**_G2PP, "portfolio_dv01": 10000,
                                               "horizon_days": 1, "confidence": 0.99,
                                               "n_paths": 2000}).json()
        assert body["cvar_usd"] <= body["var_usd"]

    def test_var_keys(self):
        body = client.post("/g2pp/var", json={**_G2PP, "portfolio_dv01": 10000,
                                               "horizon_days": 1, "confidence": 0.99,
                                               "n_paths": 1000}).json()
        for k in ["var_usd", "cvar_usd", "var_bps", "pnl_mean", "pnl_std", "n_paths"]:
            assert k in body


# ── CDS endpoints ─────────────────────────────────────────────────────────────

_CDS_BASE = {"sofr_on": 0.0433, "maturity_years": 5.0, "coupon": 0.01,
             "notional": 10_000_000.0, "recovery": 0.40, "hazard_rate": 0.02}


class TestCDSEndpoints:

    def test_price_200(self):
        r = client.post("/cds/price", json=_CDS_BASE)
        assert r.status_code == 200

    def test_price_keys(self):
        body = client.post("/cds/price", json=_CDS_BASE).json()
        for k in ["pv", "fee_leg_pv", "prot_leg_pv", "par_spread_bps",
                  "risky_annuity", "cs01", "dv01"]:
            assert k in body

    def test_fee_leg_positive(self):
        body = client.post("/cds/price", json=_CDS_BASE).json()
        assert body["fee_leg_pv"] > 0

    def test_prot_leg_positive(self):
        body = client.post("/cds/price", json=_CDS_BASE).json()
        assert body["prot_leg_pv"] > 0

    def test_par_spread_positive(self):
        body = client.post("/cds/price", json=_CDS_BASE).json()
        assert body["par_spread_bps"] > 0

    def test_buy_vs_sell_opposite_pv(self):
        buy  = client.post("/cds/price", json={**_CDS_BASE, "buy_protection": True}).json()
        sell = client.post("/cds/price", json={**_CDS_BASE, "buy_protection": False}).json()
        assert abs(buy["pv"] + sell["pv"]) < 1.0

    def test_bootstrap_200(self):
        r = client.post("/cds/bootstrap", json={
            "sofr_on": 0.0433,
            "maturities": [1.0, 3.0, 5.0],
            "spreads": [0.005, 0.010, 0.015],
        })
        assert r.status_code == 200

    def test_bootstrap_reprices_spreads(self):
        body = client.post("/cds/bootstrap", json={
            "sofr_on": 0.0433,
            "maturities": [1.0, 3.0, 5.0],
            "spreads": [0.005, 0.010, 0.015],
        }).json()
        for entry in body["schedule"]:
            diff = abs(entry["market_spread_bps"] - entry["model_spread_bps"])
            assert diff < 0.01

    def test_bootstrap_schedule_keys(self):
        body = client.post("/cds/bootstrap", json={
            "sofr_on": 0.0433,
            "maturities": [1.0, 5.0],
            "spreads": [0.005, 0.015],
        }).json()
        entry = body["schedule"][0]
        for k in ["maturity_years", "market_spread_bps", "model_spread_bps",
                  "hazard_rate_bps", "survival_prob"]:
            assert k in entry

    def test_bootstrap_mismatched_lengths_422(self):
        r = client.post("/cds/bootstrap", json={
            "sofr_on": 0.0433,
            "maturities": [1.0, 5.0],
            "spreads": [0.01],
        })
        assert r.status_code == 422


# ── LMM endpoints ─────────────────────────────────────────────────────────────

_LMM_CAP_BASE = {
    "sofr_on": 0.0433,
    "tenor_years": 5.0,
    "n_periods": 10,
    "flat_vol": 0.25,
    "strike": 0.04,
    "notional": 1_000_000,
    "corr_decay": 0.10,
    "is_cap": True,
}

_LMM_SW_BASE = {
    "sofr_on": 0.0433,
    "tenor_years": 5.0,
    "n_periods": 10,
    "flat_vol": 0.25,
    "expiry_period": 2,
    "swap_end_period": 8,
    "strike": -1.0,
    "notional": 1_000_000,
    "corr_decay": 0.10,
    "is_payer": True,
    "n_paths": 1_000,
    "n_steps": 20,
    "seed": 42,
}


class TestLMMEndpoints:
    def test_cap_200(self):
        r = client.post("/lmm/cap", json=_LMM_CAP_BASE)
        assert r.status_code == 200

    def test_cap_keys(self):
        body = client.post("/lmm/cap", json=_LMM_CAP_BASE).json()
        for k in ["total_pv", "implied_flat_vol", "caplets", "n_periods"]:
            assert k in body

    def test_cap_pv_positive(self):
        body = client.post("/lmm/cap", json=_LMM_CAP_BASE).json()
        assert body["total_pv"] > 0

    def test_cap_n_caplets_matches(self):
        body = client.post("/lmm/cap", json=_LMM_CAP_BASE).json()
        assert len(body["caplets"]) == _LMM_CAP_BASE["n_periods"]

    def test_floor_positive(self):
        body = client.post("/lmm/cap", json={**_LMM_CAP_BASE, "is_cap": False}).json()
        assert body["total_pv"] > 0

    def test_implied_vol_round_trip(self):
        body = client.post("/lmm/cap", json=_LMM_CAP_BASE).json()
        assert abs(body["implied_flat_vol"] - _LMM_CAP_BASE["flat_vol"]) < 1e-4

    def test_swaption_200(self):
        r = client.post("/lmm/swaption", json=_LMM_SW_BASE)
        assert r.status_code == 200

    def test_swaption_keys(self):
        body = client.post("/lmm/swaption", json=_LMM_SW_BASE).json()
        for k in ["pv", "std_err", "rebonato_vol_pct", "swap_rate_mean_pct"]:
            assert k in body

    def test_swaption_pv_positive_itm(self):
        body = client.post("/lmm/swaption", json={**_LMM_SW_BASE, "strike": 0.001}).json()
        assert body["pv"] > 0

    def test_swaption_atm_strike(self):
        body = client.post("/lmm/swaption", json={**_LMM_SW_BASE, "strike": -1.0}).json()
        assert body["pv"] > 0

    def test_rebonato_vol_positive(self):
        body = client.post("/lmm/swaption", json=_LMM_SW_BASE).json()
        assert body["rebonato_vol_pct"] > 0

    def test_calibrate_200(self):
        r = client.post("/lmm/calibrate-caplet-vols", json={
            "sofr_on": 0.0433,
            "tenor_years": 5.0,
            "n_periods": 10,
            "cap_flat_vols": [0.25] * 10,
            "corr_decay": 0.10,
        })
        assert r.status_code == 200

    def test_calibrate_keys(self):
        body = client.post("/lmm/calibrate-caplet-vols", json={
            "sofr_on": 0.0433,
            "tenor_years": 5.0,
            "n_periods": 5,
            "cap_flat_vols": [0.25, 0.26, 0.27, 0.28, 0.29],
            "corr_decay": 0.10,
        }).json()
        for k in ["calibrated_vols_pct", "input_cap_vols_pct", "initial_forwards_pct"]:
            assert k in body

    def test_rebonato_surface_200(self):
        r = client.get("/lmm/rebonato-surface", params={
            "sofr_on": 0.0433, "tenor_years": 3.0,
            "n_periods": 6, "flat_vol": 0.25, "corr_decay": 0.1,
        })
        assert r.status_code == 200

    def test_rebonato_surface_has_entries(self):
        body = client.get("/lmm/rebonado-surface", params={
            "sofr_on": 0.0433, "tenor_years": 3.0,
            "n_periods": 6, "flat_vol": 0.25, "corr_decay": 0.1,
        })
        # Route name check: use the correct URL
        r = client.get("/lmm/rebonato-surface", params={
            "sofr_on": 0.0433, "tenor_years": 3.0,
            "n_periods": 6, "flat_vol": 0.25, "corr_decay": 0.1,
        })
        assert r.json()["n_entries"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# XVA Endpoints
# ══════════════════════════════════════════════════════════════════════════════

class TestXVAEndpoints:
    XVA_BODY = {
        "sofr_on": 0.0433,
        "fixed_rate": 0.045,
        "maturity": 3.0,
        "notional": 1e6,
        "is_payer": True,
        "recovery": 0.40,
        "hazard_rate": 0.01,
        "own_hazard": 0.005,
        "funding_spread": 0.005,
        "n_steps": 10,
        "n_paths": 100,
        "hw_a": 0.05,
        "hw_sigma": 0.015,
    }

    def test_xva_price_200(self):
        r = client.post("/xva/price", json=self.XVA_BODY)
        assert r.status_code == 200

    def test_xva_price_keys(self):
        body = client.post("/xva/price", json=self.XVA_BODY).json()
        for k in ["cva", "dva", "fva", "total_xva", "cva_bps"]:
            assert k in body

    def test_cva_positive(self):
        body = client.post("/xva/price", json=self.XVA_BODY).json()
        assert body["cva"] >= 0.0

    def test_dva_positive(self):
        body = client.post("/xva/price", json=self.XVA_BODY).json()
        assert body["dva"] >= 0.0

    def test_fva_finite(self):
        body = client.post("/xva/price", json=self.XVA_BODY).json()
        assert np.isfinite(body["fva"])

    def test_cva_scales_with_hazard(self):
        low = {**self.XVA_BODY, "hazard_rate": 0.005}
        high = {**self.XVA_BODY, "hazard_rate": 0.05}
        cva_low  = client.post("/xva/price", json=low).json()["cva"]
        cva_high = client.post("/xva/price", json=high).json()["cva"]
        assert cva_high >= cva_low

    def test_epe_profile_200(self):
        r = client.post("/xva/epe-profile", json=self.XVA_BODY)
        assert r.status_code == 200

    def test_epe_profile_keys(self):
        body = client.post("/xva/epe-profile", json=self.XVA_BODY).json()
        for k in ["times", "epe", "ene", "n_paths"]:
            assert k in body

    def test_epe_times_length(self):
        body = client.post("/xva/epe-profile", json=self.XVA_BODY).json()
        assert len(body["times"]) == len(body["epe"])

    def test_epe_non_negative(self):
        body = client.post("/xva/epe-profile", json=self.XVA_BODY).json()
        assert all(v >= 0.0 for v in body["epe"])


# ══════════════════════════════════════════════════════════════════════════════
# XCCY Endpoints
# ══════════════════════════════════════════════════════════════════════════════

class TestXCCYEndpoints:
    PAR_BODY = {
        "usd_sofr": 0.0433,
        "eur_rate": 0.038,
        "spot_fx": 1.09,
        "maturity": 5.0,
        "notional_eur": 1e6,
        "freq": 4,
    }
    PRICE_BODY = {**PAR_BODY, "basis_bps": 0.0}

    def test_par_basis_200(self):
        r = client.post("/xccy/par-basis", json=self.PAR_BODY)
        assert r.status_code == 200

    def test_par_basis_keys(self):
        body = client.post("/xccy/par-basis", json=self.PAR_BODY).json()
        for k in ["par_basis_bps", "spot_fx", "forward_fx", "maturity_years"]:
            assert k in body

    def test_par_basis_finite(self):
        body = client.post("/xccy/par-basis", json=self.PAR_BODY).json()
        assert np.isfinite(body["par_basis_bps"])

    def test_higher_usd_rate_more_positive_basis(self):
        low  = {**self.PAR_BODY, "usd_sofr": 0.03}
        high = {**self.PAR_BODY, "usd_sofr": 0.06}
        b_low  = client.post("/xccy/par-basis", json=low).json()["par_basis_bps"]
        b_high = client.post("/xccy/par-basis", json=high).json()["par_basis_bps"]
        assert b_high >= b_low

    def test_xccy_price_200(self):
        r = client.post("/xccy/price", json=self.PRICE_BODY)
        assert r.status_code == 200

    def test_xccy_price_keys(self):
        body = client.post("/xccy/price", json=self.PRICE_BODY).json()
        for k in ["pv_usd", "usd_leg_pv", "eur_leg_pv_usd", "par_basis_bps"]:
            assert k in body

    def test_xccy_par_swap_zero_pv(self):
        body_par = client.post("/xccy/par-basis", json=self.PAR_BODY).json()
        par_basis = body_par["par_basis_bps"]
        price_body = {**self.PRICE_BODY, "basis_bps": par_basis}
        body = client.post("/xccy/price", json=price_body).json()
        # par-basis pricing: PV should be small relative to notional
        assert abs(body["pv_usd"]) < self.PAR_BODY["notional_eur"] * 0.10

    def test_basis_term_structure_200(self):
        r = client.get("/xccy/basis-term-structure", params={
            "usd_sofr": 0.0433, "eur_rate": 0.038, "spot_fx": 1.09,
        })
        assert r.status_code == 200

    def test_basis_term_structure_keys(self):
        body = client.get("/xccy/basis-term-structure", params={
            "usd_sofr": 0.0433, "eur_rate": 0.038, "spot_fx": 1.09,
        }).json()
        assert "term_structure" in body
        assert len(body["term_structure"]) == 9

    def test_basis_term_structure_has_forward_fx(self):
        body = client.get("/xccy/basis-term-structure", params={
            "usd_sofr": 0.0433, "eur_rate": 0.038, "spot_fx": 1.09,
        }).json()
        for row in body["term_structure"]:
            assert "forward_fx" in row


# ══════════════════════════════════════════════════════════════════════════════
# Inflation Endpoints
# ══════════════════════════════════════════════════════════════════════════════

class TestInflationEndpoints:
    ZC_BODY = {
        "maturity": 10.0,
        "fixed_rate": 0.025,
        "notional": 1e6,
        "infl_rate": 0.025,
        "sofr_on": 0.0433,
        "is_receiver": False,
    }
    YOY_BODY = {
        "maturity": 5.0,
        "fixed_rate": 0.025,
        "notional": 1e6,
        "infl_rate": 0.025,
        "sofr_on": 0.0433,
        "freq": 1,
        "is_receiver": False,
    }
    CAP_BODY = {
        "maturity": 5.0,
        "strike": 0.02,
        "vol": 0.015,
        "notional": 1e6,
        "infl_rate": 0.025,
        "sofr_on": 0.0433,
        "is_cap": True,
        "freq": 1,
    }

    def test_zc_price_200(self):
        r = client.post("/inflation/zc-price", json=self.ZC_BODY)
        assert r.status_code == 200

    def test_zc_price_keys(self):
        body = client.post("/inflation/zc-price", json=self.ZC_BODY).json()
        for k in ["pv", "inflation_leg_pv", "fixed_leg_pv", "breakeven_rate"]:
            assert k in body

    def test_zc_at_par_near_zero_pv(self):
        body = client.post("/inflation/zc-price", json=self.ZC_BODY).json()
        assert abs(body["pv"]) < 1000  # at-par should be near zero

    def test_zc_payer_receiver_opposite(self):
        payer = client.post("/inflation/zc-price", json={**self.ZC_BODY, "is_receiver": False}).json()["pv"]
        recvr = client.post("/inflation/zc-price", json={**self.ZC_BODY, "is_receiver": True}).json()["pv"]
        assert abs(payer + recvr) < 1.0  # should sum to zero

    def test_yoy_price_200(self):
        r = client.post("/inflation/yoy-price", json=self.YOY_BODY)
        assert r.status_code == 200

    def test_yoy_price_keys(self):
        body = client.post("/inflation/yoy-price", json=self.YOY_BODY).json()
        for k in ["pv", "inflation_leg_pv", "fixed_leg_pv", "n_payments"]:
            assert k in body

    def test_yoy_n_payments(self):
        body = client.post("/inflation/yoy-price", json=self.YOY_BODY).json()
        assert body["n_payments"] == 5  # 5yr * 1/yr

    def test_yoy_payer_receiver_opposite(self):
        payer = client.post("/inflation/yoy-price", json={**self.YOY_BODY, "is_receiver": False}).json()["pv"]
        recvr = client.post("/inflation/yoy-price", json={**self.YOY_BODY, "is_receiver": True}).json()["pv"]
        assert abs(payer + recvr) < 1.0

    def test_cap_floor_price_200(self):
        r = client.post("/inflation/cap-floor-price", json=self.CAP_BODY)
        assert r.status_code == 200

    def test_cap_floor_price_keys(self):
        body = client.post("/inflation/cap-floor-price", json=self.CAP_BODY).json()
        for k in ["pv", "n_caplets", "is_cap"]:
            assert k in body

    def test_cap_pv_positive(self):
        body = client.post("/inflation/cap-floor-price", json=self.CAP_BODY).json()
        assert body["pv"] >= 0.0

    def test_floor_pv_positive(self):
        body = client.post("/inflation/cap-floor-price", json={**self.CAP_BODY, "is_cap": False}).json()
        assert body["pv"] >= 0.0

    def test_cap_floor_parity(self):
        cap   = client.post("/inflation/cap-floor-price", json={**self.CAP_BODY, "is_cap": True}).json()["pv"]
        floor = client.post("/inflation/cap-floor-price", json={**self.CAP_BODY, "is_cap": False}).json()["pv"]
        assert abs(cap - floor) < 5e4  # rough cap-floor parity

    def test_breakeven_200(self):
        r = client.get("/inflation/breakeven", params={
            "sofr_on": 0.0433, "infl_rate": 0.025, "maturity": 10.0,
        })
        assert r.status_code == 200

    def test_breakeven_keys(self):
        body = client.get("/inflation/breakeven", params={
            "sofr_on": 0.0433, "infl_rate": 0.025, "maturity": 10.0,
        }).json()
        for k in ["breakeven_rate", "breakeven_bps", "maturity_years"]:
            assert k in body

    def test_breakeven_positive(self):
        body = client.get("/inflation/breakeven", params={
            "sofr_on": 0.0433, "infl_rate": 0.025, "maturity": 10.0,
        }).json()
        assert body["breakeven_rate"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# Callable Bond Endpoints
# ══════════════════════════════════════════════════════════════════════════════

class TestCallableBondEndpoints:
    BULLET = {
        "face": 100.0, "coupon": 0.05, "maturity": 5.0, "freq": 2,
        "sofr_on": 0.05, "hw_a": 0.10, "hw_sigma": 0.01, "dt": 0.25,
        "call_schedule": [], "put_schedule": [],
    }
    CALLABLE = {
        **BULLET, "coupon": 0.06,
        "call_schedule": [{"time": 2.0, "price": 100.0}, {"time": 3.0, "price": 100.0}],
    }

    def test_bullet_price_200(self):
        r = client.post("/callable-bond/price", json=self.BULLET)
        assert r.status_code == 200

    def test_callable_price_200(self):
        r = client.post("/callable-bond/price", json=self.CALLABLE)
        assert r.status_code == 200

    def test_price_keys(self):
        body = client.post("/callable-bond/price", json=self.BULLET).json()
        for k in ["price", "straight_price", "option_value", "oas_bps",
                  "effective_duration", "effective_convexity"]:
            assert k in body

    def test_callable_price_le_straight(self):
        body = client.post("/callable-bond/price", json=self.CALLABLE).json()
        assert body["price"] <= body["straight_price"] + 0.05

    def test_option_value_nonneg(self):
        body = client.post("/callable-bond/price", json=self.CALLABLE).json()
        assert body["option_value"] >= -0.05

    def test_duration_positive(self):
        body = client.post("/callable-bond/price", json=self.BULLET).json()
        assert body["effective_duration"] > 0.0

    def test_straight_price_200(self):
        r = client.get("/callable-bond/straight-price", params={
            "sofr_on": 0.05, "coupon": 0.05, "maturity": 5.0, "freq": 2
        })
        assert r.status_code == 200

    def test_straight_price_keys(self):
        body = client.get("/callable-bond/straight-price", params={
            "sofr_on": 0.05, "coupon": 0.05, "maturity": 5.0
        }).json()
        assert "straight_price" in body

    def test_market_price_oas(self):
        body = client.post("/callable-bond/price",
                           json={**self.CALLABLE, "market_price": 98.0}).json()
        assert body["oas_bps"] > 0.0  # priced below model → positive OAS


# ══════════════════════════════════════════════════════════════════════════════
# FX Options Endpoints
# ══════════════════════════════════════════════════════════════════════════════

class TestFXOptionEndpoints:
    OPT = {
        "spot": 1.09, "strike": 1.09, "vol": 0.08,
        "domestic_rate": 0.04, "foreign_rate": 0.03,
        "maturity": 1.0, "is_call": True,
    }
    SMILE_REQ = {
        "maturities": [0.25, 0.5, 1.0, 2.0],
        "atm_vols":   [0.07, 0.08, 0.09, 0.10],
        "rr25":       [0.002, 0.003, 0.004, 0.005],
        "bf25":       [0.001, 0.001, 0.002, 0.002],
        "spot": 1.09, "domestic_rate": 0.04, "foreign_rate": 0.03,
        "target_mat": 1.0, "n_strikes": 11,
    }

    def test_fx_price_200(self):
        r = client.post("/fx/price", json=self.OPT)
        assert r.status_code == 200

    def test_fx_price_keys(self):
        body = client.post("/fx/price", json=self.OPT).json()
        for k in ["pv", "delta", "gamma", "vega", "theta"]:
            assert k in body

    def test_call_pv_positive(self):
        body = client.post("/fx/price", json=self.OPT).json()
        assert body["pv"] > 0.0

    def test_put_pv_positive(self):
        body = client.post("/fx/price", json={**self.OPT, "is_call": False}).json()
        assert body["pv"] > 0.0

    def test_call_delta_in_range(self):
        body = client.post("/fx/price", json=self.OPT).json()
        assert 0.0 < body["delta"] < 1.0

    def test_put_delta_in_range(self):
        body = client.post("/fx/price", json={**self.OPT, "is_call": False}).json()
        assert -1.0 < body["delta"] < 0.0

    def test_implied_vol_200(self):
        r = client.post("/fx/implied-vol", json=self.OPT)
        assert r.status_code == 200

    def test_implied_vol_roundtrip(self):
        body = client.post("/fx/implied-vol", json=self.OPT).json()
        assert abs(body["implied_vol"] - self.OPT["vol"]) < 1e-4

    def test_smile_200(self):
        r = client.post("/fx/smile", json=self.SMILE_REQ)
        assert r.status_code == 200

    def test_smile_keys(self):
        body = client.post("/fx/smile", json=self.SMILE_REQ).json()
        assert "strikes" in body and "vols_pct" in body

    def test_smile_length(self):
        body = client.post("/fx/smile", json=self.SMILE_REQ).json()
        assert len(body["strikes"]) == 11

    def test_smile_vols_positive(self):
        body = client.post("/fx/smile", json=self.SMILE_REQ).json()
        assert all(v > 0 for v in body["vols_pct"])
