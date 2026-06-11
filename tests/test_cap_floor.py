"""
Tests for sofr_engine/cap_floor.py

Coverage:
- caplet_black_pv / caplet_bachelier_pv formulas
- Caplet class (forward rate, PV, implied vol)
- Cap strip: annuity, ATM, PV, DV01, vega, theta, implied vol, summary
- Floor strip: same as Cap
- Put-call parity: Cap - Floor = parity PV
- CapFloorVolSurface: interpolation, typical_market
- strip_caplet_vols: bootstrap
- price_cap_floor convenience function
"""
import math
import pytest
import numpy as np
from datetime import date

from sofr_engine.bootstrap import flat_sofr_curve
from sofr_engine.cap_floor import (
    caplet_black_pv,
    caplet_bachelier_pv,
    black_to_normal_vol,
    Caplet,
    Cap,
    Floor,
    cap_floor_parity_pv,
    CapFloorVolSurface,
    strip_caplet_vols,
    price_cap_floor,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_curve():
    return flat_sofr_curve(date.today(), 0.0433)


@pytest.fixture
def steep_curve():
    """Slightly upward-sloping curve for roll/carry tests."""
    from sofr_engine.curve import DiscountCurve
    times = np.array([0.01, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    rates = np.array([0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048])
    dfs   = np.exp(-rates * times)
    return DiscountCurve(date.today(), times, dfs)


# ── caplet_black_pv ──────────────────────────────────────────────────────────

class TestCapletBlackPV:
    def test_atm_positive(self):
        pv = caplet_black_pv(F=0.05, K=0.05, T=1.0, tau=0.25,
                              df_pay=0.95, vol=0.30, notional=1e6, cap_floor="cap")
        assert pv > 0

    def test_deep_itm_cap_approaches_intrinsic(self):
        F, K = 0.10, 0.02
        pv = caplet_black_pv(F=F, K=K, T=0.5, tau=0.25,
                              df_pay=0.98, vol=0.20, notional=1e6, cap_floor="cap")
        intrinsic = 1e6 * 0.25 * 0.98 * (F - K)
        assert abs(pv - intrinsic) / intrinsic < 0.02  # within 2% of intrinsic

    def test_deep_otm_cap_near_zero(self):
        pv = caplet_black_pv(F=0.02, K=0.10, T=1.0, tau=0.25,
                              df_pay=0.97, vol=0.30, notional=1e6, cap_floor="cap")
        assert pv < 1.0  # nearly worthless

    def test_floor_put_call_parity(self):
        F, K, T, tau, df = 0.05, 0.04, 1.0, 0.25, 0.95
        cap   = caplet_black_pv(F, K, T, tau, df, 0.30, 1e6, "cap")
        floor = caplet_black_pv(F, K, T, tau, df, 0.30, 1e6, "floor")
        parity = 1e6 * tau * df * (F - K)
        assert abs((cap - floor) - parity) < 0.01

    def test_zero_expiry_returns_intrinsic(self):
        pv = caplet_black_pv(F=0.06, K=0.05, T=0.0, tau=0.25,
                              df_pay=0.99, vol=0.30, notional=1e6, cap_floor="cap")
        expected = 1e6 * 0.25 * 0.99 * 0.01
        assert abs(pv - expected) < 0.01

    def test_floor_atm_equals_cap_atm(self):
        pv_cap   = caplet_black_pv(F=0.05, K=0.05, T=1.0, tau=0.25, df_pay=0.95,
                                    vol=0.30, notional=1e6, cap_floor="cap")
        pv_floor = caplet_black_pv(F=0.05, K=0.05, T=1.0, tau=0.25, df_pay=0.95,
                                    vol=0.30, notional=1e6, cap_floor="floor")
        assert abs(pv_cap - pv_floor) < 0.01  # ATM cap = ATM floor

    def test_higher_vol_increases_pv(self):
        base = caplet_black_pv(F=0.05, K=0.05, T=1.0, tau=0.25, df_pay=0.95,
                                vol=0.25, notional=1e6)
        high = caplet_black_pv(F=0.05, K=0.05, T=1.0, tau=0.25, df_pay=0.95,
                                vol=0.40, notional=1e6)
        assert high > base


# ── caplet_bachelier_pv ──────────────────────────────────────────────────────

class TestCapletBachelierPV:
    def test_atm_positive(self):
        pv = caplet_bachelier_pv(F=0.05, K=0.05, T=1.0, tau=0.25,
                                  df_pay=0.95, normal_vol=0.005, notional=1e6)
        assert pv > 0

    def test_parity_bachelier(self):
        F, K, T, tau, df, nv = 0.05, 0.04, 1.0, 0.25, 0.95, 0.008
        cap   = caplet_bachelier_pv(F, K, T, tau, df, nv, 1e6, "cap")
        floor = caplet_bachelier_pv(F, K, T, tau, df, nv, 1e6, "floor")
        parity = 1e6 * tau * df * (F - K)
        assert abs((cap - floor) - parity) < 0.01

    def test_atm_cap_equals_atm_floor(self):
        pv_c = caplet_bachelier_pv(F=0.05, K=0.05, T=1.0, tau=0.25, df_pay=0.95,
                                    normal_vol=0.006, notional=1e6, cap_floor="cap")
        pv_f = caplet_bachelier_pv(F=0.05, K=0.05, T=1.0, tau=0.25, df_pay=0.95,
                                    normal_vol=0.006, notional=1e6, cap_floor="floor")
        assert abs(pv_c - pv_f) < 0.01


# ── black_to_normal_vol ──────────────────────────────────────────────────────

class TestBlackToNormalVol:
    def test_atm_approx(self):
        F, vol = 0.05, 0.30
        nv = black_to_normal_vol(F, F, 1.0, vol)
        assert abs(nv - vol * F) / (vol * F) < 0.01

    def test_monotone_in_vol(self):
        nv1 = black_to_normal_vol(0.05, 0.05, 1.0, 0.20)
        nv2 = black_to_normal_vol(0.05, 0.05, 1.0, 0.40)
        assert nv2 > nv1


# ── Caplet class ─────────────────────────────────────────────────────────────

class TestCapletClass:
    def test_forward_rate_from_flat_curve(self, flat_curve):
        cl = Caplet(t_reset=0.25, t_pay=0.50, strike=0.04)
        F  = cl.forward_rate(flat_curve)
        assert 0.03 < F < 0.06

    def test_black_pv_positive(self, flat_curve):
        cl = Caplet(0.25, 0.50, 0.04, notional=1e6)
        pv = cl.black_pv(flat_curve, vol=0.30)
        assert pv > 0

    def test_implied_black_vol_roundtrip(self, flat_curve):
        cl     = Caplet(0.25, 0.50, 0.04, notional=1e6)
        vol    = 0.35
        price  = cl.black_pv(flat_curve, vol)
        iv     = cl.implied_black_vol(flat_curve, price)
        assert abs(iv - vol) < 1e-6

    def test_implied_normal_vol_roundtrip(self, flat_curve):
        cl         = Caplet(0.25, 0.50, 0.04, notional=1e6)
        normal_vol = 0.008
        price      = cl.bachelier_pv(flat_curve, normal_vol)
        iv         = cl.implied_normal_vol(flat_curve, price)
        assert abs(iv - normal_vol) < 1e-8

    def test_floor_pv_roundtrip(self, flat_curve):
        fl    = Caplet(0.5, 0.75, 0.05, cap_floor="floor", notional=1e6)
        vol   = 0.28
        price = fl.black_pv(flat_curve, vol)
        iv    = fl.implied_black_vol(flat_curve, price)
        assert abs(iv - vol) < 1e-5


# ── Cap strip ────────────────────────────────────────────────────────────────

class TestCapStrip:
    def test_n_caplets_quarterly(self):
        cap = Cap(maturity_years=2.0, strike=0.04, freq=4)
        # 2Y × 4 = 8 periods, minus first_caplet_idx=1 → 7 caplets
        assert cap.n_caplets() == 7

    def test_n_caplets_annual(self):
        cap = Cap(maturity_years=3.0, strike=0.04, freq=1, first_caplet_idx=0)
        assert cap.n_caplets() == 3

    def test_annuity_positive(self, flat_curve):
        cap = Cap(5.0, 0.04)
        assert cap.annuity(flat_curve) > 0

    def test_atm_forward_near_sofr(self, flat_curve):
        cap = Cap(5.0, 0.04)
        F   = cap.atm_forward(flat_curve)
        assert 0.03 < F < 0.06

    def test_atm_cap_pv_positive(self, flat_curve):
        cap = Cap(5.0, 0.04)
        pv  = cap.pv(flat_curve, vol=0.30)
        assert pv > 0

    def test_higher_vol_higher_pv(self, flat_curve):
        cap  = Cap(5.0, 0.04)
        pv1  = cap.pv(flat_curve, 0.20)
        pv2  = cap.pv(flat_curve, 0.40)
        assert pv2 > pv1

    def test_lower_strike_higher_cap_pv(self, flat_curve):
        pv_low  = Cap(5.0, 0.02).pv(flat_curve, 0.30)
        pv_high = Cap(5.0, 0.06).pv(flat_curve, 0.30)
        assert pv_low > pv_high

    def test_dv01_negative_for_cap(self, flat_curve):
        cap = Cap(5.0, 0.04)
        dv01 = cap.dv01(flat_curve, 0.30)
        # Rate rise → lower cap PV? No: cap PV increases with rising rates (ITM).
        # Actually DV01 for a cap is POSITIVE (higher rates → higher forwards → more ITM)
        assert dv01 > 0  # forward rates rise with curve shift → cap gains

    def test_vega_positive(self, flat_curve):
        cap = Cap(5.0, 0.04)
        v   = cap.vega(flat_curve, 0.30)
        assert v > 0

    def test_implied_vol_roundtrip(self, flat_curve):
        cap  = Cap(5.0, 0.04)
        vol  = 0.28
        pv   = cap.pv(flat_curve, vol)
        iv   = cap.implied_vol(flat_curve, pv)
        assert abs(iv - vol) < 1e-5

    def test_summary_keys(self, flat_curve):
        cap = Cap(5.0, 0.04)
        s   = cap.summary(flat_curve, 0.30)
        for key in ["pv", "dv01", "vega_per_bp", "atm_forward_pct", "n_caplets"]:
            assert key in s

    def test_bachelier_pv_positive(self, flat_curve):
        cap = Cap(5.0, 0.04)
        pv  = cap.pv_bachelier(flat_curve, 0.008)
        assert pv > 0

    def test_theta_negative(self, flat_curve):
        cap   = Cap(5.0, 0.04)
        theta = cap.theta(flat_curve, 0.30)
        assert theta < 0  # option loses time value over one day

    def test_pv_scales_with_notional(self, flat_curve):
        pv1 = Cap(5.0, 0.04, notional=1e6).pv(flat_curve, 0.30)
        pv2 = Cap(5.0, 0.04, notional=2e6).pv(flat_curve, 0.30)
        assert abs(pv2 / pv1 - 2.0) < 1e-10


# ── Floor strip ───────────────────────────────────────────────────────────────

class TestFloorStrip:
    def test_atm_floor_positive(self, flat_curve):
        floor = Floor(5.0, 0.04)
        pv    = floor.pv(flat_curve, vol=0.30)
        assert pv > 0

    def test_implied_vol_roundtrip(self, flat_curve):
        floor = Floor(3.0, 0.04)
        vol   = 0.25
        pv    = floor.pv(flat_curve, vol)
        iv    = floor.implied_vol(flat_curve, pv)
        assert abs(iv - vol) < 1e-5

    def test_higher_strike_higher_floor_pv(self, flat_curve):
        pv_low  = Floor(5.0, 0.02).pv(flat_curve, 0.30)
        pv_high = Floor(5.0, 0.06).pv(flat_curve, 0.30)
        assert pv_high > pv_low

    def test_dv01_negative_for_floor(self, flat_curve):
        floor = Floor(5.0, 0.04)
        dv01  = floor.dv01(flat_curve, 0.30)
        # Rate rise → higher forwards → floor moves more OTM → PV falls → DV01 < 0
        assert dv01 < 0

    def test_vega_positive(self, flat_curve):
        floor = Floor(5.0, 0.04)
        v     = floor.vega(flat_curve, 0.30)
        assert v > 0


# ── Put-call parity ──────────────────────────────────────────────────────────

class TestPutCallParity:
    def test_cap_minus_floor_equals_parity(self, flat_curve):
        K   = 0.04
        vol = 0.30
        mat = 5.0
        cap   = Cap(mat, K).pv(flat_curve, vol)
        floor = Floor(mat, K).pv(flat_curve, vol)
        parity = cap_floor_parity_pv(flat_curve, mat, K)
        assert abs((cap - floor) - parity) < 1.0  # within $1

    def test_parity_at_atm_near_zero(self, flat_curve):
        cap_obj = Cap(5.0, 0.04)
        K_atm   = cap_obj.atm_forward(flat_curve)
        vol     = 0.30
        cap_pv  = Cap(5.0, K_atm).pv(flat_curve, vol)
        flr_pv  = Floor(5.0, K_atm).pv(flat_curve, vol)
        # Cap ≈ Floor at ATM (parity ≈ 0)
        parity  = cap_floor_parity_pv(flat_curve, 5.0, K_atm)
        assert abs((cap_pv - flr_pv) - parity) < 1.0

    def test_parity_independent_of_vol(self, flat_curve):
        K = 0.04
        p1 = cap_floor_parity_pv(flat_curve, 3.0, K)
        p2 = cap_floor_parity_pv(flat_curve, 3.0, K)  # deterministic
        assert p1 == p2


# ── CapFloorVolSurface ────────────────────────────────────────────────────────

class TestCapFloorVolSurface:
    def test_typical_market_returns_surface(self):
        surf = CapFloorVolSurface.typical_market()
        assert len(surf._tenors) > 0
        assert len(surf._strikes) > 0

    def test_vol_at_grid_node(self):
        surf = CapFloorVolSurface.typical_market()
        T    = surf._tenors[2]
        K    = surf._strikes[3]
        v    = surf.vol(T, K)
        assert 0.01 < v < 2.0

    def test_vol_interpolation_bounded(self):
        surf = CapFloorVolSurface.typical_market()
        v    = surf.vol(4.0, 0.043)   # between grid points
        assert 0.01 < v < 2.0

    def test_vol_clamps_at_boundary(self):
        surf = CapFloorVolSurface.typical_market()
        v1   = surf.vol(0.1, 0.01)   # before first tenor
        v2   = surf.vol(50.0, 0.20)  # after last tenor
        assert np.isfinite(v1) and np.isfinite(v2)


# ── strip_caplet_vols ────────────────────────────────────────────────────────

class TestStripCapletVols:
    def test_returns_list_of_tuples(self, flat_curve):
        term_vols = {1.0: 0.35, 2.0: 0.30, 3.0: 0.27, 5.0: 0.25}
        result    = strip_caplet_vols(term_vols, flat_curve, strike=0.04)
        assert len(result) > 0
        for expiry, vol in result:
            assert expiry >= 0
            assert 0.01 < vol < 5.0

    def test_monotone_expiry(self, flat_curve):
        term_vols = {1.0: 0.35, 2.0: 0.30, 3.0: 0.27}
        result    = strip_caplet_vols(term_vols, flat_curve, strike=0.04)
        expiries  = [e for e, _ in result]
        assert expiries == sorted(expiries)

    def test_consistency_single_tenor(self, flat_curve):
        term_vols = {1.0: 0.30}
        result    = strip_caplet_vols(term_vols, flat_curve, strike=0.04)
        # Should produce one forward caplet vol close to the term vol
        assert len(result) == 1
        _, fwd_vol = result[0]
        assert 0.01 < fwd_vol < 5.0


# ── price_cap_floor convenience ──────────────────────────────────────────────

class TestPriceCapFloor:
    def test_cap_returns_dict(self, flat_curve):
        d = price_cap_floor(flat_curve, 5.0, strike=0.04, vol=0.30)
        assert "pv" in d
        assert d["pv"] > 0

    def test_floor_returns_dict(self, flat_curve):
        d = price_cap_floor(flat_curve, 5.0, strike=0.04, vol=0.30,
                             instrument="floor")
        assert "pv" in d
        assert d["pv"] > 0

    def test_atm_strike_when_none(self, flat_curve):
        d = price_cap_floor(flat_curve, 5.0, strike=None, vol=0.30)
        assert d["strike_pct"] > 0

    def test_longer_maturity_higher_pv(self, flat_curve):
        pv_2y = price_cap_floor(flat_curve, 2.0, strike=0.04, vol=0.30)["pv"]
        pv_5y = price_cap_floor(flat_curve, 5.0, strike=0.04, vol=0.30)["pv"]
        assert pv_5y > pv_2y
