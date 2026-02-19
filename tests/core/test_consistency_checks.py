"""Tests for valuation consistency checks in indicators.py."""

import pytest

from src.core.indicators import (
    check_eps_direction,
    check_growth_consistency,
    check_margin_deterioration,
    check_quarterly_eps_trend,
    run_consistency_checks,
)

# ---------------------------------------------------------------------------
# check_eps_direction
# ---------------------------------------------------------------------------


class TestCheckEpsDirection:
    """Tests for FwdEPS vs TrailEPS direction check."""

    def test_decline_detected(self):
        """FwdEPS < TrailEPS should trigger a warning."""
        stock = {"forward_eps": 151.7, "eps_current": 162.6}
        result = check_eps_direction(stock)
        assert result is not None
        assert result["code"] == "EPS_DECLINE"
        assert result["level"] == "warning"
        assert result["growth_pct"] < 0
        assert "減益" in result["message"]

    def test_growth_no_warning(self):
        """FwdEPS > TrailEPS should return None."""
        stock = {"forward_eps": 4.24, "eps_current": 2.75}
        result = check_eps_direction(stock)
        assert result is None

    def test_equal_eps_no_warning(self):
        """FwdEPS == TrailEPS should return None (not declining)."""
        stock = {"forward_eps": 100.0, "eps_current": 100.0}
        result = check_eps_direction(stock)
        assert result is None

    def test_missing_forward_eps(self):
        """Missing FwdEPS should return None."""
        stock = {"eps_current": 100.0}
        result = check_eps_direction(stock)
        assert result is None

    def test_missing_trailing_eps(self):
        """Missing TrailEPS should return None."""
        stock = {"forward_eps": 100.0}
        result = check_eps_direction(stock)
        assert result is None

    def test_zero_trailing_eps(self):
        """Zero TrailEPS should return None (avoid division by zero)."""
        stock = {"forward_eps": 100.0, "eps_current": 0}
        result = check_eps_direction(stock)
        assert result is None


# ---------------------------------------------------------------------------
# check_growth_consistency
# ---------------------------------------------------------------------------


class TestCheckGrowthConsistency:
    """Tests for PEG consistency between past growth and forward EPS."""

    def test_inconsistency_detected(self):
        """Past growth positive but forward decline should trigger warning."""
        stock = {
            "earnings_growth": 0.602,  # +60.2%
            "forward_eps": 151.7,
            "eps_current": 162.6,
            "forward_per": 24.5,
        }
        result = check_growth_consistency(stock)
        assert result is not None
        assert result["code"] == "PEG_INCONSISTENCY"
        assert "PEG矛盾" in result["message"]
        assert result["peg_past_based"] is not None

    def test_consistent_growth_no_warning(self):
        """Both past and forward growth positive should return None."""
        stock = {
            "earnings_growth": 0.20,
            "forward_eps": 120.0,
            "eps_current": 100.0,
            "forward_per": 20.0,
        }
        result = check_growth_consistency(stock)
        assert result is None

    def test_both_declining_no_peg_warning(self):
        """Past negative growth and forward decline: no PEG inconsistency."""
        stock = {
            "earnings_growth": -0.10,  # past was already declining
            "forward_eps": 90.0,
            "eps_current": 100.0,
            "forward_per": 15.0,
        }
        result = check_growth_consistency(stock)
        assert result is None

    def test_small_past_growth_no_warning(self):
        """Small past growth (<5%) with slight forward decline: no warning."""
        stock = {
            "earnings_growth": 0.03,  # only 3%
            "forward_eps": 98.0,
            "eps_current": 100.0,
            "forward_per": 20.0,
        }
        result = check_growth_consistency(stock)
        assert result is None

    def test_missing_data_returns_none(self):
        """Missing earnings_growth should return None."""
        stock = {"forward_eps": 100.0, "eps_current": 100.0}
        result = check_growth_consistency(stock)
        assert result is None

    def test_negative_trailing_eps_recovery_no_warning(self):
        """TrailEPS negative -> FwdEPS positive is recovery, not decline."""
        # COHR case: trailing -0.52, forward 7.34, earningsGrowth +73%
        stock = {
            "earnings_growth": 0.73,
            "forward_eps": 7.34,
            "eps_current": -0.52,
            "forward_per": 30.0,
        }
        result = check_growth_consistency(stock)
        assert result is None  # recovery, not a PEG inconsistency


# ---------------------------------------------------------------------------
# check_margin_deterioration
# ---------------------------------------------------------------------------


class TestCheckMarginDeterioration:
    """Tests for gross margin deterioration check."""

    def test_large_decline_detected(self):
        """Gross margin drop >= 5pt should trigger warning."""
        stock = {"gross_margins_history": [0.405, 0.558]}  # 40.5% from 55.8%
        result = check_margin_deterioration(stock)
        assert result is not None
        assert result["code"] == "MARGIN_DETERIORATION"
        assert result["delta_pt"] < -5.0
        assert "粗利率悪化" in result["message"]

    def test_small_decline_no_warning(self):
        """Gross margin drop < 5pt should not trigger warning."""
        stock = {"gross_margins_history": [0.54, 0.558]}  # only -1.8pt
        result = check_margin_deterioration(stock)
        assert result is None

    def test_improvement_no_warning(self):
        """Gross margin improvement should not trigger warning."""
        stock = {"gross_margins_history": [0.60, 0.55]}
        result = check_margin_deterioration(stock)
        assert result is None

    def test_no_history(self):
        """No gross margins history should return None."""
        stock = {}
        result = check_margin_deterioration(stock)
        assert result is None

    def test_single_period(self):
        """Single period history should return None."""
        stock = {"gross_margins_history": [0.40]}
        result = check_margin_deterioration(stock)
        assert result is None


# ---------------------------------------------------------------------------
# check_quarterly_eps_trend
# ---------------------------------------------------------------------------


class TestCheckQuarterlyEpsTrend:
    """Tests for quarterly EPS QoQ deceleration check."""

    def test_decline_detected(self):
        """QoQ EPS decline should trigger caution."""
        stock = {"quarterly_eps": [0.68, 0.71]}  # latest < previous
        result = check_quarterly_eps_trend(stock)
        assert result is not None
        assert result["code"] == "EPS_DECELERATION"
        assert result["level"] == "caution"
        assert result["change_pct"] < 0
        assert "鈍化" in result["message"]

    def test_growth_no_warning(self):
        """QoQ EPS growth should return None."""
        stock = {"quarterly_eps": [0.75, 0.71]}
        result = check_quarterly_eps_trend(stock)
        assert result is None

    def test_no_data(self):
        """No quarterly_eps should return None."""
        stock = {}
        result = check_quarterly_eps_trend(stock)
        assert result is None

    def test_single_quarter(self):
        """Single quarter should return None."""
        stock = {"quarterly_eps": [0.68]}
        result = check_quarterly_eps_trend(stock)
        assert result is None

    def test_zero_previous(self):
        """Zero previous EPS should return None (avoid division by zero)."""
        stock = {"quarterly_eps": [0.68, 0]}
        result = check_quarterly_eps_trend(stock)
        assert result is None


# ---------------------------------------------------------------------------
# run_consistency_checks
# ---------------------------------------------------------------------------


class TestRunConsistencyChecks:
    """Tests for the combined consistency check runner."""

    def test_no_warnings_for_healthy_stock(self):
        """A healthy stock should produce no warnings."""
        stock = {
            "forward_eps": 120.0,
            "eps_current": 100.0,
            "earnings_growth": 0.20,
            "forward_per": 20.0,
            "gross_margins_history": [0.55, 0.54],
            "quarterly_eps": [0.75, 0.71],
        }
        warnings = run_consistency_checks(stock)
        assert warnings == []

    def test_multiple_warnings(self):
        """A problematic stock should produce multiple warnings."""
        stock = {
            "forward_eps": 151.7,
            "eps_current": 162.6,
            "earnings_growth": 0.602,
            "forward_per": 24.5,
            "gross_margins_history": [0.405, 0.558],
            "quarterly_eps": [0.68, 0.71],
        }
        warnings = run_consistency_checks(stock)
        codes = [w["code"] for w in warnings]
        assert "EPS_DECLINE" in codes
        assert "PEG_INCONSISTENCY" in codes
        assert "MARGIN_DETERIORATION" in codes
        assert "EPS_DECELERATION" in codes

    def test_empty_stock(self):
        """An empty stock dict should produce no warnings."""
        warnings = run_consistency_checks({})
        assert warnings == []

    def test_partial_data(self):
        """Only EPS decline present should produce one warning."""
        stock = {
            "forward_eps": 90.0,
            "eps_current": 100.0,
        }
        warnings = run_consistency_checks(stock)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "EPS_DECLINE"
