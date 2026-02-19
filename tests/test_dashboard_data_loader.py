"""Tests for portfolio-dashboard data_loader module."""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# --- プロジェクトルートを sys.path に追加 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DASHBOARD_SCRIPTS = PROJECT_ROOT / ".claude" / "skills" / "portfolio-dashboard" / "scripts"
if str(DASHBOARD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_SCRIPTS))

from components.data_loader import (
    _reconstruct_daily_holdings,
    get_monthly_summary,
    get_sector_breakdown,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_trades():
    """売買履歴サンプル."""
    return [
        {
            "category": "trade",
            "date": "2026-02-16",
            "timestamp": "2026-02-16T10:00:00",
            "symbol": "7203.T",
            "trade_type": "buy",
            "shares": 100,
            "price": 2850.0,
            "currency": "JPY",
        },
        {
            "category": "trade",
            "date": "2026-02-16",
            "timestamp": "2026-02-16T10:01:00",
            "symbol": "AAPL",
            "trade_type": "buy",
            "shares": 10,
            "price": 230.0,
            "currency": "USD",
        },
        {
            "category": "trade",
            "date": "2026-02-18",
            "timestamp": "2026-02-18T10:00:00",
            "symbol": "7203.T",
            "trade_type": "buy",
            "shares": 50,
            "price": 2900.0,
            "currency": "JPY",
        },
        {
            "category": "trade",
            "date": "2026-02-19",
            "timestamp": "2026-02-19T10:00:00",
            "symbol": "AAPL",
            "trade_type": "sell",
            "shares": 5,
            "price": 240.0,
            "currency": "USD",
        },
    ]


@pytest.fixture
def sample_price_history():
    """株価履歴のサンプル DataFrame."""
    dates = pd.date_range("2026-02-14", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [2800, 2850, 2880, 2900, 2920],
            "High": [2860, 2900, 2930, 2950, 2960],
            "Low": [2780, 2830, 2860, 2880, 2900],
            "Close": [2850, 2880, 2900, 2930, 2940],
            "Volume": [1000000] * 5,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Tests for _reconstruct_daily_holdings
# ---------------------------------------------------------------------------

class TestReconstructDailyHoldings:
    def test_single_buy(self):
        trades = [
            {"date": "2026-02-16", "symbol": "7203.T", "trade_type": "buy", "shares": 100,
             "timestamp": "2026-02-16T10:00:00"},
        ]
        result = _reconstruct_daily_holdings(trades)
        assert "2026-02-16" in result
        assert result["2026-02-16"]["7203.T"] == 100

    def test_buy_then_sell(self):
        trades = [
            {"date": "2026-02-16", "symbol": "7203.T", "trade_type": "buy", "shares": 100,
             "timestamp": "2026-02-16T10:00:00"},
            {"date": "2026-02-18", "symbol": "7203.T", "trade_type": "sell", "shares": 50,
             "timestamp": "2026-02-18T10:00:00"},
        ]
        result = _reconstruct_daily_holdings(trades)
        assert result["2026-02-16"]["7203.T"] == 100
        assert result["2026-02-18"]["7203.T"] == 50

    def test_sell_all_removes_symbol(self):
        trades = [
            {"date": "2026-02-16", "symbol": "AAPL", "trade_type": "buy", "shares": 10,
             "timestamp": "2026-02-16T10:00:00"},
            {"date": "2026-02-18", "symbol": "AAPL", "trade_type": "sell", "shares": 10,
             "timestamp": "2026-02-18T10:00:00"},
        ]
        result = _reconstruct_daily_holdings(trades)
        assert "AAPL" not in result["2026-02-18"]

    def test_multiple_symbols(self, sample_trades):
        result = _reconstruct_daily_holdings(sample_trades)
        # 2/16: 7203.T=100, AAPL=10
        assert result["2026-02-16"]["7203.T"] == 100
        assert result["2026-02-16"]["AAPL"] == 10
        # 2/18: 7203.T=150
        assert result["2026-02-18"]["7203.T"] == 150
        # 2/19: AAPL=5
        assert result["2026-02-19"]["AAPL"] == 5

    def test_empty_trades(self):
        result = _reconstruct_daily_holdings([])
        assert result == {}

    def test_sell_more_than_held_clamps_to_zero(self):
        trades = [
            {"date": "2026-02-16", "symbol": "AAPL", "trade_type": "buy", "shares": 5,
             "timestamp": "2026-02-16T10:00:00"},
            {"date": "2026-02-18", "symbol": "AAPL", "trade_type": "sell", "shares": 10,
             "timestamp": "2026-02-18T10:00:00"},
        ]
        result = _reconstruct_daily_holdings(trades)
        assert "AAPL" not in result["2026-02-18"]


# ---------------------------------------------------------------------------
# Tests for get_sector_breakdown
# ---------------------------------------------------------------------------

class TestGetSectorBreakdown:
    def test_basic(self):
        snapshot = {
            "positions": [
                {"sector": "Technology", "evaluation_jpy": 1000000},
                {"sector": "Technology", "evaluation_jpy": 500000},
                {"sector": "Consumer", "evaluation_jpy": 300000},
            ]
        }
        df = get_sector_breakdown(snapshot)
        assert len(df) == 2  # 2 sectors
        tech_row = df[df["sector"] == "Technology"]
        assert tech_row["evaluation_jpy"].values[0] == 1500000

    def test_empty_positions(self):
        snapshot = {"positions": []}
        df = get_sector_breakdown(snapshot)
        assert df.empty

    def test_unknown_sector(self):
        snapshot = {
            "positions": [
                {"evaluation_jpy": 100000},  # no sector key
            ]
        }
        df = get_sector_breakdown(snapshot)
        assert df["sector"].values[0] == "Unknown"


# ---------------------------------------------------------------------------
# Tests for get_monthly_summary
# ---------------------------------------------------------------------------

class TestGetMonthlySummary:
    def test_basic(self):
        dates = pd.date_range("2025-12-01", periods=90, freq="B")
        df = pd.DataFrame({"total": range(100, 100 + len(dates))}, index=dates)
        monthly = get_monthly_summary(df)
        assert "month_end_value_jpy" in monthly.columns
        assert "change_pct" in monthly.columns
        assert len(monthly) >= 2  # at least 2 months

    def test_empty_input(self):
        df = pd.DataFrame()
        monthly = get_monthly_summary(df)
        assert monthly.empty

    def test_single_month(self):
        dates = pd.date_range("2026-02-01", periods=10, freq="B")
        df = pd.DataFrame({"total": [1000000] * 10}, index=dates)
        monthly = get_monthly_summary(df)
        assert len(monthly) == 1
