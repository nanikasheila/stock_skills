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
    _compute_invested_capital,
    _build_trade_activity,
    _load_cached_prices,
    _save_prices_cache,
    _load_prices,
    _CACHE_TTL_SECONDS,
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

    def test_with_invested_column(self):
        dates = pd.date_range("2025-12-01", periods=90, freq="B")
        df = pd.DataFrame({
            "total": range(100, 100 + len(dates)),
            "invested": range(90, 90 + len(dates)),
        }, index=dates)
        monthly = get_monthly_summary(df)
        assert "invested_jpy" in monthly.columns
        assert "unrealized_pnl" in monthly.columns


# ---------------------------------------------------------------------------
# Tests for _compute_invested_capital
# ---------------------------------------------------------------------------

class TestComputeInvestedCapital:
    def test_single_buy_jpy(self):
        trades = [
            {"date": "2026-02-16", "trade_type": "buy", "shares": 100,
             "price": 2850.0, "currency": "JPY"},
        ]
        result = _compute_invested_capital(trades, {"JPY": 1.0})
        assert result["2026-02-16"] == pytest.approx(285000.0)

    def test_buy_and_sell(self):
        trades = [
            {"date": "2026-01-01", "trade_type": "buy", "shares": 100,
             "price": 1000.0, "currency": "JPY"},
            {"date": "2026-02-01", "trade_type": "sell", "shares": 50,
             "price": 1200.0, "currency": "JPY"},
        ]
        result = _compute_invested_capital(trades, {"JPY": 1.0})
        # 100*1000 = 100000, then - 50*1200 = -60000 → 40000
        assert result["2026-02-01"] == pytest.approx(40000.0)

    def test_transfer_treated_as_buy(self):
        trades = [
            {"date": "2026-01-01", "trade_type": "transfer", "shares": 50,
             "price": 500.0, "currency": "JPY"},
        ]
        result = _compute_invested_capital(trades, {"JPY": 1.0})
        assert result["2026-01-01"] == pytest.approx(25000.0)

    def test_usd_with_fx(self):
        trades = [
            {"date": "2026-01-01", "trade_type": "buy", "shares": 10,
             "price": 230.0, "currency": "USD"},
        ]
        fx = {"USD": 150.0}
        result = _compute_invested_capital(trades, fx)
        assert result["2026-01-01"] == pytest.approx(10 * 230 * 150)

    def test_empty_trades(self):
        result = _compute_invested_capital([], {"JPY": 1.0})
        assert result == {}

    def test_clamp_to_zero(self):
        trades = [
            {"date": "2026-01-01", "trade_type": "buy", "shares": 10,
             "price": 100.0, "currency": "JPY"},
            {"date": "2026-02-01", "trade_type": "sell", "shares": 100,
             "price": 200.0, "currency": "JPY"},
        ]
        result = _compute_invested_capital(trades, {"JPY": 1.0})
        assert result["2026-02-01"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests for _build_trade_activity
# ---------------------------------------------------------------------------

class TestBuildTradeActivity:
    def test_basic(self):
        trades = [
            {"date": "2026-01-15", "trade_type": "buy", "shares": 100,
             "price": 1000.0, "currency": "JPY"},
            {"date": "2026-01-20", "trade_type": "buy", "shares": 50,
             "price": 2000.0, "currency": "JPY"},
            {"date": "2026-01-25", "trade_type": "sell", "shares": 30,
             "price": 1500.0, "currency": "JPY"},
            {"date": "2026-02-10", "trade_type": "buy", "shares": 20,
             "price": 500.0, "currency": "JPY"},
        ]
        result = _build_trade_activity(trades, {"JPY": 1.0})
        assert "2026-01" in result.index
        assert "2026-02" in result.index
        jan = result.loc["2026-01"]
        assert jan["buy_count"] == 2
        assert jan["buy_amount"] == pytest.approx(200000.0)
        assert jan["sell_count"] == 1
        assert jan["sell_amount"] == pytest.approx(45000.0)
        assert jan["net_flow"] == pytest.approx(155000.0)

    def test_empty_trades(self):
        result = _build_trade_activity([], {"JPY": 1.0})
        assert result.empty

    def test_transfer_counted_as_buy(self):
        trades = [
            {"date": "2026-03-01", "trade_type": "transfer", "shares": 10,
             "price": 300.0, "currency": "JPY"},
        ]
        result = _build_trade_activity(trades, {"JPY": 1.0})
        assert result.loc["2026-03"]["buy_count"] == 1

    def test_fx_conversion(self):
        trades = [
            {"date": "2026-01-01", "trade_type": "buy", "shares": 10,
             "price": 100.0, "currency": "USD"},
        ]
        result = _build_trade_activity(trades, {"USD": 150.0})
        assert result.loc["2026-01"]["buy_amount"] == pytest.approx(150000.0)


# ---------------------------------------------------------------------------
# Tests for transfer handling in _reconstruct_daily_holdings
# ---------------------------------------------------------------------------

class TestTransferHandling:
    def test_transfer_adds_shares(self):
        trades = [
            {"date": "2026-01-01", "symbol": "VTI", "trade_type": "transfer",
             "shares": 50, "timestamp": "2026-01-01T10:00:00"},
        ]
        result = _reconstruct_daily_holdings(trades)
        assert result["2026-01-01"]["VTI"] == 50

    def test_transfer_then_buy(self):
        trades = [
            {"date": "2026-01-01", "symbol": "VTI", "trade_type": "transfer",
             "shares": 50, "timestamp": "2026-01-01T10:00:00"},
            {"date": "2026-01-10", "symbol": "VTI", "trade_type": "buy",
             "shares": 30, "timestamp": "2026-01-10T10:00:00"},
        ]
        result = _reconstruct_daily_holdings(trades)
        assert result["2026-01-10"]["VTI"] == 80


# ---------------------------------------------------------------------------
# Tests for price cache
# ---------------------------------------------------------------------------

class TestPriceCache:
    """ディスクキャッシュの保存・読み込み・TTL テスト."""

    @pytest.fixture
    def cache_dir(self, tmp_path, monkeypatch):
        """キャッシュディレクトリを一時ディレクトリに差し替える."""
        import components.data_loader as dl
        monkeypatch.setattr(dl, "_PRICE_CACHE_DIR", tmp_path)
        return tmp_path

    def test_save_and_load(self, cache_dir):
        """保存したキャッシュを正しく読み込める."""
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        prices = pd.DataFrame(
            {"AAPL": [150, 151, 152, 153, 154],
             "MSFT": [400, 401, 402, 403, 404]},
            index=dates,
        )
        _save_prices_cache(prices, "3mo")
        loaded = _load_cached_prices("3mo")
        assert loaded is not None
        assert list(loaded.columns) == ["AAPL", "MSFT"]
        assert len(loaded) == 5

    def test_cache_ttl_expired(self, cache_dir, monkeypatch):
        """TTL 超過時は None を返す."""
        import components.data_loader as dl
        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        prices = pd.DataFrame({"AAPL": [150, 151, 152]}, index=dates)
        _save_prices_cache(prices, "1mo")

        # TTL を 0 秒に設定して必ず期限切れにする
        monkeypatch.setattr(dl, "_CACHE_TTL_SECONDS", 0)
        import time; time.sleep(0.1)
        loaded = _load_cached_prices("1mo")
        assert loaded is None

    def test_cache_miss_returns_none(self, cache_dir):
        """キャッシュファイルが無い場合は None."""
        loaded = _load_cached_prices("nonexistent")
        assert loaded is None

    def test_load_prices_full_cache_hit(self, cache_dir, monkeypatch):
        """全銘柄がキャッシュにある場合、API 呼び出しなしで返す."""
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        prices = pd.DataFrame(
            {"AAPL": [150, 151, 152, 153, 154],
             "MSFT": [400, 401, 402, 403, 404]},
            index=dates,
        )
        _save_prices_cache(prices, "3mo")

        # API が呼ばれないことを確認
        import components.data_loader as dl
        mock_batch = MagicMock(return_value=None)
        monkeypatch.setattr(dl.yahoo_client, "get_close_prices_batch", mock_batch)

        result = _load_prices(["AAPL", "MSFT"], "3mo")
        assert not result.empty
        assert list(result.columns) == ["AAPL", "MSFT"]
        mock_batch.assert_not_called()

    def test_load_prices_partial_cache(self, cache_dir, monkeypatch):
        """キャッシュに一部銘柄のみ → 不足分だけ API 取得."""
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        cached = pd.DataFrame({"AAPL": [150, 151, 152, 153, 154]}, index=dates)
        _save_prices_cache(cached, "3mo")

        new_data = pd.DataFrame({"MSFT": [400, 401, 402, 403, 404]}, index=dates)
        import components.data_loader as dl
        mock_batch = MagicMock(return_value=new_data)
        monkeypatch.setattr(dl.yahoo_client, "get_close_prices_batch", mock_batch)

        result = _load_prices(["AAPL", "MSFT"], "3mo")
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns
        # MSFT だけ取得された
        mock_batch.assert_called_once_with(["MSFT"], period="3mo")

    def test_load_prices_cache_miss_batch(self, cache_dir, monkeypatch):
        """キャッシュなし → バッチ取得して保存."""
        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        batch_result = pd.DataFrame(
            {"AAPL": [150, 151, 152], "MSFT": [400, 401, 402]},
            index=dates,
        )
        import components.data_loader as dl
        mock_batch = MagicMock(return_value=batch_result)
        monkeypatch.setattr(dl.yahoo_client, "get_close_prices_batch", mock_batch)

        result = _load_prices(["AAPL", "MSFT"], "1mo")
        assert not result.empty
        mock_batch.assert_called_once()

        # キャッシュが保存されたか確認
        reloaded = _load_cached_prices("1mo")
        assert reloaded is not None
        assert "AAPL" in reloaded.columns

    def test_load_prices_batch_fail_falls_back(self, cache_dir, monkeypatch):
        """バッチ失敗時 → 個別フォールバック."""
        import components.data_loader as dl
        mock_batch = MagicMock(return_value=None)
        monkeypatch.setattr(dl.yahoo_client, "get_close_prices_batch", mock_batch)

        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        individual_df = pd.DataFrame({"AAPL": [150, 151, 152]}, index=dates)
        mock_fetch = MagicMock(return_value=individual_df)
        monkeypatch.setattr(dl, "_fetch_price_history", mock_fetch)

        result = _load_prices(["AAPL"], "3mo")
        assert not result.empty
        mock_batch.assert_called_once()
        mock_fetch.assert_called_once()
