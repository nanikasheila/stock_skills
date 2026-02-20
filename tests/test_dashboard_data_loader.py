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
    _compute_realized_pnl,
    _build_trade_activity,
    _load_cached_prices,
    _save_prices_cache,
    _load_prices,
    _CACHE_TTL_SECONDS,
    get_monthly_summary,
    get_sector_breakdown,
    compute_daily_change,
    compute_benchmark_excess,
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


# ---------------------------------------------------------------------------
# Tests for _compute_realized_pnl
# ---------------------------------------------------------------------------

class TestComputeRealizedPnl:
    """実現損益 FIFO 計算のテスト."""

    FX = {"JPY": 1.0, "USD": 150.0}

    def test_simple_buy_sell_profit(self):
        """単純な買い→売りで利益."""
        trades = [
            {"symbol": "7203.T", "trade_type": "buy", "shares": 100,
             "price": 2800.0, "currency": "JPY", "date": "2026-01-01"},
            {"symbol": "7203.T", "trade_type": "sell", "shares": 100,
             "price": 3000.0, "currency": "JPY", "date": "2026-02-01"},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        assert result["total_jpy"] == pytest.approx(20_000.0)
        assert result["by_symbol"]["7203.T"] == pytest.approx(20_000.0)

    def test_simple_buy_sell_loss(self):
        """単純な買い→売りで損失."""
        trades = [
            {"symbol": "AAPL", "trade_type": "buy", "shares": 10,
             "price": 200.0, "currency": "USD", "date": "2026-01-01"},
            {"symbol": "AAPL", "trade_type": "sell", "shares": 10,
             "price": 180.0, "currency": "USD", "date": "2026-02-01"},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        assert result["total_jpy"] == pytest.approx(-30_000.0)

    def test_fifo_order(self):
        """FIFO: 最初のロットから先に消化."""
        trades = [
            {"symbol": "X", "trade_type": "buy", "shares": 50,
             "price": 100.0, "currency": "JPY", "date": "2026-01-01"},
            {"symbol": "X", "trade_type": "buy", "shares": 50,
             "price": 200.0, "currency": "JPY", "date": "2026-01-15"},
            {"symbol": "X", "trade_type": "sell", "shares": 50,
             "price": 150.0, "currency": "JPY", "date": "2026-02-01"},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        assert result["total_jpy"] == pytest.approx(2_500.0)

    def test_partial_lot_sell(self):
        """ロットの一部だけ売却."""
        trades = [
            {"symbol": "X", "trade_type": "buy", "shares": 100,
             "price": 100.0, "currency": "JPY", "date": "2026-01-01"},
            {"symbol": "X", "trade_type": "sell", "shares": 30,
             "price": 120.0, "currency": "JPY", "date": "2026-02-01"},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        assert result["total_jpy"] == pytest.approx(600.0)

    def test_sell_across_lots(self):
        """売却が複数ロットにまたがる."""
        trades = [
            {"symbol": "X", "trade_type": "buy", "shares": 30,
             "price": 100.0, "currency": "JPY", "date": "2026-01-01"},
            {"symbol": "X", "trade_type": "buy", "shares": 70,
             "price": 200.0, "currency": "JPY", "date": "2026-01-10"},
            {"symbol": "X", "trade_type": "sell", "shares": 50,
             "price": 180.0, "currency": "JPY", "date": "2026-02-01"},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        # lot1 30@100, lot2 70@200 → sell 50:
        # 30 from lot1: (180-100)*30=2400, 20 from lot2: (180-200)*20=-400
        assert result["total_jpy"] == pytest.approx(2_000.0)

    def test_no_sells_returns_zero(self):
        """売却がなければ実現損益ゼロ."""
        trades = [
            {"symbol": "X", "trade_type": "buy", "shares": 100,
             "price": 100.0, "currency": "JPY", "date": "2026-01-01"},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        assert result["total_jpy"] == 0.0
        assert len(result["by_symbol"]) == 0

    def test_multiple_symbols(self):
        """複数銘柄の実現損益."""
        trades = [
            {"symbol": "A", "trade_type": "buy", "shares": 10,
             "price": 100.0, "currency": "JPY", "date": "2026-01-01"},
            {"symbol": "B", "trade_type": "buy", "shares": 5,
             "price": 50.0, "currency": "USD", "date": "2026-01-01"},
            {"symbol": "A", "trade_type": "sell", "shares": 10,
             "price": 120.0, "currency": "JPY", "date": "2026-02-01"},
            {"symbol": "B", "trade_type": "sell", "shares": 5,
             "price": 60.0, "currency": "USD", "date": "2026-02-01"},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        assert result["by_symbol"]["A"] == pytest.approx(200.0)
        assert result["by_symbol"]["B"] == pytest.approx(7_500.0)
        assert result["total_jpy"] == pytest.approx(7_700.0)

    def test_cash_excluded(self):
        """CASH銘柄は実現損益計算から除外."""
        trades = [
            {"symbol": "USD.CASH", "trade_type": "buy", "shares": 1,
             "price": 100.0, "currency": "USD", "date": "2026-01-01"},
            {"symbol": "USD.CASH", "trade_type": "sell", "shares": 1,
             "price": 200.0, "currency": "USD", "date": "2026-02-01"},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        assert result["total_jpy"] == 0.0

    def test_transfer_creates_lot(self):
        """入庫 (transfer) with price > 0 も買いロットとして扱う."""
        trades = [
            {"symbol": "X", "trade_type": "transfer", "shares": 100,
             "price": 50.0, "currency": "JPY", "date": "2026-01-01"},
            {"symbol": "X", "trade_type": "sell", "shares": 100,
             "price": 80.0, "currency": "JPY", "date": "2026-02-01"},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        assert result["total_jpy"] == pytest.approx(3_000.0)

    # --- 株式分割 (Stock Split) テスト ---

    def test_stock_split_adjusts_cost_basis(self):
        """transfer(price=0) は株式分割として既存ロットの単価を調整."""
        # IXN のようなケース: 34株 → 204株 (6:1 split)
        trades = [
            {"symbol": "IXN", "trade_type": "buy", "shares": 4,
             "price": 300.0, "currency": "USD", "date": "2021-01-01",
             "settlement_jpy": 120000.0},
            {"symbol": "IXN", "trade_type": "buy", "shares": 30,
             "price": 306.0, "currency": "USD", "date": "2021-02-24",
             "settlement_jpy": 960000.0},
            # 6:1 split → 170 additional shares
            {"symbol": "IXN", "trade_type": "transfer", "shares": 170,
             "price": 0.0, "currency": "USD", "date": "2021-07-19"},
            # Sell all 204 shares at post-split price
            {"symbol": "IXN", "trade_type": "sell", "shares": 204,
             "price": 75.0, "currency": "USD", "date": "2024-03-05",
             "settlement_jpy": 2_295_000.0},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        # Total cost: 120000 + 960000 = 1,080,000 JPY
        # Proceeds: 2,295,000 JPY
        # P&L = 2,295,000 - 1,080,000 = 1,215,000
        assert result["by_symbol"]["IXN"] == pytest.approx(1_215_000.0)

    def test_stock_split_no_existing_lots(self):
        """transfer(price=0) だが既存ロットなし → 無視される."""
        trades = [
            {"symbol": "X", "trade_type": "transfer", "shares": 100,
             "price": 0.0, "currency": "USD", "date": "2021-07-19"},
            {"symbol": "X", "trade_type": "sell", "shares": 100,
             "price": 50.0, "currency": "USD", "date": "2024-01-01"},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        # No lots available → unmatched sell → no P&L
        assert result["total_jpy"] == pytest.approx(0.0)

    def test_split_then_additional_buy(self):
        """分割後に新たに購入 → 新ロットは分割影響なし."""
        trades = [
            {"symbol": "X", "trade_type": "buy", "shares": 10,
             "price": 300.0, "currency": "USD", "date": "2021-01-01",
             "fx_rate": 100.0},  # cost = 10 * 300 * 100 = 300,000
            # 3:1 split → 20 additional shares
            {"symbol": "X", "trade_type": "transfer", "shares": 20,
             "price": 0.0, "currency": "USD", "date": "2021-07-01"},
            # Post-split buy at adjusted price
            {"symbol": "X", "trade_type": "buy", "shares": 10,
             "price": 100.0, "currency": "USD", "date": "2022-01-01",
             "fx_rate": 120.0},  # cost = 10 * 100 * 120 = 120,000
            # Sell all 40 shares
            {"symbol": "X", "trade_type": "sell", "shares": 40,
             "price": 120.0, "currency": "USD", "date": "2024-01-01",
             "fx_rate": 150.0},  # proceeds = 40 * 120 * 150 = 720,000
        ]
        result = _compute_realized_pnl(trades, self.FX)
        # Lot1 after split: 30 shares @ 10,000 JPY/share (300,000/30)
        # Lot2: 10 shares @ 12,000 JPY/share (120,000/10)
        # Sell 40: proceeds = 720,000
        # FIFO: 30 from lot1 (cost 300,000) + 10 from lot2 (cost 120,000)
        # P&L = 720,000 - 420,000 = 300,000
        assert result["by_symbol"]["X"] == pytest.approx(300_000.0)

    # --- 為替レート (FX Rate) テスト ---

    def test_fx_rate_per_trade(self):
        """各取引のfx_rateで換算される（グローバルFXではなく）."""
        trades = [
            {"symbol": "VTI", "trade_type": "buy", "shares": 10,
             "price": 200.0, "currency": "USD", "date": "2021-01-01",
             "fx_rate": 105.0},  # cost = 10 * 200 * 105 = 210,000
            {"symbol": "VTI", "trade_type": "sell", "shares": 10,
             "price": 250.0, "currency": "USD", "date": "2024-01-01",
             "fx_rate": 150.0},  # proceeds = 10 * 250 * 150 = 375,000
        ]
        result = _compute_realized_pnl(trades, self.FX)
        # P&L = 375,000 - 210,000 = 165,000 (includes FX gain)
        assert result["by_symbol"]["VTI"] == pytest.approx(165_000.0)

    def test_settlement_jpy_used_directly(self):
        """settlement_jpyがある場合、直接使用される."""
        trades = [
            {"symbol": "VTI", "trade_type": "buy", "shares": 100,
             "price": 200.0, "currency": "USD", "date": "2021-01-01",
             "settlement_jpy": 2_100_000.0},  # includes fees
            {"symbol": "VTI", "trade_type": "sell", "shares": 100,
             "price": 250.0, "currency": "USD", "date": "2024-01-01",
             "settlement_jpy": 3_700_000.0},  # includes fees
        ]
        result = _compute_realized_pnl(trades, self.FX)
        # P&L = 3,700,000 - 2,100,000 = 1,600,000
        assert result["by_symbol"]["VTI"] == pytest.approx(1_600_000.0)

    def test_settlement_usd_with_fx(self):
        """settlement_usd * fx_rateでJPY換算."""
        trades = [
            {"symbol": "AAPL", "trade_type": "buy", "shares": 10,
             "price": 200.0, "currency": "USD", "date": "2021-01-01",
             "settlement_usd": 2010.0, "fx_rate": 105.0},
            {"symbol": "AAPL", "trade_type": "sell", "shares": 10,
             "price": 250.0, "currency": "USD", "date": "2024-01-01",
             "settlement_usd": 2490.0, "fx_rate": 150.0},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        # Buy cost: 2010 * 105 = 211,050
        # Sell proceeds: 2490 * 150 = 373,500
        # P&L = 373,500 - 211,050 = 162,450
        assert result["by_symbol"]["AAPL"] == pytest.approx(162_450.0)

    def test_backward_compat_no_fx_fields(self):
        """旧形式JSON（fx_rate/settlement未保存）でも動作する."""
        trades = [
            {"symbol": "X", "trade_type": "buy", "shares": 10,
             "price": 100.0, "currency": "USD", "date": "2021-01-01"},
            {"symbol": "X", "trade_type": "sell", "shares": 10,
             "price": 120.0, "currency": "USD", "date": "2024-01-01"},
        ]
        result = _compute_realized_pnl(trades, self.FX)
        # Fallback to global FX (150): P&L = 10*(120-100)*150 = 30,000
        assert result["by_symbol"]["X"] == pytest.approx(30_000.0)


# ---------------------------------------------------------------------------
# Phase 1: compute_daily_change
# ---------------------------------------------------------------------------

class TestComputeDailyChange:
    """前日比計算テスト."""

    def test_normal_change(self):
        """正常な前日比の算出."""
        dates = pd.date_range("2026-02-17", periods=5, freq="B")
        df = pd.DataFrame({"total": [1_000_000, 1_010_000, 1_020_000, 1_015_000, 1_030_000]}, index=dates)
        result = compute_daily_change(df)
        assert result["daily_change_jpy"] == pytest.approx(15_000.0)
        assert result["daily_change_pct"] == pytest.approx(
            (1_030_000 / 1_015_000 - 1) * 100, abs=0.01
        )

    def test_negative_change(self):
        """下落時の前日比."""
        dates = pd.date_range("2026-02-17", periods=3, freq="B")
        df = pd.DataFrame({"total": [1_000_000, 1_050_000, 990_000]}, index=dates)
        result = compute_daily_change(df)
        assert result["daily_change_jpy"] < 0
        assert result["daily_change_pct"] < 0

    def test_empty_df(self):
        """空DataFrame."""
        result = compute_daily_change(pd.DataFrame())
        assert result["daily_change_jpy"] == 0.0
        assert result["daily_change_pct"] == 0.0

    def test_single_row(self):
        """1行のみ（前日なし）."""
        df = pd.DataFrame({"total": [1_000_000]}, index=pd.date_range("2026-02-17", periods=1))
        result = compute_daily_change(df)
        assert result["daily_change_jpy"] == 0.0
        assert result["daily_change_pct"] == 0.0


# ---------------------------------------------------------------------------
# Phase 1: compute_benchmark_excess
# ---------------------------------------------------------------------------

class TestComputeBenchmarkExcess:
    """ベンチマーク超過リターン計算テスト."""

    def test_outperformance(self):
        """PFがベンチマークを上回る場合."""
        dates = pd.date_range("2026-01-01", periods=60, freq="B")
        df = pd.DataFrame({"total": [1_000_000 + i * 5000 for i in range(60)]}, index=dates)
        bench = pd.Series([1_000_000 + i * 3000 for i in range(60)], index=dates)
        result = compute_benchmark_excess(df, bench)
        assert result is not None
        assert result["excess_return_pct"] > 0
        assert result["portfolio_return_pct"] > result["benchmark_return_pct"]

    def test_underperformance(self):
        """PFがベンチマークを下回る場合."""
        dates = pd.date_range("2026-01-01", periods=60, freq="B")
        df = pd.DataFrame({"total": [1_000_000 + i * 2000 for i in range(60)]}, index=dates)
        bench = pd.Series([1_000_000 + i * 5000 for i in range(60)], index=dates)
        result = compute_benchmark_excess(df, bench)
        assert result is not None
        assert result["excess_return_pct"] < 0

    def test_no_benchmark(self):
        """ベンチマーク未指定."""
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        df = pd.DataFrame({"total": [1_000_000] * 10}, index=dates)
        result = compute_benchmark_excess(df, None)
        assert result is None

    def test_empty_history(self):
        """空の履歴."""
        bench = pd.Series([100, 110], index=pd.date_range("2026-01-01", periods=2))
        result = compute_benchmark_excess(pd.DataFrame(), bench)
        assert result is None
