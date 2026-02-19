"""ポートフォリオダッシュボード — データローダー.

既存の portfolio_manager / history_store / yahoo_client を活用して
ダッシュボード表示用のデータを組み立てる。
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DEFAULT_HISTORY_DIR = str(Path(_PROJECT_ROOT) / "data" / "history")

from src.core.portfolio.portfolio_manager import (
    load_portfolio,
    get_fx_rates,
    DEFAULT_CSV_PATH,
)
from src.core.models import Position
from src.core.common import is_cash
from src.core.ticker_utils import infer_currency
from src.data import yahoo_client
from src.data.history_store import load_history


# ---------------------------------------------------------------------------
# 1. 現在のスナップショット（銘柄別評価額）
# ---------------------------------------------------------------------------

def get_current_snapshot(
    csv_path: str = DEFAULT_CSV_PATH,
) -> dict:
    """現在の保有銘柄ごとの評価額を取得して dict で返す.

    Returns
    -------
    dict
        positions: list[dict]  各銘柄
        total_value_jpy: float
        fx_rates: dict
        as_of: str
    """
    positions = load_portfolio(csv_path)
    fx_rates = get_fx_rates(yahoo_client)

    result_positions: list[dict] = []
    total_value_jpy = 0.0

    for pos in positions:
        symbol = pos["symbol"]
        shares = pos["shares"]
        cost_price = pos["cost_price"]
        cost_currency = pos["cost_currency"]
        memo = pos.get("memo", "")
        purchase_date = pos.get("purchase_date", "")

        if is_cash(symbol):
            currency = symbol.replace(".CASH", "")
            rate = fx_rates.get(currency, 1.0)
            eval_jpy = shares * cost_price * rate
            result_positions.append({
                "symbol": symbol,
                "name": memo or symbol,
                "shares": shares,
                "current_price": cost_price,
                "currency": currency,
                "evaluation_jpy": eval_jpy,
                "sector": "Cash",
            })
            total_value_jpy += eval_jpy
            continue

        info = yahoo_client.get_stock_info(symbol)
        if not info:
            continue

        price = info.get("price", 0) or 0
        currency = info.get("currency") or infer_currency(symbol)
        rate = fx_rates.get(currency, 1.0)
        eval_jpy = shares * price * rate
        cost_rate = fx_rates.get(cost_currency, 1.0)
        cost_jpy = shares * cost_price * cost_rate

        result_positions.append({
            "symbol": symbol,
            "name": info.get("name", memo or symbol),
            "shares": shares,
            "current_price": price,
            "currency": currency,
            "evaluation_jpy": eval_jpy,
            "cost_jpy": cost_jpy,
            "pnl_jpy": eval_jpy - cost_jpy,
            "pnl_pct": ((eval_jpy / cost_jpy) - 1) * 100 if cost_jpy else 0,
            "sector": info.get("sector", ""),
            "purchase_date": purchase_date,
        })
        total_value_jpy += eval_jpy

    return {
        "positions": result_positions,
        "total_value_jpy": total_value_jpy,
        "fx_rates": fx_rates,
        "as_of": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# 2. 売買履歴から時系列の保有状況を復元
# ---------------------------------------------------------------------------

def _build_holdings_timeline(
    base_dir: Optional[str] = None,
) -> list[dict]:
    """trade 履歴を日時順にロードして返す."""
    trades = load_history("trade", base_dir=base_dir or _DEFAULT_HISTORY_DIR)
    # load_history は日付降順 → 昇順に
    trades.sort(key=lambda t: t.get("timestamp", t.get("date", "")))
    return trades


def _reconstruct_daily_holdings(
    trades: list[dict],
) -> dict[str, dict[str, int]]:
    """各取引日時点での全銘柄保有株数マップを返す.

    Returns
    -------
    dict
        date_str -> { symbol -> cumulative_shares }
    """
    cumulative: dict[str, int] = {}
    daily_snapshots: dict[str, dict[str, int]] = {}

    for trade in trades:
        symbol = trade["symbol"]
        shares = trade.get("shares", 0)
        trade_type = trade.get("trade_type", "buy")
        date_str = trade.get("date", "")

        if trade_type == "buy":
            cumulative[symbol] = cumulative.get(symbol, 0) + shares
        elif trade_type == "sell":
            cumulative[symbol] = max(0, cumulative.get(symbol, 0) - shares)
            if cumulative[symbol] == 0:
                del cumulative[symbol]

        daily_snapshots[date_str] = dict(cumulative)

    return daily_snapshots


# ---------------------------------------------------------------------------
# 3. 資産推移データの構築
# ---------------------------------------------------------------------------

def build_portfolio_history(
    csv_path: str = DEFAULT_CSV_PATH,
    base_dir: str = _DEFAULT_HISTORY_DIR,
    period: str = "3mo",
) -> pd.DataFrame:
    """保有銘柄の日次評価額推移を DataFrame で返す.

    売買履歴から各日の保有銘柄・株数を復元し、
    yfinance の価格履歴と掛け合わせて日次評価額を算出する。

    Parameters
    ----------
    csv_path : str
        portfolio.csv のパス
    base_dir : str | None
        history_store のベースディレクトリ
    period : str
        株価取得期間 ("1mo", "3mo", "6mo", "1y", "2y" 等)

    Returns
    -------
    pd.DataFrame
        index=Date, columns=銘柄シンボル, values=円換算評価額
        + "total" カラム
    """
    trades = _build_holdings_timeline(base_dir)
    if not trades:
        return pd.DataFrame()

    daily_snapshots = _reconstruct_daily_holdings(trades)

    # 現在保有中の銘柄（直近スナップショットから）＋過去に保有していた銘柄
    all_symbols: set[str] = set()
    for snap in daily_snapshots.values():
        all_symbols.update(snap.keys())

    # CASH 銘柄は除外（為替推移のみでの表示は別途対応可能）
    stock_symbols = [s for s in all_symbols if not is_cash(s)]

    if not stock_symbols:
        return pd.DataFrame()

    # 為替レート取得
    fx_rates = get_fx_rates(yahoo_client)

    # 各銘柄の株価履歴を取得
    price_histories: dict[str, pd.DataFrame] = {}
    for symbol in stock_symbols:
        hist = yahoo_client.get_price_history(symbol, period=period)
        if hist is not None and not hist.empty:
            price_histories[symbol] = hist[["Close"]].rename(
                columns={"Close": symbol}
            )

    if not price_histories:
        return pd.DataFrame()

    # 全銘柄の終値を結合
    price_df = pd.concat(price_histories.values(), axis=1)
    price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
    price_df = price_df.sort_index()
    price_df = price_df.ffill()  # 休日を前営業日の値で埋める

    # 売買日 → 保有株数のマッピング — 日次に展開
    dates = price_df.index
    first_trade_date = pd.Timestamp(trades[0].get("date", ""))

    # 各日の保有株数を computed
    sorted_trade_dates = sorted(daily_snapshots.keys())

    def get_holdings_at(date: pd.Timestamp) -> dict[str, int]:
        """指定日時点の保有株数を返す."""
        result: dict[str, int] = {}
        for td in sorted_trade_dates:
            if pd.Timestamp(td) <= date:
                result = daily_snapshots[td]
            else:
                break
        return result

    # 日次評価額の計算
    eval_data: dict[str, list[float]] = {s: [] for s in stock_symbols}

    for date in dates:
        holdings = get_holdings_at(date)
        for symbol in stock_symbols:
            shares = holdings.get(symbol, 0)
            if shares > 0 and symbol in price_df.columns:
                price_val = price_df.loc[date, symbol]
                if pd.notna(price_val):
                    currency = infer_currency(symbol)
                    rate = fx_rates.get(currency, 1.0)
                    eval_data[symbol].append(shares * price_val * rate)
                else:
                    eval_data[symbol].append(0.0)
            else:
                eval_data[symbol].append(0.0)

    result_df = pd.DataFrame(eval_data, index=dates)

    # 取引開始前のデータを除外
    if not first_trade_date or pd.isna(first_trade_date):
        first_trade_date = dates[0]
    result_df = result_df[result_df.index >= first_trade_date]

    # 合計列
    result_df["total"] = result_df.sum(axis=1)

    return result_df


# ---------------------------------------------------------------------------
# 4. セクター別集計
# ---------------------------------------------------------------------------

def get_sector_breakdown(snapshot: dict) -> pd.DataFrame:
    """スナップショットからセクター別評価額を集計."""
    rows = []
    for p in snapshot["positions"]:
        rows.append({
            "sector": p.get("sector") or "Unknown",
            "evaluation_jpy": p.get("evaluation_jpy", 0),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby("sector")["evaluation_jpy"].sum().reset_index()


# ---------------------------------------------------------------------------
# 5. 月次集計
# ---------------------------------------------------------------------------

def get_monthly_summary(history_df: pd.DataFrame) -> pd.DataFrame:
    """日次データから月末の total を抽出して月次テーブルを返す."""
    if history_df.empty:
        return pd.DataFrame()

    monthly = history_df[["total"]].resample("ME").last()
    monthly.index = monthly.index.strftime("%Y-%m")
    monthly.columns = ["month_end_value_jpy"]

    # 月次変動率
    monthly["change_pct"] = monthly["month_end_value_jpy"].pct_change() * 100

    return monthly
