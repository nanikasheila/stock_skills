"""ポートフォリオダッシュボード — データローダー.

既存の portfolio_manager / history_store / yahoo_client を活用して
ダッシュボード表示用のデータを組み立てる。

取引履歴（JSON/CSVインポート）から各日の保有状況を復元し、
株価履歴と掛け合わせて資産推移を構築する。
"""

from __future__ import annotations

import sys
import os
import time as _time
from collections import defaultdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import pandas as pd

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = str(Path(__file__).resolve().parents[5])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DEFAULT_HISTORY_DIR = str(Path(_PROJECT_ROOT) / "data" / "history")
_PRICE_CACHE_DIR = Path(_PROJECT_ROOT) / "data" / "cache" / "price_history"
_CACHE_TTL_SECONDS = 4 * 3600  # 4 hours

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
                "cost_jpy": eval_jpy,  # Cash: 損益ゼロ
                "pnl_jpy": 0,
                "pnl_pct": 0,
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

    # 実現損益の計算
    trades = _build_holdings_timeline()
    realized = _compute_realized_pnl(trades, fx_rates)

    return {
        "positions": result_positions,
        "total_value_jpy": total_value_jpy,
        "fx_rates": fx_rates,
        "realized_pnl": realized,
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
    # 取引日 (date) でソート。同一日は buy/transfer → sell の順に並べる
    _TRADE_TYPE_ORDER = {"transfer": 0, "buy": 1, "sell": 2}
    trades.sort(key=lambda t: (
        t.get("date", ""),
        _TRADE_TYPE_ORDER.get(t.get("trade_type", "buy"), 1),
    ))
    return trades


def _reconstruct_daily_holdings(
    trades: list[dict],
) -> dict[str, dict[str, int]]:
    """各取引日時点での全銘柄保有株数マップを返す.

    buy / transfer → 保有追加、sell → 保有削減。

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

        if trade_type in ("buy", "transfer"):
            cumulative[symbol] = cumulative.get(symbol, 0) + shares
        elif trade_type == "sell":
            cumulative[symbol] = max(0, cumulative.get(symbol, 0) - shares)
            if cumulative[symbol] == 0:
                del cumulative[symbol]

        daily_snapshots[date_str] = dict(cumulative)

    return daily_snapshots


def _compute_invested_capital(
    trades: list[dict],
    fx_rates: dict[str, float],
) -> dict[str, float]:
    """累積投資額(円換算)の推移を返す.

    buy/transfer → +投資額、sell → −売却額
    受渡金額ではなく shares*price*fx_rate で計算。

    Returns
    -------
    dict
        date_str -> cumulative_invested_jpy
    """
    cumulative = 0.0
    invested: dict[str, float] = {}

    for trade in trades:
        shares = trade.get("shares", 0)
        price = trade.get("price", 0)
        currency = trade.get("currency", "JPY")
        trade_type = trade.get("trade_type", "buy")
        date_str = trade.get("date", "")
        rate = fx_rates.get(currency, 1.0)
        amount_jpy = shares * price * rate

        if trade_type in ("buy", "transfer"):
            cumulative += amount_jpy
        elif trade_type == "sell":
            cumulative -= amount_jpy
        cumulative = max(0.0, cumulative)

        invested[date_str] = cumulative

    return invested


def _trade_cost_jpy(
    trade: dict,
    global_fx_rates: dict[str, float],
) -> float:
    """取引の約定金額をJPYで計算する.

    優先順位:
    1. settlement_jpy + settlement_usd * fx_rate (両方ある場合)
    2. settlement_jpy が正 → そのまま使用
    3. settlement_usd * fx_rate (取引日レート)
    4. shares * price * fx_rate (取引日レートで計算)
    5. shares * price * 現在のFXレート (フォールバック)
    """
    sjpy = trade.get("settlement_jpy", 0) or 0
    susd = trade.get("settlement_usd", 0) or 0
    fx = trade.get("fx_rate", 0) or 0

    if sjpy > 0 and susd > 0:
        # Mixed settlement (JPY + USD portions)
        return sjpy + susd * fx
    elif sjpy > 0:
        return sjpy
    elif susd > 0 and fx > 0:
        return susd * fx
    elif fx > 0:
        # FX rate available but no explicit settlement → use price * fx_rate
        shares = trade.get("shares", 0)
        price = trade.get("price", 0)
        return shares * price * fx
    else:
        # Final fallback: use current FX rate
        shares = trade.get("shares", 0)
        price = trade.get("price", 0)
        cur = trade.get("currency", "JPY")
        rate = global_fx_rates.get(cur, 1.0)
        return shares * price * rate


def _compute_realized_pnl(
    trades: list[dict],
    fx_rates: dict[str, float],
) -> dict:
    """FIFO方式で実現損益を計算する（為替換算・株式分割対応版）.

    改善点:
    - 為替換算: CSVの受渡金額/為替レートを使い、取引時点のJPY換算で損益を算出
    - 株式分割: transfer(入庫)でprice=0の場合、既存ロットの単価を分割比率で調整
    - フォールバック: 旧形式のJSON（fx_rate/settlement未保存）は現在レートで近似

    Returns
    -------
    dict
        by_symbol: dict[str, float]  銘柄別実現損益(JPY)
        total_jpy: float  合計実現損益(JPY)
    """
    # FIFO: 銘柄ごとに購入ロットを管理
    # 各ロット: {"shares": float, "cost_jpy_per_share": float}
    lots: dict[str, list[dict]] = defaultdict(list)
    realized_by_symbol: dict[str, float] = defaultdict(float)

    for trade in trades:
        sym = trade.get("symbol", "")
        tt = trade.get("trade_type", "buy")
        shares = trade.get("shares", 0)
        price = trade.get("price", 0)

        if is_cash(sym):
            continue

        if tt == "buy":
            total_jpy = _trade_cost_jpy(trade, fx_rates)
            cost_per_share = total_jpy / shares if shares > 0 else 0
            lots[sym].append({
                "shares": float(shares),
                "cost_jpy_per_share": cost_per_share,
            })

        elif tt == "transfer":
            if price <= 0 and lots[sym]:
                # Stock split: redistribute cost basis
                existing_shares = sum(lot["shares"] for lot in lots[sym])
                if existing_shares > 0:
                    split_ratio = (existing_shares + shares) / existing_shares
                    for lot in lots[sym]:
                        lot["cost_jpy_per_share"] /= split_ratio
                        lot["shares"] *= split_ratio
            elif price > 0:
                # Regular transfer with cost basis
                total_jpy = _trade_cost_jpy(trade, fx_rates)
                cost_per_share = total_jpy / shares if shares > 0 else 0
                lots[sym].append({
                    "shares": float(shares),
                    "cost_jpy_per_share": cost_per_share,
                })

        elif tt == "sell":
            total_jpy = _trade_cost_jpy(trade, fx_rates)
            proceeds_per_share = total_jpy / shares if shares > 0 else 0

            remaining = float(shares)
            while remaining > 0.5 and lots[sym]:
                lot = lots[sym][0]
                take = min(remaining, lot["shares"])
                pnl = take * (proceeds_per_share - lot["cost_jpy_per_share"])
                realized_by_symbol[sym] += pnl
                lot["shares"] -= take
                remaining -= take
                if lot["shares"] < 0.5:
                    lots[sym].pop(0)

    total = sum(realized_by_symbol.values())
    return {
        "by_symbol": dict(realized_by_symbol),
        "total_jpy": total,
    }


def _build_trade_activity(
    trades: list[dict],
    fx_rates: dict[str, float],
) -> pd.DataFrame:
    """月ごとの売買件数・金額をまとめた DataFrame を返す."""
    rows: list[dict] = []
    for trade in trades:
        shares = trade.get("shares", 0)
        price = trade.get("price", 0)
        currency = trade.get("currency", "JPY")
        rate = fx_rates.get(currency, 1.0)
        amount = shares * price * rate
        tt = trade.get("trade_type", "buy")
        d = trade.get("date", "")
        if not d:
            continue
        month = d[:7]  # YYYY-MM
        rows.append({
            "month": month,
            "trade_type": tt,
            "amount_jpy": amount,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    buy_df = df[df["trade_type"].isin(["buy", "transfer"])].groupby("month").agg(
        buy_count=("amount_jpy", "count"),
        buy_amount=("amount_jpy", "sum"),
    )
    sell_df = df[df["trade_type"] == "sell"].groupby("month").agg(
        sell_count=("amount_jpy", "count"),
        sell_amount=("amount_jpy", "sum"),
    )
    result = buy_df.join(sell_df, how="outer").fillna(0)
    result["net_flow"] = result["buy_amount"] - result["sell_amount"]
    result.index.name = None
    return result.sort_index()


# ---------------------------------------------------------------------------
# 3. 資産推移データの構築
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 期間 → yfinance period / start の変換
# ---------------------------------------------------------------------------

_PERIOD_MAP: dict[str, str | None] = {
    "1mo":  "1mo",
    "3mo":  "3mo",
    "6mo":  "6mo",
    "1y":   "1y",
    "2y":   "2y",
    "3y":   "3y",
    "5y":   "5y",
    "max":  "max",
    "all":  "max",
}


def _fetch_price_history(
    symbol: str, period: str,
) -> Optional[pd.DataFrame]:
    """期間指定に応じた株価履歴を取得する (個別フォールバック用)."""
    yf_period = _PERIOD_MAP.get(period, period)
    hist = yahoo_client.get_price_history(symbol, period=yf_period)
    if hist is not None and not hist.empty:
        return hist[["Close"]].rename(columns={"Close": symbol})
    return None


# ---------------------------------------------------------------------------
# 価格キャッシュ (ディスク + バッチ取得)
# ---------------------------------------------------------------------------

def _get_cache_path(period: str) -> Path:
    """期間ごとのキャッシュファイルパスを返す."""
    safe = period.replace("/", "_")
    return _PRICE_CACHE_DIR / f"close_{safe}.csv"


def _load_cached_prices(period: str) -> Optional[pd.DataFrame]:
    """ディスクキャッシュから株価を読み込む. TTL 超過時は None."""
    path = _get_cache_path(period)
    if not path.exists():
        return None
    age = _time.time() - path.stat().st_mtime
    if age > _CACHE_TTL_SECONDS:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


def _save_prices_cache(prices: pd.DataFrame, period: str) -> None:
    """株価をディスクキャッシュに保存."""
    try:
        _PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_csv(_get_cache_path(period))
    except Exception as e:
        print(f"[data_loader] Cache save error: {e}")


def _load_prices(symbols: list[str], period: str) -> pd.DataFrame:
    """キャッシュ優先で全銘柄の終値を一括取得.

    1. ディスクキャッシュが有効 (TTL 4h) → 即座に返す
    2. キャッシュに不足銘柄 → 不足分のみバッチ取得してマージ
    3. キャッシュなし → 全銘柄をバッチ取得 (yf.download 1 回)
    """
    yf_period = _PERIOD_MAP.get(period, period)

    # --- キャッシュヒット ---
    cached = _load_cached_prices(period)
    if cached is not None:
        missing = [s for s in symbols if s not in cached.columns]
        if not missing:
            available = [s for s in symbols if s in cached.columns]
            print(f"[data_loader] Cache hit: {period} ({len(available)} symbols)")
            return cached[available]
        # 不足銘柄のみ追加取得
        print(f"[data_loader] Cache partial: fetching {len(missing)} symbols")
        new_prices = yahoo_client.get_close_prices_batch(missing, period=yf_period)
        if new_prices is not None and not new_prices.empty:
            new_prices.index = pd.to_datetime(new_prices.index).tz_localize(None)
            merged = pd.concat([cached, new_prices], axis=1)
            _save_prices_cache(merged, period)
            available = [s for s in symbols if s in merged.columns]
            if available:
                return merged[available].sort_index().ffill()
        available = [s for s in symbols if s in cached.columns]
        return cached[available] if available else pd.DataFrame()

    # --- キャッシュミス: バッチ取得 ---
    print(f"[data_loader] Cache miss: batch-fetching {len(symbols)} symbols")
    prices = yahoo_client.get_close_prices_batch(symbols, period=yf_period)
    if prices is None or prices.empty:
        # フォールバック: 個別取得
        print("[data_loader] Batch failed, falling back to individual fetches")
        frames: dict[str, pd.Series] = {}
        for sym in symbols:
            hist = _fetch_price_history(sym, period)
            if hist is not None and sym in hist.columns:
                frames[sym] = hist[sym]
        if not frames:
            return pd.DataFrame()
        prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index().ffill()
    _save_prices_cache(prices, period)
    return prices


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
        株価取得期間 ("1mo" .. "5y", "max", "all")

    Returns
    -------
    pd.DataFrame
        index=Date, columns=銘柄シンボル, values=円換算評価額
        + "total" カラム + "invested" カラム
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
    stock_symbols = sorted(s for s in all_symbols if not is_cash(s))

    if not stock_symbols:
        return pd.DataFrame()

    # 為替レート取得
    fx_rates = get_fx_rates(yahoo_client)

    # 全銘柄の終値を一括取得（ディスクキャッシュ + バッチ取得）
    price_df = _load_prices(stock_symbols, period)
    if price_df.empty:
        return pd.DataFrame()

    # 売買日 → 保有株数のマッピング — 日次に展開
    dates = price_df.index
    first_trade_date = pd.Timestamp(trades[0].get("date", ""))

    # 各日の保有株数を computed
    sorted_trade_dates = sorted(daily_snapshots.keys())

    def get_holdings_at(ts: pd.Timestamp) -> dict[str, int]:
        """指定日時点の保有株数を返す."""
        result: dict[str, int] = {}
        for td in sorted_trade_dates:
            if pd.Timestamp(td) <= ts:
                result = daily_snapshots[td]
            else:
                break
        return result

    # 日次評価額の計算
    eval_data: dict[str, list[float]] = {s: [] for s in stock_symbols}

    for dt in dates:
        holdings = get_holdings_at(dt)
        for symbol in stock_symbols:
            shares = holdings.get(symbol, 0)
            if shares > 0 and symbol in price_df.columns:
                price_val = price_df.loc[dt, symbol]
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

    # 全期間ゼロの銘柄列を除外（既に売却済みで表示期間に保有がない銘柄）
    symbol_cols = [c for c in result_df.columns if c not in ("total", "invested")]
    zero_cols = [c for c in symbol_cols if (result_df[c] == 0).all()]
    if zero_cols:
        result_df = result_df.drop(columns=zero_cols)

    # 合計列
    result_df["total"] = result_df.sum(axis=1)

    # 累積投資額列の追加
    invested_map = _compute_invested_capital(trades, fx_rates)
    invested_series: list[float] = []
    sorted_inv_dates = sorted(invested_map.keys())
    for dt in result_df.index:
        inv_val = 0.0
        for inv_d in sorted_inv_dates:
            if pd.Timestamp(inv_d) <= dt:
                inv_val = invested_map[inv_d]
            else:
                break
        invested_series.append(inv_val)
    result_df["invested"] = invested_series

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

    cols = ["total"]
    if "invested" in history_df.columns:
        cols.append("invested")

    monthly = history_df[cols].resample("ME").last()
    monthly.index = monthly.index.strftime("%Y-%m")
    rename = {"total": "month_end_value_jpy"}
    if "invested" in monthly.columns:
        rename["invested"] = "invested_jpy"
    monthly = monthly.rename(columns=rename)

    # 月次変動率
    monthly["change_pct"] = monthly["month_end_value_jpy"].pct_change() * 100

    # 前年同月比 (YoY)
    monthly["yoy_pct"] = monthly["month_end_value_jpy"].pct_change(periods=12) * 100

    # 含み損益
    if "invested_jpy" in monthly.columns:
        monthly["unrealized_pnl"] = (
            monthly["month_end_value_jpy"] - monthly["invested_jpy"]
        )

    return monthly


def get_trade_activity(
    base_dir: str = _DEFAULT_HISTORY_DIR,
) -> pd.DataFrame:
    """月ごとの売買件数・金額を返す."""
    trades = _build_holdings_timeline(base_dir)
    if not trades:
        return pd.DataFrame()
    fx_rates = get_fx_rates(yahoo_client)
    return _build_trade_activity(trades, fx_rates)


# ---------------------------------------------------------------------------
# 6. 資産推定推移（楽観/ベース/悲観）
# ---------------------------------------------------------------------------

def build_projection(
    current_value: float,
    years: int = 5,
    optimistic_rate: float | None = None,
    base_rate: float | None = None,
    pessimistic_rate: float | None = None,
    csv_path: str = DEFAULT_CSV_PATH,
) -> pd.DataFrame:
    """現在の総資産から楽観/ベース/悲観の将来推定推移を生成する.

    Parameters
    ----------
    current_value : float
        現在の総資産（円）。
    years : int
        何年先まで推定するか（デフォルト5年）。
    optimistic_rate, base_rate, pessimistic_rate : float | None
        年率リターン（0.10 = 10%）。None の場合は estimate_portfolio_return から取得。
    csv_path : str
        ポートフォリオCSVパス。

    Returns
    -------
    pd.DataFrame
        index=日付, columns=[optimistic, base, pessimistic]
    """
    # リターン推定値が未指定の場合、ポートフォリオから推定
    if base_rate is None:
        try:
            from src.core.return_estimate import estimate_portfolio_return
            result = estimate_portfolio_return(csv_path, yahoo_client)
            pf = result.get("portfolio", {})
            optimistic_rate = pf.get("optimistic") or 0.15
            base_rate = pf.get("base") or 0.08
            pessimistic_rate = pf.get("pessimistic") or -0.05
        except Exception:
            optimistic_rate = 0.15
            base_rate = 0.08
            pessimistic_rate = -0.05

    if optimistic_rate is None:
        optimistic_rate = 0.15
    if pessimistic_rate is None:
        pessimistic_rate = -0.05

    # 月次ポイントで推定（years * 12 + 1 点）
    today = pd.Timestamp.now().normalize()
    months = years * 12
    dates = pd.date_range(start=today, periods=months + 1, freq="ME")
    # 先頭を今日にする
    dates = dates.insert(0, today)

    rows = []
    for d in dates:
        t_years = (d - today).days / 365.25
        rows.append({
            "date": d,
            "optimistic": current_value * (1 + optimistic_rate) ** t_years,
            "base": current_value * (1 + base_rate) ** t_years,
            "pessimistic": current_value * (1 + pessimistic_rate) ** t_years,
        })

    df = pd.DataFrame(rows).set_index("date")
    return df


# ---------------------------------------------------------------------------
# 7. リスク指標の算出
# ---------------------------------------------------------------------------

import numpy as np


def compute_risk_metrics(history_df: pd.DataFrame) -> dict:
    """日次の資産推移からリスク指標を算出する.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力。"total" 列が必須。

    Returns
    -------
    dict
        sharpe_ratio: float   年率シャープレシオ（リスクフリーレート0.5%想定）
        max_drawdown_pct: float  最大ドローダウン（%、マイナス値）
        annual_volatility_pct: float  年率ボラティリティ（%）
        annual_return_pct: float  年率リターン（%）
        calmar_ratio: float  カルマーレシオ（年率リターン / |MDD|）
    """
    if history_df.empty or "total" not in history_df.columns:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "annual_volatility_pct": 0.0,
            "annual_return_pct": 0.0,
            "calmar_ratio": 0.0,
        }

    total = history_df["total"].dropna()
    if len(total) < 2:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "annual_volatility_pct": 0.0,
            "annual_return_pct": 0.0,
            "calmar_ratio": 0.0,
        }

    # 日次リターン
    daily_returns = total.pct_change().dropna()
    if daily_returns.empty:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "annual_volatility_pct": 0.0,
            "annual_return_pct": 0.0,
            "calmar_ratio": 0.0,
        }

    trading_days = 252
    risk_free_rate = 0.005  # 0.5%

    # 年率リターン
    total_days = (total.index[-1] - total.index[0]).days
    if total_days <= 0:
        total_days = 1
    total_return = total.iloc[-1] / total.iloc[0] - 1
    annual_return = (1 + total_return) ** (365.25 / total_days) - 1

    # 年率ボラティリティ
    annual_vol = float(daily_returns.std() * np.sqrt(trading_days))

    # シャープレシオ
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0

    # 最大ドローダウン
    cummax = total.cummax()
    drawdown = (total - cummax) / cummax
    max_dd = float(drawdown.min()) * 100  # パーセント

    # カルマーレシオ
    calmar = annual_return / abs(max_dd / 100) if max_dd != 0 else 0.0

    return {
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown_pct": round(max_dd, 1),
        "annual_volatility_pct": round(annual_vol * 100, 1),
        "annual_return_pct": round(annual_return * 100, 1),
        "calmar_ratio": round(float(calmar), 2),
    }


# ---------------------------------------------------------------------------
# 8. Top/Worst パフォーマー
# ---------------------------------------------------------------------------

def compute_top_worst_performers(
    history_df: pd.DataFrame,
    top_n: int = 3,
) -> dict:
    """直近1日の銘柄別騰落率ランキングを返す.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力
    top_n : int
        上位/下位何銘柄を返すか

    Returns
    -------
    dict
        top: list[dict]  (symbol, change_pct, change_jpy)
        worst: list[dict]
    """
    if history_df.empty or len(history_df) < 2:
        return {"top": [], "worst": []}

    stock_cols = [c for c in history_df.columns if c not in ("total", "invested")]
    if not stock_cols:
        return {"top": [], "worst": []}

    latest = history_df.iloc[-1]
    previous = history_df.iloc[-2]

    performers = []
    for col in stock_cols:
        cur = float(latest.get(col, 0))
        prev = float(previous.get(col, 0))
        if prev > 0 and cur > 0:
            pct = (cur / prev - 1) * 100
            change_jpy = cur - prev
            performers.append({
                "symbol": col,
                "change_pct": round(pct, 2),
                "change_jpy": round(change_jpy, 0),
            })

    performers.sort(key=lambda x: x["change_pct"], reverse=True)

    actual_n = min(top_n, len(performers))
    return {
        "top": performers[:actual_n],
        "worst": performers[-actual_n:][::-1] if actual_n > 0 else [],
    }


# ---------------------------------------------------------------------------
# 9. 前日比計算
# ---------------------------------------------------------------------------

def compute_daily_change(history_df: pd.DataFrame) -> dict:
    """直近の前日比（金額・パーセント）を算出する.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力。"total" 列が必須。

    Returns
    -------
    dict
        daily_change_jpy: float  前日比（円）
        daily_change_pct: float  前日比（%）
    """
    if history_df.empty or "total" not in history_df.columns:
        return {"daily_change_jpy": 0.0, "daily_change_pct": 0.0}

    total = history_df["total"].dropna()
    if len(total) < 2:
        return {"daily_change_jpy": 0.0, "daily_change_pct": 0.0}

    latest = float(total.iloc[-1])
    previous = float(total.iloc[-2])
    change = latest - previous
    pct = (change / previous * 100) if previous != 0 else 0.0

    return {
        "daily_change_jpy": round(change, 0),
        "daily_change_pct": round(pct, 2),
    }


# ---------------------------------------------------------------------------
# 9. ベンチマーク超過リターン
# ---------------------------------------------------------------------------

def compute_benchmark_excess(
    history_df: pd.DataFrame,
    benchmark_series: pd.Series | None,
) -> dict | None:
    """ポートフォリオのベンチマーク超過リターンを算出する.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力。"total" 列が必須。
    benchmark_series : pd.Series | None
        get_benchmark_series() の出力（正規化済み）

    Returns
    -------
    dict | None
        portfolio_return_pct: float
        benchmark_return_pct: float
        excess_return_pct: float
    """
    if benchmark_series is None or history_df.empty or "total" not in history_df.columns:
        return None

    total = history_df["total"].dropna()
    bench = benchmark_series.dropna()
    if len(total) < 2 or len(bench) < 2:
        return None

    pf_return = (float(total.iloc[-1]) / float(total.iloc[0]) - 1) * 100
    bm_return = (float(bench.iloc[-1]) / float(bench.iloc[0]) - 1) * 100
    excess = pf_return - bm_return

    return {
        "portfolio_return_pct": round(pf_return, 2),
        "benchmark_return_pct": round(bm_return, 2),
        "excess_return_pct": round(excess, 2),
    }


# ---------------------------------------------------------------------------
# 10. ベンチマークデータ取得
# ---------------------------------------------------------------------------

def get_benchmark_series(
    symbol: str,
    history_df: pd.DataFrame,
    period: str = "3mo",
) -> pd.Series | None:
    """ベンチマーク銘柄の終値を取得し、PF の total 列と同じ基準に正規化する.

    PF 開始日の total 値を基準に、ベンチマークの相対パフォーマンスを
    同じ円スケールに変換して返す。

    Parameters
    ----------
    symbol : str
        ベンチマークのティッカー (e.g. "SPY", "^N225")
    history_df : pd.DataFrame
        build_portfolio_history() の出力
    period : str
        価格取得期間

    Returns
    -------
    pd.Series | None
        index=Date, values=正規化された評価額（PFと同スケール）
    """
    if history_df.empty or "total" not in history_df.columns:
        return None

    prices = _load_prices([symbol], period)
    if prices.empty or symbol not in prices.columns:
        return None

    bench = prices[symbol].dropna()
    if bench.empty:
        return None

    # PF の日付範囲に合わせる
    pf_start = history_df.index[0]
    bench = bench[bench.index >= pf_start]
    if bench.empty:
        return None

    # PF 開始日の total を基準に正規化
    pf_start_value = history_df["total"].iloc[0]
    bench_start_value = bench.iloc[0]
    if bench_start_value == 0:
        return None

    normalized = bench / bench_start_value * pf_start_value
    normalized.name = symbol
    return normalized


# ---------------------------------------------------------------------------
# 12. ドローダウン系列
# ---------------------------------------------------------------------------

def compute_drawdown_series(history_df: pd.DataFrame) -> pd.Series:
    """日次のドローダウン（ピークからの下落率 %）系列を返す.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力。"total" 列が必須。

    Returns
    -------
    pd.Series
        index=Date, values=ドローダウン（%、0以下の値）
    """
    if history_df.empty or "total" not in history_df.columns:
        return pd.Series(dtype=float)

    total = history_df["total"].dropna()
    if len(total) < 2:
        return pd.Series(dtype=float)

    cummax = total.cummax()
    drawdown = (total - cummax) / cummax * 100
    return drawdown


# ---------------------------------------------------------------------------
# 13. ローリングSharpe比系列
# ---------------------------------------------------------------------------

def compute_rolling_sharpe(
    history_df: pd.DataFrame,
    window: int = 60,
    risk_free_rate: float = 0.005,
) -> pd.Series:
    """ローリングSharpe比の系列を返す.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力。"total" 列が必須。
    window : int
        ローリングウィンドウ（営業日数）
    risk_free_rate : float
        年率リスクフリーレート

    Returns
    -------
    pd.Series
        index=Date, values=ローリングSharpe比（年率換算）
    """
    if history_df.empty or "total" not in history_df.columns:
        return pd.Series(dtype=float)

    total = history_df["total"].dropna()
    if len(total) < window + 1:
        return pd.Series(dtype=float)

    daily_returns = total.pct_change().dropna()
    trading_days = 252
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1

    rolling_mean = daily_returns.rolling(window=window).mean()
    rolling_std = daily_returns.rolling(window=window).std()

    rolling_sharpe = (
        (rolling_mean - daily_rf) / rolling_std * np.sqrt(trading_days)
    )
    return rolling_sharpe.dropna()


# ---------------------------------------------------------------------------
# 14. 銘柄間相関行列
# ---------------------------------------------------------------------------

def compute_correlation_matrix(
    history_df: pd.DataFrame,
    min_periods: int = 20,
) -> pd.DataFrame:
    """保有銘柄間の日次リターン相関行列を返す.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力。銘柄ごとの列を含む。
    min_periods : int
        相関計算に必要な最低データ点数。

    Returns
    -------
    pd.DataFrame
        銘柄×銘柄の相関行列。銘柄が2つ未満の場合は空DataFrame。
    """
    if history_df.empty:
        return pd.DataFrame()

    # "total" と "invested" を除いた銘柄列のみ
    stock_cols = [c for c in history_df.columns if c not in ("total", "invested")]
    if len(stock_cols) < 2:
        return pd.DataFrame()

    stock_df = history_df[stock_cols].dropna(how="all")
    if len(stock_df) < min_periods:
        return pd.DataFrame()

    # 日次リターンを計算
    daily_returns = stock_df.pct_change().dropna(how="all")

    # 相関行列
    corr = daily_returns.corr(min_periods=min_periods)
    return corr
