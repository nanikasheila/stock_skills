"""CSV trade history importer.

Parses Rakuten Securities CSV exports (JP & US formats) and saves
each trade as a JSON record compatible with history_store.

Same-day / same-symbol / same-direction trades are aggregated into a
single record (total shares, weighted-average price).
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from src.data.history_store import _safe_filename, _history_dir, _sanitize


# ---------------------------------------------------------------------------
# CSV format detection
# ---------------------------------------------------------------------------

_JP_HEADER_MARKER = "銘柄コード"
_US_HEADER_MARKER = "ティッカー"


def detect_market(header_row: list[str]) -> Literal["jp", "us"]:
    """Detect JP or US format from the header row."""
    joined = ",".join(header_row)
    if _JP_HEADER_MARKER in joined:
        return "jp"
    if _US_HEADER_MARKER in joined:
        return "us"
    raise ValueError(
        f"CSV形式を判別できません。ヘッダ: {header_row[:5]}..."
    )


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------

def _parse_number(s: str) -> float:
    """Remove commas / quotes and convert to float.  Return 0.0 for '-' etc."""
    s = s.strip().strip('"')
    if not s or s == "-":
        return 0.0
    return float(s.replace(",", ""))


def _parse_date(s: str) -> str:
    """'2024/3/5' → '2024-03-05'."""
    s = s.strip().strip('"')
    dt = datetime.strptime(s, "%Y/%m/%d")
    return dt.strftime("%Y-%m-%d")


def _trade_type(raw: str) -> str | None:
    """Map 売買区分 to buy/sell/transfer.

    Returns None for rows that should be skipped.
    """
    raw = raw.strip().strip('"')
    if raw in ("買付",):
        return "buy"
    if raw in ("売付",):
        return "sell"
    if raw in ("入庫",):
        return "transfer"
    # Unknown type — skip
    return None


def parse_jp_row(row: list[str]) -> dict | None:
    """Parse a single JP-format CSV row into a normalised trade dict.

    Returns None for rows that should be skipped (e.g. 入庫 without price).
    """
    if len(row) < 12:
        return None

    trade = _trade_type(row[7])  # 売買区分
    if trade is None:
        return None

    code = row[2].strip().strip('"')  # 銘柄コード
    symbol = f"{code}.T"
    date_str = _parse_date(row[0])      # 約定日
    shares = int(_parse_number(row[10]))  # 数量
    price = _parse_number(row[11])        # 単価[円]
    name = row[3].strip().strip('"')      # 銘柄名
    account = row[5].strip().strip('"')   # 口座区分

    # 入庫 with no meaningful price → record as transfer at stated price
    # (the price column sometimes holds an average cost for 入庫)
    if trade == "transfer" and price <= 0:
        return None

    return {
        "symbol": symbol,
        "date": date_str,
        "trade_type": trade,
        "shares": shares,
        "price": price,
        "currency": "JPY",
        "name": name,
        "account": account,
    }


def parse_us_row(row: list[str]) -> dict | None:
    """Parse a single US-format CSV row into a normalised trade dict."""
    if len(row) < 12:
        return None

    trade = _trade_type(row[6])  # 売買区分
    if trade is None:
        return None

    ticker = row[2].strip().strip('"')  # ティッカー
    date_str = _parse_date(row[0])       # 約定日
    shares = int(_parse_number(row[10])) # 数量
    price = _parse_number(row[11])       # 単価[USドル]
    name = row[3].strip().strip('"')     # 銘柄名
    account = row[4].strip().strip('"')  # 口座

    # 入庫 with no price info → skip unless price is present
    if trade == "transfer" and price <= 0:
        return None

    return {
        "symbol": ticker,
        "date": date_str,
        "trade_type": trade,
        "shares": shares,
        "price": price,
        "currency": "USD",
        "name": name,
        "account": account,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg_key(t: dict) -> tuple:
    """Grouping key: (date, symbol, trade_type)."""
    return (t["date"], t["symbol"], t["trade_type"])


def aggregate_trades(trades: list[dict]) -> list[dict]:
    """Aggregate same-day / same-symbol / same-direction trades.

    Computes total shares and volume-weighted average price.
    """
    from collections import OrderedDict

    groups: OrderedDict[tuple, list[dict]] = OrderedDict()
    for t in trades:
        key = _agg_key(t)
        groups.setdefault(key, []).append(t)

    result = []
    for key, group in groups.items():
        total_shares = sum(g["shares"] for g in group)
        total_cost = sum(g["shares"] * g["price"] for g in group)
        avg_price = total_cost / total_shares if total_shares else 0.0

        base = group[0].copy()
        base["shares"] = total_shares
        base["price"] = round(avg_price, 4)
        base["_aggregated_count"] = len(group)
        result.append(base)

    return result


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _make_filename(trade: dict, index: int | None = None) -> str:
    """Build JSON filename: {date}_{trade_type}_{safe_symbol}.json

    If *index* is provided (for deduplication), append _{index}.
    """
    safe = _safe_filename(trade["symbol"])
    base = f"{trade['date']}_{trade['trade_type']}_{safe}"
    if index is not None:
        base = f"{base}_{index}"
    return f"{base}.json"


def save_trade_record(
    trade: dict,
    base_dir: str,
    *,
    dry_run: bool = False,
) -> str:
    """Save a single aggregated trade record as JSON.

    Returns the path (absolute) of the saved/would-be-saved file.
    """
    now = datetime.now().isoformat(timespec="seconds")

    payload = {
        "category": "trade",
        "date": trade["date"],
        "timestamp": now,
        "symbol": trade["symbol"],
        "trade_type": trade["trade_type"],
        "shares": trade["shares"],
        "price": trade["price"],
        "currency": trade["currency"],
        "memo": f"CSV import: {trade.get('name', '')} ({trade.get('account', '')})",
        "_saved_at": now,
    }

    d = _history_dir("trade", base_dir)
    filename = _make_filename(trade)
    path = d / filename

    # Avoid overwriting — add index suffix if needed
    if path.exists():
        idx = 1
        while True:
            alt_filename = _make_filename(trade, index=idx)
            alt_path = d / alt_filename
            if not alt_path.exists():
                path = alt_path
                break
            idx += 1

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_sanitize(payload), f, ensure_ascii=False, indent=2)

    return str(path.resolve())


# ---------------------------------------------------------------------------
# Main import function
# ---------------------------------------------------------------------------

def import_csv(
    csv_path: str | Path,
    base_dir: str = "data/history",
    *,
    market: str | None = None,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict:
    """Import a trade-history CSV and save individual JSON files.

    Parameters
    ----------
    csv_path : path to the CSV file
    base_dir : root of the history directory
    market   : force 'jp' or 'us'; auto-detected if None
    dry_run  : if True, parse and report but don't write files
    skip_existing : if True, skip trades whose JSON already exists

    Returns
    -------
    dict with keys: imported (int), skipped (int), errors (list[str]),
                    files (list[str])
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read CSV
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        if market is None:
            market = detect_market(header)

        parse_fn = parse_jp_row if market == "jp" else parse_us_row

        raw_trades: list[dict] = []
        errors: list[str] = []
        for i, row in enumerate(reader, start=2):
            try:
                t = parse_fn(row)
                if t is not None:
                    raw_trades.append(t)
            except Exception as e:
                errors.append(f"行 {i}: {e}")

    # Aggregate same-day / same-symbol / same-direction
    aggregated = aggregate_trades(raw_trades)

    # Determine existing files for skip_existing
    existing_files: set[str] = set()
    if skip_existing:
        trade_dir = Path(base_dir) / "trade"
        if trade_dir.exists():
            existing_files = {p.name for p in trade_dir.glob("*.json")}

    # Save
    imported = 0
    skipped = 0
    files: list[str] = []

    for trade in aggregated:
        fname = _make_filename(trade)
        if skip_existing and fname in existing_files:
            skipped += 1
            continue

        path = save_trade_record(trade, base_dir, dry_run=dry_run)
        files.append(path)
        imported += 1

    return {
        "csv_file": str(csv_path),
        "market": market,
        "raw_rows": len(raw_trades),
        "aggregated": len(aggregated),
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
        "files": files,
    }
