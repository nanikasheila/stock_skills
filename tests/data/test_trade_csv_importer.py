"""Tests for src.data.trade_csv_importer."""

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest

from src.data.trade_csv_importer import (
    aggregate_trades,
    detect_market,
    import_csv,
    parse_jp_row,
    parse_us_row,
    save_trade_record,
    _parse_date,
    _parse_number,
    _trade_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JP_HEADER = [
    "約定日", "受渡日", "銘柄コード", "銘柄名", "市場名称", "口座区分",
    "取引区分", "売買区分", "信用区分", "弁済期限", "数量［株］",
    "単価［円］", "手数料［円］", "税金等［円］", "諸費用［円］",
    "税区分", "受渡金額［円］", "建約定日", "建単価［円］",
    "建手数料［円］", "建手数料消費税［円］", "金利（支払）〔円〕",
    "金利（受取）〔円〕", "逆日歩／特別空売り料（支払）〔円〕",
    "逆日歩（受取）〔円〕", "貸株料", "事務管理費〔円〕（税抜）",
    "名義書換料〔円〕（税抜）",
]

US_HEADER = [
    "約定日", "受渡日", "ティッカー", "銘柄名", "口座", "取引区分",
    "売買区分", "信用区分", "弁済期限", "決済通貨", "数量［株］",
    "単価［USドル］", "約定代金［USドル］", "為替レート",
    "手数料［USドル］", "税金［USドル］", "受渡金額［USドル］",
    "受渡金額［円］",
]


def _jp_row(
    date="2024/3/4",
    settle_date="2024/3/6",
    code="4063",
    name="信越化学",
    market="Chi-X",
    account="特定",
    txn_type="現物",
    buy_sell="買付",
    shares="100",
    price="6,688.7",
):
    """Build a minimal JP-format row."""
    return [
        date, settle_date, code, name, market, account,
        txn_type, buy_sell, "-", "-", shares, price,
        "487", "48", "0", "-", "669,405", "-",
        "0.0", "0", "0", "0", "0", "0", "0", "0", "0", "0",
    ]


def _us_row(
    date="2025/10/20",
    settle_date="2025/10/22",
    ticker="AVGO",
    name="BROADCOM INC",
    account="特定",
    txn_type="現物",
    buy_sell="買付",
    shares="30",
    price="344.9997",
):
    """Build a minimal US-format row."""
    return [
        date, settle_date, ticker, name, account, txn_type,
        buy_sell, "-", "-", "ＵＳドル", shares, price,
        "10,349.99", "151.190", "20.00", "2.00", "10,371.99", "-",
    ]


# ---------------------------------------------------------------------------
# detect_market
# ---------------------------------------------------------------------------

class TestDetectMarket:
    def test_jp(self):
        assert detect_market(JP_HEADER) == "jp"

    def test_us(self):
        assert detect_market(US_HEADER) == "us"

    def test_unknown(self):
        with pytest.raises(ValueError, match="判別できません"):
            detect_market(["col_a", "col_b", "col_c"])


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

class TestParseHelpers:
    def test_parse_number_basic(self):
        assert _parse_number("2,338.0") == 2338.0

    def test_parse_number_quoted(self):
        assert _parse_number('"100"') == 100.0

    def test_parse_number_dash(self):
        assert _parse_number("-") == 0.0

    def test_parse_number_empty(self):
        assert _parse_number("") == 0.0

    def test_parse_date(self):
        assert _parse_date("2024/3/5") == "2024-03-05"

    def test_parse_date_quoted(self):
        assert _parse_date('"2021/12/22"') == "2021-12-22"

    def test_trade_type_buy(self):
        assert _trade_type("買付") == "buy"

    def test_trade_type_sell(self):
        assert _trade_type("売付") == "sell"

    def test_trade_type_transfer(self):
        assert _trade_type("入庫") == "transfer"

    def test_trade_type_unknown(self):
        assert _trade_type("その他") is None

    def test_trade_type_quoted(self):
        assert _trade_type('"買付"') == "buy"


# ---------------------------------------------------------------------------
# parse_jp_row
# ---------------------------------------------------------------------------

class TestParseJpRow:
    def test_basic_buy(self):
        row = _jp_row()
        result = parse_jp_row(row)
        assert result is not None
        assert result["symbol"] == "4063.T"
        assert result["date"] == "2024-03-04"
        assert result["trade_type"] == "buy"
        assert result["shares"] == 100
        assert result["price"] == 6688.7
        assert result["currency"] == "JPY"

    def test_sell(self):
        row = _jp_row(buy_sell="売付")
        result = parse_jp_row(row)
        assert result["trade_type"] == "sell"

    def test_transfer_with_price(self):
        row = _jp_row(buy_sell="入庫", price="309.48")
        result = parse_jp_row(row)
        assert result is not None
        assert result["trade_type"] == "transfer"

    def test_transfer_no_price_skipped(self):
        row = _jp_row(buy_sell="入庫", price="0")
        result = parse_jp_row(row)
        assert result is None

    def test_short_row_skipped(self):
        result = parse_jp_row(["a", "b", "c"])
        assert result is None

    def test_large_shares(self):
        row = _jp_row(shares="1,200")
        result = parse_jp_row(row)
        assert result["shares"] == 1200


# ---------------------------------------------------------------------------
# parse_us_row
# ---------------------------------------------------------------------------

class TestParseUsRow:
    def test_basic_buy(self):
        row = _us_row()
        result = parse_us_row(row)
        assert result is not None
        assert result["symbol"] == "AVGO"
        assert result["date"] == "2025-10-20"
        assert result["trade_type"] == "buy"
        assert result["shares"] == 30
        assert result["price"] == 344.9997
        assert result["currency"] == "USD"

    def test_sell(self):
        row = _us_row(buy_sell="売付")
        result = parse_us_row(row)
        assert result["trade_type"] == "sell"

    def test_transfer_skipped(self):
        row = _us_row(buy_sell="入庫", price="-")
        result = parse_us_row(row)
        assert result is None

    def test_short_row_skipped(self):
        result = parse_us_row(["a", "b"])
        assert result is None


# ---------------------------------------------------------------------------
# aggregate_trades
# ---------------------------------------------------------------------------

class TestAggregateTrades:
    def test_single_trade(self):
        trades = [
            {"date": "2024-03-04", "symbol": "4063.T",
             "trade_type": "buy", "shares": 100, "price": 6688.7,
             "currency": "JPY", "name": "信越化学", "account": "特定"},
        ]
        result = aggregate_trades(trades)
        assert len(result) == 1
        assert result[0]["shares"] == 100
        assert result[0]["price"] == 6688.7

    def test_multiple_same_day(self):
        trades = [
            {"date": "2024-02-26", "symbol": "NVDA",
             "trade_type": "buy", "shares": 25, "price": 759.12,
             "currency": "USD", "name": "NVIDIA", "account": "特定"},
            {"date": "2024-02-26", "symbol": "NVDA",
             "trade_type": "buy", "shares": 15, "price": 759.12,
             "currency": "USD", "name": "NVIDIA", "account": "特定"},
            {"date": "2024-02-26", "symbol": "NVDA",
             "trade_type": "buy", "shares": 10, "price": 758.18,
             "currency": "USD", "name": "NVIDIA", "account": "特定"},
        ]
        result = aggregate_trades(trades)
        assert len(result) == 1
        assert result[0]["shares"] == 50
        # Weighted average: (25*759.12 + 15*759.12 + 10*758.18) / 50
        expected = (25 * 759.12 + 15 * 759.12 + 10 * 758.18) / 50
        assert abs(result[0]["price"] - expected) < 0.01

    def test_different_directions_not_merged(self):
        trades = [
            {"date": "2024-02-26", "symbol": "NVDA",
             "trade_type": "buy", "shares": 25, "price": 759.12,
             "currency": "USD", "name": "NVIDIA", "account": "特定"},
            {"date": "2024-02-26", "symbol": "NVDA",
             "trade_type": "sell", "shares": 75, "price": 773.32,
             "currency": "USD", "name": "NVIDIA", "account": "特定"},
        ]
        result = aggregate_trades(trades)
        assert len(result) == 2

    def test_different_dates_not_merged(self):
        trades = [
            {"date": "2024-02-26", "symbol": "VTI",
             "trade_type": "buy", "shares": 10, "price": 250.0,
             "currency": "USD", "name": "VTI", "account": "特定"},
            {"date": "2024-02-27", "symbol": "VTI",
             "trade_type": "buy", "shares": 5, "price": 251.0,
             "currency": "USD", "name": "VTI", "account": "特定"},
        ]
        result = aggregate_trades(trades)
        assert len(result) == 2

    def test_order_preserved(self):
        trades = [
            {"date": "2024-01-01", "symbol": "A",
             "trade_type": "buy", "shares": 10, "price": 100,
             "currency": "USD", "name": "A", "account": ""},
            {"date": "2024-02-01", "symbol": "B",
             "trade_type": "buy", "shares": 20, "price": 200,
             "currency": "USD", "name": "B", "account": ""},
        ]
        result = aggregate_trades(trades)
        assert result[0]["symbol"] == "A"
        assert result[1]["symbol"] == "B"


# ---------------------------------------------------------------------------
# save_trade_record
# ---------------------------------------------------------------------------

class TestSaveTradeRecord:
    def test_creates_json(self, tmp_path):
        trade = {
            "symbol": "4063.T", "date": "2024-03-04",
            "trade_type": "buy", "shares": 100, "price": 6688.7,
            "currency": "JPY", "name": "信越化学", "account": "特定",
        }
        path = save_trade_record(trade, str(tmp_path))
        assert Path(path).exists()

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["symbol"] == "4063.T"
        assert data["trade_type"] == "buy"
        assert data["shares"] == 100
        assert data["price"] == 6688.7
        assert data["currency"] == "JPY"
        assert data["category"] == "trade"

    def test_filename_format(self, tmp_path):
        trade = {
            "symbol": "AVGO", "date": "2025-10-20",
            "trade_type": "buy", "shares": 30, "price": 344.9997,
            "currency": "USD", "name": "BROADCOM", "account": "特定",
        }
        path = save_trade_record(trade, str(tmp_path))
        assert "2025-10-20_buy_AVGO.json" in path

    def test_no_overwrite_adds_index(self, tmp_path):
        trade = {
            "symbol": "VTI", "date": "2024-02-26",
            "trade_type": "buy", "shares": 10, "price": 250.0,
            "currency": "USD", "name": "VTI", "account": "特定",
        }
        path1 = save_trade_record(trade, str(tmp_path))
        path2 = save_trade_record(trade, str(tmp_path))
        assert path1 != path2
        assert "_1.json" in path2

    def test_dry_run(self, tmp_path):
        trade = {
            "symbol": "TEST", "date": "2024-01-01",
            "trade_type": "buy", "shares": 1, "price": 100,
            "currency": "USD", "name": "TEST", "account": "",
        }
        path = save_trade_record(trade, str(tmp_path), dry_run=True)
        assert not Path(path).exists()


# ---------------------------------------------------------------------------
# import_csv (end-to-end)
# ---------------------------------------------------------------------------

class TestImportCsv:
    def _write_csv(self, path: Path, header: list[str], rows: list[list[str]]):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)

    def test_jp_import(self, tmp_path):
        csv_file = tmp_path / "test_jp.csv"
        self._write_csv(csv_file, JP_HEADER, [
            _jp_row(date="2024/3/4", code="4063", price="6,688.7"),
            _jp_row(date="2024/3/4", code="7974", buy_sell="売付", price="8,649.0"),
        ])
        result = import_csv(csv_file, str(tmp_path))
        assert result["market"] == "jp"
        assert result["imported"] == 2
        assert result["skipped"] == 0
        assert len(result["files"]) == 2

    def test_us_import(self, tmp_path):
        csv_file = tmp_path / "test_us.csv"
        self._write_csv(csv_file, US_HEADER, [
            _us_row(ticker="AVGO", price="344.99"),
            _us_row(ticker="VTI", buy_sell="売付", price="302.43"),
        ])
        result = import_csv(csv_file, str(tmp_path))
        assert result["market"] == "us"
        assert result["imported"] == 2

    def test_aggregation_in_import(self, tmp_path):
        csv_file = tmp_path / "test_agg.csv"
        self._write_csv(csv_file, US_HEADER, [
            _us_row(date="2024/2/26", ticker="NVDA", shares="25", price="759.12"),
            _us_row(date="2024/2/26", ticker="NVDA", shares="15", price="759.12"),
            _us_row(date="2024/2/26", ticker="NVDA", shares="10", price="744.96"),
        ])
        result = import_csv(csv_file, str(tmp_path))
        assert result["raw_rows"] == 3
        assert result["aggregated"] == 1
        assert result["imported"] == 1

    def test_skip_existing(self, tmp_path):
        csv_file = tmp_path / "test_skip.csv"
        self._write_csv(csv_file, JP_HEADER, [
            _jp_row(date="2024/3/4", code="4063", price="6,688.7"),
        ])
        # First import
        r1 = import_csv(csv_file, str(tmp_path))
        assert r1["imported"] == 1
        # Second import — should skip
        r2 = import_csv(csv_file, str(tmp_path))
        assert r2["imported"] == 0
        assert r2["skipped"] == 1

    def test_dry_run(self, tmp_path):
        csv_file = tmp_path / "test_dry.csv"
        self._write_csv(csv_file, JP_HEADER, [_jp_row()])
        result = import_csv(csv_file, str(tmp_path), dry_run=True)
        assert result["imported"] == 1
        # Check no files were actually created
        trade_dir = tmp_path / "trade"
        if trade_dir.exists():
            assert len(list(trade_dir.glob("*.json"))) == 0

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_csv(tmp_path / "nonexistent.csv", str(tmp_path))

    def test_force_market(self, tmp_path):
        csv_file = tmp_path / "test_force.csv"
        self._write_csv(csv_file, JP_HEADER, [_jp_row()])
        result = import_csv(csv_file, str(tmp_path), market="jp")
        assert result["market"] == "jp"
        assert result["imported"] == 1
