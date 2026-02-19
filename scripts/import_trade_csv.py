#!/usr/bin/env python3
"""Import trade history from Rakuten Securities CSV exports.

Usage
-----
# Import specific CSV files:
python scripts/import_trade_csv.py data/history/trade/tradehistory(JP)_20260219.csv
python scripts/import_trade_csv.py data/history/trade/tradehistory(US)_20260219.csv

# Import all matching CSVs in a directory:
python scripts/import_trade_csv.py data/history/trade/

# Dry run (parse & report without writing):
python scripts/import_trade_csv.py --dry-run data/history/trade/tradehistory(JP)_20260219.csv

# Force market type:
python scripts/import_trade_csv.py --market jp data/history/trade/some_file.csv
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.trade_csv_importer import import_csv

_DEFAULT_HISTORY_DIR = str(Path(_PROJECT_ROOT) / "data" / "history")


def _find_csv_files(path: Path) -> list[Path]:
    """If *path* is a directory, find all trade CSV files in it."""
    if path.is_file():
        return [path]
    if path.is_dir():
        patterns = ["tradehistory*.csv", "trade_history*.csv"]
        files = []
        for pat in patterns:
            files.extend(sorted(path.glob(pat)))
        if not files:
            # Fallback: all CSVs
            files = sorted(path.glob("*.csv"))
        return files
    return []


def main():
    parser = argparse.ArgumentParser(
        description="楽天証券の取引履歴CSVをインポートする",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="CSVファイルまたはCSVファイルを含むディレクトリ",
    )
    parser.add_argument(
        "--market",
        choices=["jp", "us"],
        default=None,
        help="市場を強制指定（省略時は自動判定）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="保存せずに解析結果だけ表示",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="既存ファイルがあっても再インポート（デフォルト: スキップ）",
    )
    parser.add_argument(
        "--base-dir",
        default=_DEFAULT_HISTORY_DIR,
        help=f"履歴保存ディレクトリ（デフォルト: {_DEFAULT_HISTORY_DIR}）",
    )

    args = parser.parse_args()

    # Collect CSV files
    csv_files: list[Path] = []
    for p in args.paths:
        csv_files.extend(_find_csv_files(Path(p)))

    if not csv_files:
        print("エラー: CSVファイルが見つかりません", file=sys.stderr)
        sys.exit(1)

    total_imported = 0
    total_skipped = 0
    total_errors = 0

    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"📂 {csv_file.name}")
        print(f"{'='*60}")

        try:
            result = import_csv(
                csv_file,
                base_dir=args.base_dir,
                market=args.market,
                dry_run=args.dry_run,
                skip_existing=not args.no_skip,
            )
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            total_errors += 1
            continue

        mode = "DRY RUN" if args.dry_run else "IMPORTED"
        print(f"  市場: {result['market'].upper()}")
        print(f"  CSVの行数: {result['raw_rows']}")
        print(f"  集約後の取引数: {result['aggregated']}")
        print(f"  {mode}: {result['imported']}")
        print(f"  スキップ（既存）: {result['skipped']}")

        if result["errors"]:
            print(f"  ⚠️  パースエラー: {len(result['errors'])}")
            for err in result["errors"][:5]:
                print(f"    - {err}")
            if len(result["errors"]) > 5:
                print(f"    ... 他 {len(result['errors']) - 5} 件")

        if result["files"] and not args.dry_run:
            print(f"\n  保存されたファイル:")
            for fp in result["files"][:10]:
                print(f"    ✅ {Path(fp).name}")
            if len(result["files"]) > 10:
                print(f"    ... 他 {len(result['files']) - 10} 件")

        total_imported += result["imported"]
        total_skipped += result["skipped"]
        total_errors += len(result["errors"])

    print(f"\n{'='*60}")
    print(f"📊 合計: {total_imported} 件インポート, "
          f"{total_skipped} 件スキップ, "
          f"{total_errors} 件エラー")
    if args.dry_run:
        print("   ※ dry-run モードのため実際には保存されていません")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
