"""LLM ベースのニュース分析モジュール.

``copilot_client`` を通じて GitHub Copilot CLI を呼び出し、
経済ニュースのカテゴリ分類・ポートフォリオ影響分析を行う。

CLI の実行・モデル管理・ログ記録は ``copilot_client`` に委譲し、
本モジュールはニュース固有のプロンプト構築・レスポンス解析・キャッシュに集中する。

利用条件:
  - ``copilot`` CLI がインストール済みで GitHub 認証済み

``copilot`` が未インストールの場合は ``is_available()`` が False を返すため、
呼び出し側でキーワードベースのフォールバックに切り替える。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from components.copilot_client import (
    AVAILABLE_MODELS,  # re-export: app.py からの参照を維持
    DEFAULT_MODEL,
    call as copilot_call,
    is_available,      # re-export
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# デフォルト設定
# ---------------------------------------------------------------------------

# デフォルトの分析キャッシュ TTL（秒）。
# ニュースが変わらなければ LLM 再分析をスキップして Premium Request を節約する。
DEFAULT_CACHE_TTL_SEC: int = 3600  # 1 時間

# キャッシュ TTL の UI 選択肢: (label, seconds)
CACHE_TTL_OPTIONS: list[tuple[str, int]] = [
    ("1時間", 3600),
    ("3時間", 10800),
    ("6時間", 21600),
    ("12時間", 43200),
    ("24時間", 86400),
]

# カテゴリ → icon / label の正規マッピング
_CATEGORY_ICONS: dict[str, dict[str, str]] = {
    "金利": {"icon": "🏦", "label": "金利・金融政策"},
    "為替": {"icon": "💱", "label": "為替"},
    "地政学": {"icon": "🌍", "label": "地政学・貿易"},
    "景気": {"icon": "📊", "label": "景気・経済指標"},
    "テクノロジー": {"icon": "💻", "label": "テクノロジー"},
    "エネルギー": {"icon": "⛽", "label": "エネルギー"},
}


# ---------------------------------------------------------------------------
# 分析キャッシュ: ニュースが変わらなければ LLM 再呼び出しをスキップ
# ---------------------------------------------------------------------------
_analysis_cache: dict[str, Any] = {
    "hash": "",          # ニュースタイトル一覧の SHA-256
    "results": None,     # 前回の分析結果
    "timestamp": 0.0,    # 前回分析時刻 (time.time())
    "model": "",         # 前回使用モデル
}


def _compute_news_hash(news_items: list[dict]) -> str:
    """ニュースタイトルのリストから決定的ハッシュを生成する."""
    titles = sorted(item.get("title", "") for item in news_items if item.get("title"))
    return hashlib.sha256("\n".join(titles).encode()).hexdigest()


def get_cache_info() -> dict[str, Any]:
    """現在のキャッシュ状態を返す（UI 表示用）."""
    ts = _analysis_cache["timestamp"]
    if ts == 0:
        return {"cached": False, "age_sec": 0, "model": ""}
    age = time.time() - ts
    return {
        "cached": True,
        "age_sec": int(age),
        "model": _analysis_cache["model"],
    }


def clear_cache() -> None:
    """分析キャッシュを強制クリアする."""
    _analysis_cache["hash"] = ""
    _analysis_cache["results"] = None
    _analysis_cache["timestamp"] = 0.0
    _analysis_cache["model"] = ""


def analyze_news_batch(
    news_items: list[dict],
    positions: list[dict],
    *,
    model: str | None = None,
    timeout: int = 60,
    cache_ttl: int = DEFAULT_CACHE_TTL_SEC,
) -> list[dict] | None:
    """ニュース一覧を Copilot CLI でバッチ分析する.

    Parameters
    ----------
    news_items : list[dict]
        ``fetch_economic_news`` が収集した生ニュースリスト.
        各要素に ``title``, ``publisher``, ``source_name`` が含まれる.
    positions : list[dict]
        ポートフォリオの保有銘柄リスト.
    model : str | None
        モデル ID (``copilot --model`` の値). 省略時は ``_DEFAULT_MODEL``.
    timeout : int
        CLI タイムアウト秒数.
    cache_ttl : int
        分析結果のキャッシュ有効期間（秒）。ニュースが同一かつ TTL 内なら
        LLM を再呼び出しせずキャッシュを返す。0 でキャッシュ無効。

    Returns
    -------
    list[dict] | None
        分析結果リスト. 各要素::

            {
                "id": int,
                "categories": [...],
                "impact_level": str,
                "affected_holdings": [...],
                "reason": str,
            }

        CLI が利用不可／失敗した場合は ``None`` を返す.
        呼び出し側はキーワードベースにフォールバックすること.
    """
    if not is_available():
        return None

    mdl = model or DEFAULT_MODEL

    # --- キャッシュチェック ---
    news_hash = _compute_news_hash(news_items)
    if (
        cache_ttl > 0
        and _analysis_cache["results"] is not None
        and _analysis_cache["hash"] == news_hash
        and _analysis_cache["model"] == mdl
        and (time.time() - _analysis_cache["timestamp"]) < cache_ttl
    ):
        age = int(time.time() - _analysis_cache["timestamp"])
        logger.info(
            "[llm_analyzer] cache hit (age=%ds, ttl=%ds) — skipping LLM call",
            age, cache_ttl,
        )
        return _analysis_cache["results"]

    # ポートフォリオ概要
    pf_summary = _build_portfolio_summary(positions)

    # ニュースリスト
    news_list: list[dict[str, Any]] = []
    for i, item in enumerate(news_items):
        title = item.get("title", "")
        if not title:
            continue
        news_list.append({
            "id": i,
            "title": title,
            "publisher": item.get("publisher", ""),
            "source": item.get("source_name", ""),
        })

    if not news_list:
        return []

    prompt = _build_analysis_prompt(news_list, pf_summary)

    try:
        raw = copilot_call(
            prompt, model=mdl, timeout=timeout,
            source="news_analysis",
        )
        if raw is None:
            return None
        results = _parse_response(raw, len(news_items))
        # --- キャッシュ更新 ---
        if results is not None:
            _analysis_cache["hash"] = news_hash
            _analysis_cache["results"] = results
            _analysis_cache["timestamp"] = time.time()
            _analysis_cache["model"] = mdl
        return results
    except Exception as exc:
        logger.warning("News analysis failed: %s", exc)
        return None


# =====================================================================
# internal helpers
# =====================================================================


def _build_portfolio_summary(positions: list[dict]) -> str:
    """ポートフォリオの概要テキストを生成."""
    lines: list[str] = []
    for p in positions:
        sym = p.get("symbol", "")
        sector = p.get("sector", "")
        currency = p.get("currency", "JPY")
        weight = p.get("weight_pct", 0)
        if sector == "Cash" or not sym:
            continue
        lines.append(f"- {sym}: セクター={sector}, 通貨={currency}, 比率={weight:.1f}%")
    return "\n".join(lines) if lines else "（保有銘柄なし）"


def _build_analysis_prompt(news_list: list[dict], pf_summary: str) -> str:
    """分析用プロンプトを構築."""
    news_text = json.dumps(news_list, ensure_ascii=False, indent=2)

    return f"""あなたは経済ニュース分析の専門家です。以下のニュース一覧をポートフォリオの観点から分析してください。

## ポートフォリオ保有銘柄
{pf_summary}

## ニュース一覧
{news_text}

## タスク
各ニュースについて以下を分析し、JSON配列で返してください:

1. **categories**: ニュースの影響カテゴリ（複数可）。以下から選択:
   - 金利: 金利・中央銀行・金融政策関連
   - 為替: 為替レート・通貨関連
   - 地政学: 地政学リスク・貿易摩擦・制裁関連
   - 景気: 景気動向・経済指標・雇用関連
   - テクノロジー: テック業界・AI・半導体関連
   - エネルギー: 原油・エネルギー関連

2. **impact_level**: ポートフォリオへの影響度（"high"/"medium"/"low"/"none"）
   - high: 保有銘柄の多くに直接影響がある重大ニュース
   - medium: 一部の保有銘柄やセクターに影響
   - low: 間接的な影響の可能性
   - none: ポートフォリオへの影響なし

3. **affected_holdings**: 影響を受ける保有銘柄のシンボルリスト

4. **reason**: 影響の理由（日本語、50文字以内）

## 出力形式
以下のJSON配列のみを返してください。説明文は不要です:
```json
[
  {{"id": 0, "categories": ["金利"], "impact_level": "medium", "affected_holdings": ["7203.T"], "reason": "日銀の利上げで自動車ローン金利に影響"}}
]
```

categoriesは文字列の配列で、カテゴリ名のみを返してください（icon/labelは呼び出し側で付与します）。"""


def _parse_response(raw_text: str, expected_count: int) -> list[dict] | None:
    """Copilot CLI 応答から JSON 配列を抽出・パースする."""
    text = raw_text.strip()

    # ```json ... ``` ブロックを抽出
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # [ で始まる JSON を探す
    if not text.startswith("["):
        idx = text.find("[")
        if idx >= 0:
            text = text[idx:]
        else:
            return None

    # 末尾の ] まで
    last_bracket = text.rfind("]")
    if last_bracket >= 0:
        text = text[: last_bracket + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Copilot CLI JSON response")
        return None

    if not isinstance(parsed, list):
        return None

    # 各アイテムを正規化
    results: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue

        # categories: 文字列リスト → dict リストに変換
        raw_cats = item.get("categories", [])
        categories: list[dict] = []
        for cat in raw_cats:
            if isinstance(cat, str):
                cat_name = cat
            elif isinstance(cat, dict):
                cat_name = cat.get("category", "")
            else:
                continue
            if cat_name in _CATEGORY_ICONS:
                categories.append({
                    "category": cat_name,
                    **_CATEGORY_ICONS[cat_name],
                })

        results.append({
            "id": item.get("id", len(results)),
            "categories": categories,
            "impact_level": item.get("impact_level", "none"),
            "affected_holdings": item.get("affected_holdings", []),
            "reason": item.get("reason", ""),
        })

    return results


# =====================================================================
# ニュースサマリー生成
# =====================================================================

# サマリーキャッシュ — ニュースが変わらなければ再生成しない
_summary_cache: dict[str, Any] = {
    "hash": "",
    "result": None,
    "timestamp": 0.0,
    "model": "",
}


def get_summary_cache_info() -> dict[str, Any]:
    """サマリーキャッシュの状態を返す."""
    ts = _summary_cache["timestamp"]
    if ts == 0:
        return {"cached": False, "age_sec": 0, "model": ""}
    age = time.time() - ts
    return {"cached": True, "age_sec": int(age), "model": _summary_cache["model"]}


def clear_summary_cache() -> None:
    """サマリーキャッシュを強制クリアする."""
    _summary_cache["hash"] = ""
    _summary_cache["result"] = None
    _summary_cache["timestamp"] = 0.0
    _summary_cache["model"] = ""


def generate_news_summary(
    news_items: list[dict],
    positions: list[dict],
    *,
    model: str | None = None,
    timeout: int = 60,
    cache_ttl: int = DEFAULT_CACHE_TTL_SEC,
) -> dict | None:
    """LLM 分析済みニュースからサマリーを生成する.

    Parameters
    ----------
    news_items : list[dict]
        ``fetch_economic_news`` が返した分析済みニュースリスト.
        ``analysis_method == "llm"`` のニュースが含まれている前提.
    positions : list[dict]
        ポートフォリオ保有銘柄リスト.
    model : str | None
        モデル ID. 省略時はデフォルトモデル.
    timeout : int
        CLI タイムアウト秒数.
    cache_ttl : int
        キャッシュ有効期間（秒）.

    Returns
    -------
    dict | None
        サマリー結果::

            {
                "overview": str,          # 全体概要（2-3文）
                "key_points": [           # カテゴリ別ポイント
                    {
                        "category": str,  # カテゴリ名
                        "icon": str,      # アイコン
                        "summary": str,   # 要点
                        "news_ids": [int],# 関連ニュースID（トレース用）
                    },
                ],
                "portfolio_alert": str,   # PFへの注意点（あれば）
            }

        失敗時は ``None``.
    """
    if not is_available():
        return None

    mdl = model or DEFAULT_MODEL

    # --- キャッシュチェック ---
    news_hash = _compute_news_hash(news_items)
    if (
        cache_ttl > 0
        and _summary_cache["result"] is not None
        and _summary_cache["hash"] == news_hash
        and _summary_cache["model"] == mdl
        and (time.time() - _summary_cache["timestamp"]) < cache_ttl
    ):
        logger.info("[llm_analyzer] summary cache hit — skipping LLM call")
        return _summary_cache["result"]

    # プロンプト構築
    prompt = _build_summary_prompt(news_items, positions)

    try:
        raw = copilot_call(
            prompt, model=mdl, timeout=timeout,
            source="news_summary",
        )
        if raw is None:
            return None
        result = _parse_summary_response(raw)
        if result is not None:
            _summary_cache["hash"] = news_hash
            _summary_cache["result"] = result
            _summary_cache["timestamp"] = time.time()
            _summary_cache["model"] = mdl
        return result
    except Exception as exc:
        logger.warning("News summary generation failed: %s", exc)
        return None


def _build_summary_prompt(news_items: list[dict], positions: list[dict]) -> str:
    """サマリー生成用プロンプトを構築する."""
    pf_summary = _build_portfolio_summary(positions)

    # ニュースをコンパクトにまとめる
    news_lines: list[str] = []
    for i, item in enumerate(news_items):
        title = item.get("title", "")
        if not title:
            continue
        impact = item.get("portfolio_impact", {}).get("impact_level", "none")
        cats = ", ".join(
            c.get("category", "") if isinstance(c, dict) else str(c)
            for c in item.get("categories", [])
        )
        reason = item.get("portfolio_impact", {}).get("reason", "")
        news_lines.append(f"[{i}] {title} | 影響={impact} | 分野={cats} | 理由={reason}")

    news_text = "\n".join(news_lines)

    return f"""あなたは経済ニュースの要約アナリストです。以下の分析済みニュースを読み、投資家向けの簡潔なサマリーを作成してください。

## ポートフォリオ保有銘柄
{pf_summary}

## 分析済みニュース一覧
（各行: [ID] タイトル | 影響度 | 分野 | 理由）
{news_text}

## タスク
以下のJSON形式でサマリーを返してください:

```json
{{
  "overview": "全体概要を2-3文で。今日のニュースの全体的なトーン（リスクオン/オフ、注目テーマ等）を述べる",
  "key_points": [
    {{
      "category": "カテゴリ名（金利/為替/地政学/景気/テクノロジー/エネルギー のいずれか）",
      "summary": "そのカテゴリに関する要点を1-2文で。具体的なニュースに言及する",
      "news_ids": [0, 1]
    }}
  ],
  "portfolio_alert": "ポートフォリオへの注意点を1-2文で。影響度が高いニュースがあればその要約。なければ空文字"
}}
```

## 制約
- key_points は実際にニュースがあるカテゴリのみ。最大4カテゴリ
- news_ids はそのカテゴリに関連するニュースの[ID]番号
- overview, summary, portfolio_alert は日本語で記述
- JSONのみを返すこと。説明文は不要"""


def _parse_summary_response(raw_text: str) -> dict | None:
    """サマリー応答をパースする."""
    text = raw_text.strip()

    # ```json ... ``` ブロックを抽出
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # { で始まる JSON を探す（配列 [ は拒否）
    if text.startswith("["):
        return None
    if not text.startswith("{"):
        idx = text.find("{")
        if idx >= 0:
            text = text[idx:]
        else:
            return None

    # 末尾の } まで
    last_brace = text.rfind("}")
    if last_brace >= 0:
        text = text[: last_brace + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse summary JSON response")
        return None

    if not isinstance(parsed, dict):
        return None

    overview = parsed.get("overview", "")
    portfolio_alert = parsed.get("portfolio_alert", "")
    raw_points = parsed.get("key_points", [])

    key_points: list[dict] = []
    for pt in raw_points:
        if not isinstance(pt, dict):
            continue
        cat_name = pt.get("category", "")
        cat_info = _CATEGORY_ICONS.get(cat_name, {})
        key_points.append({
            "category": cat_name,
            "icon": cat_info.get("icon", "📌"),
            "label": cat_info.get("label", cat_name),
            "summary": pt.get("summary", ""),
            "news_ids": pt.get("news_ids", []),
        })

    return {
        "overview": overview,
        "key_points": key_points,
        "portfolio_alert": portfolio_alert,
    }


# =====================================================================
# ヘルスチェックサマリー生成
# =====================================================================

# ヘルスチェックサマリーキャッシュ
_health_summary_cache: dict[str, Any] = {
    "hash": "",
    "result": None,
    "timestamp": 0.0,
    "model": "",
}


def get_health_summary_cache_info() -> dict[str, Any]:
    """ヘルスチェックサマリーキャッシュの状態を返す."""
    ts = _health_summary_cache["timestamp"]
    if ts == 0:
        return {"cached": False, "age_sec": 0, "model": ""}
    age = time.time() - ts
    return {"cached": True, "age_sec": int(age), "model": _health_summary_cache["model"]}


def clear_health_summary_cache() -> None:
    """ヘルスチェックサマリーキャッシュを強制クリアする."""
    _health_summary_cache["hash"] = ""
    _health_summary_cache["result"] = None
    _health_summary_cache["timestamp"] = 0.0
    _health_summary_cache["model"] = ""


def _compute_health_hash(health_data: dict, news_items: list[dict] | None = None) -> str:
    """ヘルスチェックデータのハッシュを生成する."""
    positions = health_data.get("positions", [])
    parts = []
    for p in positions:
        parts.append(f"{p.get('symbol', '')}:{p.get('alert_level', '')}:{p.get('pnl_pct', 0)}")
    # ニュースタイトルもハッシュに含める（ニュースが変われば再分析）
    if news_items:
        for n in news_items:
            title = n.get("title", "")
            if title:
                parts.append(f"news:{title}")
    return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()


def generate_health_summary(
    health_data: dict,
    *,
    news_items: list[dict] | None = None,
    model: str | None = None,
    timeout: int = 60,
    cache_ttl: int = DEFAULT_CACHE_TTL_SEC,
) -> dict | None:
    """ヘルスチェック結果＋関連ニュース＋ファンダメンタルデータからLLMサマリーを生成する.

    Parameters
    ----------
    health_data : dict
        ``run_dashboard_health_check()`` が返すヘルスチェック結果.
        各 position にはファンダメンタルデータ（PER, PBR, ROE 等）が含まれる.
    news_items : list[dict] | None
        経済ニュースリスト. PF影響のあるニュースのみを渡すことを推奨.
    model : str | None
        モデル ID. 省略時はデフォルトモデル.
    timeout : int
        CLI タイムアウト秒数.
    cache_ttl : int
        キャッシュ有効期間（秒）.

    Returns
    -------
    dict | None
        サマリー結果::

            {
                "overview": str,             # 全体概要（2-3文）
                "stock_assessments": [        # 銘柄別評価
                    {
                        "symbol": str,
                        "name": str,
                        "assessment": str,   # 1-2文の評価
                        "action": str,       # 推奨アクション
                    },
                ],
                "risk_warning": str,         # リスク警告（あれば）
            }

        失敗時は ``None``.
    """
    if not is_available():
        return None

    mdl = model or DEFAULT_MODEL

    # --- キャッシュチェック ---
    health_hash = _compute_health_hash(health_data, news_items)
    if (
        cache_ttl > 0
        and _health_summary_cache["result"] is not None
        and _health_summary_cache["hash"] == health_hash
        and _health_summary_cache["model"] == mdl
        and (time.time() - _health_summary_cache["timestamp"]) < cache_ttl
    ):
        logger.info("[llm_analyzer] health summary cache hit — skipping LLM call")
        return _health_summary_cache["result"]

    prompt = _build_health_summary_prompt(health_data, news_items=news_items)

    try:
        raw = copilot_call(
            prompt, model=mdl, timeout=timeout,
            source="health_summary",
        )
        if raw is None:
            return None
        result = _parse_health_summary_response(raw)
        if result is not None:
            _health_summary_cache["hash"] = health_hash
            _health_summary_cache["result"] = result
            _health_summary_cache["timestamp"] = time.time()
            _health_summary_cache["model"] = mdl
        return result
    except Exception as exc:
        logger.warning("Health summary generation failed: %s", exc)
        return None


def _build_health_summary_prompt(
    health_data: dict,
    *,
    news_items: list[dict] | None = None,
) -> str:
    """ヘルスチェックサマリー用プロンプトを構築する.

    ヘルスチェック結果・ファンダメンタルデータ・関連ニュースを統合して
    LLM に渡すプロンプトを生成する。
    """
    summary = health_data.get("summary", {})
    positions = health_data.get("positions", [])
    sell_alerts = health_data.get("sell_alerts", [])

    # ポジション情報をコンパクトにまとめる（ヘルスチェック結果 + ファンダメンタル）
    pos_lines: list[str] = []
    for p in positions:
        symbol = p.get("symbol", "")
        name = p.get("name", symbol)
        alert = p.get("alert_level", "none")
        trend = p.get("trend", "不明")
        rsi = p.get("rsi", 0)
        pnl = p.get("pnl_pct", 0)
        reasons = " / ".join(p.get("alert_reasons", [])) if p.get("alert_reasons") else "-"
        trap = "バリュートラップ" if p.get("value_trap") else ""
        cross = p.get("cross_signal", "none")
        cross_str = ""
        if cross == "golden_cross":
            cross_str = f"GC({p.get('days_since_cross', '?')}日前)"
        elif cross == "death_cross":
            cross_str = f"DC({p.get('days_since_cross', '?')}日前)"
        quality = p.get("change_quality", "")
        stability = p.get("return_stability", "")

        extras = " / ".join(filter(None, [trap, cross_str, quality, stability]))

        # ファンダメンタルデータ
        fund_parts: list[str] = []
        per = p.get("per")
        if per is not None:
            fund_parts.append(f"PER={per:.1f}")
        pbr = p.get("pbr")
        if pbr is not None:
            fund_parts.append(f"PBR={pbr:.2f}")
        roe = p.get("roe")
        if roe is not None:
            fund_parts.append(f"ROE={roe * 100:.1f}%")
        rev_g = p.get("revenue_growth")
        if rev_g is not None:
            fund_parts.append(f"売上成長={rev_g * 100:+.1f}%")
        earn_g = p.get("earnings_growth")
        if earn_g is not None:
            fund_parts.append(f"利益成長={earn_g * 100:+.1f}%")
        div_y = p.get("dividend_yield")
        if div_y is not None:
            fund_parts.append(f"配当={div_y * 100:.2f}%")
        fwd_eps = p.get("forward_eps")
        trail_eps = p.get("trailing_eps")
        if fwd_eps is not None and trail_eps is not None:
            if trail_eps != 0:
                eps_chg = ((fwd_eps / trail_eps) - 1) * 100
                fund_parts.append(f"EPS方向={eps_chg:+.1f}%")
        sector = p.get("sector", "")
        industry = p.get("industry", "")

        fund_str = ", ".join(fund_parts) if fund_parts else ""
        sector_str = f"{sector}/{industry}" if sector else ""

        line = (
            f"- {name}({symbol}): 判定={alert}, トレンド={trend}, "
            f"RSI={rsi:.1f}, 損益={pnl:+.1f}%, 理由={reasons}"
        )
        if extras:
            line += f", 補足={extras}"
        if sector_str:
            line += f", セクター={sector_str}"
        if fund_str:
            line += f", ファンダ=[{fund_str}]"
        pos_lines.append(line)

    pos_text = "\n".join(pos_lines) if pos_lines else "（保有銘柄なし）"

    # 売りアラート情報
    alert_lines: list[str] = []
    for a in sell_alerts:
        alert_lines.append(
            f"- {a.get('name', '')}({a.get('symbol', '')}): "
            f"緊急度={a.get('urgency', '')}, アクション={a.get('action', '')}, "
            f"理由={a.get('reason', '')}"
        )
    alert_text = "\n".join(alert_lines) if alert_lines else "（なし）"

    # 関連ニュース情報
    news_section = ""
    if news_items:
        # PF影響があるニュースを優先的に含める
        impact_news = [
            n for n in news_items
            if n.get("portfolio_impact", {}).get("impact_level", "none") != "none"
        ]
        # 影響ニュースが少なければ、その他のニュースも少し含める
        other_news = [
            n for n in news_items
            if n.get("portfolio_impact", {}).get("impact_level", "none") == "none"
        ]

        news_lines: list[str] = []
        for n in impact_news[:10]:
            impact = n.get("portfolio_impact", {})
            affected = ", ".join(impact.get("affected_holdings", []))
            reason = impact.get("reason", "")
            level = impact.get("impact_level", "none")
            news_lines.append(
                f"- [{level}] {n.get('title', '')}"
                + (f" → 影響銘柄: {affected}" if affected else "")
                + (f" ({reason})" if reason else "")
            )
        for n in other_news[:5]:
            news_lines.append(f"- [参考] {n.get('title', '')}")

        if news_lines:
            news_text = "\n".join(news_lines)
            news_section = f"""
## 関連ニュース（経済ニュース & PF影響）
{news_text}
"""

    return f"""あなたはポートフォリオのヘルスチェック分析の専門家です。以下のヘルスチェック結果、ファンダメンタルデータ、関連ニュースを総合的に読み、投資家向けの簡潔なサマリーを作成してください。

## サマリー統計
- 合計: {summary.get('total', 0)}銘柄
- 健全: {summary.get('healthy', 0)}, 早期警告: {summary.get('early_warning', 0)}, 注意: {summary.get('caution', 0)}, 撤退: {summary.get('exit', 0)}

## 各銘柄のヘルスチェック結果 & ファンダメンタルデータ
{pos_text}

## 売りタイミング通知
{alert_text}
{news_section}
## タスク
以下のJSON形式でサマリーを返してください:

```json
{{
  "overview": "ポートフォリオ全体の健全性を2-3文で評価。テクニカル面・ファンダメンタル面・ニュースの3観点から総合判断する",
  "stock_assessments": [
    {{
      "symbol": "銘柄シンボル",
      "name": "銘柄名",
      "assessment": "この銘柄の現状を1-2文で評価（ヘルスチェック結果、バリュエーション、ニュースを踏まえて）",
      "action": "推奨アクション（保有継続/一部利確/損切り検討/注視 等）"
    }}
  ],
  "risk_warning": "ポートフォリオ全体のリスクに関する注意点（ニュースから読み取れるリスクも含む）。なければ空文字"
}}
```

## 制約
- stock_assessments はアラートがある銘柄（alert_level が none 以外）のみ。健全な銘柄は省略
- アラート銘柄がなければ stock_assessments は空配列
- ファンダメンタルデータ（PER, ROE, 成長率等）がある場合は、バリュエーション面の評価も含める
- 関連ニュースがある場合は、ニュースがポートフォリオに与える影響も overview や risk_warning に反映する
- overview, assessment, risk_warning は日本語で簡潔に
- JSONのみを返すこと。説明文は不要"""


def _parse_health_summary_response(raw_text: str) -> dict | None:
    """ヘルスチェックサマリー応答をパースする."""
    text = raw_text.strip()

    # ```json ... ``` ブロックを抽出
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # { で始まる JSON を探す（配列 [ は拒否）
    if text.startswith("["):
        return None
    if not text.startswith("{"):
        idx = text.find("{")
        if idx >= 0:
            text = text[idx:]
        else:
            return None

    # 末尾の } まで
    last_brace = text.rfind("}")
    if last_brace >= 0:
        text = text[: last_brace + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse health summary JSON response")
        return None

    if not isinstance(parsed, dict):
        return None

    overview = parsed.get("overview", "")
    risk_warning = parsed.get("risk_warning", "")
    raw_assessments = parsed.get("stock_assessments", [])

    stock_assessments: list[dict] = []
    for sa in raw_assessments:
        if not isinstance(sa, dict):
            continue
        stock_assessments.append({
            "symbol": sa.get("symbol", ""),
            "name": sa.get("name", ""),
            "assessment": sa.get("assessment", ""),
            "action": sa.get("action", ""),
        })

    return {
        "overview": overview,
        "stock_assessments": stock_assessments,
        "risk_warning": risk_warning,
    }
