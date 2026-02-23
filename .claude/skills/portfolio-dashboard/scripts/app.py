"""ポートフォリオダッシュボード — Streamlit アプリ.

総資産推移 / 銘柄別評価額 / セクター構成 / 月次サマリー を
インタラクティブなグラフで表示する。

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# --- コンポーネントを import ---
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from components.data_loader import (
    get_current_snapshot,
    build_portfolio_history,
    get_sector_breakdown,
    get_monthly_summary,
    get_trade_activity,
    build_projection,
    compute_risk_metrics,
    compute_daily_change,
    compute_benchmark_excess,
    compute_top_worst_performers,
    compute_drawdown_series,
    compute_rolling_sharpe,
    compute_correlation_matrix,
    compute_weight_drift,
    get_benchmark_series,
    run_dashboard_health_check,
    fetch_economic_news,
)
from components.settings_store import load_settings, save_settings, DEFAULTS
from components.llm_analyzer import (
    AVAILABLE_MODELS as LLM_MODELS,
    CACHE_TTL_OPTIONS as LLM_CACHE_OPTIONS,
    is_available as llm_is_available,
    get_cache_info as llm_get_cache_info,
    clear_cache as llm_clear_cache,
    generate_news_summary,
    get_summary_cache_info as llm_get_summary_cache_info,
    clear_summary_cache as llm_clear_summary_cache,
    generate_health_summary,
    get_health_summary_cache_info as llm_get_health_summary_cache_info,
    clear_health_summary_cache as llm_clear_health_summary_cache,
)
from components.copilot_client import (
    get_execution_logs as copilot_get_logs,
    clear_execution_logs as copilot_clear_logs,
    call as copilot_call,
    AVAILABLE_MODELS as COPILOT_MODELS,
)
from components.charts import (
    build_total_chart,
    build_invested_chart,
    build_projection_chart,
    build_sector_chart,
    build_currency_chart,
    build_individual_chart,
    build_monthly_chart,
    build_trade_flow_chart,
    build_drawdown_chart,
    build_rolling_sharpe_chart,
    build_treemap_chart,
    build_correlation_chart,
)

# =====================================================================
# ページ設定
# =====================================================================
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# カスタムCSS
st.markdown("""
<style>
    /* Smooth scroll for TOC anchor navigation */
    html { scroll-behavior: smooth; }
    .positive { color: #4ade80; }
    .negative { color: #f87171; }
    /* TOC link styling */
    .toc-link {
        display: block;
        text-decoration: none;
        padding: 7px 12px;
        border-radius: 6px;
        color: inherit;
        font-size: 0.88rem;
        transition: background 0.2s;
        margin-bottom: 2px;
    }
    .toc-link:hover {
        background: rgba(99,102,241,0.18);
        color: #a5b4fc;
    }
    /* KPI cards — theme-aware */
    .kpi-card {
        background: var(--secondary-background-color);
        border-radius: 12px;
        text-align: center;
    }
    .kpi-main {
        padding: 28px 24px 22px;
        border-bottom: 3px solid rgba(99,102,241,0.5);
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .kpi-sub {
        padding: 14px 16px;
        border-radius: 10px;
    }
    .kpi-risk {
        padding: 10px 6px;
        border-radius: 8px;
        min-width: 0;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 500;
        opacity: 0.65;
        letter-spacing: 0.02em;
        margin-bottom: 5px;
    }
    .kpi-main .kpi-label {
        font-size: 0.88rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-value-sub {
        font-size: 1.25rem;
        font-weight: 600;
    }
    .kpi-value-risk {
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 2px;
    }
    /* KPI row spacing */
    .kpi-spacer { margin-top: 10px; }
    /* Section divider */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(148,163,184,0.2);
        margin: 28px 0 20px 0;
    }
    /* Sell alert banner */
    .sell-alert {
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .sell-alert-critical {
        background: rgba(248,113,113,0.12);
        border-left-color: #f87171;
    }
    .sell-alert-warning {
        background: rgba(251,191,36,0.12);
        border-left-color: #fbbf24;
    }
    .sell-alert-info {
        background: rgba(96,165,250,0.12);
        border-left-color: #60a5fa;
    }
    .sell-alert-header {
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .sell-alert-reason {
        font-size: 0.88rem;
        opacity: 0.85;
        margin-bottom: 4px;
    }
    .sell-alert-detail {
        font-size: 0.82rem;
        opacity: 0.7;
        padding-left: 12px;
    }
    .sell-alert-ai {
        font-size: 0.82rem;
        line-height: 1.5;
        margin-top: 6px;
        padding: 6px 10px;
        background: rgba(99,102,241,0.08);
        border-radius: 6px;
        border-left: 2px solid rgba(99,102,241,0.3);
    }
    /* Health card */
    .health-card {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 8px;
        border-left: 4px solid;
    }
    .health-card-healthy { border-left-color: #4ade80; }
    .health-card-early_warning { border-left-color: #fbbf24; }
    .health-card-caution { border-left-color: #fb923c; }
    .health-card-exit { border-left-color: #f87171; }
    /* News cards */
    .news-card {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 8px;
        border-left: 4px solid #64748b;
        transition: background 0.2s;
    }
    .news-card:hover {
        filter: brightness(1.05);
    }
    .news-impact-high { border-left-color: #f87171; }
    .news-impact-medium { border-left-color: #fbbf24; }
    .news-impact-low { border-left-color: #60a5fa; }
    .news-impact-none { border-left-color: #64748b; }
    .news-title {
        font-weight: 600;
        font-size: 0.92rem;
        line-height: 1.4;
        margin-bottom: 6px;
    }
    .news-title a {
        color: inherit;
        text-decoration: none;
    }
    .news-title a:hover {
        text-decoration: underline;
        opacity: 0.9;
    }
    .news-meta {
        font-size: 0.78rem;
        opacity: 0.6;
        margin-bottom: 6px;
    }
    .news-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 4px;
        margin-bottom: 2px;
    }
    .news-badge-category {
        background: rgba(99,102,241,0.15);
        color: #a5b4fc;
    }
    .news-badge-impact-high {
        background: rgba(248,113,113,0.18);
        color: #fca5a5;
    }
    .news-badge-impact-medium {
        background: rgba(251,191,36,0.18);
        color: #fde68a;
    }
    .news-badge-impact-low {
        background: rgba(96,165,250,0.15);
        color: #93c5fd;
    }
    .news-affected {
        font-size: 0.8rem;
        opacity: 0.75;
        margin-top: 4px;
        padding-left: 4px;
    }
    .news-number {
        display: inline-block;
        background: rgba(148,163,184,0.2);
        color: #94a3b8;
        font-size: 0.68rem;
        font-weight: 700;
        border-radius: 4px;
        padding: 1px 5px;
        margin-right: 6px;
        vertical-align: middle;
    }
    /* Summary card */
    .news-summary-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(59,130,246,0.06));
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 18px 20px;
    }
    .news-summary-header {
        font-weight: 700;
        font-size: 1.0rem;
        margin-bottom: 10px;
    }
    .news-summary-overview {
        font-size: 0.9rem;
        line-height: 1.6;
        margin-bottom: 14px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(148,163,184,0.15);
    }
    .news-summary-points {
        margin-bottom: 12px;
    }
    .news-summary-point {
        margin-bottom: 8px;
        line-height: 1.5;
    }
    .news-summary-cat {
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    .news-summary-text {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .news-ref {
        display: inline-block;
        background: rgba(99,102,241,0.18);
        color: #a5b4fc;
        font-size: 0.68rem;
        font-weight: 700;
        border-radius: 4px;
        padding: 0px 4px;
        margin: 0 1px;
    }
    .news-refs {
        font-size: 0.72rem;
        opacity: 0.7;
    }
    .news-summary-alert {
        background: rgba(251,191,36,0.1);
        border: 1px solid rgba(251,191,36,0.25);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.85rem;
        margin-top: 10px;
    }
    /* Health summary card */
    .health-summary-card {
        background: linear-gradient(135deg, rgba(74,222,128,0.08), rgba(59,130,246,0.06));
        border: 1px solid rgba(74,222,128,0.2);
        border-radius: 12px;
        padding: 18px 20px;
    }
    .health-summary-header {
        font-weight: 700;
        font-size: 1.0rem;
        margin-bottom: 10px;
    }
    .health-summary-overview {
        font-size: 0.9rem;
        line-height: 1.6;
        margin-bottom: 14px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(148,163,184,0.15);
    }
    .health-summary-stocks-toggle > summary {
        font-weight: 600;
        font-size: 0.88rem;
        padding: 6px 0;
        cursor: pointer;
        list-style: none;
        display: flex;
        align-items: center;
        gap: 6px;
        color: #94a3b8;
    }
    .health-summary-stocks-toggle > summary::-webkit-details-marker { display: none; }
    .health-summary-stocks-toggle > summary::before {
        content: '▶';
        font-size: 0.7rem;
        transition: transform 0.2s;
    }
    .health-summary-stocks-toggle[open] > summary::before {
        transform: rotate(90deg);
    }
    .health-summary-stocks-toggle[open] > summary {
        margin-bottom: 8px;
    }
    .health-summary-stock {
        margin-bottom: 8px;
        padding: 8px 12px;
        background: rgba(148,163,184,0.06);
        border-radius: 8px;
        border-left: 3px solid #94a3b8;
    }
    .health-summary-stock-exit {
        border-left-color: #f87171;
    }
    .health-summary-stock-caution {
        border-left-color: #fb923c;
    }
    .health-summary-stock-early_warning {
        border-left-color: #fbbf24;
    }
    .health-summary-stock-name {
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 2px;
    }
    .health-summary-stock-text {
        font-size: 0.82rem;
        opacity: 0.85;
        line-height: 1.5;
    }
    .health-summary-action {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        color: #a5b4fc;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 4px;
        padding: 1px 6px;
        margin-left: 6px;
    }
    .health-summary-warning {
        background: rgba(248,113,113,0.1);
        border: 1px solid rgba(248,113,113,0.25);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.85rem;
        margin-top: 10px;
    }
    /* Copilot Chat */
    .copilot-chat-container {
        background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(139,92,246,0.05));
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 12px;
        padding: 18px 20px;
    }
    .copilot-chat-header {
        font-weight: 700;
        font-size: 1.0rem;
        margin-bottom: 6px;
    }
    .copilot-chat-context-badge {
        display: inline-block;
        background: rgba(74,222,128,0.12);
        color: #4ade80;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 4px;
        padding: 2px 8px;
        margin-right: 4px;
    }
    .copilot-chat-msg {
        margin-bottom: 10px;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .copilot-chat-msg-user {
        background: rgba(99,102,241,0.12);
        border-left: 3px solid rgba(99,102,241,0.5);
    }
    .copilot-chat-msg-ai {
        background: rgba(148,163,184,0.08);
        border-left: 3px solid rgba(148,163,184,0.3);
    }
    .copilot-chat-msg-role {
        font-weight: 600;
        font-size: 0.78rem;
        opacity: 0.7;
        margin-bottom: 3px;
    }
    .copilot-chat-msg-text {
        white-space: pre-wrap;
        word-break: break-word;
    }
    .copilot-chat-thinking {
        font-size: 0.82rem;
        opacity: 0.6;
        padding: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# データ取得（キャッシュ付き）— サイドバーより先に定義
# =====================================================================
@st.cache_data(ttl=300, show_spinner="データを取得中...")
def load_snapshot():
    return get_current_snapshot()


@st.cache_data(ttl=300, show_spinner="株価履歴を取得中...")
def load_history(period_val: str):
    return build_portfolio_history(period=period_val)


@st.cache_data(ttl=300, show_spinner="取引データを集計中...")
def load_trade_activity():
    return get_trade_activity()


@st.cache_data(ttl=600, show_spinner="ヘルスチェック実行中...")
def load_health_check():
    return run_dashboard_health_check()


@st.cache_data(ttl=600, show_spinner="経済ニュースを取得中...")
def load_economic_news(
    _positions_key: str,
    positions: list,
    fx_rates: dict,
    llm_enabled: bool = False,
    llm_model: str | None = None,
    llm_cache_ttl: int = 3600,
):
    """経済ニュースを取得してPF影響を分析する.

    _positions_key はキャッシュキー用（保有銘柄が変わったら再取得）。
    llm_enabled / llm_model でLLM分析の有無・モデルをキャッシュキーに含む。
    llm_cache_ttl はLLM分析結果のキャッシュ有効期間（秒）。
    """
    return fetch_economic_news(
        positions=positions,
        fx_rates=fx_rates,
        llm_enabled=llm_enabled,
        llm_model=llm_model,
        llm_cache_ttl=llm_cache_ttl,
    )


# =====================================================================
# サイドバー（タブ: 目次 / 設定）
# =====================================================================
st.sidebar.title("📊 Portfolio Dashboard")

_tab_toc, _tab_settings, _tab_help = st.sidebar.tabs(["📑 目次", "⚙️ 設定", "❓ 用語集"])

# --- 目次タブ ---
with _tab_toc:
    st.markdown(
        '<div style="display:flex; flex-direction:column; gap:2px; padding:4px 0;">'
        '<a class="toc-link" href="#summary">📈 サマリー</a>'
        '<a class="toc-link" href="#health-check">🏥 ヘルスチェック</a>'
        '<a class="toc-link" href="#economic-news">📰 経済ニュース & PF影響</a>'
        '<a class="toc-link" href="#total-chart">📊 総資産推移</a>'
        '<a class="toc-link" href="#invested-chart">💰 投資額 vs 評価額</a>'
        '<a class="toc-link" href="#projection">🔮 将来推定</a>'
        '<a class="toc-link" href="#holdings">🏢 保有銘柄・構成</a>'
        '<a class="toc-link" href="#individual-chart">📉 銘柄別チャート</a>'
        '<a class="toc-link" href="#monthly">📅 月次サマリー</a>'
        '<a class="toc-link" href="#trade-activity">🔄 売買アクティビティ</a>'
        '<a class="toc-link" href="#copilot-chat">💬 Copilot に相談</a>'
        '</div>',
        unsafe_allow_html=True,
    )

# --- 設定の読み込み ---
if "_saved_settings" not in st.session_state:
    st.session_state["_saved_settings"] = load_settings()
_saved = st.session_state["_saved_settings"]

# --- 設定タブ ---
with _tab_settings:
    _PERIOD_OPTIONS = [
        ("1ヶ月", "1mo"),
        ("3ヶ月", "3mo"),
        ("6ヶ月", "6mo"),
        ("1年", "1y"),
        ("2年", "2y"),
        ("3年", "3y"),
        ("5年", "5y"),
        ("全期間", "max"),
    ]
    _period_labels = [label for label, _ in _PERIOD_OPTIONS]
    _period_saved_idx = _period_labels.index(_saved["period_label"]) if _saved["period_label"] in _period_labels else 1

    period_label = st.selectbox(
        "📅 表示期間",
        options=_period_labels,
        index=_period_saved_idx,
        help="株価履歴の取得期間",
    )
    period = dict(_PERIOD_OPTIONS)[period_label]

    _chart_styles = ["積み上げ面", "折れ線", "積み上げ棒"]
    _chart_saved_idx = _chart_styles.index(_saved["chart_style"]) if _saved["chart_style"] in _chart_styles else 0

    chart_style = st.radio(
        "🎨 チャートスタイル",
        options=_chart_styles,
        index=_chart_saved_idx,
    )

    show_invested = st.checkbox(
        "投資額 vs 評価額を表示",
        value=_saved["show_invested"],
    )

    # ベンチマーク選択
    _BENCHMARK_OPTIONS = {
        "なし": None,
        "S&P 500 (SPY)": "SPY",
        "VTI (米国全体)": "VTI",
        "日経225 (^N225)": "^N225",
        "TOPIX (^TPX)": "1306.T",
    }
    _bench_labels = list(_BENCHMARK_OPTIONS.keys())
    _bench_saved_idx = _bench_labels.index(_saved["benchmark_label"]) if _saved["benchmark_label"] in _bench_labels else 0

    benchmark_label = st.selectbox(
        "📏 ベンチマーク比較",
        options=_bench_labels,
        index=_bench_saved_idx,
        help="総資産推移にベンチマークのパフォーマンスを重ねて表示",
    )
    benchmark_symbol = _BENCHMARK_OPTIONS[benchmark_label]

    show_individual = st.checkbox(
        "銘柄別の個別チャートを表示",
        value=_saved["show_individual"],
    )

    st.markdown("---")

    # --- 目標・推定セクション ---
    st.markdown("#### 🎯 目標・将来推定")

    show_projection = st.checkbox(
        "目標ライン & 将来推定を表示",
        value=_saved["show_projection"],
    )

    target_amount = st.number_input(
        "🎯 目標資産額（万円）",
        min_value=0,
        max_value=100000,
        value=_saved["target_amount_man"],
        step=500,
        help="総資産推移グラフに水平ラインとして表示",
    ) * 10000  # 万円→円

    projection_years = st.slider(
        "📅 推定期間（年）",
        min_value=1,
        max_value=20,
        value=_saved["projection_years"],
        help="現在の保有銘柄のリターン推定に基づく将来推移",
    )

    st.markdown("---")

    # --- データ更新セクション ---
    st.markdown("#### 🔄 データ更新")

    _REFRESH_OPTIONS = [
        ("なし（手動のみ）", 0),
        ("1分", 60),
        ("5分", 300),
        ("15分", 900),
        ("30分", 1800),
        ("1時間", 3600),
    ]
    _refresh_labels = [label for label, _ in _REFRESH_OPTIONS]
    _refresh_saved_idx = _refresh_labels.index(_saved["auto_refresh_label"]) if _saved["auto_refresh_label"] in _refresh_labels else 2

    auto_refresh_label = st.selectbox(
        "⏱ 自動更新間隔",
        options=_refresh_labels,
        index=_refresh_saved_idx,
        help="選択した間隔でダッシュボードを自動リロードします",
    )
    auto_refresh_sec = dict(_REFRESH_OPTIONS)[auto_refresh_label]

    st.markdown("---")

    # --- LLM ニュース分析セクション ---
    st.markdown("#### 🤖 ニュース分析AI")

    _llm_available = llm_is_available()

    llm_enabled = st.checkbox(
        "LLMでニュースを分析",
        value=_saved.get("llm_enabled", False),
        help=(
            "GitHub Copilot CLI を使ってニュースのカテゴリ分類・PF影響を"
            "AIで分析します。`copilot` CLI のインストールが必要です。"
        ),
        disabled=not _llm_available,
    )

    if not _llm_available:
        st.caption("⚠️ `copilot` CLI が見つかりません。GitHub Copilot CLI をインストールしてください")

    _model_ids = [m[0] for m in LLM_MODELS]
    _model_labels = [m[1] for m in LLM_MODELS]
    _saved_model = _saved.get("llm_model", "gpt-4.1")
    _model_saved_idx = (
        _model_ids.index(_saved_model)
        if _saved_model in _model_ids
        else 1
    )

    llm_model_label = st.selectbox(
        "🧠 分析モデル",
        options=_model_labels,
        index=_model_saved_idx,
        help="ニュース分析に使用するLLMモデル",
        disabled=not llm_enabled,
    )
    llm_model = _model_ids[_model_labels.index(llm_model_label)]

    # LLM 分析キャッシュ TTL
    _ttl_labels = [t[0] for t in LLM_CACHE_OPTIONS]
    _ttl_values = [t[1] for t in LLM_CACHE_OPTIONS]
    _saved_ttl_label = _saved.get("llm_cache_ttl_label", "1時間")
    _ttl_saved_idx = (
        _ttl_labels.index(_saved_ttl_label)
        if _saved_ttl_label in _ttl_labels
        else 0
    )

    llm_cache_ttl_label = st.selectbox(
        "⏳ 分析キャッシュ保持",
        options=_ttl_labels,
        index=_ttl_saved_idx,
        help=(
            "同じニュースに対して LLM 再分析をスキップする期間。"
            "Premium Request の消費を抑えます。"
        ),
        disabled=not llm_enabled,
    )
    llm_cache_ttl_sec = _ttl_values[_ttl_labels.index(llm_cache_ttl_label)]

    # --- Copilot チャットセクション ---
    st.markdown("---")
    st.markdown("#### 💬 チャットモデル")
    _chat_model_ids = [m[0] for m in COPILOT_MODELS]
    _chat_model_labels = [m[1] for m in COPILOT_MODELS]
    _saved_chat_model = _saved.get("chat_model", "claude-sonnet-4")
    _chat_model_saved_idx = (
        _chat_model_ids.index(_saved_chat_model)
        if _saved_chat_model in _chat_model_ids
        else 0
    )
    chat_model_label = st.selectbox(
        "🧠 チャットモデル",
        options=_chat_model_labels,
        index=_chat_model_saved_idx,
        help="Copilot チャットで使用するモデル（分析モデルとは独立）",
    )
    chat_model = _chat_model_ids[_chat_model_labels.index(chat_model_label)]

    # キャッシュ状態を表示
    if llm_enabled:
        _ci = llm_get_cache_info()
        if _ci["cached"]:
            _age_min = _ci["age_sec"] // 60
            if _age_min < 60:
                _age_str = f"{_age_min}分前"
            else:
                _age_str = f"{_age_min // 60}時間{_age_min % 60}分前"
            st.caption(f"💾 キャッシュあり（{_age_str}に {_ci['model']} で分析済み）")
            if st.button("🔄 今すぐ再分析", key="llm_reanalyze", help="キャッシュを破棄して LLM 分析をやり直します"):
                llm_clear_cache()
                llm_clear_summary_cache()
                llm_clear_health_summary_cache()
                st.rerun()
        else:
            st.caption("💾 キャッシュなし（次回更新時に LLM 分析を実行）")

    # --- 設定の自動保存 ---
    _current_settings = {
        "period_label": period_label,
        "chart_style": chart_style,
        "show_invested": show_invested,
        "benchmark_label": benchmark_label,
        "show_individual": show_individual,
        "show_projection": show_projection,
        "target_amount_man": int(target_amount // 10000),
        "projection_years": projection_years,
        "auto_refresh_label": auto_refresh_label,
        "llm_enabled": llm_enabled,
        "llm_model": llm_model,
        "llm_cache_ttl_label": llm_cache_ttl_label,
        "chat_model": chat_model,
    }
    if _current_settings != _saved:
        save_settings(_current_settings)
        st.session_state["_saved_settings"] = _current_settings

# --- 用語集タブ ---
with _tab_help:
    _GLOSSARY = {
        "📈 パフォーマンス指標": {
            "評価額": "保有株数 × 現在株価で算出した現在の資産価値。",
            "損益率": "（現在評価額 − 投資額）÷ 投資額 × 100。投資に対するリターン。",
            "ドローダウン": "直近の最高値からの下落率。リスク管理の重要指標で、ポートフォリオがピークから何%下がったかを示す。",
            "シャープレシオ": "リスク（値動きのばらつき）1単位あたりのリターン。1以上で良好、2以上で優秀。リスク調整後のパフォーマンスを測る。",
            "ベンチマーク": "運用成果の比較基準となる指数（S&P500、日経225等）。ベンチマークを上回っていれば市場平均以上の成績。",
        },
        "🔍 テクニカル指標": {
            "SMA（単純移動平均）": "過去N日間の終値の平均。SMA50（短期）とSMA200（長期）がよく使われる。トレンドの方向を判断する基本指標。",
            "ゴールデンクロス": "短期移動平均（SMA50）が長期移動平均（SMA200）を下から上に突き抜けること。上昇トレンドへの転換を示唆。",
            "デッドクロス": "短期移動平均（SMA50）が長期移動平均（SMA200）を上から下に突き抜けること。下降トレンドへの転換を示唆し、売り検討のサイン。",
            "RSI（相対力指数）": "0〜100で買われすぎ・売られすぎを判定。70以上で買われすぎ（売り検討）、30以下で売られすぎ（買い検討）。",
            "ボリンジャーバンド": "移動平均の上下に標準偏差の帯を描いたもの。バンド外に出ると異常値で、反転の可能性。",
        },
        "📊 ポートフォリオ分析": {
            "ウェイトドリフト": "各銘柄の構成比が均等配分からどれだけズレているか。値上がりした銘柄が膨らみ過ぎていないか確認する指標。",
            "相関係数": "2銘柄の値動きの連動性。+1で完全連動、−1で逆の動き、0で無関係。相関が高い銘柄ばかりだと分散効果が薄れる。",
            "セクター構成": "銘柄を業種別（テクノロジー、金融、ヘルスケア等）に分類した配分比率。特定セクターへの集中を防ぐために確認。",
            "通貨エクスポージャー": "保有資産の通貨別の配分。為替変動によるリスクの偏りを確認するための指標。",
        },
        "🏥 ヘルスチェック": {
            "アラートレベル": "銘柄の健全性を4段階で判定。✅ 正常 → ⚡ 早期警告 → ⚠️ 注意 → 🚨 EXIT（売却検討）。",
            "バリュートラップ": "PERが低く割安に見えるが、業績悪化が原因で株価が下がり続ける銘柄。見せかけの割安に注意。",
            "還元安定度": "配当や自社株買いの継続性を評価。✅安定 / 📈増加 / ⚠️一時的 / 📉低下の4段階。",
        },
        "💰 バリュエーション": {
            "PER（株価収益率）": "株価 ÷ 1株利益(EPS)。株価が利益の何倍かを示す。低いほど割安だが、業績悪化による低PERには注意。",
            "PBR（株価純資産倍率）": "株価 ÷ 1株純資産。1倍以下は解散価値割れで割安とされるが、万能ではない。",
            "配当利回り": "年間配当金 ÷ 株価 × 100。高いほどインカム収入が多い。ただし株価下落による見かけの高利回りに注意。",
            "総還元率": "（配当金 + 自社株買い）÷ 時価総額。配当だけでなく自社株買いも含めた株主還元の総合指標。",
        },
        "🔮 将来推定": {
            "楽観 / 基本 / 悲観": "過去リターンの平均±標準偏差で3パターンの将来推移を推計。基本＝平均、楽観＝+1σ、悲観＝−1σ。",
            "目標ライン": "設定した目標資産額を水平線で表示。将来推定と重ねて達成時期の目安を確認できる。",
        },
    }

    for _cat_name, _terms in _GLOSSARY.items():
        with st.expander(_cat_name, expanded=False):
            for _term, _desc in _terms.items():
                st.markdown(f"**{_term}**")
                st.caption(_desc)

# 自動更新タイマー（タブ外に配置）
if auto_refresh_sec > 0:
    _refresh_count = st_autorefresh(
        interval=auto_refresh_sec * 1000,
        limit=0,  # 無制限
        key="auto_refresh",
    )
else:
    _refresh_count = 0

# 手動更新ボタン（タブ外に配置）
if st.sidebar.button("🔄 今すぐ更新", width="stretch"):
    load_snapshot.clear()
    load_history.clear()
    load_trade_activity.clear()
    load_health_check.clear()
    load_economic_news.clear()
    _cache_dir = Path(_SCRIPT_DIR).resolve().parents[4] / "data" / "cache" / "price_history"
    if _cache_dir.exists():
        for f in _cache_dir.glob("*.csv"):
            f.unlink(missing_ok=True)
    st.rerun()

# 最終更新時刻を session_state で管理
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["_prev_refresh_count"] = 0

if _refresh_count > st.session_state.get("_prev_refresh_count", 0):
    load_snapshot.clear()
    load_history.clear()
    load_trade_activity.clear()
    load_health_check.clear()
    load_economic_news.clear()
    st.session_state["last_refresh"] = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["_prev_refresh_count"] = _refresh_count

st.sidebar.caption(
    f"最終更新: {st.session_state['last_refresh']}\n\n"
    f"Data Source: yfinance + portfolio.csv"
)


# =====================================================================
# メインコンテンツ
# =====================================================================
st.title("💼 ポートフォリオダッシュボード")

# --- データ読み込み ---
try:
    with st.spinner("ポートフォリオデータを読み込み中..."):
        snapshot = load_snapshot()
        history_df = load_history(period)
except Exception as _data_err:
    st.error(f"⚠️ データ取得に失敗しました: {_data_err}")
    st.info("ネットワーク接続を確認するか、「🔄 今すぐ更新」ボタンで再試行してください。")
    st.stop()

# FXレート表示（サイドバー下部）
_fx = snapshot.get("fx_rates", {})
_fx_display = {k: v for k, v in _fx.items() if k != "JPY" and v != 1.0}
if _fx_display:
    with st.sidebar.expander("💱 為替レート", expanded=False):
        for cur, rate in sorted(_fx_display.items()):
            st.caption(f"{cur}/JPY: ¥{rate:,.2f}")

# =====================================================================
# KPI メトリクスカード
# =====================================================================
st.markdown('<div id="summary"></div>', unsafe_allow_html=True)
st.markdown("### 📈 サマリー")
st.caption("ポートフォリオ全体の現在価値・損益・リスク指標を一目で把握するセクションです。")

positions = snapshot["positions"]
total_value = snapshot["total_value_jpy"]
total_cost = sum(p.get("cost_jpy", 0) for p in positions if "cost_jpy" in p)
unrealized_pnl = total_value - total_cost if total_cost else 0
unrealized_pnl_pct = ((total_value / total_cost) - 1) * 100 if total_cost else 0
realized_pnl = snapshot.get("realized_pnl", {}).get("total_jpy", 0)
total_pnl = unrealized_pnl + realized_pnl
num_holdings = len([p for p in positions if p.get("sector") != "Cash"])

# --- 大項目カード（トータル資産 / 評価損益 / 保有銘柄数） ---
def _kpi_main(label: str, value: str, sub: str = "", color: str = "") -> str:
    """大項目 KPI: テーマ追従 + 大きめフォント."""
    color_style = f"color:{color};" if color else ""
    sub_html = (
        f'<div style="font-size:0.92rem; {color_style} margin-top:4px; opacity:0.85;">{sub}</div>'
        if sub else ""
    )
    return (
        f'<div class="kpi-card kpi-main">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="{color_style}">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )

# --- 小項目カード（損益サブ指標） ---
def _kpi_sub(label: str, value: str, color: str = "") -> str:
    """小項目 KPI: テーマ追従 + コンパクト."""
    color_style = f"color:{color};" if color else ""
    return (
        f'<div class="kpi-card kpi-sub">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value-sub" style="{color_style}">{value}</div>'
        f'</div>'
    )

# --- リスク指標カード ---
def _risk_card(label: str, value: str, color: str = "") -> str:
    """リスク指標: テーマ追従 + 最小サイズ."""
    color_style = f"color:{color};" if color else ""
    return (
        f'<div class="kpi-card kpi-risk">'
        f'<div class="kpi-label" style="white-space:nowrap;'
        f' overflow:hidden; text-overflow:ellipsis;">{label}</div>'
        f'<div class="kpi-value-risk" style="{color_style}">{value}</div>'
        f'</div>'
    )

_unr_color = "#4ade80" if unrealized_pnl >= 0 else "#f87171"
_unr_sign = "+" if unrealized_pnl >= 0 else ""

# 前日比の算出
_daily = compute_daily_change(history_df)
_dc_jpy = _daily["daily_change_jpy"]
_dc_pct = _daily["daily_change_pct"]
_dc_sign = "+" if _dc_jpy >= 0 else ""
_dc_color = "#4ade80" if _dc_jpy >= 0 else "#f87171"
_dc_text = f"{_dc_sign}¥{_dc_jpy:,.0f}（{_dc_pct:+.2f}%）" if _dc_jpy != 0 else "--"
_dc_sub = f'<span style="color:{_dc_color};">前日比 {_dc_text}</span>' if _dc_jpy != 0 else ""

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(_kpi_main("トータル資産（円換算）", f"¥{total_value:,.0f}",
                          sub=_dc_sub),
                unsafe_allow_html=True)
with col2:
    st.markdown(_kpi_main(
        "評価損益（含み）",
        f"{_unr_sign}¥{unrealized_pnl:,.0f}",
        sub=f"{unrealized_pnl_pct:+.2f}%",
        color=_unr_color,
    ), unsafe_allow_html=True)
with col3:
    st.markdown(_kpi_main(
        "保有銘柄数",
        f"{num_holdings}",
        sub=f"更新: {snapshot['as_of'][:10]}",
        color="#60a5fa",
    ), unsafe_allow_html=True)

# --- 小項目: 損益 ---
realized_sign = "+" if realized_pnl >= 0 else ""
total_pnl_sign = "+" if total_pnl >= 0 else ""
realized_color = "#4ade80" if realized_pnl >= 0 else "#f87171"
total_pnl_color = "#4ade80" if total_pnl >= 0 else "#f87171"

st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

sub_col1, sub_col2 = st.columns(2)
with sub_col1:
    st.markdown(_kpi_sub(
        "トータル損益（実現＋含み）",
        f"{total_pnl_sign}¥{total_pnl:,.0f}",
        color=total_pnl_color,
    ), unsafe_allow_html=True)
with sub_col2:
    st.markdown(_kpi_sub(
        "実現損益（確定済）",
        f"{realized_sign}¥{realized_pnl:,.0f}",
        color=realized_color,
    ), unsafe_allow_html=True)

# --- リスク指標 ---
if not history_df.empty:
    risk = compute_risk_metrics(history_df)

    st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

    _sharpe_color = "#4ade80" if risk["sharpe_ratio"] >= 1.0 else (
        "#fbbf24" if risk["sharpe_ratio"] >= 0.5 else "#f87171"
    )
    _mdd_color = "#4ade80" if risk["max_drawdown_pct"] > -10 else (
        "#fbbf24" if risk["max_drawdown_pct"] > -20 else "#f87171"
    )

    rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(5)
    with rcol1:
        st.markdown(_risk_card("年率リターン", f"{risk['annual_return_pct']:+.1f}%",
                               "#4ade80" if risk["annual_return_pct"] > 0 else "#f87171"),
                    unsafe_allow_html=True)
    with rcol2:
        st.markdown(_risk_card("ボラティリティ", f"{risk['annual_volatility_pct']:.1f}%"),
                    unsafe_allow_html=True)
    with rcol3:
        st.markdown(_risk_card("Sharpe", f"{risk['sharpe_ratio']:.2f}", _sharpe_color),
                    unsafe_allow_html=True)
    with rcol4:
        st.markdown(_risk_card("最大DD", f"{risk['max_drawdown_pct']:.1f}%", _mdd_color),
                    unsafe_allow_html=True)
    with rcol5:
        st.markdown(_risk_card("Calmar", f"{risk['calmar_ratio']:.2f}"),
                    unsafe_allow_html=True)

# --- ベンチマーク超過リターン ---
if benchmark_symbol and not history_df.empty:
    _bench_for_excess = get_benchmark_series(benchmark_symbol, history_df, period)
    _excess = compute_benchmark_excess(history_df, _bench_for_excess)
    if _excess is not None:
        st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)
        _ex_color = "#4ade80" if _excess["excess_return_pct"] >= 0 else "#f87171"
        _ex_sign = "+" if _excess["excess_return_pct"] >= 0 else ""
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            st.markdown(_risk_card(
                "PFリターン",
                f"{_excess['portfolio_return_pct']:+.1f}%",
                "#4ade80" if _excess["portfolio_return_pct"] > 0 else "#f87171",
            ), unsafe_allow_html=True)
        with ecol2:
            st.markdown(_risk_card(
                f"{benchmark_label}リターン",
                f"{_excess['benchmark_return_pct']:+.1f}%",
                "#60a5fa",
            ), unsafe_allow_html=True)
        with ecol3:
            st.markdown(_risk_card(
                "超過リターン",
                f"{_ex_sign}{_excess['excess_return_pct']:.1f}%",
                _ex_color,
            ), unsafe_allow_html=True)

# --- Top / Worst パフォーマー ---
if not history_df.empty:
    _performers = compute_top_worst_performers(history_df, top_n=3)
    _top = _performers["top"]
    _worst = _performers["worst"]
    if _top or _worst:
        st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            _top_html = '<div class="kpi-card kpi-sub" style="text-align:left;">'
            _top_html += '<div class="kpi-label">🟢 本日 Best</div>'
            for p in _top:
                _c = "#4ade80" if p["change_pct"] >= 0 else "#f87171"
                _top_html += (
                    f'<div style="display:flex; justify-content:space-between;'
                    f' padding:3px 0; font-size:0.9rem;">'
                    f'<span>{p["symbol"]}</span>'
                    f'<span style="color:{_c}; font-weight:600;">'
                    f'{p["change_pct"]:+.2f}%</span></div>'
                )
            _top_html += '</div>'
            st.markdown(_top_html, unsafe_allow_html=True)
        with pcol2:
            _worst_html = '<div class="kpi-card kpi-sub" style="text-align:left;">'
            _worst_html += '<div class="kpi-label">🔴 本日 Worst</div>'
            for p in _worst:
                _c = "#4ade80" if p["change_pct"] >= 0 else "#f87171"
                _worst_html += (
                    f'<div style="display:flex; justify-content:space-between;'
                    f' padding:3px 0; font-size:0.9rem;">'
                    f'<span>{p["symbol"]}</span>'
                    f'<span style="color:{_c}; font-weight:600;">'
                    f'{p["change_pct"]:+.2f}%</span></div>'
                )
            _worst_html += '</div>'
            st.markdown(_worst_html, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# =====================================================================
# ヘルスチェック & 売りアラート
# =====================================================================
st.markdown('<div id="health-check"></div>', unsafe_allow_html=True)
st.markdown("### 🏥 ヘルスチェック")
st.caption("各銘柄のトレンド・テクニカル指標をチェックし、売りタイミングや注意が必要な銘柄を自動検出します。")

try:
    health_data = load_health_check()
except Exception as _hc_err:
    st.warning(f"ヘルスチェックの実行に失敗しました: {_hc_err}")
    health_data = None

if health_data is not None:
    hc_summary = health_data["summary"]
    hc_positions = health_data["positions"]
    sell_alerts = health_data["sell_alerts"]

    # --- サマリーカード ---
    hc_cols = st.columns(5)
    _hc_items = [
        ("合計", hc_summary["total"], ""),
        ("✅ 健全", hc_summary["healthy"], "#4ade80"),
        ("⚡ 早期警告", hc_summary["early_warning"], "#fbbf24"),
        ("⚠️ 注意", hc_summary["caution"], "#fb923c"),
        ("🚨 撤退", hc_summary["exit"], "#f87171"),
    ]
    for i, (label, count, color) in enumerate(_hc_items):
        with hc_cols[i]:
            st.markdown(_risk_card(label, str(count), color), unsafe_allow_html=True)

    # --- LLM ヘルスチェック分析（売りアラート通知より先に実行） ---
    _hc_llm_summary: dict | None = None
    _hc_llm_assessment_map: dict[str, dict] = {}
    if llm_enabled:
        # 経済ニュースを取得（st.cache_data でキャッシュされるので後のニュースセクションと重複しない）
        try:
            _hc_pos_key = ",".join(
                sorted(p.get("symbol", "") for p in positions if p.get("sector") != "Cash")
            )
            _hc_fx = snapshot.get("fx_rates", {})
            _hc_news = load_economic_news(
                _hc_pos_key, positions, _hc_fx,
                llm_enabled=llm_enabled, llm_model=llm_model,
                llm_cache_ttl=llm_cache_ttl_sec,
            )
        except Exception:
            _hc_news = []

        _hc_llm_summary = generate_health_summary(
            health_data,
            news_items=_hc_news,
            model=llm_model, timeout=120, cache_ttl=llm_cache_ttl_sec,
        )
        if _hc_llm_summary:
            st.session_state["_hc_llm_summary_data"] = _hc_llm_summary
            for _sa in _hc_llm_summary.get("stock_assessments", []):
                _sa_sym = _sa.get("symbol", "")
                if _sa_sym:
                    _hc_llm_assessment_map[_sa_sym] = _sa

    # --- 売りアラート通知 ---
    if sell_alerts:
        st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)
        st.markdown("#### 🔔 売りタイミング通知")

        for alert in sell_alerts:
            urgency = alert["urgency"]
            _urgency_emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
            _urgency_label = {"critical": "緊急", "warning": "注意", "info": "参考"}

            # Build detail HTML
            detail_html = ""
            for d in alert.get("details", []):
                detail_html += f'<div class="sell-alert-detail">• {d}</div>'

            # LLM 分析コメントを付加
            _alert_sym = alert.get("symbol", "")
            _llm_sa = _hc_llm_assessment_map.get(_alert_sym)
            if _llm_sa:
                _llm_text = _llm_sa.get("assessment", "")
                if _llm_text:
                    detail_html += (
                        f'<div class="sell-alert-ai">'
                        f'🤖 <strong>AI分析</strong>: {_llm_text}</div>'
                    )

            pnl = alert.get("pnl_pct", 0)
            pnl_color = "#4ade80" if pnl >= 0 else "#f87171"
            pnl_text = f'<span style="color:{pnl_color}; font-weight:600;">{pnl:+.1f}%</span>'

            st.markdown(
                f'<div class="sell-alert sell-alert-{urgency}">'
                f'<div class="sell-alert-header">'
                f'{_urgency_emoji.get(urgency, "")} '
                f'[{_urgency_label.get(urgency, "")}] '
                f'{alert["name"]} ({alert["symbol"]}) '
                f'— {alert["action"]} '
                f'(含み損益: {pnl_text})'
                f'</div>'
                f'<div class="sell-alert-reason">{alert["reason"]}</div>'
                f'{detail_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("🟢 現在、売りタイミングの通知はありません")

    # --- LLM ヘルスチェックサマリー表示 ---
    if _hc_llm_summary:
            st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

            _hcs_html = '<div class="health-summary-card">'
            _hcs_html += '<div class="health-summary-header">🤖 ヘルスチェックサマリー</div>'

            _hcs_overview = _hc_llm_summary.get("overview", "")
            if _hcs_overview:
                _hcs_html += f'<div class="health-summary-overview">{_hcs_overview}</div>'

            _hcs_warning = _hc_llm_summary.get("risk_warning", "")
            if _hcs_warning:
                _hcs_html += (
                    f'<div class="health-summary-warning">'
                    f'⚠️ <strong>リスク注意</strong>: {_hcs_warning}</div>'
                )

            _hcs_assessments = _hc_llm_summary.get("stock_assessments", [])
            if _hcs_assessments:
                # アラートレベルを持つ銘柄マップ
                _hc_alert_map: dict[str, str] = {}
                for _hcp in hc_positions:
                    _hc_alert_map[_hcp.get("symbol", "")] = _hcp.get("alert_level", "none")

                _hcs_html += '<details class="health-summary-stocks-toggle">'
                _hcs_html += f'<summary>📋 銘柄別コメント（{len(_hcs_assessments)}件）</summary>'

                for _sa in _hcs_assessments:
                    _sa_sym = _sa.get("symbol", "")
                    _sa_name = _sa.get("name", _sa_sym)
                    _sa_assessment = _sa.get("assessment", "")
                    _sa_action = _sa.get("action", "")
                    _sa_alert = _hc_alert_map.get(_sa_sym, "none")
                    _sa_level_class = f" health-summary-stock-{_sa_alert}" if _sa_alert != "none" else ""
                    _action_badge = (
                        f'<span class="health-summary-action">{_sa_action}</span>'
                        if _sa_action else ""
                    )
                    _hcs_html += (
                        f'<div class="health-summary-stock{_sa_level_class}">'
                        f'<div class="health-summary-stock-name">'
                        f'{_sa_name} ({_sa_sym}){_action_badge}</div>'
                        f'<div class="health-summary-stock-text">{_sa_assessment}</div>'
                        f'</div>'
                    )

                _hcs_html += '</details>'

            _hcs_html += '</div>'
            st.markdown(_hcs_html, unsafe_allow_html=True)

    # --- 銘柄別ヘルスチェック詳細 ---
    st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

    with st.expander("📋 銘柄別ヘルスチェック詳細", expanded=False):
        if hc_positions:
            # テーブル表示
            hc_table_data = []
            for pos in hc_positions:
                alert_level = pos["alert_level"]
                _level_display = {
                    "none": "✅ 健全",
                    "early_warning": "⚡ 早期警告",
                    "caution": "⚠️ 注意",
                    "exit": "🚨 撤退",
                }
                _trend_emoji = {
                    "上昇": "📈",
                    "横ばい": "➡️",
                    "下降": "📉",
                    "不明": "❓",
                }
                rsi_val = pos.get("rsi", float("nan"))
                try:
                    import math
                    rsi_str = f"{rsi_val:.1f}" if not math.isnan(rsi_val) else "N/A"
                except (TypeError, ValueError):
                    rsi_str = "N/A"

                stability_emoji = pos.get("return_stability_emoji", "")
                long_term = pos.get("long_term_label", "")

                reasons_str = " / ".join(pos.get("alert_reasons", [])) if pos.get("alert_reasons") else "-"

                hc_table_data.append({
                    "銘柄": f"{pos['name']}",
                    "シンボル": pos["symbol"],
                    "判定": _level_display.get(alert_level, alert_level),
                    "トレンド": f"{_trend_emoji.get(pos['trend'], '')} {pos['trend']}",
                    "RSI": rsi_str,
                    "変化品質": pos.get("change_quality", ""),
                    "長期適性": long_term,
                    "還元安定度": stability_emoji,
                    "含み損益(%)": pos.get("pnl_pct", 0),
                    "理由": reasons_str,
                })

            hc_df = pd.DataFrame(hc_table_data)

            # アラートレベルでソート（exit > caution > early_warning > none）
            _sort_order = {"🚨 撤退": 0, "⚠️ 注意": 1, "⚡ 早期警告": 2, "✅ 健全": 3}
            hc_df["_sort"] = hc_df["判定"].map(_sort_order).fillna(9)
            hc_df = hc_df.sort_values("_sort").drop(columns=["_sort"])

            st.dataframe(
                hc_df.style.format({
                    "含み損益(%)": "{:+.1f}%",
                }).map(
                    lambda v: "color: #4ade80" if isinstance(v, (int, float)) and v > 0
                    else ("color: #f87171" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=["含み損益(%)"],
                ),
                width="stretch",
                height=min(400, 60 + len(hc_table_data) * 38),
            )

            # --- 個別銘柄カード（アラートのみ展開） ---
            alert_positions = [p for p in hc_positions if p["alert_level"] != "none"]
            if alert_positions:
                st.markdown("##### ⚡ アラート銘柄の詳細")
                for pos in alert_positions:
                    alert_level = pos["alert_level"]
                    _card_border_color = {
                        "early_warning": "#fbbf24",
                        "caution": "#fb923c",
                        "exit": "#f87171",
                    }.get(alert_level, "#94a3b8")

                    indicators = pos.get("indicators", {})
                    ind_parts = []
                    for ind_name, ind_val in indicators.items():
                        _ind_labels = {
                            "accruals": "アクルーアルズ",
                            "revenue_acceleration": "売上加速",
                            "fcf_yield": "FCF利回り",
                            "roe_trend": "ROE趨勢",
                        }
                        label = _ind_labels.get(ind_name, ind_name)
                        if isinstance(ind_val, bool):
                            emoji = "✅" if ind_val else "❌"
                            ind_parts.append(f"{emoji} {label}")
                        elif isinstance(ind_val, (int, float)):
                            emoji = "✅" if ind_val > 0 else "❌"
                            ind_parts.append(f"{emoji} {label}")

                    ind_html = " &nbsp;|&nbsp; ".join(ind_parts) if ind_parts else ""

                    trap_html = ""
                    if pos.get("value_trap"):
                        trap_reasons = " / ".join(pos.get("value_trap_reasons", []))
                        trap_html = (
                            f'<div style="margin-top:6px; padding:6px 10px;'
                            f' background:rgba(248,113,113,0.1); border-radius:6px;'
                            f' font-size:0.82rem;">'
                            f'🪤 バリュートラップ: {trap_reasons}</div>'
                        )

                    reasons_html = ""
                    for r in pos.get("alert_reasons", []):
                        reasons_html += f'<div style="font-size:0.82rem; padding:1px 0;">• {r}</div>'

                    cross_html = ""
                    cross_signal = pos.get("cross_signal", "none")
                    if cross_signal != "none":
                        _cross_emoji = "🟡" if cross_signal == "golden_cross" else "💀"
                        _cross_label = "ゴールデンクロス" if cross_signal == "golden_cross" else "デッドクロス"
                        days = pos.get("days_since_cross", "?")
                        cross_html = f' | {_cross_emoji} {_cross_label}（{days}日前）'

                    st.markdown(
                        f'<div class="health-card health-card-{alert_level}">'
                        f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                        f'<span style="font-weight:700; font-size:1.0rem;">'
                        f'{pos["alert_emoji"]} {pos["name"]} ({pos["symbol"]})</span>'
                        f'<span style="font-size:0.85rem; opacity:0.8;">'
                        f'{pos["alert_label"]}</span>'
                        f'</div>'
                        f'<div style="font-size:0.85rem; margin-top:6px; opacity:0.8;">'
                        f'トレンド: {pos["trend"]} | RSI: {pos.get("rsi", 0):.1f} '
                        f'| SMA50: {pos.get("sma50", 0):,.1f} '
                        f'| SMA200: {pos.get("sma200", 0):,.1f}'
                        f'{cross_html}'
                        f'</div>'
                        f'<div style="font-size:0.85rem; margin-top:4px;">{ind_html}</div>'
                        f'<div style="margin-top:6px;">{reasons_html}</div>'
                        f'{trap_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        else:
            st.info("保有銘柄データがありません")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# =====================================================================
# 経済ニュース & PF影響
# =====================================================================
st.markdown('<div id="economic-news"></div>', unsafe_allow_html=True)
st.markdown("### 📰 経済ニュース & PF影響")
st.caption("主要指数・商品に関する最新ニュースと、ポートフォリオへの影響度を自動分析します。")

try:
    # キャッシュキー用にシンボルリストを文字列化
    _pos_key = ",".join(
        sorted(p.get("symbol", "") for p in positions if p.get("sector") != "Cash")
    )
    _fx_for_news = snapshot.get("fx_rates", {})
    econ_news = load_economic_news(
        _pos_key, positions, _fx_for_news,
        llm_enabled=llm_enabled, llm_model=llm_model,
        llm_cache_ttl=llm_cache_ttl_sec,
    )
except Exception as _news_err:
    st.warning(f"経済ニュースの取得に失敗しました: {_news_err}")
    econ_news = []

if econ_news:
    # 分析方法の表示
    _any_llm = any(n.get("analysis_method") == "llm" for n in econ_news)
    if _any_llm:
        _cache_info = llm_get_cache_info()
        if _cache_info["cached"] and _cache_info["age_sec"] > 10:
            _age_m = _cache_info["age_sec"] // 60
            st.caption(f"🤖 AI分析（{llm_model}）— 📦 キャッシュ済み（{_age_m}分前）")
        else:
            st.caption("🤖 AI分析（" + llm_model + "）")
    else:
        st.caption("🔑 キーワードベース分析")

    # --- サマリーカード: 影響度別件数 ---
    _n_high = sum(1 for n in econ_news if n["portfolio_impact"]["impact_level"] == "high")
    _n_med = sum(1 for n in econ_news if n["portfolio_impact"]["impact_level"] == "medium")
    _n_low = sum(1 for n in econ_news if n["portfolio_impact"]["impact_level"] == "low")
    _n_none = sum(1 for n in econ_news if n["portfolio_impact"]["impact_level"] == "none")

    ncol1, ncol2, ncol3, ncol4 = st.columns(4)
    with ncol1:
        st.markdown(_risk_card("🔴 高影響", str(_n_high),
                               "#f87171" if _n_high > 0 else ""), unsafe_allow_html=True)
    with ncol2:
        st.markdown(_risk_card("🟡 中影響", str(_n_med),
                               "#fbbf24" if _n_med > 0 else ""), unsafe_allow_html=True)
    with ncol3:
        st.markdown(_risk_card("🔵 低影響", str(_n_low),
                               "#60a5fa" if _n_low > 0 else ""), unsafe_allow_html=True)
    with ncol4:
        st.markdown(_risk_card("⚪ 影響なし", str(_n_none), ""), unsafe_allow_html=True)

    st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

    # --- LLM サマリー ---
    if _any_llm:
        _summary = generate_news_summary(
            econ_news, positions,
            model=llm_model, cache_ttl=llm_cache_ttl_sec,
        )
        if _summary:
            _overview = _summary.get("overview", "")
            _key_points = _summary.get("key_points", [])
            _pf_alert = _summary.get("portfolio_alert", "")

            # サマリーカード
            _summary_html = '<div class="news-summary-card">'
            _summary_html += '<div class="news-summary-header">📋 ニュースサマリー</div>'
            if _overview:
                _summary_html += f'<div class="news-summary-overview">{_overview}</div>'

            if _key_points:
                _summary_html += '<div class="news-summary-points">'
                for _kp in _key_points:
                    _icon = _kp.get("icon", "📌")
                    _label = _kp.get("label", _kp.get("category", ""))
                    _kp_summary = _kp.get("summary", "")
                    _news_ids = _kp.get("news_ids", [])
                    _ids_str = ""
                    if _news_ids:
                        _id_links = [f'<span class="news-ref">#{nid+1}</span>' for nid in _news_ids]
                        _ids_str = f' <span class="news-refs">{", ".join(_id_links)}</span>'
                    _summary_html += (
                        f'<div class="news-summary-point">'
                        f'<span class="news-summary-cat">{_icon} {_label}</span>'
                        f'<span class="news-summary-text">{_kp_summary}{_ids_str}</span>'
                        f'</div>'
                    )
                _summary_html += '</div>'

            if _pf_alert:
                _summary_html += (
                    f'<div class="news-summary-alert">'
                    f'⚠️ <strong>PF注意</strong>: {_pf_alert}</div>'
                )

            _summary_html += '</div>'
            st.markdown(_summary_html, unsafe_allow_html=True)
            st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

    # --- ニュースカード表示 ---
    # PF影響ありのニュースを先に表示
    _impact_news = [n for n in econ_news if n["portfolio_impact"]["impact_level"] != "none"]
    _other_news = [n for n in econ_news if n["portfolio_impact"]["impact_level"] == "none"]

    # ニュースにインデックス番号を付与（サマリーからのトレース用）
    _news_index_map: dict[int, int] = {}  # original_idx -> display_number
    for _disp_num, _news in enumerate(econ_news, 1):
        _news["_display_number"] = _disp_num

    if _impact_news:
        with st.expander(f"⚡ PF影響のあるニュース（{len(_impact_news)}件）", expanded=False):
            for news_item in _impact_news:
                _impact = news_item["portfolio_impact"]
                _impact_level = _impact["impact_level"]
                _impact_labels = {"high": "高影響", "medium": "中影響", "low": "低影響"}
                _impact_colors = {"high": "impact-high", "medium": "impact-medium", "low": "impact-low"}

                # カテゴリバッジ
                _cat_badges = ""
                for cat in news_item.get("categories", []):
                    _cat_badges += (
                        f'<span class="news-badge news-badge-category">'
                        f'{cat["icon"]} {cat["label"]}</span>'
                    )

                # 影響度バッジ
                _impact_badge = (
                    f'<span class="news-badge news-badge-{_impact_colors.get(_impact_level, "")}">'
                    f'{_impact_labels.get(_impact_level, "")} — '
                    f'{len(_impact["affected_holdings"])}銘柄</span>'
                )

                # 影響銘柄リスト
                _affected_html = ""
                if _impact["affected_holdings"]:
                    _syms = ", ".join(_impact["affected_holdings"][:8])
                    _affected_html = (
                        f'<div class="news-affected">'
                        f'📌 影響銘柄: {_syms}</div>'
                    )

                # LLM分析の理由（あれば表示）
                _reason_html = ""
                _reason = _impact.get("reason", "")
                if _reason and news_item.get("analysis_method") == "llm":
                    _reason_html = (
                        f'<div style="font-size:0.82rem; margin-top:4px; opacity:0.85;">'
                        f'💡 {_reason}</div>'
                    )

                # タイトルリンク
                _link = news_item.get("link", "")
                _disp_no = news_item.get("_display_number", "")
                _num_badge = f'<span class="news-number">#{_disp_no}</span>' if _disp_no else ""
                _title_html = (
                    f'<a href="{_link}" target="_blank">{news_item["title"]}</a>'
                    if _link else news_item["title"]
                )

                # 発行元・日時
                _pub = news_item.get("publisher", "")
                _time = news_item.get("publish_time", "")
                _source = news_item.get("source_name", "")
                _meta_parts = [p for p in [_pub, _source, _time[:16] if _time else ""] if p]
                _meta = " · ".join(_meta_parts)

                st.markdown(
                    f'<div class="news-card news-{_impact_colors.get(_impact_level, "impact-none")}">'
                    f'<div class="news-title">{_num_badge}{_title_html}</div>'
                    f'{_affected_html}'
                    f'{_reason_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    if _other_news:
        with st.expander(f"📋 その他のニュース（{len(_other_news)}件）", expanded=False):
            for news_item in _other_news:
                _link = news_item.get("link", "")
                _disp_no = news_item.get("_display_number", "")
                _num_badge = f'<span class="news-number">#{_disp_no}</span>' if _disp_no else ""
                _title_html = (
                    f'<a href="{_link}" target="_blank">{news_item["title"]}</a>'
                    if _link else news_item["title"]
                )
                _pub = news_item.get("publisher", "")
                _time = news_item.get("publish_time", "")
                _source = news_item.get("source_name", "")
                _meta_parts = [p for p in [_pub, _source, _time[:16] if _time else ""] if p]
                _meta = " · ".join(_meta_parts)

                _cat_badges = ""
                for cat in news_item.get("categories", []):
                    _cat_badges += (
                        f'<span class="news-badge news-badge-category">'
                        f'{cat["icon"]} {cat["label"]}</span>'
                    )

                st.markdown(
                    f'<div class="news-card news-impact-none">'
                    f'<div class="news-title">{_num_badge}{_title_html}</div>'
                    f'<div class="news-meta">{_meta}</div>'
                    f'<div>{_cat_badges}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
else:
    st.info("📰 経済ニュースの取得なし（ネットワーク接続を確認してください）")

# --- Copilot CLI 実行ログ ---
_cli_logs = copilot_get_logs()
if _cli_logs:
    with st.expander(f"🔍 Copilot CLI 実行ログ（{len(_cli_logs)}件）", expanded=False):
        _log_col1, _log_col2 = st.columns([6, 1])
        with _log_col2:
            if st.button("🗑️ クリア", key="clear_cli_logs"):
                copilot_clear_logs()
                st.rerun()
        for _log in _cli_logs:
            import datetime as _dt
            _ts = _dt.datetime.fromtimestamp(_log.timestamp).strftime("%H:%M:%S")
            _status = "✅" if _log.success else "❌"
            _src = f" [{_log.source}]" if _log.source else ""
            _header = f"{_status} {_ts} — {_log.model} ({_log.duration_sec:.1f}s){_src}"
            if _log.success:
                _detail = (
                    f"**プロンプト** (先頭150文字):\n```\n{_log.prompt_preview}\n```\n\n"
                    f"**応答** ({_log.response_length}文字):\n```\n{_log.response_preview}\n```"
                )
            else:
                _detail = (
                    f"**プロンプト** (先頭150文字):\n```\n{_log.prompt_preview}\n```\n\n"
                    f"**エラー**: `{_log.error}`"
                )
            with st.expander(_header, expanded=False):
                st.markdown(_detail)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# =====================================================================
# 総資産推移グラフ
# =====================================================================
st.markdown('<div id="total-chart"></div>', unsafe_allow_html=True)
st.markdown("### 📊 総資産推移")
st.caption("資産全体の値動きを時系列で確認。ドローダウンやシャープレシオの推移も合わせて表示します。")

if not history_df.empty:
    # ベンチマーク系列の取得
    bench_series = None
    if benchmark_symbol:
        bench_series = get_benchmark_series(benchmark_symbol, history_df, period)

    fig_total = build_total_chart(history_df, chart_style, bench_series, benchmark_label)
    st.plotly_chart(fig_total, key="chart_total")

    # ---------------------------------------------------------------
    # ドローダウンチャート
    # ---------------------------------------------------------------
    _dd_series = compute_drawdown_series(history_df)
    if not _dd_series.empty:
        fig_dd = build_drawdown_chart(_dd_series)
        st.plotly_chart(fig_dd, key="chart_drawdown")

    # ---------------------------------------------------------------
    # ローリングSharpe比
    # ---------------------------------------------------------------
    _rolling_window = 60
    _rolling_sharpe = compute_rolling_sharpe(history_df, window=_rolling_window)
    if not _rolling_sharpe.empty:
        fig_rs = build_rolling_sharpe_chart(_rolling_sharpe, window=_rolling_window)
        st.plotly_chart(fig_rs, key="chart_rolling_sharpe")

    # ---------------------------------------------------------------
    # 投資額 vs 評価額
    # ---------------------------------------------------------------
    if show_invested and "invested" in history_df.columns:
        st.markdown('<div id="invested-chart"></div>', unsafe_allow_html=True)
        st.markdown("### 💰 投資額 vs 評価額")
        st.caption("累計投資額と現在の評価額を比較し、投入資金に対するリターンを視覚的に確認できます。")
        fig_inv = build_invested_chart(history_df)
        st.plotly_chart(fig_inv, key="chart_invested")

    # ---------------------------------------------------------------
    # 目標ライン & 将来推定推移
    # ---------------------------------------------------------------
    if show_projection:
        st.markdown('<div id="projection"></div>', unsafe_allow_html=True)
        st.markdown("### 🔮 総資産推移 & 将来推定")
        st.caption("過去のリターン実績をもとに、楽観・基本・悲観の3シナリオで将来の資産推移を推計します。")

        projection_df = build_projection(
            current_value=total_value,
            years=projection_years,
        )

        fig_proj = build_projection_chart(history_df, projection_df, target_amount)
        st.plotly_chart(fig_proj, key="chart_projection")

        # 推定リターンのサマリー
        opt_val = projection_df["optimistic"].iloc[-1]
        base_val = projection_df["base"].iloc[-1]
        pess_val = projection_df["pessimistic"].iloc[-1]
        opt_rate = (opt_val / total_value - 1) * 100
        base_rate_pct = (base_val / total_value - 1) * 100
        pess_rate = (pess_val / total_value - 1) * 100

        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            st.markdown(
                f'<div style="text-align:center; padding:8px;">'
                f'<span style="font-size:0.85rem; opacity:0.7;">🟢 楽観（{projection_years}年後）</span><br>'
                f'<span style="font-size:1.3rem; font-weight:600; color:#4ade80;">'
                f'¥{opt_val:,.0f}</span><br>'
                f'<span style="font-size:0.8rem; color:#4ade80;">{opt_rate:+.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with scol2:
            st.markdown(
                f'<div style="text-align:center; padding:8px;">'
                f'<span style="font-size:0.85rem; opacity:0.7;">🟣 ベース（{projection_years}年後）</span><br>'
                f'<span style="font-size:1.3rem; font-weight:600; color:#a78bfa;">'
                f'¥{base_val:,.0f}</span><br>'
                f'<span style="font-size:0.8rem; color:#a78bfa;">{base_rate_pct:+.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with scol3:
            st.markdown(
                f'<div style="text-align:center; padding:8px;">'
                f'<span style="font-size:0.85rem; opacity:0.7;">🔴 悲観（{projection_years}年後）</span><br>'
                f'<span style="font-size:1.3rem; font-weight:600; color:#f87171;">'
                f'¥{pess_val:,.0f}</span><br>'
                f'<span style="font-size:0.8rem; color:#f87171;">{pess_rate:+.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

else:
    st.warning("株価履歴データが取得できませんでした。")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# =====================================================================
# 現在の保有構成
# =====================================================================
st.markdown('<div id="holdings"></div>', unsafe_allow_html=True)
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 🏢 銘柄別 評価額")
    st.caption("保有銘柄ごとの評価額・損益率を確認。構成比の偏りや損益の大きい銘柄を把握できます。")

    holdings_df = pd.DataFrame([
        {
            "銘柄": f"{p['name']} ({p['symbol']})",
            "保有数": p["shares"],
            "現在価格": f"{p['current_price']:,.2f} {p.get('currency', '')}",
            "評価額(円)": p["evaluation_jpy"],
            "構成比": p["evaluation_jpy"] / total_value * 100 if total_value else 0,
            "損益(円)": p.get("pnl_jpy", 0),
            "損益率(%)": p.get("pnl_pct", 0),
            "通貨": p.get("currency", ""),
            "セクター": p.get("sector", ""),
        }
        for p in positions
    ])

    if not holdings_df.empty:
        # 評価額でソート
        holdings_df = holdings_df.sort_values("評価額(円)", ascending=False)

        st.dataframe(
            holdings_df.style.format({
                "評価額(円)": "¥{:,.0f}",
                "構成比": "{:.1f}%",
                "損益(円)": "¥{:,.0f}",
                "損益率(%)": "{:+.1f}%",
            }).background_gradient(
                subset=["損益率(%)"],
                cmap="RdYlGn",
                vmin=-30,
                vmax=30,
            ).map(
                lambda v: "color: #4ade80" if isinstance(v, (int, float)) and v > 0
                else ("color: #f87171" if isinstance(v, (int, float)) and v < 0 else ""),
                subset=["損益(円)"],
            ),
            width="stretch",
            height=400,
        )

        # CSVダウンロード
        csv_data = holdings_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 保有一覧をCSVダウンロード",
            data=csv_data,
            file_name=f"holdings_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

with col_right:
    st.markdown("### 🥧 セクター構成")
    st.caption("セクター別の配分比率。特定業種への偏りがないか確認しましょう。")

    sector_df = get_sector_breakdown(snapshot)
    if not sector_df.empty:
        fig_sector = build_sector_chart(sector_df)
        st.plotly_chart(fig_sector, key="chart_sector")
    else:
        st.info("セクターデータなし")

    # 通貨別エクスポージャー
    st.markdown("### 💱 通貨別配分")
    st.caption("通貨エクスポージャーの確認。為替リスクの偏りを把握できます。")
    fig_cur = build_currency_chart(positions)
    if fig_cur is not None:
        st.plotly_chart(fig_cur, key="chart_currency")

# --- 構成比ツリーマップ（フルワイド表示） ---
st.markdown("### 🌳 構成比ツリーマップ")
st.caption("銘柄の評価額を面積で表現。大きいほど構成比が高く、ポートフォリオ全体像を直感的に把握できます。")
fig_treemap = build_treemap_chart(positions)
if fig_treemap is not None:
    st.plotly_chart(fig_treemap, width="stretch", key="chart_treemap")
else:
    st.info("ツリーマップの表示に必要なデータがありません")

# --- ウェイトドリフト警告 ---
drift_alerts = compute_weight_drift(positions, total_value)
if drift_alerts:
    st.markdown("### ⚖️ ウェイトドリフト警告")
    st.caption("均等配分からの乖離が大きい銘柄を表示。値上がりで膨らんだ銘柄のリバランス検討に活用できます。")
    drift_cols = st.columns(min(len(drift_alerts), 4))
    for i, alert in enumerate(drift_alerts[:4]):
        with drift_cols[i]:
            if alert["status"] == "overweight":
                icon = "🔺"
                color = "#f59e0b"
                label = "オーバーウェイト"
            else:
                icon = "🔻"
                color = "#6366f1"
                label = "アンダーウェイト"
            st.markdown(
                f'<div class="kpi-card kpi-risk" style="text-align:center;">'
                f'<span style="font-size:0.8rem; opacity:0.7;">{icon} {label}</span><br>'
                f'<span style="font-size:1.1rem; font-weight:600;">{alert["name"]}</span><br>'
                f'<span style="font-size:0.85rem;">現在 {alert["current_pct"]:.1f}% '
                f'→ 目標 {alert["target_pct"]:.1f}%</span><br>'
                f'<span style="font-size:1.0rem; font-weight:600; color:{color};">'
                f'{alert["drift_pct"]:+.1f}pp</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# --- 銘柄間相関ヒートマップ ---
if not history_df.empty:
    corr_matrix = compute_correlation_matrix(history_df)
    if not corr_matrix.empty:
        st.markdown("### 🔗 銘柄間 日次リターン相関")
        st.caption("銘柄同士の値動きの連動性を表示。相関が高い銘柄が多いと分散効果が薄れるため、確認が重要です。")
        fig_corr = build_correlation_chart(corr_matrix)
        if fig_corr is not None:
            st.plotly_chart(fig_corr, width="stretch", key="chart_correlation")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# =====================================================================
# 銘柄別個別チャート
# =====================================================================
if show_individual and not history_df.empty:
    st.markdown('<div id="individual-chart"></div>', unsafe_allow_html=True)
    st.markdown("### 📉 銘柄別 個別推移")
    st.caption("各銘柄の評価額推移を個別に確認。特定銘柄の値動きパターンを詳しく見たいときに。")

    stock_cols = [c for c in history_df.columns if c not in ("total", "invested")]
    cols_per_row = 2
    for i in range(0, len(stock_cols), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col_widget in enumerate(cols):
            idx = i + j
            if idx >= len(stock_cols):
                break
            symbol = stock_cols[idx]
            with col_widget:
                fig_ind = build_individual_chart(history_df, symbol)
                st.plotly_chart(fig_ind, key=f"chart_ind_{symbol}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# =====================================================================
# 月次サマリー
# =====================================================================
st.markdown('<div id="monthly"></div>', unsafe_allow_html=True)
st.markdown("### 📅 月次サマリー")
st.caption("月末時点の評価額と前月比変動率を一覧表示。月単位でのパフォーマンス傾向を確認できます。")

if not history_df.empty:
    monthly_df = get_monthly_summary(history_df)
    if not monthly_df.empty:
        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            fig_monthly = build_monthly_chart(monthly_df)
            st.plotly_chart(fig_monthly, key="chart_monthly")

        with col_table:
            display_cols = ["month_end_value_jpy", "change_pct"]
            col_names = {"month_end_value_jpy": "月末評価額(円)", "change_pct": "前月比(%)"}
            fmt = {"月末評価額(円)": "¥{:,.0f}", "前月比(%)": "{:+.1f}%"}
            if "invested_jpy" in monthly_df.columns:
                display_cols.insert(1, "invested_jpy")
                col_names["invested_jpy"] = "投資額(円)"
                fmt["投資額(円)"] = "¥{:,.0f}"
            if "yoy_pct" in monthly_df.columns:
                display_cols.append("yoy_pct")
                col_names["yoy_pct"] = "前年同月比(%)"
                fmt["前年同月比(%)"] = "{:+.1f}%"
            if "unrealized_pnl" in monthly_df.columns:
                display_cols.append("unrealized_pnl")
                col_names["unrealized_pnl"] = "含み損益(円)"
                fmt["含み損益(円)"] = "¥{:,.0f}"
            display_monthly = monthly_df[display_cols].rename(columns=col_names)
            st.dataframe(
                display_monthly.style.format(fmt),
                width="stretch",
            )
            # 月次CSVダウンロード
            monthly_csv = display_monthly.to_csv().encode("utf-8-sig")
            st.download_button(
                "📥 月次サマリーをCSVダウンロード",
                data=monthly_csv,
                file_name=f"monthly_summary_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
    else:
        st.info("月次データなし（データ期間が短い可能性があります）")
else:
    st.info("履歴データがありません")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# =====================================================================
# 取引アクティビティ
# =====================================================================
st.markdown('<div id="trade-activity"></div>', unsafe_allow_html=True)
st.markdown("### 🔄 月次売買アクティビティ")
st.caption("月ごとの売買件数・金額フローを表示。投資ペースや資金の出入りを振り返るのに便利です。")

trade_act_df = load_trade_activity()
if not trade_act_df.empty:
    col_flow, col_tbl = st.columns([2, 1])

    with col_flow:
        fig_flow = build_trade_flow_chart(trade_act_df)
        st.plotly_chart(fig_flow, key="chart_trade_flow")

    with col_tbl:
        display_act = trade_act_df.copy()
        display_act.columns = [
            "購入件数", "購入額(円)", "売却件数", "売却額(円)", "ネット(円)"
        ]
        st.dataframe(
            display_act.style.format({
                "購入件数": "{:.0f}",
                "購入額(円)": "¥{:,.0f}",
                "売却件数": "{:.0f}",
                "売却額(円)": "¥{:,.0f}",
                "ネット(円)": "¥{:,.0f}",
            }),
            width="stretch",
        )
else:
    st.info("取引データがありません")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# =====================================================================
# Copilot チャット
# =====================================================================
st.markdown('<div id="copilot-chat"></div>', unsafe_allow_html=True)
st.markdown("### 💬 Copilot に相談")
st.caption("ダッシュボードの全データを踏まえて、Copilot に自由に質問できます。")

# チャット履歴の初期化
if "copilot_chat_messages" not in st.session_state:
    st.session_state["copilot_chat_messages"] = []


# --- ダッシュボードコンテキストを自動構築 ---
def _build_chat_context() -> str:
    """ダッシュボード上の全情報をプロンプトコンテキストとして構築する."""
    parts: list[str] = []
    parts.append("## ポートフォリオ概要")
    parts.append(f"総資産: ¥{total_value:,.0f}")
    parts.append(f"前日比: ¥{_dc_jpy:+,.0f} ({_dc_pct:+.1f}%)")
    parts.append(f"含み損益: ¥{unrealized_pnl:,.0f} ({unrealized_pnl_pct:+.1f}%)")
    parts.append(f"実現損益: ¥{realized_pnl:,.0f}")
    parts.append(f"トータル損益: ¥{total_pnl:,.0f}")
    parts.append(f"銘柄数: {len(positions)}")

    # リスク指標
    if not history_df.empty:
        try:
            _ctx_risk = compute_risk_metrics(history_df)
            parts.append("\n## リスク指標")
            parts.append(f"シャープレシオ: {_ctx_risk['sharpe_ratio']:.2f}")
            parts.append(f"ボラティリティ: {_ctx_risk['volatility_pct']:.1f}%")
            parts.append(f"最大ドローダウン: {_ctx_risk['max_drawdown_pct']:.1f}%")
        except Exception:
            pass

    # 保有銘柄
    parts.append("\n## 保有銘柄")
    for p in positions:
        _sym = p.get("symbol", "")
        _name = p.get("name", "")
        _pnl = p.get("pnl_pct", 0)
        _eval_jpy = p.get("evaluation_jpy", 0)
        _sector = p.get("sector", "")
        _weight = (_eval_jpy / total_value * 100) if total_value else 0
        parts.append(f"- {_name} ({_sym}): 評価額¥{_eval_jpy:,.0f} 構成比{_weight:.1f}% 損益{_pnl:+.1f}% セクター:{_sector}")

    # ヘルスチェック結果
    if health_data is not None:
        _hc_pos = health_data["positions"]
        _hc_alerts_list = health_data["sell_alerts"]
        _alert_pos = [p for p in _hc_pos if p.get("alert_level") != "none"]
        if _alert_pos:
            parts.append("\n## ヘルスチェック アラート")
            for _hp in _alert_pos:
                _hp_sym = _hp.get("symbol", "")
                _hp_name = _hp.get("name", "")
                _hp_level = _hp.get("alert_level", "")
                _hp_reasons = ", ".join(_hp.get("alert_reasons", []))
                _hp_trend = _hp.get("trend", "")
                parts.append(f"- {_hp_name} ({_hp_sym}): [{_hp_level}] {_hp_reasons} トレンド:{_hp_trend}")

        # 売りアラート
        if _hc_alerts_list:
            parts.append("\n## 売りタイミング通知")
            for _sa_ctx in _hc_alerts_list:
                parts.append(f"- {_sa_ctx.get('name', '')} ({_sa_ctx.get('symbol', '')}): {_sa_ctx.get('action', '')} — {_sa_ctx.get('reason', '')}")

    # LLM ヘルスサマリー（session_stateに格納されていれば利用）
    _chat_hc_summary = st.session_state.get("_hc_llm_summary_data")
    if _chat_hc_summary:
        parts.append("\n## AI ヘルスチェック分析")
        _overview_ctx = _chat_hc_summary.get("overview", "")
        if _overview_ctx:
            parts.append(_overview_ctx)
        _warning_ctx = _chat_hc_summary.get("risk_warning", "")
        if _warning_ctx:
            parts.append(f"リスク注意: {_warning_ctx}")

    # 経済ニュース
    try:
        _chat_econ_news = econ_news  # noqa: F841 — top-level variable
    except NameError:
        _chat_econ_news = []
    if _chat_econ_news:
        _impact_items = [n for n in _chat_econ_news if n.get("portfolio_impact", {}).get("impact_level") != "none"]
        if _impact_items:
            parts.append("\n## 経済ニュース（PF影響あり）")
            for _ni in _impact_items[:10]:  # 最大10件
                _ni_title = _ni.get("title", "")
                _ni_impact = _ni.get("portfolio_impact", {})
                _ni_level = _ni_impact.get("impact_level", "")
                _ni_reason = _ni_impact.get("reason", "")
                parts.append(f"- [{_ni_level}] {_ni_title}: {_ni_reason}")

    return "\n".join(parts)


# コンテキストバッジ
_ctx_items = []
_ctx_items.append(f"銘柄 {len(positions)}")
if health_data is not None:
    _n_alerts = sum(1 for p in health_data["positions"] if p.get("alert_level") != "none")
    if _n_alerts:
        _ctx_items.append(f"アラート {_n_alerts}")
    if health_data["sell_alerts"]:
        _ctx_items.append(f"売り通知 {len(health_data['sell_alerts'])}")
if st.session_state.get("_hc_llm_summary_data"):
    _ctx_items.append("AI分析")
try:
    if econ_news:
        _ctx_items.append(f"ニュース {len(econ_news)}")
except NameError:
    pass

_badges_html = " ".join(
    f'<span class="copilot-chat-context-badge">{item}</span>'
    for item in _ctx_items
)
st.markdown(
    f'<div style="margin-bottom:10px;">'
    f'<span style="font-size:0.82rem; opacity:0.7;">📎 自動添付コンテキスト:</span> '
    f'{_badges_html}</div>',
    unsafe_allow_html=True,
)

# モデル表示 & クリアボタン
_chat_col_model, _chat_col_clear = st.columns([4, 1])
with _chat_col_model:
    _chat_model_ids = [m[0] for m in COPILOT_MODELS]
    _chat_model_labels = [m[1] for m in COPILOT_MODELS]
    _chat_model_current_idx = (
        _chat_model_ids.index(chat_model)
        if chat_model in _chat_model_ids
        else 0
    )
    st.caption(f"🧠 モデル: **{_chat_model_labels[_chat_model_current_idx]}**（設定で変更可能）")
with _chat_col_clear:
    if st.button("🗑️ クリア", key="copilot_chat_clear"):
        st.session_state["copilot_chat_messages"] = []
        st.rerun()

# チャット履歴表示
for _msg in st.session_state["copilot_chat_messages"]:
    if _msg["role"] == "user":
        st.markdown(
            f'<div class="copilot-chat-msg copilot-chat-msg-user">'
            f'<div class="copilot-chat-msg-role">👤 あなた</div>'
            f'<div class="copilot-chat-msg-text">{_msg["content"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="copilot-chat-msg copilot-chat-msg-ai">'
            '<div class="copilot-chat-msg-role">🤖 Copilot</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_msg["content"])

# 入力欄
_chat_input = st.chat_input(
    "ダッシュボードについて質問...",
    key="copilot_chat_input",
)

if _chat_input:
    # ユーザーメッセージを追加
    st.session_state["copilot_chat_messages"].append(
        {"role": "user", "content": _chat_input}
    )

    # コンテキスト付きプロンプトを構築
    _dashboard_ctx = _build_chat_context()
    _chat_prompt = (
        "あなたはポートフォリオ分析の専門家です。\n"
        "以下のダッシュボード情報を踏まえて、ユーザーの質問に日本語で回答してください。\n"
        "回答は簡潔かつ具体的に。数値データを活用してください。\n\n"
        f"--- ダッシュボードデータ ---\n{_dashboard_ctx}\n\n"
    )
    # 直近の会話履歴を含める（最大5往復）
    _recent_msgs = st.session_state["copilot_chat_messages"][-10:]
    if len(_recent_msgs) > 1:
        _chat_prompt += "--- 会話履歴 ---\n"
        for _hm in _recent_msgs[:-1]:  # 最新のユーザー入力以外
            _hm_role = "ユーザー" if _hm["role"] == "user" else "アシスタント"
            _chat_prompt += f"{_hm_role}: {_hm['content']}\n"
        _chat_prompt += "\n"

    _chat_prompt += f"--- ユーザーの質問 ---\n{_chat_input}"

    # Copilot CLI 呼び出し
    with st.spinner("🤖 Copilot が考えています..."):
        _chat_response = copilot_call(
            _chat_prompt,
            model=chat_model,
            timeout=120,
            source="dashboard_chat",
        )

    if _chat_response:
        st.session_state["copilot_chat_messages"].append(
            {"role": "assistant", "content": _chat_response}
        )
    else:
        st.session_state["copilot_chat_messages"].append(
            {"role": "assistant", "content": "⚠️ 応答を取得できませんでした。Copilot CLI の状態を確認してください。"}
        )
    st.rerun()

# =====================================================================
# フッター
# =====================================================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.caption(
    "Data provided by Yahoo Finance via yfinance. "
    "Values are estimates and may differ from actual brokerage accounts."
)
