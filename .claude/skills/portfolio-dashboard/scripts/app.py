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


# =====================================================================
# サイドバー（タブ: 目次 / 設定）
# =====================================================================
st.sidebar.title("📊 Portfolio Dashboard")

_tab_toc, _tab_settings = st.sidebar.tabs(["📑 目次", "⚙️ 設定"])

# --- 目次タブ ---
with _tab_toc:
    st.markdown(
        '<div style="display:flex; flex-direction:column; gap:2px; padding:4px 0;">'
        '<a class="toc-link" href="#summary">📈 サマリー</a>'
        '<a class="toc-link" href="#health-check">🏥 ヘルスチェック</a>'
        '<a class="toc-link" href="#total-chart">📊 総資産推移</a>'
        '<a class="toc-link" href="#invested-chart">💰 投資額 vs 評価額</a>'
        '<a class="toc-link" href="#projection">🔮 将来推定</a>'
        '<a class="toc-link" href="#holdings">🏢 保有銘柄・構成</a>'
        '<a class="toc-link" href="#individual-chart">📉 銘柄別チャート</a>'
        '<a class="toc-link" href="#monthly">📅 月次サマリー</a>'
        '<a class="toc-link" href="#trade-activity">🔄 売買アクティビティ</a>'
        '</div>',
        unsafe_allow_html=True,
    )

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

    period_label = st.selectbox(
        "📅 表示期間",
        options=[label for label, _ in _PERIOD_OPTIONS],
        index=1,
        help="株価履歴の取得期間",
    )
    period = dict(_PERIOD_OPTIONS)[period_label]

    chart_style = st.radio(
        "🎨 チャートスタイル",
        options=["積み上げ面", "折れ線", "積み上げ棒"],
        index=0,
    )

    show_invested = st.checkbox(
        "投資額 vs 評価額を表示",
        value=True,
    )

    # ベンチマーク選択
    _BENCHMARK_OPTIONS = {
        "なし": None,
        "S&P 500 (SPY)": "SPY",
        "VTI (米国全体)": "VTI",
        "日経225 (^N225)": "^N225",
        "TOPIX (^TPX)": "1306.T",
    }
    benchmark_label = st.selectbox(
        "📏 ベンチマーク比較",
        options=list(_BENCHMARK_OPTIONS.keys()),
        index=0,
        help="総資産推移にベンチマークのパフォーマンスを重ねて表示",
    )
    benchmark_symbol = _BENCHMARK_OPTIONS[benchmark_label]

    show_individual = st.checkbox(
        "銘柄別の個別チャートを表示",
        value=False,
    )

    st.markdown("---")

    # --- 目標・推定セクション ---
    st.markdown("#### 🎯 目標・将来推定")

    show_projection = st.checkbox(
        "目標ライン & 将来推定を表示",
        value=True,
    )

    target_amount = st.number_input(
        "🎯 目標資産額（万円）",
        min_value=0,
        max_value=100000,
        value=5000,
        step=500,
        help="総資産推移グラフに水平ラインとして表示",
    ) * 10000  # 万円→円

    projection_years = st.slider(
        "📅 推定期間（年）",
        min_value=1,
        max_value=20,
        value=5,
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
    auto_refresh_label = st.selectbox(
        "⏱ 自動更新間隔",
        options=[label for label, _ in _REFRESH_OPTIONS],
        index=2,  # デフォルト: 5分
        help="選択した間隔でダッシュボードを自動リロードします",
    )
    auto_refresh_sec = dict(_REFRESH_OPTIONS)[auto_refresh_label]

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
# 総資産推移グラフ
# =====================================================================
st.markdown('<div id="total-chart"></div>', unsafe_allow_html=True)
st.markdown("### 📊 総資産推移")

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
        fig_inv = build_invested_chart(history_df)
        st.plotly_chart(fig_inv, key="chart_invested")

    # ---------------------------------------------------------------
    # 目標ライン & 将来推定推移
    # ---------------------------------------------------------------
    if show_projection:
        st.markdown('<div id="projection"></div>', unsafe_allow_html=True)
        st.markdown("### 🔮 総資産推移 & 将来推定")

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

    sector_df = get_sector_breakdown(snapshot)
    if not sector_df.empty:
        fig_sector = build_sector_chart(sector_df)
        st.plotly_chart(fig_sector, key="chart_sector")
    else:
        st.info("セクターデータなし")

    # 通貨別エクスポージャー
    st.markdown("### 💱 通貨別配分")
    fig_cur = build_currency_chart(positions)
    if fig_cur is not None:
        st.plotly_chart(fig_cur, key="chart_currency")

# --- 構成比ツリーマップ（フルワイド表示） ---
st.markdown("### 🌳 構成比ツリーマップ")
fig_treemap = build_treemap_chart(positions)
if fig_treemap is not None:
    st.plotly_chart(fig_treemap, width="stretch", key="chart_treemap")
else:
    st.info("ツリーマップの表示に必要なデータがありません")

# --- ウェイトドリフト警告 ---
drift_alerts = compute_weight_drift(positions, total_value)
if drift_alerts:
    st.markdown("### ⚖️ ウェイトドリフト警告")
    st.caption("均等ウェイトからの乖離が5pp以上の銘柄")
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

# =====================================================================
# フッター
# =====================================================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.caption(
    "Data provided by Yahoo Finance via yfinance. "
    "Values are estimates and may differ from actual brokerage accounts."
)
