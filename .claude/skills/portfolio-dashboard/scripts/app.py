"""ポートフォリオダッシュボード — Streamlit アプリ.

総資産推移 / 銘柄別評価額 / セクター構成 / 月次サマリー を
インタラクティブなグラフで表示する。

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# --- コンポーネントを import ---
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from components.data_loader import (
    get_current_snapshot,
    build_portfolio_history,
    get_sector_breakdown,
    get_monthly_summary,
    get_trade_activity,
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
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .positive { color: #4ade80; }
    .negative { color: #f87171; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# サイドバー
# =====================================================================
st.sidebar.title("📊 Portfolio Dashboard")
st.sidebar.markdown("---")

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

period_label = st.sidebar.selectbox(
    "📅 表示期間",
    options=[label for label, _ in _PERIOD_OPTIONS],
    index=1,
    help="株価履歴の取得期間",
)
period = dict(_PERIOD_OPTIONS)[period_label]

chart_style = st.sidebar.radio(
    "🎨 チャートスタイル",
    options=["積み上げ面", "折れ線", "積み上げ棒"],
    index=0,
)

show_invested = st.sidebar.checkbox(
    "投資額 vs 評価額を表示",
    value=True,
)

show_individual = st.sidebar.checkbox(
    "銘柄別の個別チャートを表示",
    value=False,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data Source**: yfinance + portfolio.csv\n\n"
    "**Last Update**: Auto on page load"
)


# =====================================================================
# データ取得（キャッシュ付き）
# =====================================================================
@st.cache_data(ttl=300, show_spinner="データを取得中...")
def load_snapshot():
    return get_current_snapshot()


@st.cache_data(ttl=300, show_spinner="株価履歴を取得中...")
def load_history(period_val: str):
    return build_portfolio_history(period=period_val)


# =====================================================================
# メインコンテンツ
# =====================================================================
st.title("💼 ポートフォリオダッシュボード")

# --- データ読み込み ---
with st.spinner("ポートフォリオデータを読み込み中..."):
    snapshot = load_snapshot()
    history_df = load_history(period)

# =====================================================================
# KPI メトリクスカード
# =====================================================================
st.markdown("### 📈 サマリー")

positions = snapshot["positions"]
total_value = snapshot["total_value_jpy"]
total_cost = sum(p.get("cost_jpy", 0) for p in positions if "cost_jpy" in p)
unrealized_pnl = total_value - total_cost if total_cost else 0
unrealized_pnl_pct = ((total_value / total_cost) - 1) * 100 if total_cost else 0
realized_pnl = snapshot.get("realized_pnl", {}).get("total_jpy", 0)
total_pnl = unrealized_pnl + realized_pnl
num_holdings = len([p for p in positions if p.get("sector") != "Cash"])

# --- メイン KPI (大きく表示) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="トータル資産（円換算）",
        value=f"¥{total_value:,.0f}",
    )
with col2:
    st.metric(
        label="評価損益（含み）",
        value=f"¥{unrealized_pnl:,.0f}",
        delta=f"{unrealized_pnl_pct:+.2f}%",
    )
with col3:
    st.metric(
        label="保有銘柄数",
        value=f"{num_holdings}",
        delta=f"更新: {snapshot['as_of'][:10]}",
    )

# --- サブ KPI (小さく表示) ---
realized_sign = "+" if realized_pnl >= 0 else ""
total_pnl_sign = "+" if total_pnl >= 0 else ""
realized_color = "#4ade80" if realized_pnl >= 0 else "#f87171"
total_pnl_color = "#4ade80" if total_pnl >= 0 else "#f87171"

sub_col1, sub_col2 = st.columns(2)
with sub_col1:
    st.markdown(
        f'<div style="padding: 4px 0;">'
        f'<span style="font-size: 0.85rem; opacity: 0.7;">トータル損益（実現＋含み）</span><br>'
        f'<span style="font-size: 1.2rem; font-weight: 600; color: {total_pnl_color};">'
        f'{total_pnl_sign}¥{total_pnl:,.0f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
with sub_col2:
    st.markdown(
        f'<div style="padding: 4px 0;">'
        f'<span style="font-size: 0.85rem; opacity: 0.7;">実現損益（確定済）</span><br>'
        f'<span style="font-size: 1.2rem; font-weight: 600; color: {realized_color};">'
        f'{realized_sign}¥{realized_pnl:,.0f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# =====================================================================
# 総資産推移グラフ
# =====================================================================
st.markdown("### 📊 総資産推移")

if not history_df.empty:
    # 銘柄列（total / invested 以外）を取得
    stock_cols = [c for c in history_df.columns if c not in ("total", "invested")]

    if chart_style == "積み上げ面":
        fig_total = go.Figure()
        for col in stock_cols:
            fig_total.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df[col],
                mode="lines",
                stackgroup="one",
                name=col,
                hovertemplate="%{x}<br>%{fullData.name}: ¥%{y:,.0f}<extra></extra>",
            ))
        fig_total.update_layout(
            title="保有銘柄別 評価額推移（積み上げ面グラフ）",
            xaxis_title="日付",
            yaxis_title="評価額（円）",
            hovermode="x unified",
            height=500,
            yaxis=dict(tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )

    elif chart_style == "折れ線":
        fig_total = go.Figure()
        # 合計の太線
        fig_total.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df["total"],
            mode="lines",
            name="合計",
            line=dict(width=3, color="#fbbf24"),
            hovertemplate="合計: ¥%{y:,.0f}<extra></extra>",
        ))
        for col in stock_cols:
            fig_total.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df[col],
                mode="lines",
                name=col,
                hovertemplate="%{fullData.name}: ¥%{y:,.0f}<extra></extra>",
            ))
        fig_total.update_layout(
            title="保有銘柄別 評価額推移（折れ線グラフ）",
            xaxis_title="日付",
            yaxis_title="評価額（円）",
            hovermode="x unified",
            height=500,
            yaxis=dict(tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )

    else:  # 積み上げ棒
        # 日次だと棒が多すぎるので週次にリサンプル
        weekly = history_df[stock_cols].resample("W").last().ffill()
        fig_total = go.Figure()
        for col in stock_cols:
            fig_total.add_trace(go.Bar(
                x=weekly.index,
                y=weekly[col],
                name=col,
                hovertemplate="%{fullData.name}: ¥%{y:,.0f}<extra></extra>",
            ))
        fig_total.update_layout(
            barmode="stack",
            title="保有銘柄別 評価額推移（積み上げ棒グラフ・週次）",
            xaxis_title="日付",
            yaxis_title="評価額（円）",
            hovermode="x unified",
            height=500,
            yaxis=dict(tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )

    st.plotly_chart(fig_total, key="chart_total")

    # ---------------------------------------------------------------
    # 投資額 vs 評価額
    # ---------------------------------------------------------------
    if show_invested and "invested" in history_df.columns:
        st.markdown("### 💰 投資額 vs 評価額")

        fig_inv = go.Figure()
        fig_inv.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df["total"],
            mode="lines",
            name="評価額",
            line=dict(width=2, color="#60a5fa"),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.15)",
            hovertemplate="評価額: ¥%{y:,.0f}<extra></extra>",
        ))
        fig_inv.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df["invested"],
            mode="lines",
            name="累積投資額",
            line=dict(width=2, color="#f59e0b", dash="dot"),
            hovertemplate="投資額: ¥%{y:,.0f}<extra></extra>",
        ))
        fig_inv.update_layout(
            xaxis_title="日付",
            yaxis_title="金額（円）",
            hovermode="x unified",
            height=400,
            yaxis=dict(tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig_inv, key="chart_invested")
else:
    st.warning("株価履歴データが取得できませんでした。")

st.markdown("---")

# =====================================================================
# 現在の保有構成
# =====================================================================
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 🏢 銘柄別 評価額")

    holdings_df = pd.DataFrame([
        {
            "銘柄": f"{p['name']} ({p['symbol']})",
            "保有数": p["shares"],
            "現在価格": f"{p['current_price']:,.2f} {p.get('currency', '')}",
            "評価額(円)": p["evaluation_jpy"],
            "損益(円)": p.get("pnl_jpy", 0),
            "損益率": f"{p.get('pnl_pct', 0):+.1f}%",
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
                "損益(円)": "¥{:,.0f}",
            }).map(
                lambda v: "color: #4ade80" if isinstance(v, str) and v.startswith("+")
                else ("color: #f87171" if isinstance(v, str) and v.startswith("-") else ""),
                subset=["損益率"]
            ),
            width="stretch",
            height=400,
        )

with col_right:
    st.markdown("### 🥧 セクター構成")

    sector_df = get_sector_breakdown(snapshot)
    if not sector_df.empty:
        fig_sector = px.pie(
            sector_df,
            values="evaluation_jpy",
            names="sector",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_sector.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="%{label}<br>¥%{value:,.0f}<br>%{percent}<extra></extra>",
        )
        fig_sector.update_layout(
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig_sector, key="chart_sector")
    else:
        st.info("セクターデータなし")

st.markdown("---")

# =====================================================================
# 銘柄別個別チャート
# =====================================================================
if show_individual and not history_df.empty:
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
                fig_ind = go.Figure()
                fig_ind.add_trace(go.Scatter(
                    x=history_df.index,
                    y=history_df[symbol],
                    mode="lines",
                    fill="tozeroy",
                    name=symbol,
                    line=dict(width=2),
                    hovertemplate="¥%{y:,.0f}<extra></extra>",
                ))
                fig_ind.update_layout(
                    title=symbol,
                    height=250,
                    margin=dict(l=40, r=20, t=40, b=30),
                    yaxis=dict(tickformat=","),
                    showlegend=False,
                )
                st.plotly_chart(fig_ind, key=f"chart_ind_{symbol}")

    st.markdown("---")

# =====================================================================
# 月次サマリー
# =====================================================================
st.markdown("### 📅 月次サマリー")

if not history_df.empty:
    monthly_df = get_monthly_summary(history_df)
    if not monthly_df.empty:
        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_df.index,
                y=monthly_df["month_end_value_jpy"],
                name="月末評価額",
                marker_color=[
                    "#4ade80" if v >= 0 else "#f87171"
                    for v in monthly_df["change_pct"].fillna(0)
                ],
                hovertemplate="月末資産: ¥%{y:,.0f}<extra></extra>",
            ))
            if "invested_jpy" in monthly_df.columns:
                fig_monthly.add_trace(go.Scatter(
                    x=monthly_df.index,
                    y=monthly_df["invested_jpy"],
                    name="累積投資額",
                    mode="lines",
                    line=dict(width=2, color="#f59e0b", dash="dot"),
                    hovertemplate="投資額: ¥%{y:,.0f}<extra></extra>",
                ))
            fig_monthly.update_layout(
                title="月末資産額の推移",
                xaxis_title="月",
                yaxis_title="評価額（円）",
                height=350,
                yaxis=dict(tickformat=","),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35),
            )
            st.plotly_chart(fig_monthly, key="chart_monthly")

        with col_table:
            display_cols = ["month_end_value_jpy", "change_pct"]
            col_names = {"month_end_value_jpy": "月末評価額(円)", "change_pct": "前月比(%)"}
            fmt = {"月末評価額(円)": "¥{:,.0f}", "前月比(%)": "{:+.1f}%"}
            if "invested_jpy" in monthly_df.columns:
                display_cols.insert(1, "invested_jpy")
                col_names["invested_jpy"] = "投資額(円)"
                fmt["投資額(円)"] = "¥{:,.0f}"
            if "unrealized_pnl" in monthly_df.columns:
                display_cols.append("unrealized_pnl")
                col_names["unrealized_pnl"] = "含み損益(円)"
                fmt["含み損益(円)"] = "¥{:,.0f}"
            display_monthly = monthly_df[display_cols].rename(columns=col_names)
            st.dataframe(
                display_monthly.style.format(fmt),
                width="stretch",
            )
    else:
        st.info("月次データなし（データ期間が短い可能性があります）")
else:
    st.info("履歴データがありません")

# =====================================================================
# 取引アクティビティ
# =====================================================================
st.markdown("### 🔄 月次売買アクティビティ")


@st.cache_data(ttl=300, show_spinner="取引データを集計中...")
def load_trade_activity():
    return get_trade_activity()


trade_act_df = load_trade_activity()
if not trade_act_df.empty:
    col_flow, col_tbl = st.columns([2, 1])

    with col_flow:
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Bar(
            x=trade_act_df.index,
            y=trade_act_df["buy_amount"],
            name="購入額",
            marker_color="#60a5fa",
            hovertemplate="購入: ¥%{y:,.0f}<extra></extra>",
        ))
        fig_flow.add_trace(go.Bar(
            x=trade_act_df.index,
            y=-trade_act_df["sell_amount"],
            name="売却額",
            marker_color="#f87171",
            hovertemplate="売却: ¥%{y:,.0f}<extra></extra>",
        ))
        fig_flow.add_trace(go.Scatter(
            x=trade_act_df.index,
            y=trade_act_df["net_flow"],
            name="ネットフロー",
            mode="lines+markers",
            line=dict(color="#fbbf24", width=2),
            hovertemplate="ネット: ¥%{y:,.0f}<extra></extra>",
        ))
        fig_flow.update_layout(
            title="月次売買フロー",
            xaxis_title="月",
            yaxis_title="金額（円）",
            barmode="relative",
            height=350,
            yaxis=dict(tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        )
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
st.markdown("---")
st.caption(
    "Data provided by Yahoo Finance via yfinance. "
    "Values are estimates and may differ from actual brokerage accounts. "
    f"Generated at {snapshot.get('as_of', 'N/A')}"
)
