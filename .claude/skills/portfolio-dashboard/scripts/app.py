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

period = st.sidebar.selectbox(
    "📅 表示期間",
    options=["1mo", "3mo", "6mo", "1y", "2y"],
    index=1,
    help="株価履歴の取得期間",
)

chart_style = st.sidebar.radio(
    "🎨 チャートスタイル",
    options=["積み上げ面", "折れ線", "積み上げ棒"],
    index=0,
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
total_pnl = total_value - total_cost if total_cost else 0
total_pnl_pct = ((total_value / total_cost) - 1) * 100 if total_cost else 0
num_holdings = len([p for p in positions if p.get("sector") != "Cash"])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="トータル資産（円換算）",
        value=f"¥{total_value:,.0f}",
    )
with col2:
    st.metric(
        label="トータル損益",
        value=f"¥{total_pnl:,.0f}",
        delta=f"{total_pnl_pct:+.2f}%",
    )
with col3:
    st.metric(
        label="保有銘柄数",
        value=f"{num_holdings}",
    )
with col4:
    st.metric(
        label="最終更新",
        value=snapshot["as_of"][:10],
    )

st.markdown("---")

# =====================================================================
# 総資産推移グラフ
# =====================================================================
st.markdown("### 📊 総資産推移")

if not history_df.empty:
    # 銘柄列（total 以外）を取得
    stock_cols = [c for c in history_df.columns if c != "total"]

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

    stock_cols = [c for c in history_df.columns if c != "total"]
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
                marker_color=[
                    "#4ade80" if v >= 0 else "#f87171"
                    for v in monthly_df["change_pct"].fillna(0)
                ],
                hovertemplate="月末資産: ¥%{y:,.0f}<extra></extra>",
            ))
            fig_monthly.update_layout(
                title="月末資産額の推移",
                xaxis_title="月",
                yaxis_title="評価額（円）",
                height=350,
                yaxis=dict(tickformat=","),
            )
            st.plotly_chart(fig_monthly, key="chart_monthly")

        with col_table:
            display_monthly = monthly_df.copy()
            display_monthly.columns = ["月末評価額(円)", "前月比(%)"]
            st.dataframe(
                display_monthly.style.format({
                    "月末評価額(円)": "¥{:,.0f}",
                    "前月比(%)": "{:+.1f}%",
                }),
                width="stretch",
            )
    else:
        st.info("月次データなし（データ期間が短い可能性があります）")
else:
    st.info("履歴データがありません")

# =====================================================================
# フッター
# =====================================================================
st.markdown("---")
st.caption(
    "Data provided by Yahoo Finance via yfinance. "
    "Values are estimates and may differ from actual brokerage accounts. "
    f"Generated at {snapshot.get('as_of', 'N/A')}"
)
