import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import (
    load_vnindex, load_sector_mapping, load_stock_prices,
    load_macro_assumptions, make_synthetic_fundamentals
)
from src.features import add_market_features, latest_stock_features
from src.models import train_market_classifier, predict_market
from src.scoring import score_stocks, score_sectors
from src.utils import make_commentary, traffic_light

st.set_page_config(
    page_title="VN AI-First Market Intelligence",
    page_icon="📈",
    layout="wide"
)

st.title("📈 VN AI-First Market Intelligence")
st.caption("VNINDEX forecasting • Sector ranking • Stock screener • Risk regime")

with st.sidebar:
    st.header("Thiết lập mô hình")
    horizon = st.selectbox("Horizon", ["20 phiên", "60 phiên"], index=0)
    top_n = st.slider("Số cổ phiếu hiển thị", 5, 30, 15)
    st.info("Mô hình ưu tiên dùng vnstock. Nếu không lấy được dữ liệu live, app tự dùng dữ liệu mẫu hợp lý.")

@st.cache_data(ttl=3600)
def load_all():
    sector_map = load_sector_mapping()
    tickers = sector_map["ticker"].dropna().unique().tolist()
    vnindex = load_vnindex()
    stock_prices = load_stock_prices(tickers)
    macro = load_macro_assumptions()
    fundamentals = make_synthetic_fundamentals(tickers)
    return vnindex, stock_prices, macro, fundamentals, sector_map

vnindex, stock_prices, macro, fundamentals, sector_map = load_all()

market_features = add_market_features(vnindex)
model, metrics = train_market_classifier(market_features)
market_view = predict_market(model, market_features)

stock_features = latest_stock_features(stock_prices)
vn_ret20 = float(market_features.sort_values("date").iloc[-1]["ret_20d"])
scored_stocks = score_stocks(stock_features, fundamentals, sector_map, vn_ret20)
sector_scores = score_sectors(scored_stocks)

top_sectors = sector_scores["sector"].head(3).astype(str).tolist()
commentary = make_commentary(market_view["prob_up_20d"], market_view["regime"], top_sectors)

k1, k2, k3, k4 = st.columns(4)
k1.metric("VNINDEX", f"{market_view['vnindex']:,.0f}")
k2.metric("Xác suất tăng 20 phiên", f"{market_view['prob_up_20d']:.1%}")
k3.metric("Regime", market_view["regime"])
k4.metric("Tín hiệu", traffic_light(market_view["prob_up_20d"] * 100))

st.markdown(f"**Nhận định tự động:** {commentary}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. VNINDEX Forecast",
    "2. Sector Ranking",
    "3. Stock Screener",
    "4. Macro Inputs",
    "5. Model Notes"
])

with tab1:
    st.subheader("VNINDEX Forecast & Regime")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.line(vnindex.sort_values("date"), x="date", y="close", title="VNINDEX price history")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.write("**Expected range 20 phiên:**")
        st.write(f"{market_view['expected_lower']:,.0f} – {market_view['expected_upper']:,.0f}")
        st.write("**Model validation:**")
        st.write(metrics)
        st.write("**Feature set:** trend, momentum, volatility, liquidity.")

with tab2:
    st.subheader("Xếp hạng ngành")
    st.dataframe(sector_scores, use_container_width=True, hide_index=True)
    fig = px.bar(sector_scores.head(10), x="sector", y="sector_score", title="Top sector scores")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Lọc cổ phiếu tiềm năng")
    filters = st.multiselect(
        "Lọc theo ngành",
        options=sorted(scored_stocks["sector"].dropna().unique().tolist()),
        default=[]
    )
    view = scored_stocks.copy()
    if filters:
        view = view[view["sector"].isin(filters)]
    display_cols = [
        "ticker", "sector", "stock_score", "signal", "close",
        "momentum_20d", "momentum_60d", "roe_pct", "eps_growth_pct",
        "pe", "pb", "foreign_flow_20d_bn_vnd"
    ]
    st.dataframe(view[display_cols].head(top_n), use_container_width=True, hide_index=True)
    fig = px.scatter(
        view.head(40),
        x="momentum_20d",
        y="stock_score",
        size="market_cap_bn_vnd",
        color="sector",
        hover_name="ticker",
        title="Momentum vs AI Stock Score"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Macro assumptions")
    st.dataframe(macro.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    st.warning("Các biến macro hiện là dữ liệu giả định/fallback. Có thể thay bằng dữ liệu SBV, GSO, FiinPro, Bloomberg hoặc file Excel nội bộ.")

with tab5:
    st.subheader("Model Notes")
    st.markdown("""
### Logic mô hình

Mô hình lấy cảm hứng từ cách tiếp cận AI-first finance:

1. Không cố dự báo một điểm VNINDEX duy nhất.
2. Dự báo xác suất thị trường tăng trong 20 phiên.
3. Chuyển xác suất thành regime: Risk-on, Neutral, Risk-off.
4. Dùng regime để điều chỉnh lựa chọn ngành/cổ phiếu.
5. Ưu tiên quản trị rủi ro, tránh overfitting và look-ahead bias.

### Cần nâng cấp thêm khi dùng thật

- Thay dữ liệu fundamentals giả định bằng báo cáo tài chính thật.
- Thêm dữ liệu khối ngoại, tự doanh, margin, breadth.
- Thêm backtesting theo từng giai đoạn thị trường.
- Thêm risk module: max drawdown, VaR, stop loss, position sizing.
- Thêm explainability: SHAP hoặc feature importance.
""")
