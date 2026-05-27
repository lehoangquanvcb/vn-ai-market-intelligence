import numpy as np
import pandas as pd

def safe_pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan).fillna(0)

def traffic_light(value, green=60, yellow=45):
    if value >= green:
        return "Xanh"
    if value >= yellow:
        return "Vàng"
    return "Đỏ"

def normalize_0_100(series: pd.Series) -> pd.Series:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    if s.max() == s.min():
        return pd.Series(50, index=s.index)
    return ((s - s.min()) / (s.max() - s.min()) * 100).fillna(50)

def make_commentary(market_prob, regime, top_sectors):
    if regime.startswith("Risk-on"):
        tone = "Thị trường đang có xác suất thuận lợi, có thể ưu tiên cổ phiếu dẫn dắt và ngành có sức mạnh tương đối tốt."
    elif regime == "Neutral":
        tone = "Thị trường chưa đủ xác nhận xu hướng, nên giữ kỷ luật tỷ trọng và ưu tiên cổ phiếu có nền tảng cơ bản tốt."
    else:
        tone = "Thị trường ở trạng thái rủi ro cao, nên giảm tỷ trọng cổ phiếu và ưu tiên phòng thủ."
    return f"Xác suất VNINDEX tăng 20 phiên tới khoảng {market_prob:.1%}. Regime hiện tại: {regime}. {tone} Nhóm ngành nổi bật: {', '.join(top_sectors[:3])}."
