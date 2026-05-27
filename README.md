# VN AI-First Market Intelligence Model

Mô hình dự báo VNINDEX và lọc ngành/cổ phiếu theo tinh thần **AI-first finance**:
dữ liệu → features → machine learning → regime detection → sector ranking → stock scoring → risk control.

## Chức năng chính

- Dự báo xác suất VNINDEX tăng trong 20 phiên tới
- Phân loại thị trường: Risk-on / Neutral / Risk-off
- Xếp hạng ngành theo relative strength, momentum, valuation và dòng tiền
- Lọc cổ phiếu tiềm năng theo AI Stock Score
- Có chế độ fallback dữ liệu giả định hợp lý nếu không lấy được dữ liệu live

## Cài đặt local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy lên Streamlit Cloud

1. Upload toàn bộ package lên GitHub
2. Vào https://streamlit.io/cloud
3. Chọn repo GitHub
4. Main file: `app.py`
5. Python version khuyến nghị: 3.11
6. Deploy

## Dữ liệu

Mô hình ưu tiên lấy dữ liệu từ `vnstock`. Nếu lỗi kết nối hoặc thiếu API, hệ thống tự tạo dữ liệu mẫu hợp lý để dashboard vẫn chạy được.

## Cấu trúc

```text
app.py
requirements.txt
src/
  data_loader.py
  features.py
  models.py
  scoring.py
  utils.py
data/
  macro_assumptions.csv
  sector_mapping.csv
.streamlit/
  config.toml
```

## Lưu ý

Đây là mô hình phân tích/ra quyết định đầu tư, không phải khuyến nghị mua bán chính thức.
Cần backtest kỹ trước khi sử dụng bằng tiền thật.
