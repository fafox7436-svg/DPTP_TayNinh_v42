import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import io

# Thử import ARIMA, nếu chưa cài thư viện sẽ báo lỗi rõ ràng
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự Báo Phụ Tải Điện", layout="wide")

st.title("⚡ HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN (LONG AN CŨ)")

# Kiểm tra thư viện ngay đầu trang
if not HAS_ARIMA:
    st.error("⚠️ LỖI: Chưa cài đặt thư viện 'statsmodels'. Vui lòng thêm 'statsmodels' vào file requirements.txt và Reboot App!")

st.markdown("""
Ứng dụng so sánh 3 mô hình: **Neural Network (Chuẩn)**, **Random Forest** và **ARIMA** (Chuỗi thời gian).
""")

# ==============================================================================
# 1. HÀM XỬ LÝ
# ==============================================================================
def them_yeu_to_mua(df):
    def check_tet(row):
        try:
            nam = int(row['Năm'])
            thang = int(row['Tháng'])
            lich_tet = {2023: 1, 2024: 2, 2025: 1, 2026: 2, 2027: 2, 2028: 1, 2029: 2, 2030: 2}
            if nam in lich_tet and lich_tet[nam] == thang: return 1
            return 0
        except: return 0
            
    df['Co_Tet'] = df.apply(check_tet, axis=1)
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10, 11] else 0)
    return df

@st.cache_data
def train_and_predict(file_train, file_input):
    # Đọc dữ liệu
    try: df_train = pd.read_excel(file_train, sheet_name='Bang tinh 5 tppt')
    except: df_train = pd.read_excel(file_train, sheet_name=0)
    df_input = pd.read_excel(file_input)

    df_train = them_yeu_to_mua(df_train)
    df_input = them_yeu_to_mua(df_input)

    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua']
    target = 'Tổng thương phẩm'
    
    data_clean = df_train.dropna(subset=features + [target]).copy()
    X = data_clean[features]
    y = data_clean[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 2. Neural Network (Chuẩn)
    nn = MLPRegressor(hidden_layer_sizes=(10,15,10), activation='relu', solver='lbfgs', max_iter=5000, random_state=0) 
    nn.fit(X_scaled, y)

    # 3. ARIMA (Có bắt lỗi chi tiết)
    arima_fit = None
    arima_error_msg = None
    if HAS_ARIMA:
        try:
            ts_data = data_clean.copy()
            ts_data['Date'] = pd.to_datetime(ts_data[['Năm', 'Tháng']].assign(DAY=1))
            ts_series = ts_data.set_index('Date')[target].sort_index().asfreq('MS')
            
            # Cấu hình tự động
            order = (12, 1, 1) if len(ts_series) > 24 else (5, 1, 0)
            arima_model = ARIMA(ts_series, order=order)
            arima_fit = arima_model.fit()
        except Exception as e:
            arima_error_msg = str(e)

    # DỰ BÁO
    target_year = df_input['Năm'].max()
    df_pred = df_input[df_input['Năm'] == target_year].copy().sort_values('Tháng')
    
    if len(df_pred) == 0: return None, None, f"Không có dữ liệu năm {target_year}"

    df_pred['RF_Forecast'] = rf.predict(df_pred[features])
    df_pred['NN_Forecast'] = nn.predict(scaler.transform(df_pred[features]))
    
    if arima_fit:
        try:
            # Dự báo tiếp theo n bước
            steps = len(df_pred)
            # Lưu ý: ARIMA dự báo tiếp theo chuỗi thời gian, cần đảm bảo tháng dự báo nối tiếp tháng cuối của train
            arima_vals = arima_fit.forecast(steps=steps)
            df_pred['ARIMA_Forecast'] = arima_vals.values
        except:
            df_pred['ARIMA_Forecast'] = 0
    else:
        df_pred['ARIMA_Forecast'] = 0

    df_actual = df_train[df_train['Năm'] == target_year][['Tháng', target]]
    df_final = pd.merge(df_pred, df_actual, on='Tháng', how='left')
    df_final.rename(columns={target: 'Thuc_te'}, inplace=True)
    
    return df_final, target_year, arima_error_msg

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. File Train (Lịch sử)", type=['xlsx'])
with col2: uploaded_input = st.file_uploader("2. File Input (Dự báo)", type=['xlsx'])

if uploaded_train and uploaded_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        with st.spinner('Đang chạy mô hình...'):
            df_result, year, arima_err = train_and_predict(uploaded_train, uploaded_input)
            
        if isinstance(df_result, str): # Lỗi text
            st.error(df_result)
        else:
            # Thông báo về ARIMA
            if arima_err:
                st.warning(f"⚠️ ARIMA không chạy được do lỗi dữ liệu: {arima_err}")
            elif not HAS_ARIMA:
                st.warning("⚠️ Không chạy được ARIMA do chưa cài thư viện.")
            else:
                st.success(f"Đã chạy thành công cả 3 mô hình cho năm {year}!")

            # Bảng kết quả
            st.subheader("📊 Bảng Kết Quả")
            cols = ['Tháng', 'Thuc_te', 'NN_Forecast', 'RF_Forecast', 'ARIMA_Forecast']
            st.dataframe(df_result[cols].style.format("{:,.0f}"), use_container_width=True)
            
            # Biểu đồ
            st.subheader("📈 Biểu Đồ")
            fig, ax = plt.subplots(figsize=(12, 6))
            if df_result['Thuc_te'].notnull().any():
                ax.plot(df_result['Tháng'], df_result['Thuc_te'], 'o-', color='black', label='Thực Tế')
            ax.plot(df_result['Tháng'], df_result['NN_Forecast'], 's-', color='red', label='Neural Network')
            ax.plot(df_result['Tháng'], df_result['RF_Forecast'], 'x--', color='blue', label='Random Forest')
            ax.plot(df_result['Tháng'], df_result['ARIMA_Forecast'], '^-.', color='green', label='ARIMA')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
