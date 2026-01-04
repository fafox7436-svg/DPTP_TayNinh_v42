import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import io

# Thử import ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự Báo Phụ Tải Điện", layout="wide")
st.title("⚡ HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN")

if not HAS_ARIMA:
    st.error("⚠️ Chưa cài thư viện 'statsmodels'. Vui lòng thêm vào requirements.txt để dùng ARIMA.")

# ==============================================================================
# 1. HÀM XỬ LÝ
# ==============================================================================
def them_yeu_to_mua(df):
    def check_tet(row):
        try:
            nam = int(row['Năm'])
            thang = int(row['Tháng'])
            # Bảng tra lịch Tết (Mở rộng)
            lich_tet = {
                2023: 1, 2024: 2, 2025: 1, 2026: 2, 2027: 2, 
                2028: 1, 2029: 2, 2030: 2
            }
            if nam in lich_tet and lich_tet[nam] == thang: return 1
            return 0
        except: return 0
            
    df['Co_Tet'] = df.apply(check_tet, axis=1)
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10, 11] else 0)
    return df

@st.cache_data
def train_and_predict(file_train, file_input):
    # --- ĐỌC DỮ LIỆU ---
    try: df_train = pd.read_excel(file_train, sheet_name='Bang tinh 5 tppt')
    except: df_train = pd.read_excel(file_train, sheet_name=0)
    df_input = pd.read_excel(file_input)

    df_train = them_yeu_to_mua(df_train)
    df_input = them_yeu_to_mua(df_input)

    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua']
    target = 'Tổng thương phẩm'
    
    # Dữ liệu Train sạch
    data_clean = df_train.dropna(subset=features + [target]).copy()
    X = data_clean[features]
    y = data_clean[target]

    # Chuẩn hóa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- HUẤN LUYỆN ---
    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 2. Neural Network (Cấu hình chuẩn)
    nn = MLPRegressor(hidden_layer_sizes=(10,15,10), activation='relu', solver='lbfgs', max_iter=5000, random_state=0) 
    nn.fit(X_scaled, y)

    # 3. ARIMA (Chuỗi thời gian)
    arima_fit = None
    if HAS_ARIMA:
        try:
            ts_data = data_clean.copy()
            # Tạo index ngày tháng chuẩn
            ts_data['Date'] = pd.to_datetime(dict(year=ts_data['Năm'], month=ts_data['Tháng'], day=1))
            ts_series = ts_data.groupby('Date')[target].sum().sort_index().asfreq('MS')
            
            order = (12, 1, 1) if len(ts_series) > 24 else (5, 1, 0)
            arima_model = ARIMA(ts_series, order=order)
            arima_fit = arima_model.fit()
        except: pass

    # --- DỰ BÁO (SỬA ĐỔI: CHẠY HẾT FILE INPUT) ---
    # Lấy toàn bộ dữ liệu trong file Input, sắp xếp theo thời gian
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    
    if len(df_pred) == 0: return None, "File Input rỗng."

    # Predict RF & NN (Chạy tốt cho mọi năm)
    df_pred['RF_Forecast'] = rf.predict(df_pred[features])
    df_pred['NN_Forecast'] = nn.predict(scaler.transform(df_pred[features]))
    
    # Predict ARIMA
    # Lưu ý: ARIMA luôn dự báo tiếp diễn từ điểm cuối của Train. 
    # Nếu bạn dự báo năm 2024 mà Train đã có 2024 -> ARIMA sẽ dự báo cho tương lai xa hơn (sai lệch).
    # Code này giữ logic dự báo tiếp diễn.
    if arima_fit:
        try:
            steps = len(df_pred)
            arima_vals = arima_fit.forecast(steps=steps)
            df_pred['ARIMA_Forecast'] = arima_vals.values
        except: df_pred['ARIMA_Forecast'] = 0
    else:
        df_pred['ARIMA_Forecast'] = 0

    # --- MERGE VỚI THỰC TẾ (SỬA ĐỔI: KHỚP THEO NĂM VÀ THÁNG) ---
    # Lấy cột thực tế từ file Train để so sánh (nếu có)
    df_actual = df_train[['Năm', 'Tháng', target]].copy()
    df_final = pd.merge(df_pred, df_actual, on=['Năm', 'Tháng'], how='left')
    df_final.rename(columns={target: 'Thuc_te'}, inplace=True)
    
    # Tạo cột Date để vẽ biểu đồ liên tục
    df_final['Date'] = pd.to_datetime(dict(year=df_final['Năm'], month=df_final['Tháng'], day=1))
    
    return df_final, None

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. File Train (Lịch sử)", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. File Input (Dự báo)", type=['xlsx', 'xls'])

if uploaded_train and uploaded_input:
    if st.button("🚀 CHẠY DỰ BÁO ĐA NĂM", type="primary"):
        with st.spinner('Đang xử lý dữ liệu nhiều năm...'):
            df_result, err = train_and_predict(uploaded_train, uploaded_input)
            
        if err: st.error(err)
        else:
            years_str = ", ".join(map(str, df_result['Năm'].unique()))
            st.success(f"Đã dự báo xong cho các năm: {years_str}")

            # --- BẢNG KẾT QUẢ ---
            st.subheader("📊 Bảng Số Liệu Chi Tiết")
            
            # Tính sai số
            df_result['Lệch NN (%)'] = np.where(df_result['Thuc_te'].notnull(), 
                                                 abs(df_result['Thuc_te'] - df_result['NN_Forecast'])/df_result['Thuc_te']*100, 
                                                 np.nan)
            
            # Hiển thị bảng
            cols = ['Năm', 'Tháng', 'Thuc_te', 'NN_Forecast', 'RF_Forecast', 'ARIMA_Forecast', 'Lệch NN (%)']
            st.dataframe(df_result[cols].style.format({
                'Thuc_te': '{:,.0f}', 'NN_Forecast': '{:,.0f}', 
                'RF_Forecast': '{:,.0f}', 'ARIMA_Forecast': '{:,.0f}',
                'Lệch NN (%)': '{:.2f}%', 'Năm': '{:.0f}'
            }).background_gradient(subset=['Lệch NN (%)'], cmap='RdYlGn_r'), use_container_width=True)

            # --- BIỂU ĐỒ LIÊN TỤC ---
            st.subheader("📈 Biểu Đồ Xu Hướng Theo Thời Gian")
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Vẽ đường dự báo
            ax.plot(df_result['Date'], df_result['NN_Forecast'], 's-', color='red', label='Neural Network', linewidth=2)
            ax.plot(df_result['Date'], df_result['RF_Forecast'], 'x--', color='blue', label='Random Forest', alpha=0.5)
            
            # Vẽ đường thực tế (nếu có)
            # Lọc bỏ những điểm không có thực tế để biểu đồ không bị đứt đoạn vô lý
            mask_actual = df_result['Thuc_te'].notnull()
            if mask_actual.any():
                ax.plot(df_result.loc[mask_actual, 'Date'], df_result.loc[mask_actual, 'Thuc_te'], 'o-', color='black', label='Thực Tế', linewidth=2.5)
            
            ax.set_title("Biểu đồ Dự báo Phụ tải (Chuỗi thời gian)")
            ax.set_ylabel("Sản lượng (kWh)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Format trục thời gian cho dễ nhìn
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
            
            # Tải về
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_result.drop(columns=['Date']).to_excel(writer, sheet_name='Ket_Qua', index=False)
            st.download_button("📥 Tải kết quả Excel", buffer.getvalue(), f"Ket_qua_du_bao.xlsx", "application/vnd.ms-excel")
