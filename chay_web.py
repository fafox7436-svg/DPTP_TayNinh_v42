import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import io

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự Báo Phụ Tải Điện - Long An Cũ", layout="wide")

st.title("⚡ HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN (LONG AN CŨ)")
st.markdown("""
Ứng dụng so sánh 3 mô hình: **Neural Network (Chuẩn)**, **Random Forest** và **ARIMA** kết hợp yếu tố mùa vụ để đưa ra kết quả dự báo chính xác nhất.
""")

# ==============================================================================
# 1. HÀM XỬ LÝ (LOGIC CHUẨN CỦA BẠN)
# ==============================================================================
def them_yeu_to_mua(df):
    # Xử lý Tết theo lịch Âm (Cập nhật logic nếu cần cho các năm sau)
    def check_tet(row):
        try:
            nam = int(row['Năm'])
            thang = int(row['Tháng'])
            # Bảng tra lịch Tết (Tháng dương lịch chứa mùng 1 Tết)
            lich_tet = {2023: 1, 2024: 2, 2025: 1, 2026: 2, 2027: 2}
            
            if nam in lich_tet and lich_tet[nam] == thang:
                return 1
            return 0
        except:
            return 0
            
    df['Co_Tet'] = df.apply(check_tet, axis=1)
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10, 11] else 0)
    return df

@st.cache_data
def train_and_predict(file_train, file_input):
    # --- ĐỌC DỮ LIỆU ---
    try:
        df_train = pd.read_excel(file_train, sheet_name='Bang tinh 5 tppt')
    except:
        df_train = pd.read_excel(file_train, sheet_name=0)
    
    df_input = pd.read_excel(file_input)

    # --- THÊM YẾU TỐ MÙA VỤ ---
    df_train = them_yeu_to_mua(df_train)
    df_input = them_yeu_to_mua(df_input)

    # --- CHUẨN BỊ DỮ LIỆU MACHINE LEARNING ---
    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua']
    target = 'Tổng thương phẩm'
    
    data_clean = df_train.dropna(subset=features + [target]).copy()
    X = data_clean[features]
    y = data_clean[target]

    # --- CHUẨN HÓA (CHO NEURAL NETWORK) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==========================================================================
    # HUẤN LUYỆN MÔ HÌNH
    # ==========================================================================
    
    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 2. Neural Network (Cấu hình chuẩn của bạn)
    nn = MLPRegressor(hidden_layer_sizes=(10,15,10), 
                      activation='relu', 
                      solver='lbfgs', 
                      max_iter=5000, 
                      random_state=0) 
    nn.fit(X_scaled, y)

    # 3. ARIMA (Mới bổ sung)
    # Cần tạo chuỗi thời gian liên tục (Time Series)
    try:
        ts_data = data_clean.copy()
        # Tạo cột Date từ Năm/Tháng để làm index
        ts_data['Date'] = pd.to_datetime(ts_data[['Năm', 'Tháng']].assign(DAY=1))
        ts_series = ts_data.set_index('Date')[target].sort_index()
        ts_series = ts_series.asfreq('MS') # Đặt tần suất là Monthly Start
        
        # Cấu hình ARIMA(p,d,q). Chọn (12,1,1) để bắt chu kỳ năm nếu dữ liệu đủ dài
        # Nếu dữ liệu ít, dùng (5,1,0) để an toàn
        if len(ts_series) > 24:
            order_arima = (12, 1, 1) # Mạnh về chu kỳ năm
        else:
            order_arima = (5, 1, 0) # Đơn giản hơn cho dữ liệu ngắn
            
        arima_model = ARIMA(ts_series, order=order_arima)
        arima_fit = arima_model.fit()
    except Exception as e:
        arima_fit = None # Nếu lỗi ARIMA thì bỏ qua
        st.warning(f"Không chạy được ARIMA do dữ liệu không liên tục: {e}")

    # ==========================================================================
    # DỰ BÁO
    # ==========================================================================
    
    # Lấy năm cần dự báo (năm lớn nhất trong file input)
    target_year = df_input['Năm'].max()
    df_pred = df_input[df_input['Năm'] == target_year].copy()
    
    # Sắp xếp theo tháng để khớp với ARIMA
    df_pred = df_pred.sort_values('Tháng') 
    
    if len(df_pred) == 0:
        return None, None, f"Không tìm thấy dữ liệu dự báo cho năm {target_year}"

    # Predict RF & NN
    df_pred['RF_Forecast'] = rf.predict(df_pred[features])
    df_pred['NN_Forecast'] = nn.predict(scaler.transform(df_pred[features]))
    
    # Predict ARIMA
    if arima_fit:
        # Dự báo n bước tiếp theo (số tháng cần dự báo)
        steps = len(df_pred)
        # Forecast trả về một Series, ta lấy values
        arima_forecast_values = arima_fit.forecast(steps=steps)
        # Gán vào dataframe (đảm bảo df_pred đã sort theo tháng)
        df_pred['ARIMA_Forecast'] = arima_forecast_values.values
    else:
        df_pred['ARIMA_Forecast'] = 0

    # So sánh thực tế
    df_actual = df_train[df_train['Năm'] == target_year][['Tháng', target]]
    df_final = pd.merge(df_pred, df_actual, on='Tháng', how='left')
    df_final.rename(columns={target: 'Thuc_te'}, inplace=True)
    
    return df_final, target_year, None

# ==============================================================================
# 2. GIAO DIỆN UPLOAD FILE
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. File Dữ Liệu Lịch Sử (Train)")
    uploaded_train = st.file_uploader("Chọn file Excel Train", type=['xlsx', 'xls'])

with col2:
    st.subheader("2. File Thông Số Dự Báo (Input)")
    uploaded_input = st.file_uploader("Chọn file Excel Input", type=['xlsx', 'xls'])

# ==============================================================================
# 3. HIỂN THỊ KẾT QUẢ
# ==============================================================================
if uploaded_train and uploaded_input:
    if st.button("🚀 CHẠY DỰ BÁO NGAY", type="primary"):
        with st.spinner('Đang chạy 3 mô hình (NN, RF, ARIMA)...'):
            df_result, year, error = train_and_predict(uploaded_train, uploaded_input)
            
        if error:
            st.error(error)
        else:
            st.success(f"Đã hoàn thành dự báo năm {year}!")
            
            # --- TÍNH TOÁN SAI SỐ ---
            # Chỉ tính trên các tháng đã có số liệu thực tế
            valid_data = df_result[df_result['Thuc_te'].notnull()]
            
            mape_nn = np.nan
            if len(valid_data) > 0:
                mape_nn = (abs(valid_data['Thuc_te'] - valid_data['NN_Forecast']) / valid_data['Thuc_te']).mean() * 100
                mape_rf = (abs(valid_data['Thuc_te'] - valid_data['RF_Forecast']) / valid_data['Thuc_te']).mean() * 100
                mape_arima = (abs(valid_data['Thuc_te'] - valid_data['ARIMA_Forecast']) / valid_data['Thuc_te']).mean() * 100
            
            # --- HIỂN THỊ METRIC ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Neural Network (Chuẩn)", f"{mape_nn:.2f}%" if pd.notnull(mape_nn) else "N/A")
            m2.metric("Random Forest", f"{mape_rf:.2f}%" if pd.notnull(mape_nn) else "N/A")
            m3.metric("ARIMA (Chuỗi thời gian)", f"{mape_arima:.2f}%" if pd.notnull(mape_nn) else "N/A")

            # --- BẢNG CHI TIẾT ---
            st.subheader("📊 Bảng Kết Quả Chi Tiết")
            display_df = df_result[['Tháng', 'Thuc_te', 'NN_Forecast', 'RF_Forecast', 'ARIMA_Forecast']].copy()
            
            # Tính độ lệch cho Neural Network (Mô hình chính)
            display_df['Lệch NN (%)'] = np.where(display_df['Thuc_te'].notnull(), 
                                                 abs(display_df['Thuc_te'] - display_df['NN_Forecast'])/display_df['Thuc_te']*100, 
                                                 np.nan)

            st.dataframe(display_df.style.format({
                'Thuc_te': '{:,.0f}', 
                'NN_Forecast': '{:,.0f}',
                'RF_Forecast': '{:,.0f}', 
                'ARIMA_Forecast': '{:,.0f}',
                'Lệch NN (%)': '{:.2f}%'
            }).background_gradient(subset=['Lệch NN (%)'], cmap='RdYlGn_r'), use_container_width=True)

            # --- BIỂU ĐỒ ---
            st.subheader("📈 Biểu Đồ So Sánh 3 Mô Hình")
            fig, ax = plt.subplots(figsize=(14, 7))
            
            if display_df['Thuc_te'].notnull().any():
                ax.plot(display_df['Tháng'], display_df['Thuc_te'], 'o-', label='Thực Tế', color='black', linewidth=3)
            
            ax.plot(display_df['Tháng'], display_df['NN_Forecast'], 's-', label='Neural Network (Chuẩn)', color='red', linewidth=2)
            ax.plot(display_df['Tháng'], display_df['RF_Forecast'], 'x--', label='Random Forest', color='blue', alpha=0.6)
            ax.plot(display_df['Tháng'], display_df['ARIMA_Forecast'], '^-.', label='ARIMA', color='green', alpha=0.6)
            
            ax.set_title(f"Dự Báo Phụ Tải Năm {year}")
            ax.set_ylabel("Sản lượng (kWh)")
            ax.set_xlabel("Tháng")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

            # --- TẢI VỀ ---
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_result.to_excel(writer, sheet_name='Ket_Qua_Du_Bao', index=False)
            
            st.download_button(
                label="📥 Tải file Excel kết quả",
                data=buffer.getvalue(),
                file_name=f"Ket_qua_du_bao_3_mo_hinh_{year}.xlsx",
                mime="application/vnd.ms-excel"
            )

else:
    st.info("Vui lòng upload 2 file Excel để bắt đầu.")
