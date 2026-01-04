import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import io

# --- KIỂM TRA THƯ VIỆN BẮT BUỘC ---
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự Báo Phụ Tải (Bản Chuẩn Tích Hợp AI)", layout="wide")
st.title("⚡ HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN (BẢN CHUẨN + GEMINI)")
st.markdown("Phiên bản tích hợp: **Neural Network (Cấu hình chuẩn)** + **ARIMA** + **Trợ lý Gemini**")

# Cảnh báo nếu thiếu thư viện quan trọng
if not HAS_ARIMA:
    st.error("⚠️ CẢNH BÁO: Chưa cài `statsmodels`. Phương pháp ARIMA sẽ không chạy! Hãy thêm vào requirements.txt")
if not HAS_GEMINI:
    st.warning("⚠️ Chưa cài `google-generativeai`. Tính năng đọc báo Gemini sẽ không hoạt động.")

# ==============================================================================
# 1. HÀM GEMINI (TRỢ LÝ ĐỌC BÁO)
# ==============================================================================
def ask_gemini_to_rate_event(api_key, news_content):
    if not api_key: return None, "Thiếu API Key."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Đánh giá tác động tin tức đến phụ tải điện (-2: Giảm mạnh, -1: Giảm nhẹ, 0: Không, 1: Tăng nhẹ, 2: Tăng mạnh).
        Tin tức: "{news_content}"
        Chỉ trả về 1 con số nguyên duy nhất.
        """
        response = model.generate_content(prompt)
        return int(response.text.strip()), None
    except Exception as e: return 0, str(e)

# ==============================================================================
# 2. HÀM XỬ LÝ SỐ LIỆU (KHÔI PHỤC LOGIC CŨ)
# ==============================================================================
def them_yeu_to_mua(df):
    def check_tet(row):
        try:
            nam, thang = int(row['Năm']), int(row['Tháng'])
            lich_tet = {2023: 1, 2024: 2, 2025: 1, 2026: 2, 2027: 2, 2028: 1}
            return 1 if nam in lich_tet and lich_tet[nam] == thang else 0
        except: return 0
    df['Co_Tet'] = df.apply(check_tet, axis=1)
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10, 11] else 0)
    return df

@st.cache_data
def train_and_predict(file_train, file_input, manual_events_dict):
    # Đọc dữ liệu
    try: df_train = pd.read_excel(file_train, sheet_name='Bang tinh 5 tppt')
    except: df_train = pd.read_excel(file_train, sheet_name=0)
    df_input = pd.read_excel(file_input)

    # Gán sự kiện từ Gemini/Nhập tay (Mặc định là 0)
    def get_event(row):
        return manual_events_dict.get((int(row['Năm']), int(row['Tháng'])), 0)
    
    df_train['Su_Kien'] = df_train.apply(get_event, axis=1)
    df_input['Su_Kien'] = df_input.apply(get_event, axis=1)

    # Xử lý mùa vụ
    df_train = them_yeu_to_mua(df_train)
    df_input = them_yeu_to_mua(df_input)

    # Features (Cố định để đảm bảo ổn định)
    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Su_Kien']
    # Chỉ lấy các cột thực sự có trong file
    valid_features = [f for f in features if f in df_train.columns and f in df_input.columns]
    target = 'Tổng thương phẩm'

    # Chuẩn bị dữ liệu Train
    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- MODEL 1: RANDOM FOREST ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # --- MODEL 2: NEURAL NETWORK (KHÔI PHỤC CẤU HÌNH CHUẨN) ---
    # Cấu hình này đã được kiểm chứng cho kết quả tốt nhất (~745tr)
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=0)
    nn.fit(X_scaled, y)
    
    # --- MODEL 3: ARIMA ---
    arima_fit = None
    if HAS_ARIMA:
        try:
            ts_data = data_clean.copy()
            ts_data['Date'] = pd.to_datetime(dict(year=ts_data['Năm'], month=ts_data['Tháng'], day=1))
            ts_series = ts_data.groupby('Date')[target].sum().sort_index().asfreq('MS')
            order = (12, 1, 1) if len(ts_series) > 24 else (5, 1, 0)
            arima_model = ARIMA(ts_series, order=order)
            arima_fit = arima_model.fit()
        except: pass

    # --- DỰ BÁO ---
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    if len(df_pred) == 0: return None, "File Input rỗng."

    # Xử lý thiếu dữ liệu input bằng 0 cho an toàn
    df_pred[valid_features] = df_pred[valid_features].fillna(0)

    df_pred['RF_Forecast'] = rf.predict(df_pred[valid_features])
    df_pred['NN_Forecast'] = nn.predict(scaler.transform(df_pred[valid_features]))
    
    if arima_fit:
        try:
            steps = len(df_pred)
            arima_vals = arima_fit.forecast(steps=steps)
            df_pred['ARIMA_Forecast'] = arima_vals.values
        except: df_pred['ARIMA_Forecast'] = 0
    else: df_pred['ARIMA_Forecast'] = 0

    # Ghép với thực tế để so sánh
    df_actual = df_train[['Năm', 'Tháng', target]].copy()
    df_final = pd.merge(df_pred, df_actual, on=['Năm', 'Tháng'], how='left')
    df_final.rename(columns={target: 'Thuc_te'}, inplace=True)
    df_final['Date'] = pd.to_datetime(dict(year=df_final['Năm'], month=df_final['Tháng'], day=1))
    
    return df_final, None

# ==============================================================================
# GIAO DIỆN CHÍNH
# ==============================================================================
with st.sidebar:
    st.header("🤖 Cấu hình Gemini")
    api_key = st.text_input("Nhập API Key", type="password")

col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. File Dự Báo (Input)", type=['xlsx', 'xls'])

st.write("---")
st.subheader("📰 Cập Nhật Sự Kiện (AI)")

if 'event_list' not in st.session_state: st.session_state.event_list = {}

c1, c2 = st.columns([2, 1])
with c1: news_text = st.text_area("Dán nội dung tin tức:", height=80)
with c2:
    col_y, col_m = st.columns(2)
    sel_year = col_y.number_input("Năm", 2023, 2030, 2025)
    sel_months = col_m.multiselect("Tháng", range(1, 13), default=[4, 5])
    if st.button("Phân tích & Áp dụng"):
        score, err = ask_gemini_to_rate_event(api_key, news_text)
        if err: st.error(err)
        else:
            for m in sel_months: st.session_state.event_list[(sel_year, m)] = score
            st.success(f"Đánh giá: {score}")

# Chạy Dự Báo
st.write("---")
if uploaded_train and uploaded_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        df_result, err = train_and_predict(uploaded_train, uploaded_input, st.session_state.event_list)
        
        if err: st.error(err)
        else:
            # Bảng kết quả (Có so sánh sai số)
            st.subheader("📊 Bảng Kết Quả & So Sánh Sai Số")
            
            # Tính sai số (NN vs Thực tế)
            df_result['Lệch NN (%)'] = np.where(df_result['Thuc_te'].notnull(), 
                                                 abs(df_result['Thuc_te'] - df_result['NN_Forecast'])/df_result['Thuc_te']*100, 
                                                 np.nan)

            cols = ['Năm', 'Tháng', 'Thuc_te', 'NN_Forecast', 'RF_Forecast', 'ARIMA_Forecast', 'Lệch NN (%)']
            st.dataframe(df_result[cols].style.format({
                'Thuc_te': '{:,.0f}', 'NN_Forecast': '{:,.0f}', 
                'RF_Forecast': '{:,.0f}', 'ARIMA_Forecast': '{:,.0f}',
                'Lệch NN (%)': '{:.2f}%', 'Năm': '{:.0f}'
            }).background_gradient(subset=['Lệch NN (%)'], cmap='RdYlGn_r'), use_container_width=True)

            # Biểu đồ
            st.subheader("📈 Biểu Đồ")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_result['Date'], df_result['NN_Forecast'], 's-', color='red', label='Neural Network', linewidth=2)
            ax.plot(df_result['Date'], df_result['RF_Forecast'], 'x--', color='blue', label='Random Forest', alpha=0.5)
            ax.plot(df_result['Date'], df_result['ARIMA_Forecast'], '^-.', color='green', label='ARIMA', alpha=0.7)
            
            mask_actual = df_result['Thuc_te'].notnull()
            if mask_actual.any():
                ax.plot(df_result.loc[mask_actual, 'Date'], df_result.loc[mask_actual, 'Thuc_te'], 'o-', color='black', label='Thực Tế', linewidth=2)
            
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            # Tải về
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_result.drop(columns=['Date']).to_excel(writer, sheet_name='Ket_Qua', index=False)
            st.download_button("📥 Tải Excel", buffer.getvalue(), "Ket_qua.xlsx")
