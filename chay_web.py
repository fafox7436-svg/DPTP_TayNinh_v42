import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
# Import thư viện Holt-Winters chuẩn
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự Báo Phụ Tải", layout="wide")
st.title("⚡ HỆ THỐNG DỰ BÁO PHỤ TẢI")
st.markdown("So sánh 3 phương pháp: **Neural Network**, **Random Forest** và **Holt-Winters** (Chuyên trị chuỗi thời gian & mùa vụ).")

# ==============================================================================
# 1. HÀM GEMINI
# ==============================================================================
def ask_gemini_to_rate_event(api_key, news_content):
    if not api_key: return None, "Chưa nhập API Key."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Đánh giá tác động tin tức đến phụ tải điện (-2 đến 2). Tin: '{news_content}'. Trả về 1 số nguyên."
        response = model.generate_content(prompt)
        return int(response.text.strip()), None
    except Exception as e: return 0, str(e)

# ==============================================================================
# 2. HÀM XỬ LÝ SỐ LIỆU
# ==============================================================================
def them_yeu_to_mua(df):
    def check_tet(row):
        try:
            nam, thang = int(row['Năm']), int(row['Tháng'])
            lich_tet = {2023: 1, 2024: 2, 2025: 1, 2026: 2, 2027: 2}
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

    # Xử lý sự kiện (Nếu có)
    use_event = False
    if manual_events_dict and any(v != 0 for v in manual_events_dict.values()):
        use_event = True
    
    def get_event(row): return manual_events_dict.get((int(row['Năm']), int(row['Tháng'])), 0)
    if use_event:
        df_train['Su_Kien'] = df_train.apply(get_event, axis=1)
        df_input['Su_Kien'] = df_input.apply(get_event, axis=1)
    
    df_train = them_yeu_to_mua(df_train)
    df_input = them_yeu_to_mua(df_input)

    # Features
    if use_event:
        features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Su_Kien']
    else:
        features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua']

    valid_features = [f for f in features if f in df_train.columns and f in df_input.columns]
    target = 'Tổng thương phẩm'

    # Dữ liệu Train Machine Learning
    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 1. RANDOM FOREST ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # --- 2. NEURAL NETWORK (Cấu hình chuẩn) ---
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=0)
    nn.fit(X_scaled, y)
    
    # --- 3. HOLT-WINTERS (Thay thế ARIMA) ---
    # Phương pháp chuyên dụng: Holt-Winters Exponential Smoothing
    hw_fit = None
    try:
        ts_data = data_clean.copy()
        ts_data['Date'] = pd.to_datetime(dict(year=ts_data['Năm'], month=ts_data['Tháng'], day=1))
        ts_data = ts_data.sort_values('Date')
        ts_series = ts_data.set_index('Date')[target].asfreq('MS')

        # Cấu hình mạnh mẽ nhất cho dữ liệu điện:
        # - seasonal='mul': Mùa vụ nhân (Biên độ dao động lớn theo sản lượng)
        # - trend='add': Xu hướng tăng
        # - damped_trend=True: Tắt dần xu hướng (Tránh tăng vọt vô lý)
        # - use_boxcox=True: Tự động chuẩn hóa số liệu cho "mượt"
        hw_model = ExponentialSmoothing(
            ts_series, 
            seasonal_periods=12, 
            trend='add', 
            seasonal='mul', 
            damped_trend=True, 
            use_boxcox=True
        )
        hw_fit = hw_model.fit(optimized=True, remove_bias=True)
    except: pass

    # --- DỰ BÁO ---
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    if len(df_pred) == 0: return None, "File Input rỗng."

    df_pred[valid_features] = df_pred[valid_features].fillna(0)

    # Predict ML
    df_pred['RF_Forecast'] = rf.predict(df_pred[valid_features])
    df_pred['NN_Forecast'] = nn.predict(scaler.transform(df_pred[valid_features]))
    
    # Predict Holt-Winters
    if hw_fit:
        try:
            steps = len(df_pred)
            df_pred['HW_Forecast'] = hw_fit.forecast(steps).values
        except: df_pred['HW_Forecast'] = 0
    else: df_pred['HW_Forecast'] = 0

    # Merge Kết quả
    df_actual = df_train[['Năm', 'Tháng', target]].copy()
    df_final = pd.merge(df_pred, df_actual, on=['Năm', 'Tháng'], how='left')
    df_final.rename(columns={target: 'Thuc_te'}, inplace=True)
    df_final['Date'] = pd.to_datetime(dict(year=df_final['Năm'], month=df_final['Tháng'], day=1))
    
    return df_final, None

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
with st.sidebar:
    st.header("Cấu hình")
    # Điền Key mặc định của bạn vào đây nếu muốn
    api_key = st.text_input("Gemini API Key", value="", type="password")

col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. File Dự Báo (Input)", type=['xlsx', 'xls'])

# Phần Tin tức
st.write("---")
if 'event_list' not in st.session_state: st.session_state.event_list = {}
c1, c2 = st.columns([2, 1])
with c1: news = st.text_area("Dán tin tức (Gemini phân tích):", height=80)
with c2:
    if st.button("Phân tích ngay"):
        s, e = ask_gemini_to_rate_event(api_key, news)
        if e: st.error(e)
        else: 
            st.session_state.event_list[(2025, 4)] = s
            st.session_state.event_list[(2025, 5)] = s
            st.success(f"Đánh giá tác động: {s}")

# CHẠY DỰ BÁO
st.write("---")
if uploaded_train and uploaded_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        df_result, err = train_and_predict(uploaded_train, uploaded_input, st.session_state.event_list)
        
        if err: st.error(err)
        else:
            # --- TÍNH TOÁN SAI SỐ CHI TIẾT ---
            mask = df_result['Thuc_te'].notnull()
            
            # Tính % Lệch
            df_result['Lệch NN (%)'] = np.where(mask, abs(df_result['Thuc_te'] - df_result['NN_Forecast'])/df_result['Thuc_te']*100, np.nan)
            df_result['Lệch RF (%)'] = np.where(mask, abs(df_result['Thuc_te'] - df_result['RF_Forecast'])/df_result['Thuc_te']*100, np.nan)
            df_result['Lệch HW (%)'] = np.where(mask, abs(df_result['Thuc_te'] - df_result['HW_Forecast'])/df_result['Thuc_te']*100, np.nan)

            # --- TÌM PHƯƠNG PHÁP TỐT NHẤT ---
            def tim_best(row):
                if pd.isna(row['Thuc_te']): return ""
                errors = {
                    'Neural Net': row['Lệch NN (%)'],
                    'Random Forest': row['Lệch RF (%)'],
                    'Holt-Winters': row['Lệch HW (%)'] if row['HW_Forecast'] > 0 else 999
                }
                return min(errors, key=errors.get)

            df_result['Tốt nhất'] = df_result.apply(tim_best, axis=1)

            # --- HIỂN THỊ BẢNG ---
            st.subheader("📊 Bảng Kết Quả & So Sánh")
            
            cols_show = ['Tháng', 'Thuc_te', 
                         'NN_Forecast', 'Lệch NN (%)', 
                         'HW_Forecast', 'Lệch HW (%)', 
                         'Tốt nhất']
            
            st.dataframe(df_result[cols_show].style.format({
                'Thuc_te': '{:,.0f}', 
                'NN_Forecast': '{:,.0f}', 'Lệch NN (%)': '{:.2f}%',
                'HW_Forecast': '{:,.0f}', 'Lệch HW (%)': '{:.2f}%'
            }).applymap(lambda x: 'background-color: #d4edda; color: green; font-weight: bold' if isinstance(x, str) and len(x)>0 else '', subset=['Tốt nhất']), 
            use_container_width=True)

            # --- BIỂU ĐỒ ---
            st.subheader("📈 Biểu Đồ So Sánh")
            fig, ax = plt.subplots(figsize=(14, 7))
            
            if mask.any():
                ax.plot(df_result.loc[mask, 'Date'], df_result.loc[mask, 'Thuc_te'], 'o-', color='black', linewidth=3, label='Thực Tế')
            
            ax.plot(df_result['Date'], df_result['NN_Forecast'], 's-', color='red', label='Neural Network', alpha=0.8)
            
            # Vẽ đường Holt-Winters (Màu tím cho khác biệt)
            if df_result['HW_Forecast'].max() > 0:
                ax.plot(df_result['Date'], df_result['HW_Forecast'], '^-.', color='purple', label='Holt-Winters (Trend + Season)', alpha=0.8, linewidth=2)
            
            ax.set_title("So Sánh: Neural Network vs Holt-Winters")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
            # Tải về
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_result.drop(columns=['Date']).to_excel(writer, index=False)
            st.download_button("📥 Tải Báo Cáo Excel", buffer.getvalue(), "Ket_qua_final.xlsx")
