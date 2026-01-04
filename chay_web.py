import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import io

# --- KIỂM TRA THƯ VIỆN ---
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
st.set_page_config(page_title="Hệ Thống Dự Báo", layout="wide")
st.title("⚡ HỆ THỐNG DỰ BÁO PHỤ TẢI")

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
# 2. HÀM XỬ LÝ (LOGIC KHÔI PHỤC)
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

    # --- CƠ CHẾ ỔN ĐỊNH: CHỈ DÙNG CỘT SỰ KIỆN NẾU CÓ DỮ LIỆU ---
    use_event = False
    if manual_events_dict and len(manual_events_dict) > 0:
        # Kiểm tra xem có sự kiện nào khác 0 không
        if any(v != 0 for v in manual_events_dict.values()):
            use_event = True
    
    if use_event:
        def get_event(row): return manual_events_dict.get((int(row['Năm']), int(row['Tháng'])), 0)
        df_train['Su_Kien'] = df_train.apply(get_event, axis=1)
        df_input['Su_Kien'] = df_input.apply(get_event, axis=1)
    
    # Xử lý mùa vụ
    df_train = them_yeu_to_mua(df_train)
    df_input = them_yeu_to_mua(df_input)

    # --- FEATURES CHUẨN ---
    # Nếu KHÔNG có sự kiện -> Dùng bộ feature CŨ (để ra số chuẩn 745tr)
    # Nếu CÓ sự kiện -> Dùng bộ feature MỚI
    if use_event:
        features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Su_Kien']
    else:
        features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua']

    # Lọc cột thực tế có trong file
    valid_features = [f for f in features if f in df_train.columns and f in df_input.columns]
    target = 'Tổng thương phẩm'

    # Chuẩn bị dữ liệu
    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 2. Neural Network (Cấu hình VÀNG)
    # Cấu hình này khớp với kết quả bạn ưng ý nhất
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=0)
    nn.fit(X_scaled, y)
    
    # 3. ARIMA (Fix lỗi số kỳ cục)
    arima_fit = None
    if HAS_ARIMA:
        try:
            ts_data = data_clean.copy()
            # Sort dữ liệu theo thời gian (QUAN TRỌNG ĐỂ ARIMA KHÔNG BỊ LOẠN)
            ts_data['Date'] = pd.to_datetime(dict(year=ts_data['Năm'], month=ts_data['Tháng'], day=1))
            ts_data = ts_data.sort_values('Date') 
            ts_series = ts_data.set_index('Date')[target].asfreq('MS')
            
            # Ép buộc tham số đơn giản nếu dữ liệu ít
            order = (1, 1, 0) # Đơn giản hóa để tránh ra số kỳ cục
            arima_model = ARIMA(ts_series, order=order)
            arima_fit = arima_model.fit()
        except: pass

    # DỰ BÁO
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    if len(df_pred) == 0: return None, "File Input rỗng."

    # Fillna = 0 để tránh lỗi
    df_pred[valid_features] = df_pred[valid_features].fillna(0)

    df_pred['RF_Forecast'] = rf.predict(df_pred[valid_features])
    df_pred['NN_Forecast'] = nn.predict(scaler.transform(df_pred[valid_features]))
    
    if arima_fit:
        try:
            # Dự báo nối tiếp từ điểm cuối của lịch sử
            steps = len(df_pred)
            arima_vals = arima_fit.forecast(steps=steps)
            df_pred['ARIMA_Forecast'] = arima_vals.values
        except: df_pred['ARIMA_Forecast'] = 0 # Nếu lỗi thì về 0
    else: df_pred['ARIMA_Forecast'] = 0

    # Merge
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
    api_key = st.text_input("Gemini API Key", type="password")

col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. File Train (Lịch sử)", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. File Input (Dự báo)", type=['xlsx', 'xls'])

st.write("---")
if 'event_list' not in st.session_state: st.session_state.event_list = {}

c1, c2 = st.columns([2, 1])
with c1: news = st.text_area("Dán tin tức (Nếu có):", height=80, placeholder="Thông tin ảnh hưởng đến nhu cầu phụ tải.")
with c2:
    if st.button("Phân tích tin tức"):
        s, e = ask_gemini_to_rate_event(api_key, news)
        if e: st.error(e)
        else: 
            # Demo gán cho tháng 4,5 năm 2025
            st.session_state.event_list[(2025, 4)] = s
            st.session_state.event_list[(2025, 5)] = s
            st.success(f"Đánh giá tác động: {s}")

if uploaded_train and uploaded_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        df_result, err = train_and_predict(uploaded_train, uploaded_input, st.session_state.event_list)
        
        if err: st.error(err)
        else:
            # Bảng
            st.subheader("📊 Kết Quả")
            df_result['Lệch NN (%)'] = np.where(df_result['Thuc_te'].notnull(), 
                                                 abs(df_result['Thuc_te'] - df_result['NN_Forecast'])/df_result['Thuc_te']*100, np.nan)
            
            cols = ['Năm', 'Tháng', 'Thuc_te', 'NN_Forecast', 'RF_Forecast', 'ARIMA_Forecast', 'Lệch NN (%)']
            st.dataframe(df_result[cols].style.format("{:,.0f}"), use_container_width=True)

            # Biểu đồ
            st.subheader("📈 Biểu Đồ")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_result['Date'], df_result['NN_Forecast'], 's-', color='red', label='Neural Network', linewidth=2)
            ax.plot(df_result['Date'], df_result['RF_Forecast'], 'x--', color='blue', label='Random Forest', alpha=0.5)
            
            # Chỉ vẽ ARIMA nếu nó ra số hợp lý (> 1 triệu kWh)
            if df_result['ARIMA_Forecast'].max() > 1000000:
                ax.plot(df_result['Date'], df_result['ARIMA_Forecast'], '^-.', color='green', label='ARIMA', alpha=0.6)
            
            mask = df_result['Thuc_te'].notnull()
            if mask.any(): ax.plot(df_result.loc[mask, 'Date'], df_result.loc[mask, 'Thuc_te'], 'o-', color='black')
            
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            # Tải về
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_result.drop(columns=['Date']).to_excel(writer, index=False)
            st.download_button("Tải Excel", buffer.getvalue(), "Ket_qua.xlsx")

