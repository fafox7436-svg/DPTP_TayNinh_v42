import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import io

# Kiểm tra thư viện Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ Thống Dự Báo Thông Minh", layout="wide")
st.title("⚡ HỆ THỐNG DỰ BÁO")
st.markdown("So sánh hiệu suất: **Neural Network** vs **Random Forest** vs **XGBoost**.")

# ==============================================================================
# 1. HÀM GEMINI THÔNG MINH (TỰ ĐỘNG DÒ TÌM MODEL SỐNG)
# ==============================================================================
def ask_gemini_to_rate_event(api_key, news_content):
    if not api_key: return None, "Chưa nhập API Key."
    
    # Danh sách các tên model có thể có (Thử từ mới nhất đến cũ nhất)
    candidate_models = [
        "gemini-1.5-flash", 
        "gemini-1.5-pro", 
        "gemini-1.0-pro", 
        "gemini-pro",
        "models/gemini-1.5-flash",
        "models/gemini-pro"
    ]
    
    genai.configure(api_key=api_key)
    
    last_error = ""
    
    # Vòng lặp thử từng model
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            prompt = f"Đánh giá tác động tin tức đến phụ tải điện (-2 đến 2). Tin: '{news_content}'. Trả về 1 số nguyên."
            response = model.generate_content(prompt)
            # Nếu chạy đến đây là thành công, trả về kết quả ngay
            return int(response.text.strip()), None
        except Exception as e:
            # Nếu lỗi, lưu lại lỗi và thử model tiếp theo
            last_error = str(e)
            continue
            
    # Nếu thử hết danh sách mà vẫn lỗi
    return 0, f"Không tìm thấy model phù hợp. Lỗi cuối cùng: {last_error}"

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

    # Xử lý sự kiện
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

    # Data Clean
    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- MODEL 1: RANDOM FOREST ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # --- MODEL 2: NEURAL NETWORK ---
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=0)
    nn.fit(X_scaled, y)
    
    # --- MODEL 3: XGBOOST (Cấu hình "Anti-Overfitting") ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=50,       # Ít cây để tránh học vẹt
        learning_rate=0.1,     # Học chậm
        max_depth=2,           # Cây nông (chỉ học ý chính)
        subsample=0.7,         # Che 30% dữ liệu khi học
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X, y)

    # --- DỰ BÁO ---
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    if len(df_pred) == 0: return None, "File Input rỗng."

    df_pred[valid_features] = df_pred[valid_features].fillna(0)

    # Predict
    df_pred['RF_Forecast'] = rf.predict(df_pred[valid_features])
    df_pred['NN_Forecast'] = nn.predict(scaler.transform(df_pred[valid_features]))
    df_pred['XGB_Forecast'] = xgb_model.predict(df_pred[valid_features])

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
    api_key = st.text_input("Gemini API Key", value="", type="password")

col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. File Dự Báo (Input)", type=['xlsx', 'xls'])

st.write("---")
# Tin tức Gemini
if 'event_list' not in st.session_state: st.session_state.event_list = {}
c1, c2 = st.columns([2, 1])
with c1: news = st.text_area("Nhập nội dung tin tức:", height=80)
with c2:
    if st.button("Phân tích"):
        # Code mới sẽ tự động thử các model đến khi nào được thì thôi
        with st.spinner("Đang thử kết nối Gemini..."):
            s, e = ask_gemini_to_rate_event(api_key, news)
        
        if e: st.error(f"Lỗi: {e}")
        else: 
            st.session_state.event_list[(2025, 4)] = s
            st.session_state.event_list[(2025, 5)] = s
            st.success(f"Đã kết nối thành công! Đánh giá: {s}")

# Nút chạy
st.write("---")
if uploaded_train and uploaded_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        df_result, err = train_and_predict(uploaded_train, uploaded_input, st.session_state.event_list)
        
        if err: st.error(err)
        else:
            # --- TÍNH TOÁN SAI SỐ ---
            mask = df_result['Thuc_te'].notnull()
            
            # Tính % Lệch
            df_result['Sai số NN (%)'] = np.where(mask, abs(df_result['Thuc_te'] - df_result['NN_Forecast'])/df_result['Thuc_te']*100, np.nan)
            df_result['Sai số RF (%)'] = np.where(mask, abs(df_result['Thuc_te'] - df_result['RF_Forecast'])/df_result['Thuc_te']*100, np.nan)
            df_result['Sai số XGB (%)'] = np.where(mask, abs(df_result['Thuc_te'] - df_result['XGB_Forecast'])/df_result['Thuc_te']*100, np.nan)

            # Chọn mô hình tốt nhất
            def select_best_model(row):
                if pd.isna(row['Thuc_te']): return ""
                errors = {}
                if row['NN_Forecast'] > 0: errors['Neural Net'] = row['Sai số NN (%)']
                if row['RF_Forecast'] > 0: errors['Random Forest'] = row['Sai số RF (%)']
                if row['XGB_Forecast'] > 0: errors['XGBoost'] = row['Sai số XGB (%)']
                
                if not errors: return ""
                return min(errors, key=errors.get)

            df_result['Mô hình tối ưu'] = df_result.apply(select_best_model, axis=1)

            # --- HIỂN THỊ BẢNG ---
            st.subheader("📊 Bảng So Sánh Hiệu Suất")
            
            cols_show = ['Tháng', 'Thuc_te', 
                         'NN_Forecast', 'Sai số NN (%)', 
                         'RF_Forecast', 'Sai số RF (%)',
                         'XGB_Forecast', 'Sai số XGB (%)', 
                         'Mô hình tối ưu']
            
            st.dataframe(df_result[cols_show].style.format({
                'Thuc_te': '{:,.0f}', 
                'NN_Forecast': '{:,.0f}', 'Sai số NN (%)': '{:.2f}%',
                'RF_Forecast': '{:,.0f}', 'Sai số RF (%)': '{:.2f}%',
                'XGB_Forecast': '{:,.0f}', 'Sai số XGB (%)': '{:.2f}%'
            }).applymap(lambda x: 'background-color: #d4edda; color: green; font-weight: bold' if isinstance(x, str) and len(x)>0 else '', subset=['Mô hình tối ưu']), 
            use_container_width=True)

            # --- BIỂU ĐỒ ---
            st.subheader("📈 Biểu Đồ So Sánh")
            fig, ax = plt.subplots(figsize=(14, 7))
            
            if mask.any():
                ax.plot(df_result.loc[mask, 'Date'], df_result.loc[mask, 'Thuc_te'], 'o-', color='black', linewidth=3, label='Thực Tế', zorder=10)
            
            ax.plot(df_result['Date'], df_result['NN_Forecast'], 's-', color='#d62728', label='Neural Network', alpha=0.8, linewidth=2)
            ax.plot(df_result['Date'], df_result['RF_Forecast'], 'x--', color='#1f77b4', label='Random Forest', alpha=0.6)
            
            # XGBoost (Màu xanh lá đậm)
            ax.plot(df_result['Date'], df_result['XGB_Forecast'], '^-.', color='#2ca02c', label='XGBoost', alpha=0.9, linewidth=2)
            
            ax.set_title("So Sánh: Neural Network vs Random Forest vs XGBoost")
            ax.set_ylabel("Sản lượng (kWh)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
            # Tải về
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_result.drop(columns=['Date']).to_excel(writer, index=False)
            st.download_button("📥 Tải Báo Cáo Excel", buffer.getvalue(), "Ket_qua_XGBoost.xlsx")
