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
st.set_page_config(page_title="Hệ Thống Dự Báo (Safe Mode)", layout="wide")
st.title("⚡ HỆ THỐNG DỰ BÁO: CUỘC ĐUA CÔNG NGHỆ")
st.markdown("So sánh hiệu suất: **Neural Network** vs **Random Forest** vs **XGBoost**.")

# ==============================================================================
# 1. HÀM GEMINI (CHẾ ĐỘ AN TOÀN - KHÔNG BAO GIỜ CRASH APP)
# ==============================================================================
def ask_gemini_to_rate_event(api_key, news_content):
    if not api_key: return 0, "Chưa nhập API Key. Mặc định là 0."
    
    try:
        genai.configure(api_key=api_key)
        
        # Tự động lấy model đầu tiên hỗ trợ 'generateContent'
        # Đây là cách chắc chắn nhất để không bao giờ sai tên model
        available_models = []
        target_model_name = "gemini-1.5-flash" # Ưu tiên dùng cái này nếu có
        
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        except: pass
        
        # Logic chọn model: Nếu tìm thấy flash thì dùng, không thì dùng cái đầu tiên tìm được
        final_model = target_model_name
        if available_models:
            # Kiểm tra xem flash có trong danh sách không
            found = False
            for m in available_models:
                if "flash" in m:
                    final_model = m
                    found = True
                    break
            if not found:
                final_model = available_models[0] # Lấy đại cái đầu tiên
        
        # Gọi Gemini
        model = genai.GenerativeModel(final_model)
        prompt = f"Đánh giá tác động tin tức đến phụ tải điện (-2 đến 2). Tin: '{news_content}'. Trả về duy nhất 1 số nguyên."
        response = model.generate_content(prompt)
        
        # Xử lý kết quả trả về
        text = response.text.strip()
        # Cố gắng tìm số trong chuỗi kết quả (đề phòng nó trả lời dài dòng)
        import re
        match = re.search(r'-?\d+', text)
        if match:
            return int(match.group()), f"Thành công (Model: {final_model})"
        else:
            return 0, "Gemini trả lời nhưng không tìm thấy số."
            
    except Exception as e:
        # NẾU LỖI: TRẢ VỀ 0 ĐỂ APP KHÔNG BỊ DỪNG
        return 0, f"Lỗi Gemini: {str(e)} -> Đã tự chuyển về 0 để chạy tiếp."

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
    try: df_train = pd.read_excel(file_train, sheet_name='Bang tinh 5 tppt')
    except: df_train = pd.read_excel(file_train, sheet_name=0)
    df_input = pd.read_excel(file_input)

    use_event = False
    if manual_events_dict and any(v != 0 for v in manual_events_dict.values()):
        use_event = True
    
    def get_event(row): return manual_events_dict.get((int(row['Năm']), int(row['Tháng'])), 0)
    if use_event:
        df_train['Su_Kien'] = df_train.apply(get_event, axis=1)
        df_input['Su_Kien'] = df_input.apply(get_event, axis=1)
    
    df_train = them_yeu_to_mua(df_train)
    df_input = them_yeu_to_mua(df_input)

    if use_event:
        features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Su_Kien']
    else:
        features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua']

    valid_features = [f for f in features if f in df_train.columns and f in df_input.columns]
    target = 'Tổng thương phẩm'

    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 2. Neural Network
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=0)
    nn.fit(X_scaled, y)
    
    # 3. XGBoost (Cấu hình chống Overfitting)
    xgb_model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, subsample=0.7, random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)

    # Dự báo
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    if len(df_pred) == 0: return None, "File Input rỗng."

    df_pred[valid_features] = df_pred[valid_features].fillna(0)

    df_pred['RF_Forecast'] = rf.predict(df_pred[valid_features])
    df_pred['NN_Forecast'] = nn.predict(scaler.transform(df_pred[valid_features]))
    df_pred['XGB_Forecast'] = xgb_model.predict(df_pred[valid_features])

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
    
    # TÍNH NĂNG SOI MODEL (Debug)
    if api_key and HAS_GEMINI:
        try:
            genai.configure(api_key=api_key)
            st.caption("Các Model khả dụng:")
            found_any = False
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    st.code(m.name)
                    found_any = True
            if not found_any:
                st.error("Key đúng nhưng không thấy model nào!")
        except:
            st.warning("Key lỗi hoặc chưa kích hoạt API.")

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
        with st.spinner("Đang gọi AI..."):
            s, msg = ask_gemini_to_rate_event(api_key, news)
        
        if "Lỗi" in msg:
            st.warning(msg) # Hiện cảnh báo nhưng vẫn gán s=0 để chạy tiếp
            st.info("Hệ thống đã tự động gán tác động = 0 để không gián đoạn.")
        else:
            st.success(msg)
            
        st.session_state.event_list[(2025, 4)] = s
        st.session_state.event_list[(2025, 5)] = s

# Nút chạy
st.write("---")
if uploaded_train and uploaded_input:
    if st.button("🚀 CHẠY SO SÁNH", type="primary"):
        df_result, err = train_and_predict(uploaded_train, uploaded_input, st.session_state.event_list)
        
        if err: st.error(err)
        else:
            mask = df_result['Thuc_te'].notnull()
            
            df_result['Sai số NN (%)'] = np.where(mask, abs(df_result['Thuc_te'] - df_result['NN_Forecast'])/df_result['Thuc_te']*100, np.nan)
            df_result['Sai số RF (%)'] = np.where(mask, abs(df_result['Thuc_te'] - df_result['RF_Forecast'])/df_result['Thuc_te']*100, np.nan)
            df_result['Sai số XGB (%)'] = np.where(mask, abs(df_result['Thuc_te'] - df_result['XGB_Forecast'])/df_result['Thuc_te']*100, np.nan)

            def select_best_model(row):
                if pd.isna(row['Thuc_te']): return ""
                errors = {}
                if row['NN_Forecast'] > 0: errors['Neural Net'] = row['Sai số NN (%)']
                if row['RF_Forecast'] > 0: errors['Random Forest'] = row['Sai số RF (%)']
                if row['XGB_Forecast'] > 0: errors['XGBoost'] = row['Sai số XGB (%)']
                if not errors: return ""
                return min(errors, key=errors.get)

            df_result['Mô hình tối ưu'] = df_result.apply(select_best_model, axis=1)

            st.subheader("📊 Bảng So Sánh Hiệu Suất")
            cols_show = ['Tháng', 'Thuc_te', 'NN_Forecast', 'Sai số NN (%)', 'RF_Forecast', 'Sai số RF (%)', 'XGB_Forecast', 'Sai số XGB (%)', 'Mô hình tối ưu']
            st.dataframe(df_result[cols_show].style.format("{:,.0f}", subset=['Thuc_te', 'NN_Forecast', 'RF_Forecast', 'XGB_Forecast']).format("{:.2f}%", subset=['Sai số NN (%)', 'Sai số RF (%)', 'Sai số XGB (%)']).applymap(lambda x: 'background-color: #d4edda; color: green; font-weight: bold' if isinstance(x, str) and len(x)>0 else '', subset=['Mô hình tối ưu']), use_container_width=True)

            st.subheader("📈 Biểu Đồ So Sánh")
            fig, ax = plt.subplots(figsize=(14, 7))
            if mask.any(): ax.plot(df_result.loc[mask, 'Date'], df_result.loc[mask, 'Thuc_te'], 'o-', color='black', linewidth=3, label='Thực Tế', zorder=10)
            ax.plot(df_result['Date'], df_result['NN_Forecast'], 's-', color='#d62728', label='Neural Network', alpha=0.8, linewidth=2)
            ax.plot(df_result['Date'], df_result['RF_Forecast'], 'x--', color='#1f77b4', label='Random Forest', alpha=0.6)
            ax.plot(df_result['Date'], df_result['XGB_Forecast'], '^-.', color='#2ca02c', label='XGBoost', alpha=0.9, linewidth=2)
            ax.set_title("So Sánh: Neural Network vs Random Forest vs XGBoost")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer: df_result.drop(columns=['Date']).to_excel(writer, index=False)
            st.download_button("📥 Tải Báo Cáo Excel", buffer.getvalue(), "Ket_qua_XGBoost.xlsx")
