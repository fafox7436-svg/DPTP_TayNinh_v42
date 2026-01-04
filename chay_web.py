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
st.set_page_config(page_title="Hệ Thống Dự Báo Phụ Tải", layout="wide")
st.title("HỆ THỐNG DỰ BÁO PHỤ TẢI")
st.markdown("---")

# ==============================================================================
# 1. MODULE XỬ LÝ NLP
# ==============================================================================
def xu_ly_du_lieu_dinh_tinh(api_key, text_input):
    if not api_key: return 0, "⚠️ Chưa nhập khóa API. Giá trị mặc định là 0."
    try:
        genai.configure(api_key=api_key)
        candidate_models = ["gemini-1.5-flash", "gemini-pro", "gemini-1.0-pro"]
        final_model = "gemini-pro"
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if "flash" in m.name: final_model = m.name; break
                    final_model = m.name
        except: pass

        model = genai.GenerativeModel(final_model)
        prompt = f"Đánh giá mức độ ảnh hưởng của thông tin sau đến nhu cầu tiêu thụ điện (-2 đến 2). Nội dung: '{text_input}'. Chỉ trả về 1 số nguyên."
        response = model.generate_content(prompt)
        
        import re
        match = re.search(r'-?\d+', response.text)
        if match: return int(match.group()), f"✅ Mức độ tác động: {match.group()}"
        return 0, "⚠️ Không xác định được mức độ."
    except Exception as e: return 0, f"❌ Lỗi xử lý: {str(e)}"

# ==============================================================================
# 2. HÀM TÍNH TOÁN (CẤU HÌNH 10-15-10 + RANDOM STATE 42 -> 749tr)
# ==============================================================================
def feature_engineering(df):
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
def chay_mo_phong(df_train_origin, df_input_origin, exogenous_params, scenario_label="Kịch bản"):
    df_train = df_train_origin.copy()
    df_input = df_input_origin.copy()

    def get_exogenous(row): return exogenous_params.get((int(row['Năm']), int(row['Tháng'])), 0)
    df_train['Bien_Ngoai_Sinh'] = df_train.apply(get_exogenous, axis=1)
    df_input['Bien_Ngoai_Sinh'] = df_input.apply(get_exogenous, axis=1)

    df_train = feature_engineering(df_train)
    df_input = feature_engineering(df_input)

    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_features = [f for f in features if f in df_train.columns and f in df_input.columns]
    target = 'Tổng thương phẩm'

    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 1. NEURAL NETWORK (CHUẨN 10-15-10 & RANDOM_STATE=42) ---
    nn = MLPRegressor(
        hidden_layer_sizes=(10, 15, 10), # Đúng cấu trúc bạn yêu cầu
        activation='relu', 
        solver='lbfgs', 
        max_iter=5000, 
        random_state=42 # Đổi từ 0 sang 42 để quay về kết quả cũ (749tr)
    )
    nn.fit(X_scaled, y)
    
    # --- 2. RANDOM FOREST ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # --- 3. XGBOOST ---
    xgb_model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, subsample=0.7, random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)

    # Dự báo
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    df_pred[valid_features] = df_pred[valid_features].fillna(0)
    
    suffix = "" if scenario_label == "Final" else f"_{scenario_label}"
    
    # Predict
    df_pred[f'NN{suffix}'] = nn.predict(scaler.transform(df_pred[valid_features]))
    df_pred[f'RF{suffix}'] = rf.predict(df_pred[valid_features]) 
    df_pred[f'XGB{suffix}'] = xgb_model.predict(df_pred[valid_features])
    
    return df_pred[['Năm', 'Tháng', f'NN{suffix}', f'RF{suffix}', f'XGB{suffix}']]

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    api_key = st.text_input("Nhập Khóa API", value="", type="password")

col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. Dữ liệu Lịch sử (Train)", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. Dữ liệu Dự báo (Input)", type=['xlsx', 'xls'])

st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# PHẦN TIN TỨC
st.subheader("📰 Phân Tích Thông Tin Đầu Vào")
c1, c2 = st.columns([2, 1])
with c1: text_data = st.text_area("Nội dung thông tin:", height=100)
with c2:
    if st.button("Phân Tích"):
        val, log = xu_ly_du_lieu_dinh_tinh(api_key, text_data)
        if "Lỗi" in log: st.warning(log)
        else: st.success(log)
        st.session_state.param_dict[(2025, 5)] = val
        st.session_state.param_dict[(2025, 6)] = val

if st.session_state.param_dict:
    st.info(f"Đã ghi nhận yếu tố tác động: {st.session_state.param_dict}")

st.write("---")

if uploaded_train and uploaded_input:
    if st.button("🚀 THỰC HIỆN DỰ BÁO", type="primary"):
        try:
            try: df_train_org = pd.read_excel(uploaded_train, sheet_name='Bang tinh 5 tppt')
            except: df_train_org = pd.read_excel(uploaded_train, sheet_name=0)
            df_input_org = pd.read_excel(uploaded_input)

            with st.spinner("Đang chạy so sánh 3 mô hình..."):
                # 1. Chạy Kịch bản Điều chỉnh
                df_final = chay_mo_phong(df_train_org, df_input_org, st.session_state.param_dict, "Final")
                
                # Ghép với Thực tế
                df_actual = df_train_org[['Năm', 'Tháng', 'Tổng thương phẩm']].copy()
                df_final = pd.merge(df_final, df_actual, on=['Năm', 'Tháng'], how='left')
                df_final.rename(columns={'Tổng thương phẩm': 'Thuc_te'}, inplace=True)
                df_final['ThoiGian'] = pd.to_datetime(dict(year=df_final['Năm'], month=df_final['Tháng'], day=1))

                # --- TÍNH TOÁN SAI SỐ (%) ---
                mask = df_final['Thuc_te'].notnull()
                df_final['Loi_NN(%)'] = np.where(mask, abs(df_final['Thuc_te'] - df_final['NN'])/df_final['Thuc_te']*100, np.nan)
                df_final['Loi_RF(%)'] = np.where(mask, abs(df_final['Thuc_te'] - df_final['RF'])/df_final['Thuc_te']*100, np.nan)
                df_final['Loi_XGB(%)'] = np.where(mask, abs(df_final['Thuc_te'] - df_final['XGB'])/df_final['Thuc_te']*100, np.nan)

                # --- TAB 1: BẢNG TỔNG HỢP ---
                st.subheader("📊 Bảng So Sánh Sai Số Các Phương Pháp")
                
                cols_display = {
                    'Tháng': 'Tháng',
                    'Thuc_te': 'Thực Tế',
                    'NN': 'Neural Net', 'Loi_NN(%)': 'Sai số NN (%)',
                    'RF': 'Random Forest', 'Loi_RF(%)': 'Sai số RF (%)',
                    'XGB': 'XGBoost', 'Loi_XGB(%)': 'Sai số XGB (%)'
                }
                
                df_show = df_final[['Năm'] + list(cols_display.keys())].rename(columns=cols_display)
                
                # --- LOGIC TÔ MÀU (<= 1.5%) ---
                def highlight_accuracy(val):
                    if isinstance(val, float) and val <= 1.5:
                        return 'background-color: #ccffcc; color: green; font-weight: bold' 
                    return ''

                st.dataframe(df_show.style.format({
                    'Thực Tế': '{:,.0f}',
                    'Neural Net': '{:,.0f}', 'Sai số NN (%)': '{:.2f}%',
                    'Random Forest': '{:,.0f}', 'Sai số RF (%)': '{:.2f}%',
                    'XGBoost': '{:,.0f}', 'Sai số XGB (%)': '{:.2f}%'
                }).applymap(highlight_accuracy, subset=['Sai số NN (%)', 'Sai số RF (%)', 'Sai số XGB (%)']), 
                use_container_width=True)

                # --- TAB 2: BIỂU ĐỒ ĐA ĐƯỜNG ---
                st.subheader("📈 Biểu Đồ So Sánh 3 Phương Pháp")
                fig, ax = plt.subplots(figsize=(14, 7))
                
                ax.plot(df_final['ThoiGian'], df_final['NN'], 's-', color='#d62728', label='Neural Network', linewidth=2, alpha=0.8)
                ax.plot(df_final['ThoiGian'], df_final['RF'], 'x--', color='#1f77b4', label='Random Forest', linewidth=1.5, alpha=0.7)
                ax.plot(df_final['ThoiGian'], df_final['XGB'], '^-.', color='#2ca02c', label='XGBoost', linewidth=2, alpha=0.9)

                if df_final['Thuc_te'].notnull().any():
                    ax.plot(df_final['ThoiGian'], df_final['Thuc_te'], 'o-', color='black', linewidth=3, label='Thực Tế', zorder=10)

                ax.set_title("So Sánh Độ Bám Sát Của 3 Mô Hình")
                ax.set_ylabel("Sản lượng điện (kWh)")
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
                
                # Nút tải về
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_show.to_excel(writer, index=False, sheet_name='Ket_qua')
                st.download_button("📥 Tải Báo Cáo Excel", buffer.getvalue(), "Ket_qua_Du_bao.xlsx")

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
