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
# 1. MODULE XỬ LÝ NLP (AI)
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
        # Prompt kỹ: Yêu cầu đánh giá mức độ từ -3 đến 3
        prompt = f"Đánh giá tác động của tin tức sau đến phụ tải điện (Số nguyên từ -3 đến 3). -3 là giảm rất mạnh, 0 là không đổi, 3 là tăng rất mạnh. Tin: '{text_input}'. Chỉ trả về 1 số nguyên."
        response = model.generate_content(prompt)
        
        import re
        match = re.search(r'-?\d+', response.text)
        if match: return int(match.group()), f"✅ Điểm tác động Gemini: {match.group()}"
        return 0, "⚠️ Không xác định được mức độ."
    except Exception as e: return 0, f"❌ Lỗi xử lý: {str(e)}"

# ==============================================================================
# 2. HÀM TÍNH TOÁN (DỰ BÁO CHUẨN + ĐIỀU CHỈNH %)
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
def chay_mo_phong(df_train_origin, df_input_origin, exogenous_params, user_seed, sensitivity):
    """
    sensitivity: Độ nhạy (ví dụ 0.012 tức là 1.2%)
    """
    df_train = df_train_origin.copy()
    df_input = df_input_origin.copy()

    # Feature Engineering
    df_train = feature_engineering(df_train)
    df_input = feature_engineering(df_input)

    # Train mô hình trên dữ liệu sạch (Baseline)
    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua']
    valid_features = [f for f in features if f in df_train.columns and f in df_input.columns]
    target = 'Tổng thương phẩm'

    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. NEURAL NETWORK
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=user_seed)
    nn.fit(X_scaled, y)
    
    # 2. RANDOM FOREST
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 3. XGBOOST
    xgb_model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, subsample=0.7, random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)

    # --- DỰ BÁO & ĐIỀU CHỈNH ---
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    df_pred[valid_features] = df_pred[valid_features].fillna(0)
    
    # Bước 1: Dự báo nền
    base_nn = nn.predict(scaler.transform(df_pred[valid_features]))
    base_rf = rf.predict(df_pred[valid_features])
    base_xgb = xgb_model.predict(df_pred[valid_features])
    
    # Bước 2: Tính hệ số điều chỉnh (Sensitivity)
    def get_adj_factor(row):
        score = exogenous_params.get((int(row['Năm']), int(row['Tháng'])), 0)
        return 1 + (score * sensitivity)

    df_pred['Adj_Factor'] = df_pred.apply(get_adj_factor, axis=1)

    # Bước 3: Áp dụng
    df_pred['NN'] = base_nn * df_pred['Adj_Factor']
    df_pred['RF'] = base_rf * df_pred['Adj_Factor']
    df_pred['XGB'] = base_xgb * df_pred['Adj_Factor']
    
    return df_pred[['Năm', 'Tháng', 'NN', 'RF', 'XGB', 'Adj_Factor']]

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình Hệ Thống")
    api_key = st.text_input("Nhập Khóa API (Google GenAI)", value="", type="password")
    
    st.markdown("---")
    st.caption("Thông số kỹ thuật:")
    selected_seed = st.number_input("Random Seed (Hạt giống)", value=42, step=1)
    
    # --- CẬP NHẬT: ĐỘ NHẠY MẶC ĐỊNH 1.2% ---
    st.markdown("---")
    st.caption("🎛️ Điều chỉnh độ nhạy Gemini:")
    # value=1.2 (Mặc định), step=0.1 (Chỉnh nhuyễn hơn)
    sensitivity_pct = st.slider("Mỗi 1 điểm Gemini thay đổi bao nhiêu % sản lượng?", 0.1, 5.0, 1.2, 0.1)
    sensitivity = sensitivity_pct / 100.0

col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. Dữ liệu Lịch sử (Train)", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. Dữ liệu Dự báo (Input)", type=['xlsx', 'xls'])

st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# PHẦN 1: PHÂN TÍCH TIN TỨC
st.subheader("📰 Phân Tích Thông Tin & Kịch Bản")
c1, c2 = st.columns([2, 1])
with c1: text_data = st.text_area("Nội dung thông tin / Sự kiện:", height=100, placeholder="Ví dụ: Kinh tế suy giảm nhẹ, cắt giảm sản xuất...")
with c2:
    if st.button("Phân Tích AI"):
        with st.spinner("AI đang đọc hiểu..."):
            val, log = xu_ly_du_lieu_dinh_tinh(api_key, text_data)
        if "Lỗi" in log: st.warning(log)
        else: st.success(log)
        st.session_state.param_dict[(2025, 5)] = val
        st.session_state.param_dict[(2025, 6)] = val

if st.session_state.param_dict:
    st.info(f"⚡ Đang áp dụng kịch bản: {st.session_state.param_dict} | Độ nhạy: {sensitivity_pct}% mỗi điểm")

st.write("---")

# PHẦN 2: CHẠY DỰ BÁO
if uploaded_train and uploaded_input:
    if st.button("🚀 THỰC HIỆN DỰ BÁO", type="primary"):
        try:
            try: df_train_org = pd.read_excel(uploaded_train, sheet_name='Bang tinh 5 tppt')
            except: df_train_org = pd.read_excel(uploaded_train, sheet_name=0)
            df_input_org = pd.read_excel(uploaded_input)

            with st.spinner(f"Đang chạy mô hình (Seed={selected_seed})..."):
                # Gọi hàm tính toán
                df_final = chay_mo_phong(df_train_org, df_input_org, st.session_state.param_dict, selected_seed, sensitivity)
                
                # Ghép dữ liệu thực tế
                df_actual = df_train_org[['Năm', 'Tháng', 'Tổng thương phẩm']].copy()
                df_final = pd.merge(df_final, df_actual, on=['Năm', 'Tháng'], how='left')
                df_final.rename(columns={'Tổng thương phẩm': 'Thuc_te'}, inplace=True)
                df_final['ThoiGian'] = pd.to_datetime(dict(year=df_final['Năm'], month=df_final['Tháng'], day=1))

                # Tính sai số
                mask = df_final['Thuc_te'].notnull()
                df_final['Loi_NN(%)'] = np.where(mask, abs(df_final['Thuc_te'] - df_final['NN'])/df_final['Thuc_te']*100, np.nan)
                df_final['Loi_RF(%)'] = np.where(mask, abs(df_final['Thuc_te'] - df_final['RF'])/df_final['Thuc_te']*100, np.nan)
                df_final['Loi_XGB(%)'] = np.where(mask, abs(df_final['Thuc_te'] - df_final['XGB'])/df_final['Thuc_te']*100, np.nan)

                # HIỂN THỊ
                st.subheader("📊 Bảng So Sánh Sai Số (Đã Điều Chỉnh Theo Kịch Bản)")
                
                # Cột Hệ số điều chỉnh
                df_final['Hệ số ĐC'] = df_final['Adj_Factor'].apply(lambda x: f"{x:.3f}x") # Hiện 3 số lẻ cho chi tiết (vd: 0.988x)
                
                cols_display = {
                    'Tháng': 'Tháng',
                    'Thuc_te': 'Thực Tế',
                    'Hệ số ĐC': 'Hệ số Gemini',
                    'NN': 'Neural Net', 'Loi_NN(%)': 'Sai số NN (%)',
                    'RF': 'Random Forest', 'Loi_RF(%)': 'Sai số RF (%)',
                    'XGB': 'XGBoost', 'Loi_XGB(%)': 'Sai số XGB (%)'
                }
                
                df_show = df_final[['Năm'] + list(cols_display.keys())].rename(columns=cols_display)
                
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

                st.subheader("📈 Biểu Đồ Dự Báo")
                fig, ax = plt.subplots(figsize=(14, 7))
                
                ax.plot(df_final['ThoiGian'], df_final['NN'], 's-', color='#d62728', label='Neural Network', linewidth=2, alpha=0.8)
                ax.plot(df_final['ThoiGian'], df_final['RF'], 'x--', color='#1f77b4', label='Random Forest', linewidth=1.5, alpha=0.7)
                ax.plot(df_final['ThoiGian'], df_final['XGB'], '^-.', color='#2ca02c', label='XGBoost', linewidth=2, alpha=0.9)

                if df_final['Thuc_te'].notnull().any():
                    ax.plot(df_final['ThoiGian'], df_final['Thuc_te'], 'o-', color='black', linewidth=3, label='Thực Tế', zorder=10)

                ax.set_title(f"Dự Báo Với Kịch Bản Điều Chỉnh (Độ nhạy {sensitivity_pct}%/điểm)")
                ax.set_ylabel("Sản lượng điện (kWh)")
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_show.to_excel(writer, index=False, sheet_name='Ket_qua_Dieu_chinh')
                st.download_button("📥 Tải Báo Cáo Excel", buffer.getvalue(), "Ket_qua_Forecast.xlsx")

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
