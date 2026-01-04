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

# --- CẤU HÌNH TRANG (Dùng thuật ngữ chuyên ngành) ---
st.set_page_config(page_title="Load Forecasting System", layout="wide")
st.title("HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN")
st.markdown("Module: **Phân tích Kịch bản & Đánh giá Hiệu suất Mô hình (Scenario Analysis & Model Evaluation)**")

# ==============================================================================
# 1. MODULE XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)
# ==============================================================================
def process_qualitative_data(api_key, text_input):
    """
    Chuyển đổi thông tin định tính (văn bản) thành tham số định lượng (score).
    """
    if not api_key: return 0, "Chưa có API Key. Tham số mặc định = 0."
    try:
        genai.configure(api_key=api_key)
        # Tự động chọn model khả dụng
        candidate_models = ["gemini-1.5-flash", "gemini-pro", "gemini-1.0-pro"]
        final_model = "gemini-pro"
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if "flash" in m.name: final_model = m.name; break
                    final_model = m.name
        except: pass

        model = genai.GenerativeModel(final_model)
        # Prompt kỹ thuật: Yêu cầu lượng hóa mức độ ảnh hưởng
        prompt = f"Lượng hóa tác động của thông tin sau đến nhu cầu phụ tải điện trên thang đo số nguyên từ -2 đến 2. Nội dung: '{text_input}'. Chỉ trả về giá trị số."
        response = model.generate_content(prompt)
        
        import re
        match = re.search(r'-?\d+', response.text)
        if match: return int(match.group()), f"Mã hóa thành công. Giá trị: {match.group()}"
        return 0, "Không trích xuất được giá trị số."
    except Exception as e: return 0, f"Lỗi xử lý: {str(e)}"

# ==============================================================================
# 2. HÀM TÍNH TOÁN & DỰ BÁO (CORE)
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
def run_simulation(df_train_origin, df_input_origin, exogenous_params, scenario_label="Base Case"):
    """
    Chạy mô phỏng dự báo cho một kịch bản cụ thể.
    """
    df_train = df_train_origin.copy()
    df_input = df_input_origin.copy()

    # Gán biến ngoại sinh (Exogenous Variable) từ tham số đầu vào
    def get_exogenous(row): return exogenous_params.get((int(row['Năm']), int(row['Tháng'])), 0)
    
    df_train['Bien_Ngoai_Sinh'] = df_train.apply(get_exogenous, axis=1)
    df_input['Bien_Ngoai_Sinh'] = df_input.apply(get_exogenous, axis=1)

    # Feature Engineering
    df_train = feature_engineering(df_train)
    df_input = feature_engineering(df_input)

    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_features = [f for f in features if f in df_train.columns and f in df_input.columns]
    target = 'Tổng thương phẩm'

    # Prepare Training Data
    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model 1: MLP Regressor (Neural Network)
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=0)
    nn.fit(X_scaled, y)
    
    # Model 2: XGBoost Regressor (Gradient Boosting)
    xgb_model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, subsample=0.7, random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)

    # Forecasting
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    df_pred[valid_features] = df_pred[valid_features].fillna(0)
    
    suffix = "" if scenario_label == "Final" else f"_{scenario_label}"
    
    df_pred[f'NN{suffix}'] = nn.predict(scaler.transform(df_pred[valid_features]))
    df_pred[f'XGB{suffix}'] = xgb_model.predict(df_pred[valid_features])
    
    return df_pred[['Năm', 'Tháng', f'NN{suffix}', f'XGB{suffix}']]

# ==============================================================================
# GIAO DIỆN (UI)
# ==============================================================================
with st.sidebar:
    st.header("Cấu hình tham số")
    api_key = st.text_input("API Key (Google GenAI)", value="", type="password")

col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. Dữ liệu Huấn luyện (Train Set)", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. Dữ liệu Đầu vào (Input Set)", type=['xlsx', 'xls'])

st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# PHẦN NHẬP LIỆU ĐỊNH TÍNH
st.subheader("1. Tham số hóa thông tin định tính")
c1, c2 = st.columns([2, 1])
with c1: 
    text_data = st.text_area("Thông tin đầu vào (Văn bản):", height=100, placeholder="Nhập các yếu tố thời tiết, kinh tế xã hội ảnh hưởng đến phụ tải...")
with c2:
    if st.button("Xử lý & Mã hóa"):
        with st.spinner("Đang xử lý..."):
            val, log = process_qualitative_data(api_key, text_data)
        if "Lỗi" in log: st.warning(log)
        else: st.success(log)
        
        # Gán tham số cho kỳ dự báo (Demo)
        st.session_state.param_dict[(2025, 5)] = val
        st.session_state.param_dict[(2025, 6)] = val

if st.session_state.param_dict:
    st.info(f"Ma trận tham số ngoại sinh (Exogenous Matrix): {st.session_state.param_dict}")
else:
    st.caption("Chưa có tham số điều chỉnh. Hệ thống chạy ở chế độ Kịch bản Cơ sở (Base Case).")

st.write("---")
if uploaded_train and uploaded_input:
    if st.button("Thực hiện Dự báo & So sánh Kịch bản", type="primary"):
        try: df_train_org = pd.read_excel(uploaded_train, sheet_name='Bang tinh 5 tppt')
        except: df_train_org = pd.read_excel(uploaded_train, sheet_name=0)
        df_input_org = pd.read_excel(uploaded_input)

        with st.spinner("Đang tính toán..."):
            # 1. Chạy Kịch bản Cơ sở (Base Case) - Không có biến ngoại sinh
            df_base = run_simulation(df_train_org, df_input_org, {}, "Base")
            
            # 2. Chạy Kịch bản Điều chỉnh (Adjusted Case) - Có biến ngoại sinh
            df_adj = run_simulation(df_train_org, df_input_org, st.session_state.param_dict, "Adj")
            
            # 3. Tổng hợp kết quả
            df_final = pd.merge(df_base, df_adj, on=['Năm', 'Tháng'])
            
            # Merge dữ liệu thực tế (Ground Truth) để tính sai số
            df_actual = df_train_org[['Năm', 'Tháng', 'Tổng thương phẩm']].copy()
            df_final = pd.merge(df_final, df_actual, on=['Năm', 'Tháng'], how='left')
            df_final.rename(columns={'Tổng thương phẩm': 'Thuc_te'}, inplace=True)
            df_final['Date'] = pd.to_datetime(dict(year=df_final['Năm'], month=df_final['Tháng'], day=1))

            # Tính độ lệch (Deviation)
            df_final['Độ lệch (NN)'] = df_final['NN_Adj'] - df_final['NN_Base']
            
            # --- HIỂN THỊ KẾT QUẢ ---
            st.subheader("2. Kết quả So sánh Kịch bản (Scenario Comparison)")
            
            cols = ['Tháng', 'NN_Base', 'NN_Adj', 'Độ lệch (NN)', 'XGB_Base', 'XGB_Adj']
            st.dataframe(df_final[['Năm'] + cols].style.format("{:,.0f}").applymap(
                lambda x: 'background-color: #f0f0f0' if x != 0 else '', 
                subset=['Độ lệch (NN)']
            ), use_container_width=True)

            # --- BIỂU ĐỒ ---
            st.subheader("3. Biểu đồ Phân tích (Visual Analysis)")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Kịch bản cơ sở (Nét đứt)
            ax.plot(df_final['Date'], df_final['NN_Base'], '--', color='gray', label='Kịch bản Cơ sở (Base Case)', alpha=0.7)
            
            # Kịch bản điều chỉnh (Nét liền)
            ax.plot(df_final['Date'], df_final['NN_Adj'], 's-', color='#d62728', label='Kịch bản Điều chỉnh (Adjusted Case)', linewidth=2)
            
            # Tô vùng chênh lệch
            ax.fill_between(df_final['Date'], df_final['NN_Base'], df_final['NN_Adj'], color='orange', alpha=0.15, label='Độ lệch do Biến ngoại sinh')

            if df_final['Thuc_te'].notnull().any():
                ax.plot(df_final['Date'], df_final['Thuc_te'], 'k-', linewidth=2, label='Giá trị Thực tế (Actual)', zorder=10)

            ax.set_title("So sánh Kịch bản Dự báo (NN Model)")
            ax.set_ylabel("Sản lượng (kWh)")
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Lỗi tính toán: {e}")
