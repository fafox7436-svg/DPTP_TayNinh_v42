import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import io
import time
import requests
from bs4 import BeautifulSoup

# --- CẤU HÌNH ---
st.set_page_config(page_title="Hệ Thống Dự Báo Phụ Tải", layout="wide")
st.title("HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN")
st.markdown("---")

# --- KIỂM TRA THƯ VIỆN ---
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except: HAS_GEMINI = False

# ==============================================================================
# 1. MODULE AI & WEB (GIỮ NGUYÊN)
# ==============================================================================
def lay_noi_dung_tu_link(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = " ".join([p.get_text() for p in soup.find_all('p')])
            return text if len(text) > 50 else None, "✅ Đã đọc link!"
        return None, "⚠️ Lỗi link."
    except: return None, "⚠️ Lỗi đọc web."

def xu_ly_du_lieu_dinh_tinh(api_key, input_data):
    if not api_key: return 0.0, "⚠️ Thiếu API Key.", ""
    text_data = input_data
    status = ""
    
    if input_data.strip().startswith("http"):
        with st.spinner("Đang đọc link..."):
            extracted, msg = lay_noi_dung_tu_link(input_data)
            if extracted: 
                text_data = extracted
                status = msg + "\n"
            else: return 0.0, msg, ""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")
        prompt = (f"Đọc tin: '{text_data[:4000]}'. Xác định % tăng/giảm phụ tải điện. "
                  "Trả về: SỐ | LÝ DO. Ví dụ: -1.5 | Giảm 1.5%")
        res = model.generate_content(prompt).text.strip()
        
        if "|" in res:
            parts = res.split("|")
            val = float(parts[0].strip())
            return val, f"{status}✅ Xong!", parts[1].strip()
        return 0.0, "⚠️ Không rõ số.", ""
    except Exception as e: return 0.0, f"❌ Lỗi: {e}", ""

# ==============================================================================
# 2. HÀM TÍNH TOÁN (CÁCH LY TUYỆT ĐỐI)
# ==============================================================================
def feature_engineering(df):
    # Tạo các đặc trưng cơ bản
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3,4,5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6,7,8,9,10,11] else 0)
    # Lịch Tết đơn giản
    def check_tet(row):
        return 1 if (row['Năm']==2025 and row['Tháng']==1) or (row['Năm']==2024 and row['Tháng']==2) else 0
    df['Co_Tet'] = df.apply(check_tet, axis=1)
    return df

# HÀM NÀY CHỈ CHẠY MÔ HÌNH TRÊN DỮ LIỆU SẠCH (KHÔNG CÓ GEMINI)
def chay_mo_phong_sach(df_train_origin, df_input_origin, user_seed):
    df_train = df_train_origin.copy()
    df_input = df_input_origin.copy()

    # --- QUAN TRỌNG: LUÔN GÁN BIẾN NGOẠI SINH = 0 ĐỂ MODEL ỔN ĐỊNH ---
    df_train['Bien_Ngoai_Sinh'] = 0
    df_input['Bien_Ngoai_Sinh'] = 0

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

    # 1. Train NN
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=user_seed)
    nn.fit(X_scaled, y)
    
    # 2. Train RF
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 3. Train XGB
    xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)

    # DỰ BÁO BASELINE (GỐC)
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    df_pred[valid_features] = df_pred[valid_features].fillna(0)
    
    df_pred['NN_Goc'] = nn.predict(scaler.transform(df_pred[valid_features]))
    df_pred['RF_Goc'] = rf.predict(df_pred[valid_features])
    df_pred['XGB_Goc'] = xgb_model.predict(df_pred[valid_features])
    
    return df_pred[['Năm', 'Tháng', 'NN_Goc', 'RF_Goc', 'XGB_Goc']]

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    api_key = st.text_input("API Key", type="password")
    st.markdown("---")
    selected_seed = st.number_input("Random Seed", value=42)
    
    if st.button("🗑️ XÓA CACHE"):
        st.cache_data.clear()
        st.rerun()

col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. Dữ liệu Lịch sử", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. Dữ liệu Dự báo", type=['xlsx', 'xls'])

st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# --- PHẦN 1: GEMINI TÍNH % ---
st.subheader("1️⃣ Phân Tích Tác Động (Gemini)")
c1, c2 = st.columns([2, 1])
if 'detected_months' not in st.session_state: st.session_state.detected_months = []

with c1: 
    text_data = st.text_area("Nhập Link hoặc Tin tức:", height=80)

with c2:
    if uploaded_input:
        try:
            df_temp = pd.read_excel(uploaded_input)
            st.session_state.detected_months = sorted(list(set(zip(df_temp['Năm'], df_temp['Tháng']))))
        except: pass
    
    if st.button("Phân Tích Ngay"):
        with st.spinner("AI đang đọc..."):
            val, log, reason = xu_ly_du_lieu_dinh_tinh(api_key, text_data)
        if "Lỗi" in log: st.warning(log)
        else:
            st.success(f"{log} -> {val}%")
            st.session_state.temp_score = val
            st.session_state.temp_reason = reason

# --- PHẦN 2: THIẾT LẬP CỘNG TRỪ ---
if st.session_state.detected_months:
    c_a, c_b = st.columns(2)
    with c_a:
        months_str = [f"Tháng {m}/{y}" for y, m in st.session_state.detected_months]
        selected = st.multiselect("Chọn tháng áp dụng:", months_str, default=months_str)
    with c_b:
        cur_val = st.session_state.get('temp_score', 0.0)
        final_pct = st.number_input("Mức Tăng/Giảm (%) - Cộng trừ bên ngoài:", value=float(cur_val), step=0.1)
    
    if st.button("💾 Lưu Kịch Bản"):
        temp = {}
        for s in selected:
            m = int(s.split('/')[0].replace('Tháng ', ''))
            y = int(s.split('/')[1])
            temp[(y, m)] = (final_pct, st.session_state.get('temp_reason', ''))
        st.session_state.param_dict = temp
        st.success("Đã lưu! Số này sẽ được cộng/trừ vào kết quả cuối cùng.")

st.write("---")

# --- PHẦN 3: CHẠY DỰ BÁO VÀ TỰ CỘNG TRỪ ---
if uploaded_train and uploaded_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        # 1. Chạy mô hình trên dữ liệu sạch (Ra số Gốc ổn định)
        with st.spinner("Đang chạy mô hình gốc..."):
            df_final = chay_mo_phong_sach(pd.read_excel(uploaded_train), pd.read_excel(uploaded_input), selected_seed)

        # 2. Cộng trừ thủ công bên ngoài (Post-processing)
        def apply_adjustment(row):
            # Lấy % từ kịch bản
            param = st.session_state.param_dict.get((row['Năm'], row['Tháng']), (0.0, ""))
            pct = param[0]
            reason = param[1]
            factor = 1.0 + (pct / 100.0)
            
            # Nhân hệ số cho từng phương pháp
            return (
                row['NN_Goc'] * factor, 
                row['RF_Goc'] * factor, 
                row['XGB_Goc'] * factor, 
                pct, 
                reason
            )

        # Áp dụng hàm cộng trừ
        adj_res = df_final.apply(apply_adjustment, axis=1, result_type='expand')
        df_final['NN_Final'] = adj_res[0]
        df_final['RF_Final'] = adj_res[1]
        df_final['XGB_Final'] = adj_res[2]
        df_final['Gemini_Pct'] = adj_res[3]
        df_final['Lý_do'] = adj_res[4]

        # Ghép thực tế để so sánh
        df_actual = pd.read_excel(uploaded_train)[['Năm', 'Tháng', 'Tổng thương phẩm']]
        df_show = pd.merge(df_final, df_actual, on=['Năm', 'Tháng'], how='left')
        df_show['Thuc_te'] = df_show['Tổng thương phẩm']
        
        # HIỂN THỊ
        st.subheader("📊 Bảng Kết Quả (Gốc vs Đã Điều Chỉnh)")
        
        cols = {
            'Tháng': 'Tháng', 'Thuc_te': 'Thực Tế',
            'Gemini_Pct': '% Đ.Chỉnh',
            'NN_Goc': 'NN Gốc', 'NN_Final': 'NN (Đã chỉnh)',
            'RF_Goc': 'RF Gốc', 'RF_Final': 'RF (Đã chỉnh)',
            'XGB_Goc': 'XGB Gốc', 'XGB_Final': 'XGB (Đã chỉnh)',
            'Lý_do': 'Lý do'
        }
        
        # Format bảng đẹp
        df_display = df_show[['Năm'] + list(cols.keys())].rename(columns=cols)
        df_display['% Đ.Chỉnh'] = df_display['% Đ.Chỉnh'].apply(lambda x: f"{x:+.1f}%" if x!=0 else "-")
        
        st.dataframe(df_display.style.format({
            'Thực Tế': '{:,.0f}',
            'NN Gốc': '{:,.0f}', 'NN (Đã chỉnh)': '{:,.0f}',
            'RF Gốc': '{:,.0f}', 'RF (Đã chỉnh)': '{:,.0f}',
            'XGB Gốc': '{:,.0f}', 'XGB (Đã chỉnh)': '{:,.0f}',
        }), use_container_width=True)

        # Biểu đồ
        st.subheader("📈 Biểu Đồ")
        df_show['Date'] = pd.to_datetime(dict(year=df_show['Năm'], month=df_show['Tháng'], day=1))
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(df_show['Date'], df_show['NN_Goc'], '--', color='gray', label='NN Gốc', alpha=0.5)
        ax.plot(df_show['Date'], df_show['NN_Final'], 's-', color='red', label='NN Đã Chỉnh')
        ax.plot(df_show['Date'], df_show['XGB_Final'], '^-', color='green', label='XGB Đã Chỉnh')
        if df_show['Thuc_te'].notnull().any():
            ax.plot(df_show['Date'], df_show['Thuc_te'], 'o-', color='black', label='Thực tế')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
