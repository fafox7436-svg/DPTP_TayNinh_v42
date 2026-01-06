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
import re

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
# 1. MODULE AI (GIỮ NGUYÊN)
# ==============================================================================
def lay_noi_dung_tu_link(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = " ".join([p.get_text() for p in soup.find_all('p')])
            return text if len(text) > 50 else None, "✅ Đã đọc link!"
        return None, "⚠️ Lỗi link."
    except: return None, "⚠️ Lỗi đọc web."

def trich_xuat_so(text):
    try:
        matches = re.findall(r'-?\d+(?:\.\d+)?', str(text))
        if matches:
            val = float(matches[0])
            return max(min(val, 30.0), -30.0)
        return 0.0
    except: return 0.0

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
        
        # Auto-detect model
        found_model = "models/gemini-pro"
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    found_model = m.name
                    break
        except: pass

        model = genai.GenerativeModel(found_model)
        prompt = (f"Đọc tin: '{text_data[:3000]}'. Xác định % tăng/giảm phụ tải điện. "
                  "Trả về: SỐ | LÝ DO. Ví dụ: -1.5 | Giảm 1.5%")
        
        response = model.generate_content(prompt)
        res = response.text.strip()
        
        if "|" in res:
            parts = res.split("|")
            val = trich_xuat_so(parts[0]) 
            reason = parts[1].strip()
            return val, f"{status}✅ Xong! (Model: {found_model})", reason
            
        val = trich_xuat_so(res)
        if val != 0.0:
             return val, f"{status}✅ Xong (Tự bắt số)!", res

        return 0.0, f"⚠️ AI chạy OK nhưng không ra số.", ""
    except Exception as e: return 0.0, f"❌ Lỗi: {str(e)[:100]}", ""

# ==============================================================================
# 2. HÀM TÍNH TOÁN (ĐÃ SỬA LỖI KEY ERROR)
# ==============================================================================

# --- HÀM MỚI: TỰ ĐỘNG SỬA TÊN CỘT ---
def chuan_hoa_ten_cot(df):
    """
    Hàm này tự động đổi tên cột về chuẩn: 'Năm', 'Tháng', 'Tổng thương phẩm'
    Bất chấp file Excel ghi là 'Month', 'thang', 'Year', 'nam', v.v.
    """
    # Xóa khoảng trắng thừa ở tên cột
    df.columns = df.columns.str.strip()
    
    # Map các tên có thể gặp sang tên chuẩn
    col_map = {
        # Tháng
        'month': 'Tháng', 'thang': 'Tháng', 'tháng': 'Tháng', 'Thang': 'Tháng',
        # Năm
        'year': 'Năm', 'nam': 'Năm', 'năm': 'Năm', 'Nam': 'Năm',
        # Sản lượng
        'tổng thương phẩm': 'Tổng thương phẩm', 'tong thuong pham': 'Tổng thương phẩm',
        'thuong pham': 'Tổng thương phẩm', 'sản lượng': 'Tổng thương phẩm',
        'san luong': 'Tổng thương phẩm', 'commercial': 'Tổng thương phẩm'
    }
    
    new_cols = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower in col_map:
            new_cols[col] = col_map[col_lower]
            
    df = df.rename(columns=new_cols)
    return df

def feature_engineering(df):
    # Trước khi xử lý, phải đảm bảo cột tồn tại
    if 'Tháng' not in df.columns or 'Năm' not in df.columns:
        # Nếu vẫn không tìm thấy sau khi chuẩn hóa -> Báo lỗi mềm thay vì sập
        st.error(f"❌ Lỗi dữ liệu: File thiếu cột 'Tháng' hoặc 'Năm'. Các cột hiện có: {list(df.columns)}")
        st.stop()
        
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3,4,5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6,7,8,9,10,11] else 0)
    
    def check_tet(row):
        try:
            return 1 if (row['Năm']==2025 and row['Tháng']==1) or (row['Năm']==2024 and row['Tháng']==2) else 0
        except: return 0
        
    df['Co_Tet'] = df.apply(check_tet, axis=1)
    return df

def chay_mo_phong_sach(df_train_origin, df_input_origin, user_seed):
    # --- ÁP DỤNG CHUẨN HÓA TÊN CỘT NGAY ĐẦU VÀO ---
    df_train = chuan_hoa_ten_cot(df_train_origin.copy())
    df_input = chuan_hoa_ten_cot(df_input_origin.copy())

    df_train['Bien_Ngoai_Sinh'] = 0
    df_input['Bien_Ngoai_Sinh'] = 0

    df_train = feature_engineering(df_train)
    df_input = feature_engineering(df_input)

    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_features = [f for f in features if f in df_train.columns and f in df_input.columns]
    target = 'Tổng thương phẩm'
    
    # Kiểm tra target tồn tại
    if target not in df_train.columns:
        st.error("❌ Lỗi: Không tìm thấy cột 'Tổng thương phẩm' (hoặc tương đương) trong file Train.")
        st.stop()

    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=user_seed)
    nn.fit(X_scaled, y)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)

    # Predict
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

# --- PHẦN 1: AI ---
st.subheader("1️⃣ Phân Tích Tác Động (AI)")
c1, c2 = st.columns([2, 1])
if 'detected_months' not in st.session_state: st.session_state.detected_months = []

with c1: 
    text_data = st.text_area("Nhập Link hoặc Tin tức:", height=80)

with c2:
    if uploaded_input:
        try:
            # Đọc thử để lấy tháng, nhớ chuẩn hóa cột trước khi đọc
            df_temp = pd.read_excel(uploaded_input)
            df_temp = chuan_hoa_ten_cot(df_temp) # SỬA LỖI: Chuẩn hóa ngay tại đây
            if 'Năm' in df_temp.columns and 'Tháng' in df_temp.columns:
                st.session_state.detected_months = sorted(list(set(zip(df_temp['Năm'], df_temp['Tháng']))))
        except: pass
    
    if st.button("Phân Tích Ngay"):
        with st.spinner("Đang xử lý..."):
            val, log, reason = xu_ly_du_lieu_dinh_tinh(api_key, text_data)
        
        if "Lỗi" in log: 
            st.warning(log)
            st.session_state.temp_score = 0.0
            st.session_state.temp_reason = "Lỗi, nhập tay"
        else:
            st.success(log)
            st.session_state.temp_score = val
            st.session_state.temp_reason = reason

# --- PHẦN 2: THIẾT LẬP ---
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
        st.success("Đã lưu!")

st.write("---")

# --- PHẦN 3: CHẠY DỰ BÁO ---
if uploaded_train and uploaded_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        with st.spinner("Đang chạy mô hình gốc..."):
            # Sửa lỗi: Đọc file xong phải pass qua hàm chuẩn hóa bên trong chay_mo_phong_sach
            df_final = chay_mo_phong_sach(pd.read_excel(uploaded_train), pd.read_excel(uploaded_input), selected_seed)

        def apply_adjustment(row):
            param = st.session_state.param_dict.get((row['Năm'], row['Tháng']), (0.0, ""))
            pct = param[0]
            reason = param[1]
            factor = 1.0 + (pct / 100.0)
            return (row['NN_Goc']*factor, row['RF_Goc']*factor, row['XGB_Goc']*factor, pct, reason)

        adj_res = df_final.apply(apply_adjustment, axis=1, result_type='expand')
        df_final['NN_Final'] = adj_res[0]
        df_final['RF_Final'] = adj_res[1]
        df_final['XGB_Final'] = adj_res[2]
        df_final['Gemini_Pct'] = adj_res[3]
        df_final['Lý_do'] = adj_res[4]

        # Chuẩn hóa cả file train gốc để merge cho đúng
        df_actual_raw = pd.read_excel(uploaded_train)
        df_actual_raw = chuan_hoa_ten_cot(df_actual_raw)
        
        df_actual = df_actual_raw[['Năm', 'Tháng', 'Tổng thương phẩm']]
        df_show = pd.merge(df_final, df_actual, on=['Năm', 'Tháng'], how='left')
        df_show['Thuc_te'] = df_show['Tổng thương phẩm']
        
        # HIỂN THỊ
        st.subheader("📊 Bảng Kết Quả")
        cols = {'Tháng': 'Tháng', 'Thuc_te': 'Thực Tế', 'Gemini_Pct': '% Đ.Chỉnh',
                'NN_Goc': 'NN Gốc', 'NN_Final': 'NN (Đã chỉnh)',
                'RF_Goc': 'RF Gốc', 'RF_Final': 'RF (Đã chỉnh)',
                'XGB_Goc': 'XGB Gốc', 'XGB_Final': 'XGB (Đã chỉnh)', 'Lý_do': 'Lý do'}
        
        df_display = df_show[['Năm'] + list(cols.keys())].rename(columns=cols)
        df_display['% Đ.Chỉnh'] = df_display['% Đ.Chỉnh'].apply(lambda x: f"{x:+.1f}%" if x!=0 else "-")
        
        st.dataframe(df_display.style.format({
            'Thực Tế': '{:,.0f}',
            'NN Gốc': '{:,.0f}', 'NN (Đã chỉnh)': '{:,.0f}',
            'RF Gốc': '{:,.0f}', 'RF (Đã chỉnh)': '{:,.0f}',
            'XGB Gốc': '{:,.0f}', 'XGB (Đã chỉnh)': '{:,.0f}',
        }), use_container_width=True)

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
