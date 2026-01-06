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
st.set_page_config(page_title="Dự Báo Phụ Tải Tây Ninh", layout="wide")
st.title("HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN TỈNH TÂY NINH")
st.markdown("---")

# --- KIỂM TRA THƯ VIỆN AI ---
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except: 
    HAS_GEMINI = False

# ==============================================================================
# 1. MODULE AI & XỬ LÝ TEXT (SIÊU BỀN)
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
    # Dùng Regex để bắt số bất chấp văn bản (VD: "Tăng 5%" -> lấy 5.0)
    try:
        matches = re.findall(r'-?\d+(?:\.\d+)?', str(text))
        if matches:
            val = float(matches[0])
            # Chặn số ảo quá lớn
            return max(min(val, 50.0), -50.0)
        return 0.0
    except: return 0.0

def xu_ly_du_lieu_dinh_tinh(api_key, input_data):
    if not api_key: return 0.0, "⚠️ Chưa nhập API Key.", ""
    
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
        
        # Cơ chế "Vơ bèo gạt tép": Có model nào dùng model đó
        found_model = "gemini-pro" # Mặc định
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
        
        # Ưu tiên 1: Lấy theo format chuẩn
        if "|" in res:
            parts = res.split("|")
            val = trich_xuat_so(parts[0]) 
            reason = parts[1].strip()
            return val, f"{status}✅ Xong! (Model: {found_model})", reason
            
        # Ưu tiên 2: Tự tìm số trong văn bản
        val = trich_xuat_so(res)
        if val != 0.0:
             return val, f"{status}✅ Xong (Tự bắt số)!", res

        return 0.0, f"⚠️ AI chạy OK nhưng không ra số.", ""
    except Exception as e: return 0.0, f"❌ Lỗi: {str(e)[:100]}", ""

# ==============================================================================
# 2. XỬ LÝ FILE EXCEL (CHÌA KHÓA VẠN NĂNG)
# ==============================================================================
def chuan_hoa_ten_cot(df):
    """Đưa mọi tên cột về chuẩn: Năm, Tháng, Tổng thương phẩm"""
    if df is None: return None
    df.columns = df.columns.astype(str).str.strip()
    
    col_map = {
        'month': 'Tháng', 'thang': 'Tháng', 'tháng': 'Tháng', 'Thang': 'Tháng',
        'year': 'Năm', 'nam': 'Năm', 'năm': 'Năm', 'Nam': 'Năm',
        'tổng thương phẩm': 'Tổng thương phẩm', 'tong thuong pham': 'Tổng thương phẩm',
        'thuong pham': 'Tổng thương phẩm', 'sản lượng': 'Tổng thương phẩm',
        'san luong': 'Tổng thương phẩm', 'commercial': 'Tổng thương phẩm'
    }
    
    new_cols = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in col_map:
            new_cols[col] = col_map[col_lower]
            
    return df.rename(columns=new_cols)

def read_user_file(uploaded_file, header_idx):
    """Đọc file với dòng tiêu đề do người dùng chọn"""
    try:
        # Reset con trỏ file
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, header=header_idx)
        df = chuan_hoa_ten_cot(df)
        return df
    except Exception as e:
        return None

# ==============================================================================
# 3. TÍNH TOÁN DỰ BÁO (CÁCH LY TUYỆT ĐỐI)
# ==============================================================================
def feature_engineering(df):
    if df is None: return None
    
    # Kiểm tra cột bắt buộc
    required = ['Tháng', 'Năm']
    if not all(col in df.columns for col in required):
        st.error(f"❌ Thiếu cột 'Tháng' hoặc 'Năm'. Các cột hiện có: {list(df.columns)}")
        st.stop()

    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3,4,5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6,7,8,9,10,11] else 0)
    
    def check_tet(row):
        try: return 1 if (row['Năm']==2025 and row['Tháng']==1) or (row['Năm']==2024 and row['Tháng']==2) else 0
        except: return 0
    df['Co_Tet'] = df.apply(check_tet, axis=1)
    
    return df

def chay_mo_phong_sach(df_train, df_input, user_seed):
    # Đảm bảo sạch 100%
    df_train['Bien_Ngoai_Sinh'] = 0
    df_input['Bien_Ngoai_Sinh'] = 0

    df_train = feature_engineering(df_train)
    df_input = feature_engineering(df_input)

    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_features = [f for f in features if f in df_train.columns and f in df_input.columns]
    target = 'Tổng thương phẩm'
    
    if target not in df_train.columns:
        st.error("❌ Không tìm thấy cột 'Tổng thương phẩm' trong file Train.")
        st.stop()

    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # HUẤN LUYỆN 3 MÔ HÌNH
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=user_seed)
    nn.fit(X_scaled, y)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)

    # DỰ BÁO
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    df_pred[valid_features] = df_pred[valid_features].fillna(0)
    
    df_pred['NN_Goc'] = nn.predict(scaler.transform(df_pred[valid_features]))
    df_pred['RF_Goc'] = rf.predict(df_pred[valid_features])
    df_pred['XGB_Goc'] = xgb_model.predict(df_pred[valid_features])
    
    return df_pred[['Năm', 'Tháng', 'NN_Goc', 'RF_Goc', 'XGB_Goc']]

# ==============================================================================
# GIAO DIỆN NGƯỜI DÙNG
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    api_key = st.text_input("API Key (Tùy chọn)", type="password")
    
    st.markdown("---")
    st.write("### 📂 Cấu hình File Excel")
    # Cho phép chọn dòng tiêu đề để tránh lỗi
    header_idx = st.number_input("Dòng tiêu đề (Header Row):", min_value=0, max_value=10, value=0, help="Dòng chứa chữ Năm, Tháng là dòng số mấy? (Bắt đầu từ 0)")
    
    st.markdown("---")
    selected_seed = st.number_input("Random Seed", value=42)
    
    if st.button("🗑️ XÓA CACHE & LÀM MỚI"):
        st.cache_data.clear()
        st.rerun()

col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. Dữ liệu Lịch sử (Train)", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. Dữ liệu Dự báo (Input)", type=['xlsx', 'xls'])

st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# --- PHẦN 1: KỊCH BẢN (AI HOẶC TAY) ---
st.subheader("1️⃣ Thiết Lập Kịch Bản (AI / Thủ Công)")
c1, c2 = st.columns([2, 1])
if 'detected_months' not in st.session_state: st.session_state.detected_months = []

with c1: 
    text_data = st.text_area("Nhập Link báo hoặc Tin tức:", height=80)

with c2:
    if uploaded_input:
        try:
            # Đọc file để lấy tháng (dùng header user chọn)
            df_temp = read_user_file(uploaded_input, header_idx)
            if df_temp is not None and 'Năm' in df_temp.columns:
                st.session_state.detected_months = sorted(list(set(zip(df_temp['Năm'], df_temp['Tháng']))))
        except: pass
    
    if st.button("Phân Tích Ngay"):
        with st.spinner("Đang xử lý..."):
            val, log, reason = xu_ly_du_lieu_dinh_tinh(api_key, text_data)
        
        if "Lỗi" in log: 
            st.warning(log)
            st.session_state.temp_score = 0.0
            st.session_state.temp_reason = "Lỗi, hãy nhập tay"
        else:
            st.success(log)
            st.session_state.temp_score = val
            st.session_state.temp_reason = reason

# --- PHẦN 2: CHỌN THÁNG & NHẬP SỐ ---
if st.session_state.detected_months:
    c_a, c_b = st.columns(2)
    with c_a:
        months_str = [f"Tháng {m}/{y}" for y, m in st.session_state.detected_months]
        selected = st.multiselect("Chọn tháng áp dụng:", months_str, default=months_str)
    with c_b:
        cur_val = st.session_state.get('temp_score', 0.0)
        final_pct = st.number_input("Mức Tăng/Giảm (%) - NHẬP SỐ Ở ĐÂY:", value=float(cur_val), step=0.1, format="%.2f")
    
    if st.button("💾 Lưu Kịch Bản"):
        temp = {}
        for s in selected:
            m = int(s.split('/')[0].replace('Tháng ', ''))
            y = int(s.split('/')[1])
            temp[(y, m)] = (final_pct, st.session_state.get('temp_reason', 'Thủ công'))
        st.session_state.param_dict = temp
        st.success(f"Đã lưu: {final_pct}%")

st.write("---")

# --- PHẦN 3: CHẠY DỰ BÁO ---
if uploaded_train and uploaded_input:
    # Preview để user biết mình chọn đúng dòng tiêu đề chưa
    st.caption(f"👀 Xem trước dữ liệu (Header Row = {header_idx}):")
    df_preview = read_user_file(uploaded_train, header_idx)
    if df_preview is not None:
        st.dataframe(df_preview.head(2), use_container_width=True)

    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        with st.spinner("Đang chạy mô hình gốc..."):
            # Đọc lại file chuẩn chỉnh
            df_train_clean = read_user_file(uploaded_train, header_idx)
            df_input_clean = read_user_file(uploaded_input, header_idx)
            
            # Chạy mô hình
            df_final = chay_mo_phong_sach(df_train_clean, df_input_clean, selected_seed)

        # HÀM CỘNG TRỪ (POST-PROCESSING)
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

        # Merge thực tế
        df_actual_raw = read_user_file(uploaded_train, header_idx)
        if 'Tổng thương phẩm' in df_actual_raw.columns:
            df_actual = df_actual_raw[['Năm', 'Tháng', 'Tổng thương phẩm']]
            df_show = pd.merge(df_final, df_actual, on=['Năm', 'Tháng'], how='left')
            df_show['Thuc_te'] = df_show['Tổng thương phẩm']
        else:
            df_show = df_final.copy()
            df_show['Thuc_te'] = np.nan
        
        # HIỂN THỊ KẾT QUẢ
        st.subheader("📊 Bảng Kết Quả Chi Tiết")
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

        st.subheader("📈 Biểu Đồ So Sánh")
        df_show['Date'] = pd.to_datetime(dict(year=df_show['Năm'], month=df_show['Tháng'], day=1))
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(df_show['Date'], df_show['NN_Goc'], '--', color='gray', label='NN Gốc', alpha=0.5)
        ax.plot(df_show['Date'], df_show['NN_Final'], 's-', color='red', label='NN Đã Chỉnh', linewidth=2)
        ax.plot(df_show['Date'], df_show['XGB_Final'], '^-', color='green', label='XGB Đã Chỉnh', linewidth=2)
        if df_show['Thuc_te'].notnull().any():
            ax.plot(df_show['Date'], df_show['Thuc_te'], 'o-', color='black', label='Thực tế', linewidth=3)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
