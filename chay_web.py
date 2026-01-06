import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import requests
from bs4 import BeautifulSoup
import re

# --- CẤU HÌNH ---
st.set_page_config(page_title="Dự Báo Phụ Tải Tây Ninh", layout="wide")
st.title("⚡ HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN TỈNH TÂY NINH")
st.markdown("---")

# --- KIỂM TRA THƯ VIỆN AI ---
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except: 
    HAS_GEMINI = False

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
            return max(min(val, 50.0), -50.0)
        return 0.0
    except: return 0.0

def xu_ly_du_lieu_dinh_tinh(api_key, input_data):
    if not api_key: return 0.0, "⚠️ Chưa nhập API Key (Sẽ nhập tay).", "Thủ công"
    
    text_data = input_data
    status = ""
    if input_data.strip().startswith("http"):
        with st.spinner("Đang đọc bài báo..."):
            extracted, msg = lay_noi_dung_tu_link(input_data)
            if extracted: 
                text_data = extracted
                status = msg + "\n"
            else: return 0.0, msg, ""

    try:
        genai.configure(api_key=api_key)
        found_model = "gemini-1.5-flash"
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
            return val, f"{status}✅ Xong! (AI: {found_model})", reason
            
        val = trich_xuat_so(res)
        if val != 0.0:
             return val, f"{status}✅ Xong (AI tự bắt số)!", res

        return 0.0, f"⚠️ AI không tìm thấy số liệu cụ thể.", ""
    except Exception as e:
        if "429" in str(e): return 0.0, "⚠️ Hết hạn mức AI (Quota). Hãy nhập tay.", "Hết Quota"
        return 0.0, f"❌ Lỗi AI: {str(e)[:50]}...", "Lỗi"

# ==============================================================================
# 2. XỬ LÝ FILE (THÔNG MINH)
# ==============================================================================
def chuan_hoa_ten_cot(df):
    if df is None: return None
    df.columns = df.columns.astype(str).str.strip()
    col_map = {
        'month': 'Tháng', 'thang': 'Tháng', 'tháng': 'Tháng',
        'year': 'Năm', 'nam': 'Năm', 'năm': 'Năm',
        'tổng thương phẩm': 'Tổng thương phẩm', 'tong thuong pham': 'Tổng thương phẩm', 'sản lượng': 'Tổng thương phẩm',
        'nhiệt độ tb': 'Nhiệt độ TB', 'nhiet do tb': 'Nhiệt độ TB',
        'độ ẩm': 'Độ ẩm', 'do am': 'Độ ẩm',
        'số ngày': 'Số ngày', 'so ngay': 'Số ngày'
    }
    new_cols = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in col_map:
            new_cols[col] = col_map[col_lower]
    return df.rename(columns=new_cols)

def ultra_scan_read_excel(uploaded_file):
    try:
        xl = pd.ExcelFile(uploaded_file)
        for sheet_name in xl.sheet_names:
            preview = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, nrows=10)
            for i, row in preview.iterrows():
                row_str = str(row.values).lower()
                if 'tháng' in row_str and 'năm' in row_str:
                    uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=i)
                    return chuan_hoa_ten_cot(df)
        uploaded_file.seek(0)
        return chuan_hoa_ten_cot(pd.read_excel(uploaded_file, header=0))
    except: return None

def tao_dac_trung(df):
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3,4,5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6,7,8,9,10,11] else 0)
    def check_tet(row):
        try: return 1 if (row['Năm']==2025 and row['Tháng']==1) or (row['Năm']==2024 and row['Tháng']==2) or (row['Năm']==2026 and row['Tháng']==2) else 0
        except: return 0
    df['Co_Tet'] = df.apply(check_tet, axis=1)
    df['Bien_Ngoai_Sinh'] = 0
    return df

# ==============================================================================
# 3. CHẠY DỰ BÁO (CÓ CACHE)
# ==============================================================================
@st.cache_data(show_spinner=False)
def chay_mo_hinh_goc(df_train, df_input, seed=42):
    # Tạo đặc trưng
    df_train = tao_dac_trung(df_train.copy())
    df_input = tao_dac_trung(df_input.copy())
    
    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_cols = [c for c in features if c in df_train.columns and c in df_input.columns]
    target = 'Tổng thương phẩm'
    
    # Train
    data_train = df_train.dropna(subset=valid_cols + [target])
    X_train = data_train[valid_cols]
    y_train = data_train[target]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # --- MODEL 1: Neural Network ---
    nn = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=5000, random_state=seed)
    nn.fit(X_train_scaled, y_train)
    
    # --- MODEL 2 & 3: RF & XGB ---
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    
    xg = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xg.fit(X_train, y_train)
    
    # Predict
    df_pred = df_input.copy()
    X_pred = df_pred[valid_cols].fillna(0)
    
    pred_nn = nn.predict(scaler.transform(X_pred))
    pred_rf = rf.predict(X_pred)
    pred_xg = xg.predict(X_pred)
    
    return pred_nn, pred_rf, pred_xg

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    api_key = st.text_input("API Key (Cho AI)", type="password")
    
    st.markdown("---")
    st.write("### 🛠️ Điều chỉnh chung")
    seed_val = st.number_input("Random Seed (Mặc định 42)", value=42)
    if st.button("🗑️ Xóa Cache (Reset App)"):
        st.cache_data.clear()
        st.rerun()

col1, col2 = st.columns(2)
with col1: f_train = st.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
# --- SỬA LỖI Ở ĐÂY (Thay c2 bằng col2) ---
with col2: f_input = st.file_uploader("2. File Dự Báo (Input)", type=['xlsx', 'xls'])

st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# --- PHẦN 1: AI ---
st.subheader("1️⃣ Phân Tích Thông Tin (AI / Thủ Công)")
c1, c2 = st.columns([2, 1])
if 'detected_months' not in st.session_state: st.session_state.detected_months = []

with c1: 
    text_data = st.text_area("Nhập Link hoặc Tin tức:", height=80)

with c2:
    if f_input:
        try:
            df_temp = ultra_scan_read_excel(f_input)
            if df_temp is not None:
                st.session_state.detected_months = sorted(list(set(zip(df_temp['Năm'], df_temp['Tháng']))))
        except: pass
    
    if st.button("Phân Tích AI"):
        with st.spinner("AI đang đọc..."):
            val, log, reason = xu_ly_du_lieu_dinh_tinh(api_key, text_data)
        
        if "Lỗi" in log or "429" in log: 
            st.warning(log)
            st.session_state.temp_score = 0.0
            st.session_state.temp_reason = "Lỗi/Thủ công"
        else:
            st.success(log)
            st.session_state.temp_score = val
            st.session_state.temp_reason = reason

# --- PHẦN 2: LƯU KỊCH BẢN ---
if st.session_state.detected_months:
    c_a, c_b = st.columns(2)
    with c_a:
        months_str = [f"Tháng {m}/{y}" for y, m in st.session_state.detected_months]
        selected = st.multiselect("Chọn tháng áp dụng:", months_str, default=months_str)
    with c_b:
        cur_val = st.session_state.get('temp_score', 0.0)
        final_pct = st.number_input("Mức Tăng/Giảm (%) - NHẬP SỐ:", value=float(cur_val), step=0.1)
    
    if st.button("💾 Lưu Kịch Bản"):
        temp = {}
        for s in selected:
            m = int(s.split('/')[0].replace('Tháng ', ''))
            y = int(s.split('/')[1])
            temp[(y, m)] = (final_pct, st.session_state.get('temp_reason', 'Thủ công'))
        st.session_state.param_dict = temp
        st.success(f"Đã lưu: {final_pct}%")

st.write("---")

# --- PHẦN 3: DỰ BÁO ---
if f_train and f_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        with st.spinner("Đang tính toán..."):
            # 1. Đọc file
            df_train = ultra_scan_read_excel(f_train)
            df_input = ultra_scan_read_excel(f_input)
            
            if df_train is not None and df_input is not None:
                # 2. Chạy mô hình gốc
                pred_nn, pred_rf, pred_xg = chay_mo_hinh_goc(df_train, df_input, seed_val)
                
                # 3. Tạo DataFrame kết quả
                res = df_input[['Năm', 'Tháng']].copy()
                res['NN_Goc'] = pred_nn
                res['RF_Goc'] = pred_rf
                res['XGB_Goc'] = pred_xg

                # 4. Áp dụng điều chỉnh
                def apply_adj(row):
                    param = st.session_state.param_dict.get((row['Năm'], row['Tháng']), (0.0, ""))
                    factor = 1.0 + (param[0] / 100.0)
                    
                    nn_adj = row['NN_Goc'] * factor
                    rf_adj = row['RF_Goc'] * factor
                    xgb_adj = row['XGB_Goc'] * factor
                    
                    # CÔNG THỨC CHỐT: Trung bình cộng
                    chot = (nn_adj + rf_adj + xgb_adj) / 3
                    return chot, nn_adj, rf_adj, xgb_adj, param[0], param[1]

                adj_data = res.apply(apply_adj, axis=1, result_type='expand')
                res['Dự Báo Chốt'] = adj_data[0]
                res['NN'] = adj_data[1]
                res['RF'] = adj_data[2]
                res['XGB'] = adj_data[3]
                res['Tác Động %'] = adj_data[4]
                res['Lý do'] = adj_data[5]

                # 5. Merge thực tế
                if 'Tổng thương phẩm' in df_train.columns:
                    actual = df_train[['Năm', 'Tháng', 'Tổng thương phẩm']]
                    res = pd.merge(res, actual, on=['Năm', 'Tháng'], how='left')
                    res.rename(columns={'Tổng thương phẩm': 'Thực Tế'}, inplace=True)
                
                # --- HIỂN THỊ KẾT QUẢ DỄ HIỂU ---
                st.subheader("📊 Bảng Kết Quả Dự Báo")
                
                cols_display = {
                    'Tháng': 'Tháng', 'Năm': 'Năm',
                    'Thực Tế': 'Thực Tế (Năm ngoái)',
                    'Dự Báo Chốt': 'Dự Báo Chính Thức',
                    'Tác Động %': 'Điều Chỉnh (%)',
                    'Lý do': 'Ghi Chú'
                }
                
                cols_to_use = [c for c in cols_display.keys() if c in res.columns]
                df_show = res[cols_to_use].rename(columns=cols_display)
                df_show['Điều Chỉnh (%)'] = df_show['Điều Chỉnh (%)'].apply(lambda x: f"{x:+.1f}%" if x!=0 else "-")
                
                st.dataframe(df_show.style.format({
                    'Thực Tế (Năm ngoái)': '{:,.0f}', 'Dự Báo Chính Thức': '{:,.0f}'
                }), use_container_width=True)
                
                # GIẢI THÍCH
                st.info("""
                ℹ️ **Giải thích:** Kết quả "Dự Báo Chính Thức" là trung bình của 3 mô hình (Neural Network + Random Forest + XGBoost).
                * **Tại sao RF/XGB thấp hơn NN?** Do mô hình Cây (RF/XGB) có tính chất "an toàn", không dám dự báo cao hơn mức đỉnh lịch sử.
                * **Neural Network (NN):** Thông minh hơn trong việc bắt xu hướng tăng trưởng của phụ tải.
                👉 Hệ thống đã tự động cân bằng cả 3 để ra con số hợp lý nhất.
                """)

                # BIỂU ĐỒ
                st.subheader("📈 Biểu Đồ So Sánh")
                res['Date'] = pd.to_datetime(dict(year=res['Năm'], month=res['Tháng'], day=1))
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(res['Date'], res['Dự Báo Chốt'], 'o-', color='#d62728', linewidth=3, label='DỰ BÁO CHÍNH THỨC')
                ax.plot(res['Date'], res['NN'], '--', color='blue', alpha=0.3, label='Neural Network (Xu hướng tăng)')
                ax.plot(res['Date'], res['RF'], '--', color='green', alpha=0.3, label='Random Forest (Bảo thủ)')
                
                if 'Thực Tế' in res.columns:
                    mask = res['Thực Tế'].notnull()
                    ax.plot(res.loc[mask, 'Date'], res.loc[mask, 'Thực Tế'], 'ko', label='Thực Tế')
                    
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.error("Lỗi đọc file Excel. Vui lòng kiểm tra lại định dạng.")
