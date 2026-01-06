import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import requests
from bs4 import BeautifulSoup
import re
import time

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
# 2. XỬ LÝ FILE (CÓ KIỂM SOÁT LỖI)
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

def kiem_tra_chat_luong(df, ten_file):
    errors = []
    required = ['Tháng', 'Năm']
    for col in required:
        if col not in df.columns:
            st.error(f"❌ File {ten_file} thiếu cột '{col}'")
            st.stop()
            
    check_cols = ['Nhiệt độ TB', 'Độ ẩm', 'Số ngày']
    for col in check_cols:
        if col in df.columns:
            if df[col].isnull().any():
                rows = df[df[col].isnull()].index + 2
                errors.append(f"❌ Cột '{col}' bị TRỐNG ở dòng: {rows.tolist()}")
            if (df[col] == 0).any():
                rows = df[df[col] == 0].index + 2
                errors.append(f"❌ Cột '{col}' bằng 0 (Vô lý) ở dòng: {rows.tolist()}")
                
    if errors:
        st.error(f"⚠️ Dữ liệu file {ten_file} không đạt chuẩn. Vui lòng sửa file Excel:")
        for e in errors: st.write(e)
        st.stop()

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
# 3. CHẠY DỰ BÁO (LOGIC CHUẨN: SCALING + DETRENDING)
# ==============================================================================
@st.cache_data(show_spinner=False)
def chay_mo_hinh_goc(df_train, df_input, seed=42):
    # 1. Chuẩn bị dữ liệu
    df_train = tao_dac_trung(df_train.copy())
    df_input = tao_dac_trung(df_input.copy())
    
    # Tạo Time Index để bắt xu hướng (Trend)
    start_year = df_train['Năm'].min()
    def create_time_index(row):
        return (row['Năm'] - start_year) * 12 + row['Tháng']
    
    df_train['Time_Index'] = df_train.apply(create_time_index, axis=1)
    df_input['Time_Index'] = df_input.apply(create_time_index, axis=1)

    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_cols = [c for c in features if c in df_train.columns and c in df_input.columns]
    target = 'Tổng thương phẩm'
    
    # Train Data
    data_train = df_train.dropna(subset=valid_cols + [target])
    X_train = data_train[valid_cols]
    y_train = data_train[target]
    
    # --- QUAN TRỌNG: CHUYỂN ĐỔI ĐƠN VỊ VỀ TRIỆU kWh ---
    # Việc này giúp cả Neural Net và Linear Regression (Trend) hoạt động chính xác
    y_train_scaled = y_train / 1_000_000.0
    
    # Scale X cho NN
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    
    # Input Data
    df_pred = df_input.copy()
    X_pred = df_pred[valid_cols].fillna(0)
    
    # ---------------------------------------------------------
    # MODEL 1: NEURAL NETWORK (CẤU TRÚC 10-15-10 CHUẨN)
    # ---------------------------------------------------------
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), max_iter=10000, random_state=seed)
    # Train trên dữ liệu đã chia 1 triệu
    nn.fit(X_train_std, y_train_scaled)
    pred_nn_scaled = nn.predict(scaler.transform(X_pred))
    
    # ---------------------------------------------------------
    # MODEL 2 & 3: RF & XGB (CÓ DETRENDING - BẮT XU HƯỚNG)
    # ---------------------------------------------------------
    # B1: Tìm xu hướng trên dữ liệu đã chia 1 triệu
    trend_model = LinearRegression()
    trend_model.fit(data_train[['Time_Index']], y_train_scaled)
    
    trend_train = trend_model.predict(data_train[['Time_Index']])
    trend_future = trend_model.predict(df_pred[['Time_Index']])
    
    # B2: Trừ xu hướng (Chỉ còn dao động mùa vụ)
    y_residual = y_train_scaled - trend_train
    
    # B3: Train RF/XGB trên phần dư
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_residual)
    
    xg = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xg.fit(X_train, y_residual)
    
    # B4: Dự báo = Phần dư + Xu hướng tương lai
    pred_rf_scaled = rf.predict(X_pred) + trend_future
    pred_xg_scaled = xg.predict(X_pred) + trend_future
    
    # --- NHÂN NGƯỢC LẠI 1 TRIỆU ĐỂ TRẢ VỀ SỐ GỐC ---
    return pred_nn_scaled * 1_000_000, pred_rf_scaled * 1_000_000, pred_xg_scaled * 1_000_000

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    api_key = st.text_input("API Key (Cho AI)", type="password")
    
    st.markdown("---")
    st.write("### 🛠️ Cài đặt chạy")
    seed_val = st.number_input("Random Seed (Mặc định 42)", value=42)
    if st.button("🗑️ Xóa Cache (Reset App)"):
        st.cache_data.clear()
        st.rerun()

col1, col2 = st.columns(2)
with col1: f_train = st.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
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
        with st.spinner("Đang tính toán (Đã xử lý số lớn + Tách xu hướng)..."):
            # 1. Đọc file
            df_train = ultra_scan_read_excel(f_train)
            df_input = ultra_scan_read_excel(f_input)
            
            if df_train is not None and df_input is not None:
                # 2. KIỂM TRA LỖI
                kiem_tra_chat_luong(df_train, "Lịch Sử")
                kiem_tra_chat_luong(df_input, "Dự Báo")
                
                # 3. Chạy 3 mô hình riêng biệt
                pred_nn, pred_rf, pred_xg = chay_mo_hinh_goc(df_train, df_input, seed_val)
                
                # 4. Tạo DataFrame kết quả
                res = df_input[['Năm', 'Tháng']].copy()
                res['Neural Network'] = pred_nn
                res['Random Forest'] = pred_rf
                res['XGBoost'] = pred_xg

                # 5. Áp dụng điều chỉnh % cho TỪNG CỘT RIÊNG
                def apply_adj(row):
                    param = st.session_state.param_dict.get((row['Năm'], row['Tháng']), (0.0, ""))
                    factor = 1.0 + (param[0] / 100.0)
                    
                    # Nhân hệ số riêng
                    nn_adj = row['Neural Network'] * factor
                    rf_adj = row['Random Forest'] * factor
                    xgb_adj = row['XGBoost'] * factor
                    
                    return nn_adj, rf_adj, xgb_adj, param[0], param[1]

                adj_data = res.apply(apply_adj, axis=1, result_type='expand')
                res['Neural Network'] = adj_data[0]
                res['Random Forest'] = adj_data[1]
                res['XGBoost'] = adj_data[2]
                res['Tác Động %'] = adj_data[3]
                res['Ghi chú'] = adj_data[4]

                # 6. Merge thực tế
                if 'Tổng thương phẩm' in df_train.columns:
                    actual = df_train[['Năm', 'Tháng', 'Tổng thương phẩm']]
                    res = pd.merge(res, actual, on=['Năm', 'Tháng'], how='left')
                    res.rename(columns={'Tổng thương phẩm': 'Thực Tế'}, inplace=True)
                
                # --- HIỂN THỊ KẾT QUẢ TÁCH BẠCH ---
                st.subheader("📊 Bảng Kết Quả Chi Tiết (Từng Phương Pháp)")
                
                cols_display = {
                    'Tháng': 'Tháng', 'Năm': 'Năm',
                    'Thực Tế': 'Thực Tế',
                    'Neural Network': 'Neural Network',
                    'Random Forest': 'Random Forest',
                    'XGBoost': 'XGBoost',
                    'Tác Động %': 'Điều Chỉnh (%)',
                    'Ghi chú': 'Ghi Chú'
                }
                
                # Chỉ lấy các cột có dữ liệu
                cols_to_use = [c for c in cols_display.keys() if c in res.columns]
                df_show = res[cols_to_use].rename(columns=cols_display)
                
                df_show['Điều Chỉnh (%)'] = df_show['Điều Chỉnh (%)'].apply(lambda x: f"{x:+.1f}%" if x!=0 else "-")
                
                # Format số liệu
                format_dict = {
                    'Thực Tế': '{:,.0f}', 
                    'Neural Network': '{:,.0f}',
                    'Random Forest': '{:,.0f}',
                    'XGBoost': '{:,.0f}'
                }
                
                st.dataframe(df_show.style.format(format_dict), use_container_width=True)
                
                st.info("✅ **Đã xử lý:** Code tự động chuyển đổi đơn vị về 'Triệu kWh' để tính toán, sau đó nhân ngược lại. Điều này giúp các mô hình (đặc biệt là Neural Network và XGBoost) hoạt động ổn định và chính xác với con số lớn.")

                # BIỂU ĐỒ 3 ĐƯỜNG RIÊNG BIỆT
                st.subheader("📈 Biểu Đồ So Sánh 3 Phương Pháp")
                res['Date'] = pd.to_datetime(dict(year=res['Năm'], month=res['Tháng'], day=1))
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Vẽ 3 đường riêng biệt
                ax.plot(res['Date'], res['Neural Network'], 'o-', color='blue', linewidth=2, label='Neural Network')
                ax.plot(res['Date'], res['Random Forest'], 's--', color='green', linewidth=1.5, label='Random Forest (Có Trend)')
                ax.plot(res['Date'], res['XGBoost'], '^-.', color='purple', linewidth=1.5, label='XGBoost (Có Trend)')
                
                if 'Thực Tế' in res.columns:
                    mask = res['Thực Tế'].notnull()
                    ax.plot(res.loc[mask, 'Date'], res.loc[mask, 'Thực Tế'], 'ko', label='Thực Tế', zorder=10, linewidth=2.5)
                    
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
            else:
                st.error("Lỗi đọc file Excel. Vui lòng kiểm tra lại định dạng.")
