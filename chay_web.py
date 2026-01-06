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
import calendar

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
# 1. CẤU HÌNH NGÀY NGHỈ (DẠNG BẢNG CHO USER TỰ SỬA)
# ==============================================================================
# Dữ liệu mặc định (Làm mẫu)
DEFAULT_HOLIDAYS = [
    {"Năm": 2023, "Tháng": 1, "Số ngày nghỉ lễ": 8, "Ghi chú": "Tết Quý Mão"},
    {"Năm": 2023, "Tháng": 4, "Số ngày nghỉ lễ": 1, "Ghi chú": "Giỗ tổ"},
    {"Năm": 2023, "Tháng": 5, "Số ngày nghỉ lễ": 2, "Ghi chú": "30/4 - 1/5"},
    {"Năm": 2023, "Tháng": 9, "Số ngày nghỉ lễ": 2, "Ghi chú": "Quốc khánh"},
    
    {"Năm": 2024, "Tháng": 1, "Số ngày nghỉ lễ": 1, "Ghi chú": "Tết Dương"},
    {"Năm": 2024, "Tháng": 2, "Số ngày nghỉ lễ": 7, "Ghi chú": "Tết Giáp Thìn"},
    {"Năm": 2024, "Tháng": 4, "Số ngày nghỉ lễ": 3, "Ghi chú": "Giỗ tổ + 30/4"},
    {"Năm": 2024, "Tháng": 5, "Số ngày nghỉ lễ": 1, "Ghi chú": "1/5"},
    {"Năm": 2024, "Tháng": 9, "Số ngày nghỉ lễ": 2, "Ghi chú": "Quốc khánh"},

    {"Năm": 2025, "Tháng": 1, "Số ngày nghỉ lễ": 7, "Ghi chú": "Tết Ất Tỵ"},
    {"Năm": 2025, "Tháng": 2, "Số ngày nghỉ lễ": 2, "Ghi chú": "Mùng 4,5 Tết"},
    {"Năm": 2025, "Tháng": 4, "Số ngày nghỉ lễ": 2, "Ghi chú": "Giỗ tổ + 30/4"},
    {"Năm": 2025, "Tháng": 5, "Số ngày nghỉ lễ": 1, "Ghi chú": "1/5"},
    {"Năm": 2025, "Tháng": 9, "Số ngày nghỉ lễ": 2, "Ghi chú": "Quốc khánh"},

    {"Năm": 2026, "Tháng": 1, "Số ngày nghỉ lễ": 1, "Ghi chú": "Chỉ nghỉ Tết Dương (Tải cao)"},
    {"Năm": 2026, "Tháng": 2, "Số ngày nghỉ lễ": 5, "Ghi chú": "Tết Bính Ngọ"},
]

# Hàm đếm T7/CN
def dem_ngay_nghi_cuoi_tuan(year, month):
    num_days = calendar.monthrange(year, month)[1]
    saturdays, sundays = 0, 0
    for day in range(1, num_days + 1):
        weekday = calendar.weekday(year, month, day)
        if weekday == 5: saturdays += 1
        elif weekday == 6: sundays += 1
    return saturdays, sundays

# ==============================================================================
# 2. MODULE AI (GIỮ NGUYÊN)
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
        if matches: return max(min(float(matches[0]), 50.0), -50.0)
        return 0.0
    except: return 0.0

def xu_ly_du_lieu_dinh_tinh(api_key, input_data):
    if not api_key: return 0.0, "⚠️ Chưa nhập API Key.", "Thủ công"
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
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (f"Đọc tin: '{text_data[:3000]}'. Xác định % tăng/giảm phụ tải điện. "
                  "Trả về: SỐ | LÝ DO. Ví dụ: -1.5 | Giảm 1.5%")
        response = model.generate_content(prompt)
        res = response.text.strip()
        if "|" in res:
            parts = res.split("|")
            val = trich_xuat_so(parts[0]) 
            return val, f"{status}✅ Xong!", parts[1].strip()
        val = trich_xuat_so(res)
        return val, f"{status}✅ Xong (Tự bắt số)!", res
    except Exception as e:
        if "429" in str(e): return 0.0, "⚠️ Hết hạn mức AI.", "Hết Quota"
        return 0.0, f"❌ Lỗi AI: {str(e)[:50]}...", "Lỗi"

# ==============================================================================
# 3. XỬ LÝ FILE & TẠO ĐẶC TRƯNG
# ==============================================================================
def chuan_hoa_ten_cot(df):
    if df is None: return None
    df.columns = df.columns.astype(str).str.strip()
    col_map = {
        'month': 'Tháng', 'thang': 'Tháng', 'tháng': 'Tháng',
        'year': 'Năm', 'nam': 'Năm', 'năm': 'Năm',
        'tổng thương phẩm': 'Tổng thương phẩm', 'tong thuong pham': 'Tổng thương phẩm',
        'nhiệt độ tb': 'Nhiệt độ TB', 'nhiet do tb': 'Nhiệt độ TB',
        'độ ẩm': 'Độ ẩm', 'do am': 'Độ ẩm', 'số ngày': 'Số ngày', 'so ngay': 'Số ngày',
        'số ngày nghỉ': 'So_Ngay_Nghi', 'so ngay nghi': 'So_Ngay_Nghi' # Hỗ trợ nếu file có sẵn
    }
    new_cols = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in col_map: new_cols[col] = col_map[col_lower]
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
                errors.append(f"❌ Cột '{col}' có ô Trống.")
            if (df[col] == 0).any():
                errors.append(f"❌ Cột '{col}' bằng 0.")
    if errors:
        st.error(f"⚠️ Lỗi dữ liệu file {ten_file}:")
        for e in errors: st.write(e)
        st.stop()

# --- TẠO ĐẶC TRƯNG VỚI BẢNG LỊCH NGƯỜI DÙNG NHẬP ---
def tao_dac_trung(df, holidays_map):
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3,4,5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6,7,8,9,10,11] else 0)
    
    def get_calendar_info(row):
        y, m = int(row['Năm']), int(row['Tháng'])
        t7, cn = dem_ngay_nghi_cuoi_tuan(y, m)
        # Lấy từ bảng người dùng nhập, nếu không có thì bằng 0
        le_tet = holidays_map.get((y, m), 0)
        return pd.Series([t7, cn, le_tet])

    df[['So_Ngay_T7', 'So_Ngay_CN', 'So_Ngay_Le_Tet']] = df.apply(get_calendar_info, axis=1)
    df['Bien_Ngoai_Sinh'] = 0
    return df

# ==============================================================================
# 3. CHẠY DỰ BÁO
# ==============================================================================
def chay_mo_hinh_goc(df_train, df_input, holidays_map, seed=42):
    # Pass holidays_map vào hàm tạo đặc trưng
    df_train = tao_dac_trung(df_train.copy(), holidays_map)
    df_input = tao_dac_trung(df_input.copy(), holidays_map)
    
    start_year = df_train['Năm'].min()
    def create_time_index(row): return (row['Năm'] - start_year) * 12 + row['Tháng']
    
    df_train['Time_Index'] = df_train.apply(create_time_index, axis=1)
    df_input['Time_Index'] = df_input.apply(create_time_index, axis=1)

    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 
                'So_Ngay_T7', 'So_Ngay_CN', 'So_Ngay_Le_Tet', # <- Quan trọng
                'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_cols = [c for c in features if c in df_train.columns and c in df_input.columns]
    target = 'Tổng thương phẩm'
    
    data_train = df_train.dropna(subset=valid_cols + [target])
    X_train = data_train[valid_cols]
    y_train = data_train[target]
    
    y_train_log = np.log1p(y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    df_pred = df_input.copy()
    X_pred = df_pred[valid_cols].fillna(0)
    
    # NN
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', alpha=0.1, max_iter=5000, random_state=seed)
    nn.fit(X_train_scaled, y_train_log)
    pred_nn = np.expm1(nn.predict(scaler.transform(X_pred)))
    
    # RF/XGB (Detrend)
    trend_model = LinearRegression()
    trend_model.fit(data_train[['Time_Index']], y_train_log)
    trend_train = trend_model.predict(data_train[['Time_Index']])
    trend_future = trend_model.predict(df_pred[['Time_Index']])
    y_residual = y_train_log - trend_train
    
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_residual)
    pred_rf = np.expm1(rf.predict(X_pred) + trend_future)
    
    xg = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xg.fit(X_train, y_residual)
    pred_xg = np.expm1(xg.predict(X_pred) + trend_future)
    
    return pred_nn, pred_rf, pred_xg

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    api_key = st.text_input("API Key (Cho AI)", type="password")
    st.markdown("---")
    
    # --- BẢNG NHẬP LIỆU NGÀY NGHỈ ---
    st.write("### 📅 Cập nhật Lịch Nghỉ Lễ")
    st.info("Nhập số ngày nghỉ Lễ/Tết vào bảng dưới. Bấm (+) để thêm năm/tháng mới.")
    
    df_default = pd.DataFrame(DEFAULT_HOLIDAYS)
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
    
    # Chuyển đổi bảng nhập liệu thành Dictionary để code dùng
    # Key: (Năm, Tháng) -> Value: Số ngày nghỉ
    USER_HOLIDAYS_MAP = dict(zip(zip(edited_df['Năm'], edited_df['Tháng']), edited_df['Số ngày nghỉ lễ']))
    
    st.markdown("---")
    seed_val = st.number_input("Random Seed", value=42)
    if st.button("🗑️ Xóa Cache"):
        st.cache_data.clear()
        st.rerun()

col1, col2 = st.columns(2)
with col1: f_train = st.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
with col2: f_input = st.file_uploader("2. File Dự Báo (Input)", type=['xlsx', 'xls'])

st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# --- AI ---
st.subheader("1️⃣ Phân Tích Thông Tin")
c1, c2 = st.columns([2, 1])
if 'detected_months' not in st.session_state: st.session_state.detected_months = []
with c1: text_data = st.text_area("Link/Tin tức:", height=80)
with c2:
    if f_input:
        try:
            df_temp = ultra_scan_read_excel(f_input)
            if df_temp is not None:
                st.session_state.detected_months = sorted(list(set(zip(df_temp['Năm'], df_temp['Tháng']))))
        except: pass
    if st.button("Phân Tích AI"):
        with st.spinner("Đang đọc..."):
            val, log, reason = xu_ly_du_lieu_dinh_tinh(api_key, text_data)
        if "Lỗi" in log: st.warning(log)
        else:
            st.success(log)
            st.session_state.temp_score = val
            st.session_state.temp_reason = reason

# --- KỊCH BẢN ---
if st.session_state.detected_months:
    c_a, c_b = st.columns(2)
    with c_a:
        months_str = [f"Tháng {m}/{y}" for y, m in st.session_state.detected_months]
        selected = st.multiselect("Tháng áp dụng:", months_str, default=months_str)
    with c_b:
        cur_val = st.session_state.get('temp_score', 0.0)
        final_pct = st.number_input("Tăng/Giảm (%):", value=float(cur_val), step=0.1)
    if st.button("💾 Lưu Kịch Bản"):
        temp = {}
        for s in selected:
            m = int(s.split('/')[0].replace('Tháng ', ''))
            y = int(s.split('/')[1])
            temp[(y, m)] = (final_pct, st.session_state.get('temp_reason', 'Thủ công'))
        st.session_state.param_dict = temp
        st.success("Đã lưu!")

st.write("---")

# --- DỰ BÁO ---
if f_train and f_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        with st.spinner("Đang tính toán với Lịch Nghỉ Của Bạn..."):
            df_train = ultra_scan_read_excel(f_train)
            df_input = ultra_scan_read_excel(f_input)
            
            if df_train is not None and df_input is not None:
                kiem_tra_chat_luong(df_train, "Lịch Sử")
                kiem_tra_chat_luong(df_input, "Dự Báo")
                
                # Truyền USER_HOLIDAYS_MAP vào hàm chạy mô hình
                pred_nn, pred_rf, pred_xg = chay_mo_hinh_goc(df_train, df_input, USER_HOLIDAYS_MAP, seed_val)
                
                res = df_input[['Năm', 'Tháng']].copy()
                
                # Kiểm tra lại số ngày nghỉ đã dùng
                df_check = tao_dac_trung(df_input.copy(), USER_HOLIDAYS_MAP)
                res['Ngày nghỉ Lễ/Tết'] = df_check['So_Ngay_Le_Tet']
                
                res['Neural Network'] = pred_nn
                res['Random Forest'] = pred_rf
                res['XGBoost'] = pred_xg

                def apply_adj(row):
                    param = st.session_state.param_dict.get((row['Năm'], row['Tháng']), (0.0, ""))
                    factor = 1.0 + (param[0] / 100.0)
                    return row['Neural Network']*factor, row['Random Forest']*factor, row['XGBoost']*factor, param[0], param[1]

                adj_data = res.apply(apply_adj, axis=1, result_type='expand')
                res['Neural Network'] = adj_data[0]
                res['Random Forest'] = adj_data[1]
                res['XGBoost'] = adj_data[2]
                res['%'] = adj_data[3]
                res['Ghi chú'] = adj_data[4]

                if 'Tổng thương phẩm' in df_train.columns:
                    actual = df_train[['Năm', 'Tháng', 'Tổng thương phẩm']]
                    res = pd.merge(res, actual, on=['Năm', 'Tháng'], how='left')
                    res.rename(columns={'Tổng thương phẩm': 'Thực Tế'}, inplace=True)
                
                st.subheader("📊 Kết Quả & Ngày Nghỉ (Theo cấu hình của bạn)")
                cols = ['Tháng', 'Năm', 'Thực Tế', 'Ngày nghỉ Lễ/Tết', 'Neural Network', 'Random Forest', 'XGBoost', '%', 'Ghi chú']
                cols = [c for c in cols if c in res.columns]
                
                st.dataframe(res[cols].style.format({
                    'Thực Tế': '{:,.0f}', 'Neural Network': '{:,.0f}', 
                    'Random Forest': '{:,.0f}', 'XGBoost': '{:,.0f}',
                    '%': '{:+.1f}%', 'Ngày nghỉ Lễ/Tết': '{:.0f}'
                }), use_container_width=True)
                
                res['Date'] = pd.to_datetime(dict(year=res['Năm'], month=res['Tháng'], day=1))
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(res['Date'], res['Neural Network'], 'o-', color='blue', label='NN')
                ax.plot(res['Date'], res['Random Forest'], 's--', color='green', label='RF')
                ax.plot(res['Date'], res['XGBoost'], '^-.', color='purple', label='XGB')
                if 'Thực Tế' in res.columns:
                    mask = res['Thực Tế'].notnull()
                    ax.plot(res.loc[mask, 'Date'], res.loc[mask, 'Thực Tế'], 'ko', label='Thực Tế')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
