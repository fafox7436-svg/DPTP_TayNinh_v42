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

# >>> THÊM ĐOẠN NÀY VÀO <<<
try:
    # Vì file ảnh đã upload lên cùng thư mục với file code trên GitHub
    st.image("image_1.png", width=500) 
except:
    st.write("Đang tải logo...")
# >>> KẾT THÚC ĐOẠN THÊM <<<

st.title("HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN TỈNH TÂY NINH")
st.markdown("---")

# --- KIỂM TRA THƯ VIỆN AI ---
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except: 
    HAS_GEMINI = False

# ==============================================================================
# 1. CẤU HÌNH NGÀY NGHỈ (DỮ LIỆU MẶC ĐỊNH)
# ==============================================================================
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

    {"Năm": 2026, "Tháng": 1, "Số ngày nghỉ lễ": 1, "Ghi chú": "Tết Dương (Ít nghỉ -> Tải cao)"},
    {"Năm": 2026, "Tháng": 2, "Số ngày nghỉ lễ": 5, "Ghi chú": "Tết Bính Ngọ"},
    {"Năm": 2026, "Tháng": 4, "Số ngày nghỉ lễ": 1, "Ghi chú": "Giỗ Tổ"},
    {"Năm": 2026, "Tháng": 5, "Số ngày nghỉ lễ": 2, "Ghi chú": "30/4 - 1/5"},
]

def dem_ngay_nghi_cuoi_tuan(year, month):
    num_days = calendar.monthrange(year, month)[1]
    saturdays, sundays = 0, 0
    for day in range(1, num_days + 1):
        weekday = calendar.weekday(year, month, day)
        if weekday == 5: saturdays += 1
        elif weekday == 6: sundays += 1
    return saturdays, sundays

# ==============================================================================
# 2. MODULE AI (CHỈ ĐÁNH GIÁ, KHÔNG CAN THIỆP)
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
        
        # Tự động tìm model
        found_model = "gemini-pro"
        try:
            models = genai.list_models()
            for m in models:
                if 'generateContent' in m.supported_generation_methods:
                    if 'flash' in m.name: 
                        found_model = m.name
                        break
                    found_model = m.name
        except: pass
        
        model = genai.GenerativeModel(found_model)
        # Prompt nhấn mạnh việc so sánh với cùng kỳ năm trước
        prompt = (f"Đọc thông tin sau: '{text_data[:3000]}'. "
                  "Hãy đánh giá xem phụ tải điện tháng này sẽ TĂNG hay GIẢM bao nhiêu % so với CÙNG KỲ NĂM TRƯỚC. "
                  "Chỉ đưa ra con số ước lượng dựa trên tác động (thời tiết, kinh tế...). "
                  "Trả về định dạng: SỐ | LÝ DO NGẮN GỌN. Ví dụ: +5.5 | Nắng nóng hơn năm ngoái.")
        
        response = model.generate_content(prompt)
        res = response.text.strip()
        
        if "|" in res:
            parts = res.split("|")
            val = trich_xuat_so(parts[0]) 
            return val, f"{status}✅ AI Đã xong ({found_model})", parts[1].strip()
            
        val = trich_xuat_so(res)
        if val != 0.0: return val, f"{status}✅ AI Đã xong (Tự bắt số)!", res
        return 0.0, f"⚠️ AI không tìm thấy số liệu cụ thể.", res
        
    except Exception as e:
        if "429" in str(e): return 0.0, "⚠️ Hết hạn mức AI.", "Hết Quota"
        return 0.0, f"❌ Lỗi AI: {str(e)[:50]}...", "Lỗi"

# ==============================================================================
# 3. XỬ LÝ FILE
# ==============================================================================
def chuan_hoa_ten_cot(df):
    if df is None: return None
    df.columns = df.columns.astype(str).str.strip()
    col_map = {
        'month': 'Tháng', 'thang': 'Tháng', 'tháng': 'Tháng',
        'year': 'Năm', 'nam': 'Năm', 'năm': 'Năm',
        'tổng thương phẩm': 'Tổng thương phẩm', 'tong thuong pham': 'Tổng thương phẩm', 'sản lượng': 'Tổng thương phẩm',
        'nhiệt độ tb': 'Nhiệt độ TB', 'nhiet do tb': 'Nhiệt độ TB',
        'độ ẩm': 'Độ ẩm', 'do am': 'Độ ẩm', 'số ngày': 'Số ngày', 'so ngay': 'Số ngày',
        'số ngày nghỉ': 'So_Ngay_Nghi', 'so ngay nghi': 'So_Ngay_Nghi'
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
            if df[col].isnull().any(): errors.append(f"❌ Cột '{col}' có ô Trống.")
            if (df[col] == 0).any(): errors.append(f"❌ Cột '{col}' bằng 0.")
    if errors:
        st.error(f"⚠️ Lỗi dữ liệu file {ten_file}:")
        for e in errors: st.write(e)
        st.stop()

def tao_dac_trung(df, holidays_map):
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3,4,5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6,7,8,9,10,11] else 0)
    
    def get_calendar_info(row):
        y, m = int(row['Năm']), int(row['Tháng'])
        t7, cn = dem_ngay_nghi_cuoi_tuan(y, m)
        le_tet = holidays_map.get((y, m), 0)
        return pd.Series([t7, cn, le_tet])

    df[['So_Ngay_T7', 'So_Ngay_CN', 'So_Ngay_Le_Tet']] = df.apply(get_calendar_info, axis=1)
    df['Bien_Ngoai_Sinh'] = 0
    return df

# ==============================================================================
# 3. CHẠY DỰ BÁO (LOGIC CHUẨN NHẤT)
# ==============================================================================
@st.cache_data(show_spinner=False)
def chay_mo_hinh_goc(df_train, df_input, holidays_map, seed=42):
    df_train = tao_dac_trung(df_train.copy(), holidays_map)
    df_input = tao_dac_trung(df_input.copy(), holidays_map)
    
    start_year = df_train['Năm'].min()
    def create_time_index(row): return (row['Năm'] - start_year) * 12 + row['Tháng']
    
    df_train['Time_Index'] = df_train.apply(create_time_index, axis=1)
    df_input['Time_Index'] = df_input.apply(create_time_index, axis=1)

    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 
                'So_Ngay_T7', 'So_Ngay_CN', 'So_Ngay_Le_Tet', 
                'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    
    valid_cols = [c for c in features if c in df_train.columns and c in df_input.columns]
    target = 'Tổng thương phẩm'
    
    data_train = df_train.dropna(subset=valid_cols + [target])
    X_train = data_train[valid_cols]
    y_train = data_train[target]
    
    # Log Transform
    y_train_log = np.log1p(y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    df_pred = df_input.copy()
    X_pred = df_pred[valid_cols].fillna(0)
    
    # NN (10-15-10)
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
    
    xg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, reg_alpha=0.1, random_state=42)
    xg.fit(X_train, y_residual)
    
    pred_rf_log = rf.predict(X_pred) + trend_future
    pred_xg_log = xg.predict(X_pred) + trend_future
    
    # Quy luật tháng 1 < tháng 12
    is_jan = df_pred['Tháng'] == 1
    if is_jan.any():
        pred_rf_log[is_jan] *= 0.995 
        pred_xg_log[is_jan] *= 0.995

    pred_rf = np.expm1(pred_rf_log)
    pred_xg = np.expm1(pred_xg_log)
    
    return pred_nn, pred_rf, pred_xg

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    api_key = st.text_input("API Key (Cho AI)", type="password")
    
    st.markdown("---")
    st.write("### 📅 Cập nhật Lịch Nghỉ Lễ")
    st.info("Nhập số ngày nghỉ vào đây. Mô hình sẽ dùng dữ liệu này để tính toán.")
    df_default = pd.DataFrame(DEFAULT_HOLIDAYS)
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
    USER_HOLIDAYS_MAP = dict(zip(zip(edited_df['Năm'], edited_df['Tháng']), edited_df['Số ngày nghỉ lễ']))
    
    st.markdown("---")
    if st.button("🗑️ Xóa Cache & Reset"):
        st.cache_data.clear()
        st.rerun()

col1, col2 = st.columns(2)
with col1: f_train = st.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
with col2: f_input = st.file_uploader("2. File Dự Báo (Input)", type=['xlsx', 'xls'])

st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# --- PHẦN QUAN TRỌNG: AI GỢI Ý - BẠN QUYẾT ĐỊNH ---
st.subheader("1️⃣ Tham khảo AI & Chốt phương án")
c1, c2 = st.columns([2, 1])
if 'detected_months' not in st.session_state: st.session_state.detected_months = []

with c1: 
    text_data = st.text_area("Dán link báo hoặc tin tức vào đây:", height=100, 
                             placeholder="Ví dụ: Dự báo nắng nóng gay gắt tháng tới, nhu cầu điện tăng cao so với cùng kỳ...")

with c2:
    if f_input:
        try:
            df_temp = ultra_scan_read_excel(f_input)
            if df_temp is not None:
                st.session_state.detected_months = sorted(list(set(zip(df_temp['Năm'], df_temp['Tháng']))))
        except: pass
    
    # Nút bấm chạy AI
    if st.button("🤖 AI Phân Tích Ngay"):
        with st.spinner("AI đang đọc và so sánh với cùng kỳ..."):
            val, log, reason = xu_ly_du_lieu_dinh_tinh(api_key, text_data)
        
        # Lưu kết quả AI vào session state riêng
        st.session_state.ai_suggestion_val = val
        st.session_state.ai_suggestion_reason = reason
        st.session_state.ai_log = log

    # Hiển thị kết quả AI (Chỉ để tham khảo)
    if 'ai_suggestion_val' in st.session_state:
        if st.session_state.ai_suggestion_val != 0:
            st.success(f"{st.session_state.ai_log}")
            st.metric(label="AI Đề Xuất (So với cùng kỳ)", value=f"{st.session_state.ai_suggestion_val}%")
            st.info(f"📝 **Lý do:** {st.session_state.ai_suggestion_reason}")
        else:
            st.warning("AI không tìm thấy con số cụ thể. Hãy tự nhập.")

# --- NGƯỜI DÙNG QUYẾT ĐỊNH ---
st.write("---")
st.write("### ✍️ CHỐT SỐ LIỆU")

if st.session_state.detected_months:
    c_a, c_b = st.columns(2)
    with c_a:
        months_str = [f"Tháng {m}/{y}" for y, m in st.session_state.detected_months]
        selected = st.multiselect("Chọn tháng áp dụng:", months_str, default=months_str)
    
    with c_b:
        # Ô nhập liệu này độc lập hoàn toàn với AI
        user_val = st.number_input("Nhập % bạn muốn Tăng/Giảm vào kết quả cuối cùng:", 
                                   value=0.0, step=0.1, help="Số dương là tăng, số âm là giảm")
        user_note = st.text_input("Ghi chú cho quyết định này:", value="Thủ công")
    
    if st.button("💾 LƯU QUYẾT ĐỊNH"):
        temp = {}
        for s in selected:
            m = int(s.split('/')[0].replace('Tháng ', ''))
            y = int(s.split('/')[1])
            # Lưu số của USER nhập, không phải số của AI
            temp[(y, m)] = (user_val, user_note)
        st.session_state.param_dict = temp
        st.success(f"Đã lưu: {user_val}% cho các tháng đã chọn!")

st.write("---")

# --- DỰ BÁO ---
if f_train and f_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        with st.spinner("Đang tính toán..."):
            df_train = ultra_scan_read_excel(f_train)
            df_input = ultra_scan_read_excel(f_input)
            
            if df_train is not None and df_input is not None:
                kiem_tra_chat_luong(df_train, "Lịch Sử")
                kiem_tra_chat_luong(df_input, "Dự Báo")
                
                # Chạy mô hình
                pred_nn, pred_rf, pred_xg = chay_mo_hinh_goc(df_train, df_input, USER_HOLIDAYS_MAP)
                
                res = df_input[['Năm', 'Tháng']].copy()
                df_check = tao_dac_trung(df_input.copy(), USER_HOLIDAYS_MAP)
                res['T7+CN'] = df_check['So_Ngay_T7'] + df_check['So_Ngay_CN']
                res['Lễ Tết'] = df_check['So_Ngay_Le_Tet']
                
                res['Neural Network'] = pred_nn
                res['Random Forest'] = pred_rf
                res['XGBoost'] = pred_xg

                # Áp dụng con số của USER
                def apply_adj(row):
                    # Lấy % từ dictionary đã lưu
                    param = st.session_state.param_dict.get((row['Năm'], row['Tháng']), (0.0, ""))
                    user_pct = param[0]
                    user_note = param[1]
                    
                    factor = 1.0 + (user_pct / 100.0)
                    
                    return row['Neural Network']*factor, row['Random Forest']*factor, row['XGBoost']*factor, user_pct, user_note

                adj_data = res.apply(apply_adj, axis=1, result_type='expand')
                res['Neural Network'] = adj_data[0]
                res['Random Forest'] = adj_data[1]
                res['XGBoost'] = adj_data[2]
                res['Điều Chỉnh (%)'] = adj_data[3]
                res['Ghi chú'] = adj_data[4]

                if 'Tổng thương phẩm' in df_train.columns:
                    actual = df_train[['Năm', 'Tháng', 'Tổng thương phẩm']]
                    res = pd.merge(res, actual, on=['Năm', 'Tháng'], how='left')
                    res.rename(columns={'Tổng thương phẩm': 'Thực Tế'}, inplace=True)
                
                st.subheader("📊 Kết Quả Dự Báo")
                cols = ['Tháng', 'Năm', 'Thực Tế', 'T7+CN', 'Lễ Tết', 'Neural Network', 'Random Forest', 'XGBoost', 'Điều Chỉnh (%)', 'Ghi chú']
                cols = [c for c in cols if c in res.columns]
                
                st.dataframe(res[cols].style.format({
                    'Thực Tế': '{:,.0f}', 'Neural Network': '{:,.0f}', 
                    'Random Forest': '{:,.0f}', 'XGBoost': '{:,.0f}',
                    'Điều Chỉnh (%)': '{:+.1f}%', 'T7+CN': '{:.0f}', 'Lễ Tết': '{:.0f}'
                }), use_container_width=True)
                
                res['Date'] = pd.to_datetime(dict(year=res['Năm'], month=res['Tháng'], day=1))
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(res['Date'], res['Neural Network'], 'o-', color='blue', label='NN (10-15-10)')
                ax.plot(res['Date'], res['Random Forest'], 's--', color='green', label='RF (Detrend)')
                ax.plot(res['Date'], res['XGBoost'], '^-.', color='purple', label='XGB (Detrend)')
                if 'Thực Tế' in res.columns:
                    mask = res['Thực Tế'].notnull()
                    ax.plot(res.loc[mask, 'Date'], res.loc[mask, 'Thực Tế'], 'ko', label='Thực Tế')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)


