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
import calendar
import warnings

warnings.filterwarnings('ignore')

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự Báo Phụ Tải Tây Ninh", layout="wide")

# ==============================================================================
# 🎨 GIAO DIỆN & CSS
# ==============================================================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #ffffff 0%, #cce6ff 100%); }
    h1, h2, h3, h4 { color: #003366 !important; font-family: 'Segoe UI', Tahoma, sans-serif; font-weight: 700; }
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    div[data-testid="stSelectbox"] label { color: #d63384 !important; font-weight: bold; font-size: 1.1em; }
    .author-subtitle { 
        color: #444; 
        font-size: 16px; 
        font-style: italic; 
        margin-top: -15px;
        margin-bottom: 10px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- BỐ CỤC TIÊU ĐỀ ---
col1, col2 = st.columns([8, 2], vertical_alignment="center")
with col1: 
    st.markdown("<h3>HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN TỈNH TÂY NINH</h3>", unsafe_allow_html=True)
    st.markdown('<p class="author-subtitle">Tác giả: Lê Minh Trí</p>', unsafe_allow_html=True)
with col2: 
    try: st.image("image_1.png", use_column_width=True)
    except: st.write("EVN SPC")
st.markdown("---")

# --- KIỂM TRA THƯ VIỆN AI ---
try: import google.generativeai as genai; HAS_GEMINI = True
except: HAS_GEMINI = False
try: from openai import OpenAI; HAS_OPENAI = True
except: HAS_OPENAI = False

# ==============================================================================
# 1. CẤU HÌNH NGÀY NGHỈ
# ==============================================================================
DEFAULT_HOLIDAYS = [
    {"Năm": 2023, "Tháng": 1, "Tết Âm": 7, "Lễ Nhỏ": 1}, {"Năm": 2023, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2023, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 3}, {"Năm": 2023, "Tháng": 9, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2024, "Tháng": 1, "Tết Âm": 0, "Lễ Nhỏ": 1}, {"Năm": 2024, "Tháng": 2, "Tết Âm": 7, "Lễ Nhỏ": 0},
    {"Năm": 2024, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 3}, {"Năm": 2024, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2024, "Tháng": 9, "Tết Âm": 0, "Lễ Nhỏ": 2}, {"Năm": 2025, "Tháng": 1, "Tết Âm": 7, "Lễ Nhỏ": 1},
    {"Năm": 2025, "Tháng": 2, "Tết Âm": 2, "Lễ Nhỏ": 0}, {"Năm": 2025, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2025, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 2}, {"Năm": 2025, "Tháng": 9, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2026, "Tháng": 1, "Tết Âm": 0, "Lễ Nhỏ": 1}, {"Năm": 2026, "Tháng": 2, "Tết Âm": 7, "Lễ Nhỏ": 0},
    {"Năm": 2026, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 3}, {"Năm": 2026, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2026, "Tháng": 9, "Tết Âm": 0, "Lễ Nhỏ": 2}, {"Năm": 2027, "Tháng": 1, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2027, "Tháng": 2, "Tết Âm": 7, "Lễ Nhỏ": 0}, {"Năm": 2027, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2027, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 1},
]

# ==============================================================================
# 2. XỬ LÝ DỮ LIỆU & ĐẶC TRƯNG CẬP NHẬT
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

def xu_ly_du_lieu_dinh_tinh(api_key, input_data, target_model_name, provider):
    if not api_key: return 0.0, "⚠️ Chưa nhập API Key.", "Thủ công"
    text_data = input_data
    status = ""
    if input_data.strip().startswith("http"):
        with st.spinner("Đang đọc bài báo..."):
            extracted, msg = lay_noi_dung_tu_link(input_data)
            if extracted: 
                text_data = extracted; status = msg + "\n"
            else: return 0.0, msg, ""

    prompt = (f"Đọc thông tin sau: '{text_data[:3000]}'. "
              "Hãy đánh giá xem phụ tải điện tháng này sẽ TĂNG hay GIẢM bao nhiêu % so với CÙNG KỲ NĂM TRƯỚC. "
              "Chỉ đưa ra con số ước lượng dựa trên tác động (thời tiết, kinh tế...). "
              "Trả về định dạng: SỐ | LÝ DO NGẮN GỌN. Ví dụ: +5.5 | Nắng nóng hơn năm ngoái.")
    res = ""
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(target_model_name)
            response = model.generate_content(prompt)
            res = response.text.strip()
        elif provider == "OpenAI ChatGPT":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=target_model_name,
                messages=[{"role": "system", "content": "Bạn là chuyên gia dự báo phụ tải điện."},
                          {"role": "user", "content": prompt}],
                temperature=0.5
            )
            res = response.choices[0].message.content.strip()

        if "|" in res:
            parts = res.split("|")
            val = trich_xuat_so(parts[0]) 
            return val, f"{status}✅ AI Đã xong ({provider} - {target_model_name})", parts[1].strip()
        val = trich_xuat_so(res)
        if val != 0.0: return val, f"{status}✅ AI Đã xong (Tự bắt số)!", res
        return 0.0, f"⚠️ AI không tìm thấy số.", res
    except Exception as e:
        if "429" in str(e): return 0.0, "⚠️ Hết hạn mức (Quota).", "Hết Quota"
        return 0.0, f"❌ Lỗi AI: {str(e)[:50]}...", "Lỗi"

def chuan_hoa_ten_cot(df):
    if df is None: return None
    df.columns = df.columns.astype(str).str.strip()
    col_map = {
        'month': 'Tháng', 'thang': 'Tháng', 'tháng': 'Tháng', 'year': 'Năm', 'nam': 'Năm', 'năm': 'Năm',
        'tổng thương phẩm': 'Tổng thương phẩm', 'tong thuong pham': 'Tổng thương phẩm', 'sản lượng': 'Tổng thương phẩm',
        'nhiệt độ tb': 'Nhiệt độ TB', 'nhiet do tb': 'Nhiệt độ TB', 'độ ẩm': 'Độ ẩm', 'do am': 'Độ ẩm', 'số ngày': 'Số ngày'
    }
    new_cols = {}
    for col in df.columns:
        if col.lower() in col_map: new_cols[col] = col_map[col.lower()]
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
    for col in ['Tháng', 'Năm']:
        if col not in df.columns: st.error(f"❌ File {ten_file} thiếu cột '{col}'"); st.stop()
    for col in ['Nhiệt độ TB', 'Độ ẩm', 'Số ngày']:
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
        num_days = calendar.monthrange(y, m)[1]
        t2 = t7 = cn = 0
        for day in range(1, num_days + 1):
            wd = calendar.weekday(y, m, day)
            if wd == 0: t2 += 1
            elif wd == 5: t7 += 1
            elif wd == 6: cn += 1
        tet_am, le_nho = holidays_map.get((y, m), (0, 0))
        so_ngay_thuong = num_days - t2 - t7 - cn - tet_am - le_nho
        if so_ngay_thuong < 0: so_ngay_thuong = 0
        return pd.Series([so_ngay_thuong, t2, t7, cn, tet_am, le_nho])
        
    df[['So_Ngay_Thuong', 'So_Ngay_T2', 'So_Ngay_T7', 'So_Ngay_CN', 'So_Ngay_Tet_Am', 'So_Ngay_Le_Nho']] = df.apply(get_calendar_info, axis=1)
    df['Bien_Ngoai_Sinh'] = 0
    return df

# ==============================================================================
# 3. CHẠY DỰ BÁO CỐT LÕI AI (TỐI ƯU HÓA THEO NGÀY CƠ SỞ TƯƠNG ĐƯƠNG)
# ==============================================================================
@st.cache_data(show_spinner=False)
def chay_mo_hinh_goc(df_train, df_input, holidays_map, k_dict, seed=42):
    df_train = tao_dac_trung(df_train.copy(), holidays_map)
    df_input = tao_dac_trung(df_input.copy(), holidays_map)
    
    start_year = df_train['Năm'].min()
    def create_time_index(row): return (row['Năm'] - start_year) * 12 + row['Tháng']
    
    df_train['Time_Index'] = df_train.apply(create_time_index, axis=1)
    df_input['Time_Index'] = df_input.apply(create_time_index, axis=1)

    # Tính TỔNG NGÀY CƠ SỞ TƯƠNG ĐƯƠNG dựa trên hệ số k
    def calc_equiv_days(df):
        return (df['So_Ngay_Thuong'] * 1.0 + 
                df['So_Ngay_T2'] * k_dict['T2'] + 
                df['So_Ngay_T7'] * k_dict['T7'] + 
                df['So_Ngay_CN'] * k_dict['CN'] + 
                df['So_Ngay_Le_Nho'] * k_dict['Le'] + 
                df['So_Ngay_Tet_Am'] * k_dict['Tet'])
                
    df_train['Ngay_Tuong_Duong'] = calc_equiv_days(df_train)
    df_input['Ngay_Tuong_Duong'] = calc_equiv_days(df_input)

    # Chạy mô hình học trên Sản lượng Ngày Tương đương
    features = ['Tháng', 'Năm', 'Nhiệt độ TB', 'Độ ẩm', 'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_cols = [c for c in features if c in df_train.columns and c in df_input.columns]
    
    data_train = df_train.dropna(subset=valid_cols + ['Tổng thương phẩm'])
    X_train = data_train[valid_cols]
    
    y_train_base_daily = data_train['Tổng thương phẩm'] / data_train['Ngay_Tuong_Duong']
    y_train_log = np.log1p(y_train_base_daily)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred = df_input[valid_cols].fillna(0)
    
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', alpha=0.1, max_iter=5000, random_state=seed)
    nn.fit(X_train_scaled, y_train_log)
    
    trend_model = LinearRegression()
    trend_model.fit(data_train[['Time_Index']], y_train_log)
    trend_future = trend_model.predict(df_input[['Time_Index']])
    y_residual = y_train_log - trend_model.predict(data_train[['Time_Index']])
    
    rf = RandomForestRegressor(n_estimators=200, random_state=seed)
    rf.fit(X_train, y_residual)
    
    xg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, reg_alpha=0.1, random_state=seed)
    xg.fit(X_train, y_residual)
    
    # Dự báo ra SẢN LƯỢNG NGÀY THƯỜNG
    pred_nn_base = np.expm1(nn.predict(scaler.transform(X_pred)))
    pred_rf_base = np.expm1(rf.predict(X_pred) + trend_future)
    pred_xg_base = np.expm1(xg.predict(X_pred) + trend_future)
    pred_trend_base = np.expm1(trend_future)
    
    # Quy ra tổng tháng 
    pred_nn = pred_nn_base * df_input['Ngay_Tuong_Duong']
    pred_rf = pred_rf_base * df_input['Ngay_Tuong_Duong']
    pred_xg = pred_xg_base * df_input['Ngay_Tuong_Duong']
    pred_trend = pred_trend_base * df_input['Ngay_Tuong_Duong'] 
    
    return pred_nn, pred_rf, pred_xg, pred_trend

# ==============================================================================
# GIAO DIỆN CHÍNH
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    provider = st.selectbox("Chọn Nhà Cung Cấp AI:", ["Google Gemini", "OpenAI ChatGPT"])
    api_key = st.text_input(f"API Key ({provider})", type="password")
    
    available_models = []
    selected_model = None
    if provider == "Google Gemini" and api_key:
        try:
            genai.configure(api_key=api_key)
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
                    available_models.append(m.name)
            available_models.sort(key=lambda x: 100 if 'pro' in x.lower() else (50 if '1.5' in x else 10), reverse=True)
        except: st.error("Lỗi Key Gemini hoặc mạng!")
    elif provider == "OpenAI ChatGPT":
        available_models = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    
    if available_models: selected_model = st.selectbox("🤖 Chọn Model:", available_models, index=0)
    
    st.markdown("---")
    st.write("### 📅 Cập nhật Lịch Nghỉ Lễ")
    df_default = pd.DataFrame(DEFAULT_HOLIDAYS)
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
    USER_HOLIDAYS_MAP = {}
    for _, row in edited_df.iterrows():
        USER_HOLIDAYS_MAP[(int(row['Năm']), int(row['Tháng']))] = (int(row['Tết Âm']), int(row['Lễ Nhỏ']))
    
    # --- CẤU HÌNH HỆ SỐ NGÀY (K) MẶC ĐỊNH ---
    st.markdown("---")
    st.write("### ⚖️ Hệ số Ngày Mặc định (k)")
    c_k1, c_k2 = st.columns(2)
    with c_k1:
        k_t2 = st.number_input("Hệ số T2", value=0.96, step=0.01)
        k_t7 = st.number_input("Hệ số T7", value=0.96, step=0.01)
        k_cn = st.number_input("Hệ số CN", value=0.85, step=0.01)
    with c_k2:
        k_le = st.number_input("Hệ số Lễ", value=0.80, step=0.01)
        k_tet = st.number_input("Hệ số Tết", value=0.55, step=0.01)
        
    K_DICT_DEFAULT = {'T2': k_t2, 'T7': k_t7, 'CN': k_cn, 'Le': k_le, 'Tet': k_tet}
    
    st.markdown("---")
    seed_val = st.number_input("Random Seed", value=42)
    if st.button("🗑️ Xóa Cache & Reset"):
        st.cache_data.clear(); st.rerun()

col1, col2 = st.columns(2)
with col1: f_train = st.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
with col2: f_input = st.file_uploader("2. File Dự Báo (Input)", type=['xlsx', 'xls'])

st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# --- AI & CHỐT PHƯƠNG ÁN ---
st.subheader("1️⃣ Tham khảo AI & Phân tích Động thái")
c1, c2 = st.columns([2, 1])
if 'detected_months' not in st.session_state: st.session_state.detected_months = []

with c1: 
    text_data = st.text_area("Dán link báo hoặc tin tức vào đây:", height=150, 
                             placeholder="Ví dụ: Dự báo nắng nóng gay gắt tháng tới tại khu vực miền Nam...")

with c2:
    if f_input:
        try:
            df_temp = ultra_scan_read_excel(f_input)
            if df_temp is not None:
                st.session_state.detected_months = sorted(list(set(zip(df_temp['Năm'], df_temp['Tháng']))))
        except: pass
    
    btn_ai = st.button("🤖 AI Phân Tích Ngay", disabled=not selected_model, use_container_width=True)
    if btn_ai:
        with st.spinner(f"Đang hỏi {provider}..."):
            val, log, reason = xu_ly_du_lieu_dinh_tinh(api_key, text_data, selected_model, provider)
            st.session_state.ai_suggestion_val = val
            st.session_state.ai_suggestion_reason = reason
            st.session_state.ai_log = log

    if 'ai_suggestion_val' in st.session_state:
        if st.session_state.ai_suggestion_val != 0:
            st.success(f"{st.session_state.ai_log}")
            if st.session_state.detected_months:
                options = [f"Tháng {m}/{y}" for y, m in st.session_state.detected_months]
                sel_ref = st.selectbox("Soi cùng kỳ cho:", options, index=len(options)-1)
                
                m_val = int(sel_ref.split('/')[0].replace('Tháng ', ''))
                y_val = int(sel_ref.split('/')[1])
                y_past = y_val - 1
                
                if f_train:
                    try:
                        df_hist = ultra_scan_read_excel(f_train)
                        row_old = df_hist[(df_hist['Năm'] == y_past) & (df_hist['Tháng'] == m_val)]
                        if not row_old.empty:
                            v_old = row_old['Tổng thương phẩm'].values[0]
                            pct = st.session_state.ai_suggestion_val
                            v_new = v_old * (1 + pct/100)
                            
                            st.markdown(f"**Đối chiếu cùng kỳ ({m_val}/{y_past}):**")
                            m1, m2 = st.columns(2)
                            m1.metric(f"Thực tế {y_past}", f"{v_old:,.0f}")
                            m2.metric(f"Dự tính {y_val}", f"{v_new:,.0f}", f"{pct:+.1f}%")
                        else: st.warning(f"Không tìm thấy số liệu tháng {m_val}/{y_past}")
                    except: st.error("Không thể đối chiếu số liệu cũ.")
            st.info(f"📝 **Lý do:** {st.session_state.ai_suggestion_reason}")
        else:
            st.warning("⚠️ AI không trích xuất được con số %.")
            with st.expander("Xem chi tiết phản hồi"):
                st.write(st.session_state.ai_suggestion_reason)

st.write("---")

# --- CHẠY DỰ BÁO ---
if f_train and f_input:
    if st.button("🚀 CHẠY DỰ BÁO AI", type="primary"):
        with st.spinner("Đang cho AI quét và tìm ra Tổng sản lượng..."):
            df_train_main = ultra_scan_read_excel(f_train)
            df_input_main = ultra_scan_read_excel(f_input)
            
            if df_train_main is not None and df_input_main is not None:
                kiem_tra_chat_luong(df_train_main, "Lịch Sử")
                kiem_tra_chat_luong(df_input_main, "Dự Báo")
                
                pred_nn, pred_rf, pred_xg, p_trend = chay_mo_hinh_goc(df_train_main, df_input_main, USER_HOLIDAYS_MAP, K_DICT_DEFAULT, seed_val)
                
                res = df_input_main[['Năm', 'Tháng']].copy()
                df_check = tao_dac_trung(df_input_main.copy(), USER_HOLIDAYS_MAP)
                
                res['Ngày Thường'] = df_check['So_Ngay_Thuong']
                res['T2'] = df_check['So_Ngay_T2']
                res['T7'] = df_check['So_Ngay_T7']
                res['CN'] = df_check['So_Ngay_CN']
                res['Lễ'] = df_check['So_Ngay_Le_Nho']
                res['Tết'] = df_check['So_Ngay_Tet_Am']
                
                res['Neural Network'] = pred_nn
                res['Random Forest'] = pred_rf
                res['XGBoost'] = pred_xg

                if 'Tổng thương phẩm' in df_train_main.columns:
                    actual = df_train_main[['Năm', 'Tháng', 'Tổng thương phẩm']]
                    res = pd.merge(res, actual, on=['Năm', 'Tháng'], how='left')
                    res.rename(columns={'Tổng thương phẩm': 'Thực Tế'}, inplace=True)
                
                st.session_state.res_output = res.copy()
                st.session_state.trend_val = p_trend
                st.success("✅ Đã chạy xong Mô hình AI. Hãy chuyển xuống Bảng tinh chỉnh bên dưới!")
                
                st.subheader("📊 Kết Quả Dự Báo Tổng Của Từng Thuật Toán (Tham khảo)")
                cols = ['Tháng', 'Năm', 'Thực Tế', 'Neural Network', 'Random Forest', 'XGBoost']
                cols = [c for c in cols if c in res.columns]
                
                st.dataframe(res[cols].style.format({
                    'Thực Tế': '{:,.0f}', 'Neural Network': '{:,.0f}', 
                    'Random Forest': '{:,.0f}', 'XGBoost': '{:,.0f}'
                }), use_container_width=True)
                
# ==============================================================================
# BẢNG TƯƠNG TÁC: CHỐT SỐ & TINH CHỈNH HỆ SỐ THEO NGƯỜI DÙNG
# ==============================================================================
if 'res_output' in st.session_state:
    st.markdown("---")
    st.header("🎛️ BẢNG TINH CHỈNH HỆ SỐ & CHỐT SẢN LƯỢNG CUỐI CÙNG")
    
    # Cho phép người dùng chọn mô hình muốn áp dụng
    model_for_base = st.selectbox(
        "🎯 Chọn Mô hình áp dụng để phân bổ ngày:", 
        ["XGBoost", "Random Forest", "Neural Network"], 
        index=0
    )
    
    st.write("""
    **Quy trình tính toán:**
    1. Lấy kết quả từ mô hình **bạn vừa chọn ở trên** làm **Tổng Dự Báo Gốc ($Q$)**.
    2. Dựa trên số lượng ngày và **Hệ số mặc định**, hệ thống tính ngược ra **Sản lượng ngày cơ sở ($x$)**.
    3. Bạn có thể **sửa trực tiếp** các Hệ số $k$ trong Bảng 1. Ngay lập tức, Bảng 2 sẽ phân tách ngày và tính tổng cuối cùng!
    """)
    
    res = st.session_state.res_output
    
    # SỬA LỖI MẤT BẢNG: Lọc chỉ lấy các tháng Tương lai, nếu không có thì lấy toàn bộ
    if 'Thực Tế' in res.columns:
        res_forecast = res[res['Thực Tế'].isnull()].copy()
    else:
        res_forecast = res.copy()
        
    if res_forecast.empty:
        st.warning("⚠️ BỘ LỌC TỰ ĐỘNG: Toàn bộ dữ liệu bạn nhập đều là lịch sử (Đã có số Thực Tế). Hệ thống tạm thời hiển thị lại toàn bộ các tháng để bạn thao tác.")
        res_forecast = res.copy() # Trả lại toàn bộ dữ liệu để bảng không bị trắng
        
    # Chuẩn bị dữ liệu cho Bảng Edit dựa vào model được chọn (CHỈ hiển thị tháng dự báo)
    edit_data = []
    orig_indices = [] # Theo dõi chỉ mục gốc để ánh xạ lại
    for idx, row in res_forecast.iterrows():
        orig_indices.append(idx)
        edit_data.append({
            'Tháng': f"{int(row['Tháng'])}/{int(row['Năm'])}",
            'Tổng Dự Báo Gốc ($Q$)': row[model_for_base],
            'k_T2': K_DICT_DEFAULT['T2'],
            'k_T7': K_DICT_DEFAULT['T7'],
            'k_CN': K_DICT_DEFAULT['CN'],
            'k_Lễ': K_DICT_DEFAULT['Le'],
            'k_Tết': K_DICT_DEFAULT['Tet']
        })
    df_edit = pd.DataFrame(edit_data)
    
    st.write("✍️ **BẢNG 1: Nhấp đúp vào các ô Hệ số để thay đổi theo ý muốn của bạn**")
    
    # KHẮC PHỤC LỖI HIỂN THỊ CHỮ e+07: Ép định dạng NumberColumn thành số nguyên
    # SỬA LỖI TRẮNG BẢNG DO CACHE: Thêm Key linh động phụ thuộc vào model đang chọn
    edited_df = st.data_editor(
        df_edit,
        column_config={
            "Tổng Dự Báo Gốc ($Q$)": st.column_config.NumberColumn(
                "Tổng Dự Báo Gốc ($Q$)",
                format="%d" 
            )
        },
        disabled=['Tháng', 'Tổng Dự Báo Gốc ($Q$)'],
        hide_index=True,
        use_container_width=True,
        key=f"editor_k_{model_for_base}" 
    )
    
    # Logic tính toán lại từ Bảng 1 đổ xuống Bảng 2
    final_results = []
    for i, e_row in edited_df.iterrows():
        orig_idx = orig_indices[i]
        r_orig = res.loc[orig_idx]
        
        n_thuong = r_orig['Ngày Thường']
        n_t2 = r_orig['T2']
        n_t7 = r_orig['T7']
        n_cn = r_orig['CN']
        n_le = r_orig['Lễ']
        n_tet = r_orig['Tết']
        
        # 1. Số ngày chuẩn cũ AI đã dùng
        eq_days_standard = n_thuong + n_t2*K_DICT_DEFAULT['T2'] + n_t7*K_DICT_DEFAULT['T7'] + n_cn*K_DICT_DEFAULT['CN'] + n_le*K_DICT_DEFAULT['Le'] + n_tet*K_DICT_DEFAULT['Tet']
        
        # 2. Sản lượng 1 Ngày Cơ sở (x)
        base_x = e_row['Tổng Dự Báo Gốc ($Q$)'] / eq_days_standard if eq_days_standard else 0
        
        # 3. Số ngày chuẩn MỚI (User sửa)
        eq_days_new = n_thuong + n_t2*e_row['k_T2'] + n_t7*e_row['k_T7'] + n_cn*e_row['k_CN'] + n_le*e_row['k_Lễ'] + n_tet*e_row['k_Tết']
        
        # 4. Tính Chốt Cuối
        final_total = base_x * eq_days_new
        
        final_results.append({
            'Tháng': e_row['Tháng'],
            'Ngày Thường (T3-T6)': base_x,
            'Thứ 2': base_x * e_row['k_T2'],
            'Thứ 7': base_x * e_row['k_T7'],
            'Chủ Nhật': base_x * e_row['k_CN'],
            'Ngày Lễ': base_x * e_row['k_Lễ'],
            'Tết Âm': base_x * e_row['k_Tết'],
            'TỔNG CUỐI CÙNG': final_total,
            'Chênh lệch so Gốc': final_total - e_row['Tổng Dự Báo Gốc ($Q$)']
        })
        
    df_final = pd.DataFrame(final_results)
    
    st.write("📊 **BẢNG 2: Cơ cấu Sản lượng Từng Loại Ngày & Kết Quả Chốt Cuối (kWh)**")
    st.dataframe(df_final.style.format({
        'Ngày Thường (T3-T6)': '{:,.0f}',
        'Thứ 2': '{:,.0f}',
        'Thứ 7': '{:,.0f}',
        'Chủ Nhật': '{:,.0f}',
        'Ngày Lễ': '{:,.0f}',
        'Tết Âm': '{:,.0f}',
        'TỔNG CUỐI CÙNG': '{:,.0f}',
        'Chênh lệch so Gốc': '{:+,.0f}'
    }).apply(lambda x: ['background-color: #d4edda; font-weight: bold' if i == 'TỔNG CUỐI CÙNG' else ('color: #d9534f; font-weight:bold' if i == 'Chênh lệch so Gốc' and x[i] < 0 else ('color: #5cb85c; font-weight:bold' if i == 'Chênh lệch so Gốc' and x[i] > 0 else '')) for i in x.index], axis=1), use_container_width=True)

    if not df_final.empty:
        st.markdown("---")
        st.subheader("📈 Biểu Đồ Phụ Tải Ngày Của Các Tháng Dự Báo")
        
        # FIX LỖI HIỂN THỊ ĐÈ CHỮ Ở TRỤC X: Tăng figsize và xoay nhãn 45 độ
        fig, ax = plt.subplots(figsize=(14, 6))
        
        months = df_final['Tháng']
        base_loads = df_final['Ngày Thường (T3-T6)']
        t7_loads = df_final['Thứ 7']
        cn_loads = df_final['Chủ Nhật']
        
        x = np.arange(len(months))
        width = 0.25
        
        ax.bar(x - width, base_loads, width, label='Ngày Thường (T3-T6)', color='#4c72b0')
        ax.bar(x, t7_loads, width, label='Thứ 7', color='#dd8452')
        ax.bar(x + width, cn_loads, width, label='Chủ Nhật', color='#c44e52')
        
        ax.set_ylabel('Sản lượng (kWh)')
        ax.set_title('So sánh Cơ cấu Phụ tải Ngày theo Tháng')
        
        # Cập nhật trục X để không bị lỗi đè chữ
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha='right')
        
        ax.legend()
        plt.tight_layout() # Giúp các nhãn trục X không bị cắt lẹm ra ngoài khung ảnh
        st.pyplot(fig)

# ==============================================================================
# 4. MÁY SOI SEED
# ==============================================================================
st.markdown("---")
st.header("🔬 Find Seed")
if 'scan_current_seed' not in st.session_state: st.session_state.scan_current_seed = 0
if 'scan_history' not in st.session_state: st.session_state.scan_history = pd.DataFrame()

if f_train and f_input:
    with st.expander("BẢNG ĐIỀU KHIỂN & CHỌN MỤC TIÊU", expanded=False):
        df_scan_preview = ultra_scan_read_excel(f_input)
        if df_scan_preview is not None:
            list_dates = [f"Tháng {int(row['Tháng'])}/{int(row['Năm'])}" for i, row in df_scan_preview.iterrows()]
            
            col_target, col_val = st.columns(2)
            with col_target:
                selected_month_str = st.selectbox("🎯 Chọn tháng muốn soi:", list_dates, index=len(list_dates)-1)
                target_index = list_dates.index(selected_month_str)
            with col_val:
                target_val = st.number_input(f"Giá trị mong muốn", value=740000000.0, step=1000000.0, format="%.0f")

            c1, c2, c3 = st.columns(3)
            with c1: model_choice = st.selectbox("Mô hình ưu tiên:", ["Neural Network", "Random Forest", "XGBoost", "Trung Bình Cộng"])
            with c2: acc = st.slider("Sai số (+/- %)", 0.1, 10.0, 1.0)
            with c3: batch_size = st.number_input("Số lượng seed/lô", value=20, step=10)

            limit_min, limit_max = target_val * (1 - acc/100), target_val * (1 + acc/100)
            start_seed = st.session_state.scan_current_seed
            end_seed = start_seed + batch_size - 1
            
            st.info(f"📍 Đang kiểm tra từ Seed {start_seed} đến {end_seed}")
            if st.button(f"▶️ Chạy {start_seed}-{end_seed}"):
                df_train_scan = ultra_scan_read_excel(f_train)
                df_input_scan = ultra_scan_read_excel(f_input)
                batch_data = []
                p_bar = st.progress(0)
                for i, seed in enumerate(range(start_seed, end_seed + 1)):
                    try:
                        p_nn, p_rf, p_xg, _ = chay_mo_hinh_goc(df_train_scan, df_input_scan, USER_HOLIDAYS_MAP, K_DICT_DEFAULT, seed=seed)
                        v_nn, v_rf, v_xg = p_nn[target_index], p_rf[target_index], p_xg[target_index]
                        v_avg = (v_nn + v_rf + v_xg) / 3
                        val = {"Neural Network": v_nn, "Random Forest": v_rf, "XGBoost": v_xg, "Trung Bình Cộng": v_avg}[model_choice]
                        is_pass = limit_min <= val <= limit_max
                        batch_data.append({"Seed": seed, "Tháng": selected_month_str, "Kết quả (NN)": v_nn, "Kết quả (RF)": v_rf, "Kết quả (XGB)": v_xg, "Độ lệch": val - target_val, "Trạng thái": "✅ ĐẠT" if is_pass else "❌"})
                    except: batch_data.append({"Seed": seed, "Trạng thái": "⚠️ Lỗi"})
                    p_bar.progress((i + 1) / batch_size)
                
                st.session_state.scan_history = pd.concat([st.session_state.scan_history, pd.DataFrame(batch_data)], ignore_index=True)
                st.session_state.scan_current_seed = end_seed + 1
                st.rerun()

            if st.button("🗑️ Xóa lịch sử"):
                st.session_state.scan_current_seed = 0
                st.session_state.scan_history = pd.DataFrame()
                st.rerun()

    if not st.session_state.scan_history.empty:
        st.subheader("📋 Kết Quả Lọc Seed")
        df_show = st.session_state.scan_history.sort_values(by='Seed')
        st.dataframe(df_show.style.apply(lambda r: ['background-color: #d4edda' if "ĐẠT" in str(r['Trạng thái']) else '']*len(r), axis=1).format({
            "Seed": "{:.0f}", "Kết quả (NN)": "{:,.0f}", "Kết quả (RF)": "{:,.0f}", "Kết quả (XGB)": "{:,.0f}", "Độ lệch": "{:+,.0f}"
        }), use_container_width=True)

# ==============================================================================
# 5. BẢNG GIẢI TRÌNH ĐỘ TIN CẬY
# ==============================================================================
st.markdown("---")
if 'res_output' in st.session_state:
    st.header("🛡️ GIẢI TRÌNH LOGIC DỰ BÁO (Dựa trên mô hình bạn chọn)")
    
    r = st.session_state.res_output
    df_p = r[['Tháng', 'Năm']].copy()
    
    # Xu hướng
    df_p['Xu hướng nền (A)'] = st.session_state.trend_val
    
    # Lấy Kết quả dự báo gốc từ mô hình đang được chọn (thay vì trung bình cộng)
    try:
        model_val = r[model_for_base]
    except:
        model_val = r['Tổng AI (Q)']
        
    # Biến động do ML
    df_p['Biến động ML (B)'] = model_val - df_p['Xu hướng nền (A)']
    
    if 'Thực Tế' in r.columns:
        df_p['Thực Tế (E)'] = r['Thực Tế']
        
    df_p['DỰ BÁO CỦA MÔ HÌNH'] = model_val
    
    format_dict = {
        'Xu hướng nền (A)': '{:,.0f}', 
        'Biến động ML (B)': '{:+,.0f}', 
        'DỰ BÁO CỦA MÔ HÌNH': '{:,.0f}',
        'Thực Tế (E)': '{:,.0f}'
    }
    
    cols_to_show = [c for c in ['Tháng', 'Năm', 'Thực Tế (E)', 'Xu hướng nền (A)', 'Biến động ML (B)', 'DỰ BÁO CỦA MÔ HÌNH'] if c in df_p.columns]
    
    st.dataframe(df_p[cols_to_show].style.format(format_dict).apply(
        lambda x: ['background-color: #f0f2f6' if i == 'DỰ BÁO CỦA MÔ HÌNH' else '' for i in x.index], axis=1
    ), use_container_width=True)
    
    st.caption("🔍 **A**: Tăng trưởng tự nhiên dựa trên hồi quy 3 năm | **B**: Sai lệch do thời tiết/lễ tết do máy học tự tìm | **Dự báo của Mô hình (Q) = A + B**")
