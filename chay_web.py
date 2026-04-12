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
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([8, 2], vertical_alignment="center")
with col1: st.markdown("<h3>HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN TỈNH TÂY NINH</h3>", unsafe_allow_html=True)
with col2: 
    try: st.image("image_1.png", use_column_width=True)
    except: st.write("EVN SPC")
st.markdown("---")

# --- KIỂM TRA THƯ VIỆN AI ---
# 1. Google Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except: HAS_GEMINI = False

# 2. OpenAI ChatGPT
try:
    from openai import OpenAI
    HAS_OPENAI = True
except: HAS_OPENAI = False

# ==============================================================================
# 1. CẤU HÌNH NGÀY NGHỈ
# ==============================================================================
DEFAULT_HOLIDAYS = [
    {"Năm": 2023, "Tháng": 1, "Số ngày nghỉ lễ": 8}, {"Năm": 2023, "Tháng": 4, "Số ngày nghỉ lễ": 1},
    {"Năm": 2023, "Tháng": 5, "Số ngày nghỉ lễ": 2}, {"Năm": 2023, "Tháng": 9, "Số ngày nghỉ lễ": 2},
    {"Năm": 2024, "Tháng": 1, "Số ngày nghỉ lễ": 1}, {"Năm": 2024, "Tháng": 2, "Số ngày nghỉ lễ": 7},
    {"Năm": 2024, "Tháng": 4, "Số ngày nghỉ lễ": 3}, {"Năm": 2024, "Tháng": 5, "Số ngày nghỉ lễ": 1},
    {"Năm": 2024, "Tháng": 9, "Số ngày nghỉ lễ": 2}, {"Năm": 2025, "Tháng": 1, "Số ngày nghỉ lễ": 7},
    {"Năm": 2025, "Tháng": 2, "Số ngày nghỉ lễ": 2}, {"Năm": 2025, "Tháng": 4, "Số ngày nghỉ lễ": 2},
    {"Năm": 2025, "Tháng": 5, "Số ngày nghỉ lễ": 1}, {"Năm": 2025, "Tháng": 9, "Số ngày nghỉ lễ": 2},
    {"Năm": 2026, "Tháng": 1, "Số ngày nghỉ lễ": 1}, {"Năm": 2026, "Tháng": 2, "Số ngày nghỉ lễ": 5},
    {"Năm": 2026, "Tháng": 4, "Số ngày nghỉ lễ": 1}, {"Năm": 2026, "Tháng": 5, "Số ngày nghỉ lễ": 2},
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
# 2. MODULE AI (GEMINI + CHATGPT)
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
    
    # 1. Xử lý đầu vào (Link hoặc Text)
    text_data = input_data
    status = ""
    if input_data.strip().startswith("http"):
        with st.spinner("Đang đọc bài báo..."):
            extracted, msg = lay_noi_dung_tu_link(input_data)
            if extracted: 
                text_data = extracted
                status = msg + "\n"
            else: return 0.0, msg, ""

    # 2. Tạo Prompt chung
    prompt = (f"Đọc thông tin sau: '{text_data[:3000]}'. "
              "Hãy đánh giá xem phụ tải điện tháng này sẽ TĂNG hay GIẢM bao nhiêu % so với CÙNG KỲ NĂM TRƯỚC. "
              "Chỉ đưa ra con số ước lượng dựa trên tác động (thời tiết, kinh tế...). "
              "Trả về định dạng: SỐ | LÝ DO NGẮN GỌN. Ví dụ: +5.5 | Nắng nóng hơn năm ngoái.")

    res = ""
    
    # 3. Gọi API theo Nhà cung cấp
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
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia dự báo phụ tải điện."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            res = response.choices[0].message.content.strip()

        # 4. Xử lý kết quả trả về
        if "|" in res:
            parts = res.split("|")
            val = trich_xuat_so(parts[0]) 
            return val, f"{status}✅ AI Đã xong ({provider} - {target_model_name})", parts[1].strip()
            
        val = trich_xuat_so(res)
        if val != 0.0: return val, f"{status}✅ AI Đã xong (Tự bắt số)!", res
        
        return 0.0, f"⚠️ AI ({provider}) không tìm thấy số liệu cụ thể.", res
        
    except Exception as e:
        if "429" in str(e): return 0.0, "⚠️ Hết hạn mức (Quota).", "Hết Quota"
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
# 4. CHẠY DỰ BÁO
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
    
    y_train_log = np.log1p(y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    df_pred = df_input.copy()
    X_pred = df_pred[valid_cols].fillna(0)
    
    # 1. Neural Network
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', 
                      alpha=0.1, max_iter=5000, random_state=seed)
    nn.fit(X_train_scaled, y_train_log)
    pred_nn_log = nn.predict(scaler.transform(X_pred))
    
    # 2. Detrending
    trend_model = LinearRegression()
    trend_model.fit(data_train[['Time_Index']], y_train_log)
    trend_future = trend_model.predict(df_pred[['Time_Index']])
    y_residual = y_train_log - trend_model.predict(data_train[['Time_Index']])
    
    # 3. Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=seed)
    rf.fit(X_train, y_residual)
    
    # 4. XGBoost
    xg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, 
                          reg_alpha=0.1, random_state=seed)
    xg.fit(X_train, y_residual)
    
    pred_rf_log = rf.predict(X_pred) + trend_future
    pred_xg_log = xg.predict(X_pred) + trend_future
    
    pred_nn = np.expm1(pred_nn_log)
    pred_rf = np.expm1(pred_rf_log)
    pred_xg = np.expm1(pred_xg_log)
    pred_trend = np.expm1(trend_future) # TÍNH NĂNG THÊM 1: TRẢ VỀ TREND
    
    return pred_nn, pred_rf, pred_xg, pred_trend

# ==============================================================================
# GIAO DIỆN CHÍNH
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    
    # --- CHỌN PROVIDER ---
    provider = st.selectbox("Chọn Nhà Cung Cấp AI:", ["Google Gemini", "OpenAI ChatGPT"])
    
    api_key = st.text_input(f"API Key ({provider})", type="password")
    
    available_models = []
    selected_model = None
    
    # --- LOGIC HIỂN THỊ MODEL ---
    if provider == "Google Gemini":
        if api_key:
            try:
                genai.configure(api_key=api_key)
                models_obj = genai.list_models()
                for m in models_obj:
                    if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
                        available_models.append(m.name)
                # Sắp xếp ưu tiên
                def sort_key(name):
                    val = 0
                    if 'pro' in name.lower(): val += 100
                    if '1.5' in name: val += 50
                    if 'flash' in name.lower(): val += 10
                    return val
                available_models.sort(key=sort_key, reverse=True)
            except: st.error("Lỗi Key Gemini hoặc mạng!")
            
    elif provider == "OpenAI ChatGPT":
        # Với OpenAI, ta liệt kê tĩnh các model phổ biến để tránh lỗi API khi chưa có key
        available_models = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    
    if available_models:
        selected_model = st.selectbox("🤖 Chọn Model:", available_models, index=0)
    elif api_key:
        st.warning("Không tìm thấy model phù hợp.")
    
    st.markdown("---")
    st.write("### 📅 Cập nhật Lịch Nghỉ Lễ")
    df_default = pd.DataFrame(DEFAULT_HOLIDAYS)
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
    USER_HOLIDAYS_MAP = dict(zip(zip(edited_df['Năm'], edited_df['Tháng']), edited_df['Số ngày nghỉ lễ']))
    
    st.markdown("---")
    seed_val = st.number_input("Random Seed", value=42)
    if st.button("🗑️ Xóa Cache & Reset"):
        st.cache_data.clear()
        st.rerun()

col1, col2 = st.columns(2)
with col1: f_train = st.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
with col2: f_input = st.file_uploader("2. File Dự Báo (Input)", type=['xlsx', 'xls'])

st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# --- AI & CHỐT PHƯƠNG ÁN ---
st.subheader("1️⃣ Tham khảo AI & Chốt phương án")
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
                # Lưu danh sách các tháng tìm thấy trong file dự báo vào session_state
                st.session_state.detected_months = sorted(list(set(zip(df_temp['Năm'], df_temp['Tháng']))))
        except:
            pass
    
    # Nút bấm kích hoạt AI
    btn_ai = st.button("🤖 AI Phân Tích Ngay", disabled=not selected_model, use_container_width=True)
    if btn_ai:
        with st.spinner(f"Đang hỏi {provider}..."):
            val, log, reason = xu_ly_du_lieu_dinh_tinh(api_key, text_data, selected_model, provider)
            st.session_state.ai_suggestion_val = val
            st.session_state.ai_suggestion_reason = reason
            st.session_state.ai_log = log

    # --- KHỐI HIỂN THỊ KẾT QUẢ AI ---
    if 'ai_suggestion_val' in st.session_state:
        if st.session_state.ai_suggestion_val != 0:
            st.success(f"{st.session_state.ai_log}")
            
            # TÍNH NĂNG THÊM 2: CHO CHỌN THÁNG ĐỐI CHIẾU
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
                        else:
                            st.warning(f"Không tìm thấy số liệu tháng {m_val}/{y_past}")
                    except:
                        st.error("Không thể đối chiếu số liệu cũ.")
            
            st.info(f"📝 **Lý do:** {st.session_state.ai_suggestion_reason}")
        
        else:
            st.warning("⚠️ AI không trích xuất được con số %.")
            with st.expander("Xem chi tiết phản hồi"):
                st.write(st.session_state.ai_suggestion_reason)

# --- NGƯỜI DÙNG QUYẾT ĐỊNH ---
st.write("---")
st.write("### ✍️ CHỐT SỐ LIỆU")

if st.session_state.detected_months:
    c_a, c_b = st.columns(2)
    with c_a:
        months_str = [f"Tháng {m}/{y}" for y, m in st.session_state.detected_months]
        selected = st.multiselect("Chọn tháng áp dụng:", months_str, default=months_str)
    
    with c_b:
        user_val = st.number_input("Nhập % Điều Chỉnh:", value=0.0, step=0.1)
        user_note = st.text_input("Ghi chú:", value="Thủ công")
    
    if st.button("💾 LƯU QUYẾT ĐỊNH"):
        temp = {}
        for s in selected:
            m = int(s.split('/')[0].replace('Tháng ', ''))
            y = int(s.split('/')[1])
            temp[(y, m)] = (user_val, user_note)
        st.session_state.param_dict = temp
        st.success(f"Đã lưu: {user_val}%")

st.write("---")

# --- DỰ BÁO ---
if f_train and f_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        with st.spinner("Đang tính toán..."):
            df_train_main = ultra_scan_read_excel(f_train)
            df_input_main = ultra_scan_read_excel(f_input)
            
            if df_train_main is not None and df_input_main is not None:
                kiem_tra_chat_luong(df_train_main, "Lịch Sử")
                kiem_tra_chat_luong(df_input_main, "Dự Báo")
                
                pred_nn, pred_rf, pred_xg, p_trend = chay_mo_hinh_goc(df_train_main, df_input_main, USER_HOLIDAYS_MAP, seed_val)
                
                res = df_input_main[['Năm', 'Tháng']].copy()
                df_check = tao_dac_trung(df_input_main.copy(), USER_HOLIDAYS_MAP)
                res['T7+CN'] = df_check['So_Ngay_T7'] + df_check['So_Ngay_CN']
                res['Lễ Tết'] = df_check['So_Ngay_Le_Tet']
                
                res['Neural Network'] = pred_nn
                res['Random Forest'] = pred_rf
                res['XGBoost'] = pred_xg

                def apply_adj(row):
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

                # GIỮ NGUYÊN CODE TÌM "THỰC TẾ" CỦA ÔNG
                if 'Tổng thương phẩm' in df_train_main.columns:
                    actual = df_train_main[['Năm', 'Tháng', 'Tổng thương phẩm']]
                    res = pd.merge(res, actual, on=['Năm', 'Tháng'], how='left')
                    res.rename(columns={'Tổng thương phẩm': 'Thực Tế'}, inplace=True)
                
                # LƯU DỮ LIỆU CHO BẢNG GIẢI TRÌNH Ở CUỐI
                st.session_state.res_output = res.copy()
                st.session_state.trend_val = p_trend

                st.subheader("📊 Kết Quả Dự Báo")
                cols = ['Tháng', 'Năm', 'Thực Tế', 'T7+CN', 'Lễ Tết', 'Neural Network', 'Random Forest', 'XGBoost', 'Điều Chỉnh (%)', 'Ghi chú']
                cols = [c for c in cols if c in res.columns]
                
                # GIỮ NGUYÊN FORMAT BẢNG CỦA ÔNG
                st.dataframe(res[cols].style.format({
                    'Thực Tế': '{:,.0f}', 'Neural Network': '{:,.0f}', 
                    'Random Forest': '{:,.0f}', 'XGBoost': '{:,.0f}',
                    'Điều Chỉnh (%)': '{:+.1f}%', 'T7+CN': '{:.0f}', 'Lễ Tết': '{:.0f}'
                }), use_container_width=True)
                
                # GIỮ NGUYÊN CODE VẼ 4 ĐƯỜNG ĐỒ THỊ CỦA ÔNG
                res['Date'] = pd.to_datetime(dict(year=res['Năm'], month=res['Tháng'], day=1))
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(res['Date'], res['Neural Network'], 'o-', color='blue', label='NN')
                ax.plot(res['Date'], res['Random Forest'], 's--', color='green', label='RF')
                ax.plot(res['Date'], res['XGBoost'], '^-.', color='purple', label='XGB')
                if 'Thực Tế' in res.columns:
                    mask = res['Thực Tế'].notnull()
                    ax.plot(res.loc[mask, 'Date'], res.loc[mask, 'Thực Tế'], 'ko', label='Thực Tế')
                ax.legend(); ax.grid(True, alpha=0.3)
                st.pyplot(fig)

# ==============================================================================
# 4. MÁY SOI SEED
# ==============================================================================
st.markdown("---")
st.header("🔬 Find Seed")
if 'scan_current_seed' not in st.session_state: st.session_state.scan_current_seed = 0
if 'scan_history' not in st.session_state: st.session_state.scan_history = pd.DataFrame()

if f_train and f_input:
    with st.expander("BẢNG ĐIỀU KHIỂN & CHỌN MỤC TIÊU", expanded=True):
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
                        p_nn, p_rf, p_xg, _ = chay_mo_hinh_goc(df_train_scan, df_input_scan, USER_HOLIDAYS_MAP, seed=seed)
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
# 5. BẢNG GIẢI TRÌNH ĐỘ TIN CẬY (TÍNH NĂNG THÊM 3)
# ==============================================================================
st.markdown("---")
if 'res_output' in st.session_state:
    st.header("🛡️ GIẢI TRÌNH LOGIC DỰ BÁO")
    
    r = st.session_state.res_output
    df_p = r[['Tháng', 'Năm']].copy()
    
    # 1. Xu hướng
    df_p['Xu hướng nền (A)'] = st.session_state.trend_val
    
    # 2. Biến động ML (lấy trung bình cộng 3 mô hình)
    avg_ml = (r['Neural Network'] + r['Random Forest'] + r['XGBoost']) / 3
    # Phải chia ngược lại hệ số điều chỉnh để tìm ra con số GỐC của Máy học trước khi con người can thiệp
    avg_ml_truoc_khi_chinh = avg_ml / (1 + r['Điều Chỉnh (%)'] / 100)
    df_p['Biến động ML (B)'] = avg_ml_truoc_khi_chinh - df_p['Xu hướng nền (A)']
    
    # 3. Điều chỉnh
    df_p['Điều chỉnh (%) (C)'] = r['Điều Chỉnh (%)']
    df_p['Giá trị điều chỉnh (D)'] = avg_ml - avg_ml_truoc_khi_chinh
    
    # 4. Thực tế (nếu có)
    if 'Thực Tế' in r.columns:
        df_p['Thực Tế (E)'] = r['Thực Tế']
    
    # Chốt
    df_p['DỰ BÁO CUỐI CÙNG'] = avg_ml
    
    # Hiển thị
    format_dict = {
        'Xu hướng nền (A)': '{:,.0f}', 
        'Biến động ML (B)': '{:+,.0f}', 
        'Điều chỉnh (%) (C)': '{:+.1f}%',
        'Giá trị điều chỉnh (D)': '{:+,.0f}', 
        'DỰ BÁO CUỐI CÙNG': '{:,.0f}',
        'Thực Tế (E)': '{:,.0f}'
    }
    
    cols_to_show = [c for c in ['Tháng', 'Năm', 'Thực Tế (E)', 'Xu hướng nền (A)', 'Biến động ML (B)', 'Điều chỉnh (%) (C)', 'Giá trị điều chỉnh (D)', 'DỰ BÁO CUỐI CÙNG'] if c in df_p.columns]
    
    st.dataframe(df_p[cols_to_show].style.format(format_dict).apply(
        lambda x: ['background-color: #f0f2f6' if i == 'DỰ BÁO CUỐI CÙNG' else '' for i in x.index], axis=1
    ), use_container_width=True)
    
    st.caption("🔍 **A**: Tăng trưởng tự nhiên dựa trên hồi quy 3 năm | **B**: Sai lệch do thời tiết/lễ tết do máy học tự tìm | **D**: Trí tuệ con người & AI điều chỉnh | **Tổng chốt = A + B + D**")
