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
# 🎨 GIAO DIỆN & CSS (GIỮ NGUYÊN GỐC)
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
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except: HAS_GEMINI = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except: HAS_OPENAI = False

# ==============================================================================
# 1. CẤU HÌNH NGÀY NGHỈ (GIỮ NGUYÊN GỐC)
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
# 2. MODULE AI (GIỮ NGUYÊN GỐC)
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
                text_data = extracted
                status = msg + "\n"
            else: return 0.0, msg, ""

    prompt = (f"Đọc thông tin sau: '{text_data[:3000]}'. "
              "Hãy đánh giá xem phụ tải điện tháng này sẽ TĂNG hay GIẢM bao nhiêu % so với CÙNG KỲ NĂM TRƯỚC. "
              "Chỉ đưa ra con số ước lượng dựa trên tác động (thời tiết, kinh tế...). "
              "Trả về định dạng: SỐ | LÝ DO NGẮN GỌN. Ví dụ: +5.5 | Nắng nóng hơn năm ngoái.")

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
                messages=[{"role": "system", "content": "Bạn là chuyên gia dự báo phụ tải điện."}, {"role": "user", "content": prompt}],
                temperature=0.5
            )
            res = response.choices[0].message.content.strip()

        if "|" in res:
            parts = res.split("|")
            val = trich_xuat_so(parts[0]) 
            return val, f"{status}✅ AI Đã xong!", parts[1].strip()
        val = trich_xuat_so(res)
        return (val, f"{status}✅ AI Xong (Tự bắt số)", res) if val != 0.0 else (0.0, "⚠️ AI không có số cụ thể.", res)
    except Exception as e: return 0.0, f"❌ Lỗi AI: {str(e)[:50]}", "Lỗi"

# ==============================================================================
# 3. XỬ LÝ FILE (GIỮ NGUYÊN GỐC)
# ==============================================================================
def chuan_hoa_ten_cot(df):
    if df is None: return None
    df.columns = df.columns.astype(str).str.strip()
    col_map = {
        'month': 'Tháng', 'thang': 'Tháng', 'tháng': 'Tháng', 'year': 'Năm', 'nam': 'Năm', 'năm': 'Năm',
        'tổng thương phẩm': 'Tổng thương phẩm', 'tong thuong pham': 'Tổng thương phẩm', 'sản lượng': 'Tổng thương phẩm',
        'nhiệt độ tb': 'Nhiệt độ TB', 'nhiet do tb': 'Nhiệt độ TB', 'độ ẩm': 'Độ ẩm', 'do am': 'Độ ẩm',
        'số ngày': 'Số ngày', 'so ngay': 'Số ngày', 'số ngày nghỉ': 'So_Ngay_Nghi', 'so ngay nghi': 'So_Ngay_Nghi'
    }
    new_cols = {col: col_map[col.lower()] for col in df.columns if col.lower() in col_map}
    return df.rename(columns=new_cols)

def ultra_scan_read_excel(uploaded_file):
    try:
        xl = pd.ExcelFile(uploaded_file)
        for sheet_name in xl.sheet_names:
            preview = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, nrows=10)
            for i, row in preview.iterrows():
                if 'tháng' in str(row.values).lower() and 'năm' in str(row.values).lower():
                    uploaded_file.seek(0)
                    return chuan_hoa_ten_cot(pd.read_excel(uploaded_file, sheet_name=sheet_name, header=i))
        uploaded_file.seek(0)
        return chuan_hoa_ten_cot(pd.read_excel(uploaded_file, header=0))
    except: return None

def kiem_tra_chat_luong(df, ten_file):
    required = ['Tháng', 'Năm']
    for col in required:
        if col not in df.columns: st.error(f"❌ File {ten_file} thiếu '{col}'"); st.stop()

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
# 4. CHẠY DỰ BÁO (TINH CHỈNH: TRẢ VỀ THÊM TREND)
# ==============================================================================
@st.cache_data(show_spinner=False)
def chay_mo_hinh_goc(df_train, df_input, holidays_map, seed=42):
    df_train = tao_dac_trung(df_train.copy(), holidays_map)
    df_input = tao_dac_trung(df_input.copy(), holidays_map)
    start_year = df_train['Năm'].min()
    df_train['Time_Index'] = (df_train['Năm'] - start_year) * 12 + df_train['Tháng']
    df_input['Time_Index'] = (df_input['Năm'] - start_year) * 12 + df_input['Tháng']
    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'So_Ngay_T7', 'So_Ngay_CN', 'So_Ngay_Le_Tet', 'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_cols = [c for c in features if c in df_train.columns and c in df_input.columns]
    target = 'Tổng thương phẩm'
    data_train = df_train.dropna(subset=valid_cols + [target])
    X_train, y_train = data_train[valid_cols], data_train[target]
    y_train_log = np.log1p(y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 1. NN
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', alpha=0.1, max_iter=5000, random_state=seed)
    nn.fit(X_train_scaled, y_train_log)
    
    # 2. Trend & Residuals
    trend_model = LinearRegression().fit(data_train[['Time_Index']], y_train_log)
    trend_future = trend_model.predict(df_input[['Time_Index']])
    y_residual = y_train_log - trend_model.predict(data_train[['Time_Index']])
    
    # 3. RF & XGB on Residuals
    rf = RandomForestRegressor(n_estimators=200, random_state=seed).fit(X_train, y_residual)
    xg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=seed).fit(X_train, y_residual)
    
    X_pred = df_input[valid_cols].fillna(0)
    pred_nn = np.expm1(nn.predict(scaler.transform(X_pred)))
    pred_rf = np.expm1(rf.predict(X_pred) + trend_future)
    pred_xg = np.expm1(xg.predict(X_pred) + trend_future)
    pred_trend_only = np.expm1(trend_future)
    
    return pred_nn, pred_rf, pred_xg, pred_trend_only

# ==============================================================================
# GIAO DIỆN CHÍNH (GIỮ NGUYÊN CẤU TRÚC GỐC)
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    provider = st.selectbox("Chọn Nhà Cung Cấp AI:", ["Google Gemini", "OpenAI ChatGPT"])
    api_key = st.text_input(f"API Key ({provider})", type="password")
    available_models = ["gemini-1.5-flash", "gemini-1.5-pro"] if provider == "Google Gemini" else ["gpt-4o", "gpt-3.5-turbo"]
    selected_model = st.selectbox("🤖 Chọn Model:", available_models, index=0)
    st.markdown("---")
    st.write("### 📅 Cập nhật Lịch Nghỉ Lễ")
    df_default = pd.DataFrame(DEFAULT_HOLIDAYS)
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
    USER_HOLIDAYS_MAP = dict(zip(zip(edited_df['Năm'], edited_df['Tháng']), edited_df['Số ngày nghỉ lễ']))
    st.markdown("---")
    seed_val = st.number_input("Random Seed", value=42)
    if st.button("🗑️ Xóa Cache & Reset"): st.cache_data.clear(); st.rerun()

col1, col2 = st.columns(2)
with col1: f_train = st.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
with col2: f_input = st.file_uploader("2. File Dự Báo (Input)", type=['xlsx', 'xls'])
st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# --- 1️⃣ THAM KHẢO AI (BỔ SUNG ĐỐI CHIẾU CÙNG KỲ) ---
st.subheader("1️⃣ Tham khảo AI & Chốt phương án")
c1, c2 = st.columns([2, 1])
if 'detected_months' not in st.session_state: st.session_state.detected_months = []

with c1: 
    text_data = st.text_area("Dán link báo hoặc tin tức vào đây:", height=150, placeholder="Ví dụ: Dự báo nắng nóng gay gắt tháng tới...")

with c2:
    if f_input:
        try:
            df_temp = ultra_scan_read_excel(f_input)
            if df_temp is not None:
                st.session_state.detected_months = sorted(list(set(zip(df_temp['Năm'], df_temp['Tháng']))))
        except: pass
    
    if st.button("🤖 AI Phân Tích Ngay", disabled=not selected_model, use_container_width=True):
        with st.spinner(f"Đang hỏi {provider}..."):
            val, log, reason = xu_ly_du_lieu_dinh_tinh(api_key, text_data, selected_model, provider)
            st.session_state.ai_suggestion_val = val
            st.session_state.ai_suggestion_reason = reason
            st.session_state.ai_log = log

    if 'ai_suggestion_val' in st.session_state:
        if st.session_state.ai_suggestion_val != 0:
            st.success(f"{st.session_state.ai_log}")
            # --- PHẦN BỔ SUNG: CHỌN THÁNG VÀ ĐỐI CHIẾU KWH ---
            if st.session_state.detected_months:
                opts = [f"Tháng {m}/{y}" for y, m in st.session_state.detected_months]
                sel_ref = st.selectbox("Soi cùng kỳ cho:", opts, index=len(opts)-1)
                m_val = int(sel_ref.split('/')[0].replace('Tháng ', ''))
                y_val = int(sel_ref.split('/')[1])
                y_past = y_val - 1
                if f_train:
                    try:
                        df_hist = ultra_scan_read_excel(f_train)
                        row_old = df_hist[(df_hist['Năm'] == y_past) & (df_hist['Tháng'] == m_val)]
                        if not row_old.empty:
                            v_old = row_old['Tổng thương phẩm'].values[0]
                            v_new = v_old * (1 + st.session_state.ai_suggestion_val/100)
                            st.markdown(f"**Đối chiếu ({m_val}/{y_past}):**")
                            ma, mb = st.columns(2)
                            ma.metric(f"Thực tế {y_past}", f"{v_old:,.0f}")
                            mb.metric(f"Dự tính {y_val}", f"{v_new:,.0f}", f"{st.session_state.ai_suggestion_val:+.1f}%")
                        else: st.warning(f"Không có dữ liệu {m_val}/{y_past}")
                    except: pass
            st.info(f"📝 {st.session_state.ai_suggestion_reason}")
        else: st.warning("AI không tìm thấy số cụ thể."); st.write(st.session_state.ai_suggestion_reason)

# --- CHỐT SỐ LIỆU ---
st.write("---")
st.write("### ✍️ CHỐT SỐ LIỆU")
if st.session_state.detected_months:
    ca, cb = st.columns(2)
    with ca: 
        m_str = [f"Tháng {m}/{y}" for y, m in st.session_state.detected_months]
        selected = st.multiselect("Chọn tháng áp dụng:", m_str, default=m_str)
    with cb: u_val = st.number_input("Nhập % Điều Chỉnh:", value=0.0); u_note = st.text_input("Ghi chú:", value="Thủ công")
    if st.button("💾 LƯU QUYẾT ĐỊNH"):
        for s in selected:
            m, y = int(s.split('/')[0].replace('Tháng ', '')), int(s.split('/')[1])
            st.session_state.param_dict[(y, m)] = (u_val, u_note)
        st.success(f"Đã lưu: {u_val}%")

# --- DỰ BÁO ---
if f_train and f_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        with st.spinner("Đang tính toán..."):
            df_tr, df_in = ultra_scan_read_excel(f_train), ultra_scan_read_excel(f_input)
            kiem_tra_chat_luong(df_tr, "Lịch Sử"); kiem_tra_chat_luong(df_in, "Dự Báo")
            p_nn, p_rf, p_xg, p_trend = chay_mo_hinh_goc(df_tr, df_in, USER_HOLIDAYS_MAP, seed_val)
            
            res = df_in[['Năm', 'Tháng']].copy()
            df_ck = tao_dac_trung(df_in.copy(), USER_HOLIDAYS_MAP)
            res['T7+CN'] = df_ck['So_Ngay_T7'] + df_ck['So_Ngay_CN']
            res['Lễ Tết'] = df_ck['So_Ngay_Le_Tet']
            res['Neural Network'], res['Random Forest'], res['XGBoost'] = p_nn, p_rf, p_xg

            def apply_adj(row):
                pct, note = st.session_state.param_dict.get((row['Năm'], row['Tháng']), (0.0, ""))
                f = 1 + pct/100
                return row['Neural Network']*f, row['Random Forest']*f, row['XGBoost']*f, pct, note
            
            adj_data = res.apply(apply_adj, axis=1, result_type='expand')
            res['Neural Network'], res['Random Forest'], res['XGBoost'] = adj_data[0], adj_data[1], adj_data[2]
            res['Điều Chỉnh (%)'], res['Ghi chú'] = adj_data[3], adj_data[4]
            
            # Lưu để hiện bảng giải trình XAI
            st.session_state.res_output = res
            st.session_state.trend_val = p_trend

            st.subheader("📊 Kết Quả Dự Báo")
            st.dataframe(res.style.format({'Neural Network': '{:,.0f}', 'Random Forest': '{:,.0f}', 'XGBoost': '{:,.0f}', 'Điều Chỉnh (%)': '{:+.1f}%'}))
            res['Date'] = pd.to_datetime(dict(year=res['Năm'], month=res['Tháng'], day=1))
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(res['Date'], res['Random Forest'], 's--', label='RF'); ax.legend(); st.pyplot(fig)

# ==============================================================================
# 4. MÁY SOI SEED (GIỮ NGUYÊN GỐC)
# ==============================================================================
st.markdown("---")
st.header("🔬 Find Seed")
if 'scan_current_seed' not in st.session_state: st.session_state.scan_current_seed = 0
if 'scan_history' not in st.session_state: st.session_state.scan_history = pd.DataFrame()
if f_train and f_input:
    with st.expander("BẢNG ĐIỀU KHIỂN & CHỌN MỤC TIÊU", expanded=True):
        df_sp = ultra_scan_read_excel(f_input)
        if df_sp is not None:
            l_dates = [f"Tháng {int(r['Tháng'])}/{int(r['Năm'])}" for i, r in df_sp.iterrows()]
            ct, cv = st.columns(2)
            with ct: s_m_str = st.selectbox("🎯 Chọn tháng:", l_dates, index=len(l_dates)-1); t_idx = l_dates.index(s_m_str)
            with cv: t_v = st.number_input("Giá trị mong muốn", value=740000000.0)
            c1, c2, c3 = st.columns(3)
            with c1: m_ch = st.selectbox("Mô hình:", ["Neural Network", "Random Forest", "XGBoost", "Trung Bình Cộng"])
            with c2: acc = st.slider("Sai số %", 0.1, 10.0, 1.0)
            with c3: b_size = st.number_input("Số lượng seed/lô", value=20)
            if st.button(f"▶️ Chạy Seed {st.session_state.scan_current_seed}"):
                batch = []
                for s in range(st.session_state.scan_current_seed, st.session_state.scan_current_seed + b_size):
                    p_nn, p_rf, p_xg, _ = chay_mo_hinh_goc(df_tr, df_in, USER_HOLIDAYS_MAP, seed=s)
                    v = {"Neural Network": p_nn[t_idx], "Random Forest": p_rf[t_idx], "XGBoost": p_xg[t_idx], "Trung Bình Cộng": (p_nn[t_idx]+p_rf[t_idx]+p_xg[t_idx])/3}[m_ch]
                    batch.append({"Seed": s, "Tháng": s_m_str, "Kết quả": v, "Độ lệch": v - t_v, "Trạng thái": "✅ ĐẠT" if abs(v-t_v)/t_v <= acc/100 else "❌"})
                st.session_state.scan_history = pd.concat([st.session_state.scan_history, pd.DataFrame(batch)])
                st.session_state.scan_current_seed += b_size; st.rerun()

# ==============================================================================
# 5. BẢNG GIẢI TRÌNH ĐỘ TIN CẬY (BỔ SUNG DƯỚI CÙNG)
# ==============================================================================
st.markdown("---")
if 'res_output' in st.session_state:
    st.header("🛡️ GIẢI TRÌNH LOGIC DỰ BÁO")
    r = st.session_state.res_output
    df_p = r[['Tháng', 'Năm']].copy()
    df_p['Xu hướng nền (A)'] = st.session_state.trend_val
    avg_ml = (r['Neural Network'] + r['Random Forest'] + r['XGBoost']) / 3
    df_p['Biến động ML (B)'] = avg_ml - df_p['Xu hướng nền (A)']
    df_p['Điều chỉnh (%) (C)'] = r['Điều Chỉnh (%)']
    df_p['Giá trị điều chỉnh (D)'] = avg_ml * (r['Điều Chỉnh (%)'] / 100)
    df_p['DỰ BÁO CUỐI CÙNG'] = avg_ml + df_p['Giá trị điều chỉnh (D)']
    st.dataframe(df_p.style.format({'Xu hướng nền (A)': '{:,.0f}', 'Biến động ML (B)': '{:+,.0f}', 'Giá trị điều chỉnh (D)': '{:+,.0f}', 'DỰ BÁO CUỐI CÙNG': '{:,.0f}'}), use_container_width=True)
    st.caption("A: Tăng trưởng tự nhiên | B: Máy học xử lý thời tiết/lễ | D: Tác động ngoại biên (AI/Con người).")
else: st.info("Chạy dự báo để xem giải trình tại đây.")
