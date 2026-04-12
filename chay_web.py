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

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #ffffff 0%, #cce6ff 100%); }
    h1, h2, h3, h4 { color: #003366 !important; font-family: 'Segoe UI', Tahoma, sans-serif; font-weight: 700; }
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
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
# 1. CẤU HÌNH NGÀY NGHỈ & XỬ LÝ FILE
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

def chuan_hoa_ten_cot(df):
    if df is None: return None
    df.columns = df.columns.astype(str).str.strip()
    col_map = {
        'month': 'Tháng', 'thang': 'Tháng', 'tháng': 'Tháng',
        'year': 'Năm', 'nam': 'Năm', 'năm': 'Năm',
        'tổng thương phẩm': 'Tổng thương phẩm', 'tong thuong pham': 'Tổng thương phẩm', 'sản lượng': 'Tổng thương phẩm',
        'nhiệt độ tb': 'Nhiệt độ TB', 'nhiet do tb': 'Nhiệt độ TB',
        'độ ẩm': 'Độ ẩm', 'do am': 'Độ ẩm', 'số ngày': 'Số ngày', 'so ngay': 'Số ngày'
    }
    new_cols = {col: col_map[col.lower()] for col in df.columns if col.lower() in col_map}
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

def xu_ly_du_lieu_dinh_tinh(api_key, input_data, target_model_name, provider):
    if not api_key: return 0.0, "⚠️ Chưa nhập API Key.", "Thủ công"
    text_data = input_data
    if input_data.strip().startswith("http"):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(input_data, headers=headers, timeout=5)
            soup = BeautifulSoup(resp.content, 'html.parser')
            text_data = " ".join([p.get_text() for p in soup.find_all('p')])
        except: return 0.0, "⚠️ Lỗi đọc web.", ""

    prompt = (f"Dựa trên: '{text_data[:2000]}', phụ tải điện tháng tới tại Tây Ninh tăng/giảm bao nhiêu % so với CÙNG KỲ NĂM TRƯỚC? Trả về: SỐ | LÝ DO.")
    
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(target_model_name)
            res = model.generate_content(prompt).text.strip()
        else:
            client = OpenAI(api_key=api_key)
            res = client.chat.completions.create(model=target_model_name, messages=[{"role": "user", "content": prompt}]).choices[0].message.content.strip()
        
        if "|" in res:
            parts = res.split("|")
            val = float(re.findall(r'-?\d+(?:\.\d+)?', parts[0])[0])
            return val, "✅ AI Đã phân tích!", parts[1].strip()
        return 0.0, "⚠️ AI không tìm thấy số liệu.", res
    except: return 0.0, "❌ Lỗi AI.", ""

# ==============================================================================
# 2. MÔ HÌNH DỰ BÁO (ĐÃ CẬP NHẬT TRẢ VỀ TREND)
# ==============================================================================
def tao_dac_trung(df, holidays_map):
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3,4,5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6,7,8,9,10,11] else 0)
    def get_cal(row):
        y, m = int(row['Năm']), int(row['Tháng'])
        t7, cn = dem_ngay_nghi_cuoi_tuan(y, m)
        le = holidays_map.get((y, m), 0)
        return pd.Series([t7, cn, le])
    df[['So_Ngay_T7', 'So_Ngay_CN', 'So_Ngay_Le_Tet']] = df.apply(get_cal, axis=1)
    return df

@st.cache_data(show_spinner=False)
def chay_mo_hinh_goc(df_train, df_input, holidays_map, seed=42):
    df_train = tao_dac_trung(df_train.copy(), holidays_map)
    df_input = tao_dac_trung(df_input.copy(), holidays_map)
    start_y = df_train['Năm'].min()
    df_train['Time_Index'] = (df_train['Năm'] - start_y) * 12 + df_train['Tháng']
    df_input['Time_Index'] = (df_input['Năm'] - start_y) * 12 + df_input['Tháng']
    
    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'So_Ngay_T7', 'So_Ngay_CN', 'So_Ngay_Le_Tet', 'Mua_Nong', 'Mua_Mua']
    target = 'Tổng thương phẩm'
    X_train, y_train = df_train[features], df_train[target]
    y_log = np.log1p(y_train)
    
    # ML Models
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), solver='lbfgs', random_state=seed).fit(X_sc, y_log)
    
    trend_m = LinearRegression().fit(df_train[['Time_Index']], y_log)
    y_res = y_log - trend_m.predict(df_train[['Time_Index']])
    
    rf = RandomForestRegressor(n_estimators=200, random_state=seed).fit(X_train, y_res)
    xg = xgb.XGBRegressor(n_estimators=100, random_state=seed).fit(X_train, y_res)
    
    # Predict
    X_f = df_input[features].fillna(0)
    T_f = df_input[['Time_Index']]
    p_trend = trend_m.predict(T_f)
    p_nn = nn.predict(scaler.transform(X_f))
    p_rf = rf.predict(X_f) + p_trend
    p_xg = xg.predict(X_f) + p_trend
    
    return np.expm1(p_nn), np.expm1(p_rf), np.expm1(p_xg), np.expm1(p_trend)

# ==============================================================================
# 3. GIAO DIỆN CHÍNH
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    provider = st.selectbox("Chọn AI:", ["Google Gemini", "OpenAI ChatGPT"])
    api_key = st.text_input(f"API Key ({provider})", type="password")
    
    # Model selection logic
    available_models = ["gemini-1.5-flash", "gemini-1.5-pro"] if provider == "Google Gemini" else ["gpt-4o", "gpt-3.5-turbo"]
    selected_model = st.selectbox("🤖 Model:", available_models)
    
    st.markdown("---")
    df_default = pd.DataFrame(DEFAULT_HOLIDAYS)
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
    USER_HOLIDAYS_MAP = dict(zip(zip(edited_df['Năm'], edited_df['Tháng']), edited_df['Số ngày nghỉ lễ']))
    seed_val = st.number_input("Seed", value=42)

col1, col2 = st.columns(2)
with col1: f_train = st.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
with col2: f_input = st.file_uploader("2. File Dự Báo (Input)", type=['xlsx', 'xls'])

# --- AI & CHỐT PHƯƠNG ÁN ---
st.write("---")
st.subheader("1️⃣ Tham khảo AI & Chốt phương án")
c1, c2 = st.columns([2, 1])
if 'detected_months' not in st.session_state: st.session_state.detected_months = []

with c1: 
    text_data = st.text_area("Link báo/Tin tức:", height=150, placeholder="Dán nội dung thời tiết/kinh tế...")

with c2:
    if f_input:
        try:
            df_temp = ultra_scan_read_excel(f_input)
            if df_temp is not None:
                st.session_state.detected_months = sorted(list(set(zip(df_temp['Năm'], df_temp['Tháng']))))
        except: pass
    
    if st.button("🤖 AI Phân Tích", disabled=not selected_model, use_container_width=True):
        with st.spinner("Đang tính..."):
            val, log, reason = xu_ly_du_lieu_dinh_tinh(api_key, text_data, selected_model, provider)
            st.session_state.ai_suggestion_val = val
            st.session_state.ai_suggestion_reason = reason
            st.session_state.ai_log = log

    if 'ai_suggestion_val' in st.session_state:
        if st.session_state.ai_suggestion_val != 0:
            st.success(st.session_state.ai_log)
            if st.session_state.detected_months:
                opts = [f"Tháng {m}/{y}" for y, m in st.session_state.detected_months]
                sel_ref = st.selectbox("Soi cùng kỳ cho:", opts, index=len(opts)-1)
                m_v = int(sel_ref.split('/')[0].replace('Tháng ', ''))
                y_v = int(sel_ref.split('/')[1])
                y_p = y_v - 1
                
                if f_train:
                    try:
                        df_h = ultra_scan_read_excel(f_train)
                        r_o = df_h[(df_h['Năm'] == y_p) & (df_h['Tháng'] == m_v)]
                        if not r_o.empty:
                            v_o = r_o['Tổng thương phẩm'].values[0]
                            v_n = v_o * (1 + st.session_state.ai_suggestion_val/100)
                            st.markdown(f"**Đối chiếu ({m_v}/{y_p}):**")
                            ma, mb = st.columns(2)
                            ma.metric(f"Thực tế {y_p}", f"{v_o:,.0f}")
                            mb.metric(f"Dự tính {y_v}", f"{v_n:,.0f}", f"{st.session_state.ai_suggestion_val:+.1f}%")
                    except: pass
            st.info(f"📝 Lý do: {st.session_state.ai_suggestion_reason}")

# --- CHỐT SỐ & CHẠY ---
st.write("---")
if st.session_state.detected_months:
    ca, cb = st.columns(2)
    with ca: sel_m = st.multiselect("Tháng áp dụng:", [f"Tháng {m}/{y}" for y, m in st.session_state.detected_months])
    with cb: u_val = st.number_input("% Điều Chỉnh:", value=0.0)
    if st.button("💾 LƯU"):
        if 'param_dict' not in st.session_state: st.session_state.param_dict = {}
        for s in sel_m:
            m = int(s.split('/')[0].replace('Tháng ', ''))
            y = int(s.split('/')[1])
            st.session_state.param_dict[(y, m)] = (u_val, "Manual")
        st.success("Đã lưu!")

if f_train and f_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        df_tr = ultra_scan_read_excel(f_train)
        df_in = ultra_scan_read_excel(f_input)
        p_nn, p_rf, p_xg, p_trd = chay_mo_hinh_goc(df_tr, df_in, USER_HOLIDAYS_MAP, seed_val)
        
        res = df_in[['Năm', 'Tháng']].copy()
        res['Neural Network'], res['Random Forest'], res['XGBoost'] = p_nn, p_rf, p_xg
        
        def adjust(row):
            adj, note = st.session_state.get('param_dict', {}).get((row['Năm'], row['Tháng']), (0.0, ""))
            f = 1 + adj/100
            return row['Neural Network']*f, row['Random Forest']*f, row['XGBoost']*f, adj
        
        res[['Neural Network', 'Random Forest', 'XGBoost', 'Điều Chỉnh (%)']] = res.apply(adjust, axis=1, result_type='expand')
        
        # Lưu Session để hiện bảng giải trình ở cuối
        st.session_state.res_output = res
        st.session_state.trend_val = p_trd
        
        st.subheader("📊 Kết Quả")
        st.dataframe(res.style.format({'Neural Network': '{:,.0f}', 'Random Forest': '{:,.0f}', 'XGBoost': '{:,.0f}', 'Điều Chỉnh (%)': '{:+.1f}%'}))
        
        # Đồ thị
        res['Date'] = pd.to_datetime(dict(year=res['Năm'], month=res['Tháng'], day=1))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(res['Date'], res['Random Forest'], label='Dự báo (RF)')
        ax.legend(); st.pyplot(fig)

# ==============================================================================
# 4. BẢNG GIẢI TRÌNH (DƯỚI CÙNG)
# ==============================================================================
st.markdown("---")
if 'res_output' in st.session_state:
    st.header("🔬 GIẢI TRÌNH LOGIC & ĐỘ TIN CẬY")
    res = st.session_state.res_output
    df_p = res[['Tháng', 'Năm']].copy()
    df_p['Xu hướng nền (A)'] = st.session_state.trend_val
    avg_raw = (res['Neural Network'] + res['Random Forest'] + res['XGBoost']) / 3
    df_p['Biến động ML (B)'] = avg_raw - df_p['Xu hướng nền (A)']
    df_p['Điều chỉnh (%) (C)'] = res['Điều Chỉnh (%)']
    df_p['Giá trị điều chỉnh (D)'] = avg_raw * (res['Điều Chỉnh (%)'] / 100)
    df_p['TỔNG DỰ BÁO CHỐT'] = avg_raw + df_p['Giá trị điều chỉnh (D)']
    
    st.dataframe(df_p.style.format({
        'Xu hướng nền (A)': '{:,.0f}', 'Biến động ML (B)': '{:+,.0f}', 
        'Điều chỉnh (%) (C)': '{:+.1f}%', 'Giá trị điều chỉnh (D)': '{:+,.0f}', 
        'TỔNG DỰ BÁO CHỐT': '{:,.0f}'
    }))
    with st.expander("📚 Cách đọc"):
        st.write("A: Tăng trưởng tự nhiên | B: Máy học xử lý thời tiết/lễ | D: AI/Người điều chỉnh.")
else:
    st.info("💡 Chạy dự báo để xem bảng giải trình logic tại đây.")
