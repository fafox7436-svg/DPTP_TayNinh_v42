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
    .author-subtitle { color: #444; font-size: 16px; font-style: italic; margin-top: -15px; margin-bottom: 10px; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([8, 2], vertical_alignment="center")
with col1: 
    st.markdown("<h3>HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN TỈNH TÂY NINH</h3>", unsafe_allow_html=True)
    st.markdown('<p class="author-subtitle">Tác giả: Lê Minh Trí</p>', unsafe_allow_html=True)

with col2: 
    try: st.image("image_1.png", use_column_width=True)
    except: st.write("EVN SPC")
st.markdown("---")

try: import google.generativeai as genai; HAS_GEMINI = True
except: HAS_GEMINI = False
try: from openai import OpenAI; HAS_OPENAI = True
except: HAS_OPENAI = False

# ==============================================================================
# 1. CẤU HÌNH NGÀY NGHỈ
# ==============================================================================
DEFAULT_HOLIDAYS = [
    {"Năm": 2023, "Tháng": 1, "Tết Âm": 7, "Lễ Nhỏ": 1},
    {"Năm": 2023, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2023, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 3},
    {"Năm": 2023, "Tháng": 9, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2024, "Tháng": 1, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2024, "Tháng": 2, "Tết Âm": 7, "Lễ Nhỏ": 0},
    {"Năm": 2024, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 3},
    {"Năm": 2024, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2024, "Tháng": 9, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2025, "Tháng": 1, "Tết Âm": 7, "Lễ Nhỏ": 1},
    {"Năm": 2025, "Tháng": 2, "Tết Âm": 2, "Lễ Nhỏ": 0},
    {"Năm": 2025, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2025, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2025, "Tháng": 9, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2026, "Tháng": 1, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2026, "Tháng": 2, "Tết Âm": 7, "Lễ Nhỏ": 0},
    {"Năm": 2026, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 3},
    {"Năm": 2026, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2026, "Tháng": 9, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2027, "Tháng": 1, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2027, "Tháng": 2, "Tết Âm": 7, "Lễ Nhỏ": 0},
    {"Năm": 2027, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2027, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 1},
]

# ==============================================================================
# 2. XỬ LÝ DỮ LIỆU & ĐẶC TRƯNG CẬP NHẬT
# ==============================================================================
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
    new_cols = {col: col_map.get(col.lower(), col) for col in df.columns}
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
                    return chuan_hoa_ten_cot(pd.read_excel(uploaded_file, sheet_name=sheet_name, header=i))
        uploaded_file.seek(0)
        return chuan_hoa_ten_cot(pd.read_excel(uploaded_file, header=0))
    except: return None

def tao_dac_trung(df, holidays_map):
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3,4,5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6,7,8,9,10,11] else 0)
    
    def get_calendar_info(row):
        y, m = int(row['Năm']), int(row['Tháng'])
        num_days = calendar.monthrange(y, m)[1]
        t2, t7, cn = 0, 0, 0
        for day in range(1, num_days + 1):
            wd = calendar.weekday(y, m, day)
            if wd == 0: t2 += 1      # Đếm Thứ 2
            elif wd == 5: t7 += 1    # Đếm Thứ 7
            elif wd == 6: cn += 1    # Đếm Chủ nhật
            
        tet_am, le_nho = holidays_map.get((y, m), (0, 0))
        return pd.Series([t2, t7, cn, tet_am, le_nho])
        
    df[['So_Ngay_T2', 'So_Ngay_T7', 'So_Ngay_CN', 'So_Ngay_Tet_Am', 'So_Ngay_Le_Nho']] = df.apply(get_calendar_info, axis=1)
    df['Bien_Ngoai_Sinh'] = 0
    return df

# ==============================================================================
# 3. MÔ HÌNH HỌC THEO SẢN LƯỢNG TRUNG BÌNH NGÀY
# ==============================================================================
@st.cache_data(show_spinner=False)
def chay_mo_hinh_goc(df_train, df_input, holidays_map, seed=42):
    df_train = tao_dac_trung(df_train.copy(), holidays_map)
    df_input = tao_dac_trung(df_input.copy(), holidays_map)
    
    start_year = df_train['Năm'].min()
    df_train['Time_Index'] = (df_train['Năm'] - start_year) * 12 + df_train['Tháng']
    df_input['Time_Index'] = (df_input['Năm'] - start_year) * 12 + df_input['Tháng']

    # Đưa biến Thứ 2 vào bộ dữ liệu huấn luyện
    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 
                'So_Ngay_T2', 'So_Ngay_T7', 'So_Ngay_CN', 'So_Ngay_Tet_Am', 'So_Ngay_Le_Nho', 
                'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_cols = [c for c in features if c in df_train.columns and c in df_input.columns]
    
    data_train = df_train.dropna(subset=valid_cols + ['Tổng thương phẩm', 'Số ngày'])
    X_train = data_train[valid_cols]
    
    # BƯỚC NGOẶT: AI học theo TRUNG BÌNH NGÀY thay vì Tổng tháng
    y_train_daily = data_train['Tổng thương phẩm'] / data_train['Số ngày']
    y_train_log = np.log1p(y_train_daily)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    df_pred = df_input.copy()
    X_pred = df_pred[valid_cols].fillna(0)
    
    # 1. Huấn luyện 
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', alpha=0.1, max_iter=5000, random_state=seed)
    nn.fit(X_train_scaled, y_train_log)
    
    trend_model = LinearRegression()
    trend_model.fit(data_train[['Time_Index']], y_train_log)
    trend_future = trend_model.predict(df_pred[['Time_Index']])
    y_residual = y_train_log - trend_model.predict(data_train[['Time_Index']])
    
    rf = RandomForestRegressor(n_estimators=200, random_state=seed)
    rf.fit(X_train, y_residual)
    
    xg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, reg_alpha=0.1, random_state=seed)
    xg.fit(X_train, y_residual)
    
    # 2. Dự báo TRUNG BÌNH NGÀY
    pred_nn_daily_log = nn.predict(scaler.transform(X_pred))
    pred_rf_daily_log = rf.predict(X_pred) + trend_future
    pred_xg_daily_log = xg.predict(X_pred) + trend_future
    
    # Khôi phục giá trị thực và nhân lại với Số ngày để ra TỔNG THÁNG
    pred_nn = np.expm1(pred_nn_daily_log) * df_pred['Số ngày']
    pred_rf = np.expm1(pred_rf_daily_log) * df_pred['Số ngày']
    pred_xg = np.expm1(pred_xg_daily_log) * df_pred['Số ngày']
    pred_trend = np.expm1(trend_future) * df_pred['Số ngày']
    
    # 3. TRÍCH XUẤT HỆ SỐ NGÀY (k)
    coef_list = []
    for idx, row in df_pred.iterrows():
        D = row['Số ngày']
        t_idx = df_pred[['Time_Index']].loc[[idx]]
        t_val = trend_model.predict(t_idx)[0]
        
        # Dòng dữ liệu cơ sở: Ép toàn bộ các ngày đặc biệt về 0 (Chỉ còn T3, 4, 5, 6)
        row_base = X_pred.loc[[idx]].copy()
        for col in ['So_Ngay_T2', 'So_Ngay_T7', 'So_Ngay_CN', 'So_Ngay_Tet_Am', 'So_Ngay_Le_Nho']:
            row_base[col] = 0
            
        # Tìm giá trị x (Sản lượng 1 ngày cơ sở)
        x_nn = np.expm1(nn.predict(scaler.transform(row_base))[0])
        x_rf = np.expm1(rf.predict(row_base)[0] + t_val)
        x_xg = np.expm1(xg.predict(row_base)[0] + t_val)
        
        # Hàm tính toán ngược hệ số k
        def get_k(col_name):
            row_test = row_base.copy()
            row_test[col_name] = 1 # Chèn thử 1 ngày đặc biệt vào tháng
            
            y_nn = np.expm1(nn.predict(scaler.transform(row_test))[0])
            y_rf = np.expm1(rf.predict(row_test)[0] + t_val)
            y_xg = np.expm1(xg.predict(row_test)[0] + t_val)
            
            # Giải phương trình tìm k: (D*y - (D-1)*x) / x
            k_nn = (D * y_nn - (D - 1) * x_nn) / x_nn if x_nn else 1.0
            k_rf = (D * y_rf - (D - 1) * x_rf) / x_rf if x_rf else 1.0
            k_xg = (D * y_xg - (D - 1) * x_xg) / x_xg if x_xg else 1.0
            return k_nn, k_rf, k_xg

        k_t2_nn, k_t2_rf, k_t2_xg = get_k('So_Ngay_T2')
        k_t7_nn, k_t7_rf, k_t7_xg = get_k('So_Ngay_T7')
        k_cn_nn, k_cn_rf, k_cn_xg = get_k('So_Ngay_CN')
        k_tet_nn, k_tet_rf, k_tet_xg = get_k('So_Ngay_Tet_Am')
        k_le_nn, k_le_rf, k_le_xg = get_k('So_Ngay_Le_Nho')
        
        coef_list.append({
            'Tháng': f"{int(row['Tháng'])}/{int(row['Năm'])}",
            'x (Sản lượng ngày T3-T6) kWh': x_xg, # Lấy số của XGBoost làm đại diện hiển thị
            'T2 (NN)': k_t2_nn, 'T2 (RF)': k_t2_rf, 'T2 (XGB)': k_t2_xg,
            'T7 (NN)': k_t7_nn, 'T7 (RF)': k_t7_rf, 'T7 (XGB)': k_t7_xg,
            'CN (NN)': k_cn_nn, 'CN (RF)': k_cn_rf, 'CN (XGB)': k_cn_xg,
            'Tết (NN)': k_tet_nn, 'Tết (RF)': k_tet_rf, 'Tết (XGB)': k_tet_xg,
            'Lễ (NN)': k_le_nn, 'Lễ (RF)': k_le_rf, 'Lễ (XGB)': k_le_xg,
        })
        
    df_coef = pd.DataFrame(coef_list)
    return pred_nn, pred_rf, pred_xg, pred_trend, df_coef

# ==============================================================================
# GIAO DIỆN CHÍNH
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    st.write("### 📅 Cập nhật Lịch Nghỉ Lễ")
    df_default = pd.DataFrame(DEFAULT_HOLIDAYS)
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
    USER_HOLIDAYS_MAP = {}
    for _, row in edited_df.iterrows():
        USER_HOLIDAYS_MAP[(int(row['Năm']), int(row['Tháng']))] = (int(row['Tết Âm']), int(row['Lễ Nhỏ']))
    
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

# --- CHẠY DỰ BÁO ---
if f_train and f_input:
    if st.button("🚀 CHẠY DỰ BÁO", type="primary"):
        with st.spinner("Đang tính toán lại thuật toán Hệ số ngày..."):
            df_train_main = ultra_scan_read_excel(f_train)
            df_input_main = ultra_scan_read_excel(f_input)
            
            if df_train_main is not None and df_input_main is not None:
                # Chạy mô hình
                pred_nn, pred_rf, pred_xg, p_trend, df_coef = chay_mo_hinh_goc(df_train_main, df_input_main, USER_HOLIDAYS_MAP, seed_val)
                
                res = df_input_main[['Năm', 'Tháng']].copy()
                df_check = tao_dac_trung(df_input_main.copy(), USER_HOLIDAYS_MAP)
                res['T2'] = df_check['So_Ngay_T2']
                res['T7'] = df_check['So_Ngay_T7']
                res['CN'] = df_check['So_Ngay_CN']
                res['Tết'] = df_check['So_Ngay_Tet_Am']
                res['Lễ'] = df_check['So_Ngay_Le_Nho']
                
                res['Neural Network'] = pred_nn
                res['Random Forest'] = pred_rf
                res['XGBoost'] = pred_xg

                if 'Tổng thương phẩm' in df_train_main.columns:
                    actual = df_train_main[['Năm', 'Tháng', 'Tổng thương phẩm']]
                    res = pd.merge(res, actual, on=['Năm', 'Tháng'], how='left')
                    res.rename(columns={'Tổng thương phẩm': 'Thực Tế'}, inplace=True)
                
                st.session_state.res_output = res.copy()
                st.session_state.df_coef = df_coef # Lưu bảng Hệ số k
                
                st.subheader("📊 Kết Quả Dự Báo Tổng Tháng")
                cols = ['Tháng', 'Năm', 'Thực Tế', 'T2', 'T7', 'CN', 'Tết', 'Lễ', 'Neural Network', 'Random Forest', 'XGBoost']
                cols = [c for c in cols if c in res.columns]
                
                st.dataframe(res[cols].style.format({
                    'Thực Tế': '{:,.0f}', 'Neural Network': '{:,.0f}', 
                    'Random Forest': '{:,.0f}', 'XGBoost': '{:,.0f}'
                }), use_container_width=True)

# ==============================================================================
# BẢNG PHÂN TÍCH HỆ SỐ NGÀY ĐỘNG (DAY-TYPE COEFFICIENT)
# ==============================================================================
if 'df_coef' in st.session_state:
    st.markdown("---")
    st.header("⚙️ THEO DÕI HỆ SỐ NGÀY ÁP DỤNG ($k$)")
    st.write("Bảng phân tích mức độ học sâu của AI. Hệ số **$k = 1.0$** đại diện cho sản lượng của ngày cơ sở (T3, T4, T5, T6). Nếu $k = 0.85$, nghĩa là ngày đó tiêu thụ bằng 85% so với ngày thường.")
    
    df_show_coef = st.session_state.df_coef
    
    def color_coef(val):
        if pd.isna(val) or type(val) == str: return ''
        if val < 0.95: return 'color: #d9534f; font-weight: bold;' # Đỏ cho hệ số giảm (Nghỉ)
        elif val > 1.05: return 'color: #5cb85c; font-weight: bold;' # Xanh cho tăng đột biến
        return 'color: #777777;' # Xám nếu gần như ngày thường (1.0)

    tab1, tab2, tab3 = st.tabs(["🚀 XGBoost", "🌳 Random Forest", "📊 Neural Network"])
    
    with tab1:
        cols_xg = ['Tháng', 'x (Sản lượng ngày T3-T6) kWh', 'T2 (XGB)', 'T7 (XGB)', 'CN (XGB)', 'Tết (XGB)', 'Lễ (XGB)']
        st.dataframe(df_show_coef[cols_xg].style.format({
            'x (Sản lượng ngày T3-T6) kWh': '{:,.0f}',
            'T2 (XGB)': '{:.2f}', 'T7 (XGB)': '{:.2f}', 'CN (XGB)': '{:.2f}', 'Tết (XGB)': '{:.2f}', 'Lễ (XGB)': '{:.2f}'
        }).map(color_coef), use_container_width=True)
        
    with tab2:
        cols_rf = ['Tháng', 'x (Sản lượng ngày T3-T6) kWh', 'T2 (RF)', 'T7 (RF)', 'CN (RF)', 'Tết (RF)', 'Lễ (RF)']
        st.dataframe(df_show_coef[cols_rf].style.format({
            'x (Sản lượng ngày T3-T6) kWh': '{:,.0f}',
            'T2 (RF)': '{:.2f}', 'T7 (RF)': '{:.2f}', 'CN (RF)': '{:.2f}', 'Tết (RF)': '{:.2f}', 'Lễ (RF)': '{:.2f}'
        }).map(color_coef), use_container_width=True)
        
    with tab3:
        cols_nn = ['Tháng', 'x (Sản lượng ngày T3-T6) kWh', 'T2 (NN)', 'T7 (NN)', 'CN (NN)', 'Tết (NN)', 'Lễ (NN)']
        st.dataframe(df_show_coef[cols_nn].style.format({
            'x (Sản lượng ngày T3-T6) kWh': '{:,.0f}',
            'T2 (NN)': '{:.2f}', 'T7 (NN)': '{:.2f}', 'CN (NN)': '{:.2f}', 'Tết (NN)': '{:.2f}', 'Lễ (NN)': '{:.2f}'
        }).map(color_coef), use_container_width=True)
