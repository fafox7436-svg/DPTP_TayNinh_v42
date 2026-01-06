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
import random # Thêm thư viện ngẫu nhiên

# --- KIỂM TRA THƯ VIỆN ---
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPER = True
except ImportError:
    HAS_SCRAPER = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ Thống Dự Báo Phụ Tải", layout="wide")
st.title("HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN")
st.markdown("---")

# ==============================================================================
# 1. MODULE ĐỌC LINK & AI
# ==============================================================================
def lay_noi_dung_tu_link(url):
    if not HAS_SCRAPER: return None, "⚠️ Thiếu thư viện đọc web. Hãy dán Text."
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = " ".join([p.get_text() for p in soup.find_all('p')])
            if len(text_content) < 50: return None, "⚠️ Web chặn bot, hãy copy text."
            return text_content, "✅ Đã đọc link!"
        return None, f"⚠️ Lỗi link: {response.status_code}"
    except Exception as e: return None, f"❌ Lỗi: {str(e)}"

def xu_ly_du_lieu_dinh_tinh(api_key, input_data):
    if not api_key: return 0.0, "⚠️ Chưa nhập API Key.", ""
    
    final_text = input_data
    status_msg = ""
    if input_data.strip().startswith("http"):
        with st.spinner("Đang đọc bài báo..."):
            extracted_text, msg = lay_noi_dung_tu_link(input_data)
            if extracted_text:
                final_text = extracted_text
                status_msg = f"{msg}\n\n"
            else: return 0.0, msg, ""

    try:
        genai.configure(api_key=api_key)
        final_model = "gemini-pro"
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if "flash" in m.name: final_model = m.name; break
                    final_model = m.name
        except: pass

        model = genai.GenerativeModel(final_model)
        prompt = (
            f"Đọc tin sau: '{final_text[:4000]}'. "
            "Xác định % tăng/giảm sản lượng điện. "
            "Trả về: SỐ | LÝ DO. (VD: -1.5 | Giảm 1.5%)"
        )
        response = model.generate_content(prompt)
        text_res = response.text.strip()
        
        if "|" in text_res:
            parts = text_res.split("|")
            val_str = parts[0].strip()
            reason = parts[1].strip()
            import re
            match = re.search(r'-?\d+(\.\d+)?', val_str)
            if match:
                val = float(match.group())
                val = max(min(val, 20.0), -20.0)
                return val, f"{status_msg}✅ Xong!", reason
        return 0.0, "⚠️ Không rõ số liệu.", ""
    except Exception as e: return 0.0, f"❌ Lỗi AI: {str(e)}", ""

# ==============================================================================
# 2. HÀM TÍNH TOÁN (KHÔNG CACHE)
# ==============================================================================
def feature_engineering(df):
    def check_tet(row):
        try:
            nam, thang = int(row['Năm']), int(row['Tháng'])
            lich_tet = {2023: 1, 2024: 2, 2025: 1, 2026: 2, 2027: 2}
            return 1 if nam in lich_tet and lich_tet[nam] == thang else 0
        except: return 0
    df['Co_Tet'] = df.apply(check_tet, axis=1)
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10, 11] else 0)
    return df

# KHÔNG DÙNG @st.cache_data
def chay_mo_phong(df_train_origin, df_input_origin, exogenous_params, user_seed):
    df_train = df_train_origin.copy()
    df_input = df_input_origin.copy()

    df_train['Bien_Ngoai_Sinh'] = 0 
    df_input['Bien_Ngoai_Sinh'] = 0

    df_train = feature_engineering(df_train)
    df_input = feature_engineering(df_input)

    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_features = [f for f in features if f in df_train.columns and f in df_input.columns]
    target = 'Tổng thương phẩm'

    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # MODELS
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=user_seed)
    nn.fit(X_scaled, y)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    xgb_model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, subsample=0.7, random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)

    # PREDICT
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    df_pred[valid_features] = df_pred[valid_features].fillna(0)
    
    base_nn = nn.predict(scaler.transform(df_pred[valid_features]))
    base_rf = rf.predict(df_pred[valid_features])
    base_xgb = xgb_model.predict(df_pred[valid_features])
    
    # ADJUSTMENT
    def get_adjustment_details(row):
        data = exogenous_params.get((int(row['Năm']), int(row['Tháng'])), (0.0, ""))
        try: pct_change = float(data[0])
        except: pct_change = 0.0
        reason = data[1]
        factor = 1.0 + (pct_change / 100.0)
        return pct_change, factor, reason

    adj_data = df_pred.apply(get_adjustment_details, axis=1, result_type='expand')
    df_pred['Gemini_Pct'] = adj_data[0]
    df_pred['Adj_Factor'] = adj_data[1]
    df_pred['Lý do AI'] = adj_data[2]

    df_pred['NN'] = base_nn * df_pred['Adj_Factor']
    df_pred['RF'] = base_rf * df_pred['Adj_Factor']
    df_pred['XGB'] = base_xgb * df_pred['Adj_Factor']
    
    return df_pred[['Năm', 'Tháng', 'NN', 'RF', 'XGB', 'Gemini_Pct', 'Adj_Factor', 'Lý do AI']]

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    api_key = st.text_input("Nhập Khóa API", value="", type="password")
    st.markdown("---")
    
    # --- CHẾ ĐỘ NGẪU NHIÊN ĐỂ THOÁT KHỎI SỐ CŨ ---
    use_random = st.checkbox("🎲 Chế độ Ngẫu Nhiên (Mỗi lần bấm ra số khác)", value=False)
    if use_random:
        # Nếu chọn ngẫu nhiên, tạo số mới mỗi giây
        selected_seed = int(time.time()) % 1000 
        st.caption(f"Đang dùng Seed ngẫu nhiên: {selected_seed}")
    else:
        # Nếu không, dùng số cố định (Mặc định 42)
        selected_seed = st.number_input("Random Seed (Cố định)", value=42, step=1)

col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. Dữ liệu Lịch sử (Train)", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. Dữ liệu Dự báo (Input)", type=['xlsx', 'xls'])

st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# PHẦN 1: PHÂN TÍCH
st.subheader("📰 Phân Tích Thông Tin")
c1, c2 = st.columns([2, 1])

if 'detected_months' not in st.session_state: st.session_state.detected_months = []

with c1: 
    placeholder_text = "Dán Link bài báo vào đây." if HAS_SCRAPER else "Hãy dán nội dung văn bản vào đây."
    text_data = st.text_area("Nhập Link hoặc Nội dung:", height=100, placeholder=placeholder_text)

with c2:
    if uploaded_input:
        try:
            df_in_temp = pd.read_excel(uploaded_input)
            pairs = sorted(list(set(zip(df_in_temp['Năm'], df_in_temp['Tháng']))))
            st.session_state.detected_months = pairs
        except: pass
    
    if st.button("Phân Tích AI"):
        with st.spinner("Gemini đang đọc..."):
            val, log, reason = xu_ly_du_lieu_dinh_tinh(api_key, text_data)
        if "Lỗi" in log: st.warning(log)
        else: 
            st.success(log)
            st.session_state.temp_score = val 
            st.session_state.temp_reason = reason 

# CHỌN THÁNG VÀ SỬA ĐIỂM
if 'detected_months' in st.session_state and st.session_state.detected_months:
    st.write("---")
    st.write("### 🛠️ Thiết Lập Kịch Bản:")
    
    c_a, c_b = st.columns(2)
    with c_a:
        months_str = [f"Tháng {m}/{y}" for y, m in st.session_state.detected_months]
        selected_months_str = st.multiselect("Chọn tháng áp dụng:", months_str, default=months_str)
    
    with c_b:
        current_val = st.session_state.get('temp_score', 0.0)
        final_score = st.number_input("Điều chỉnh % (Nhập 0 để Hủy):", value=float(current_val), step=0.1, format="%.2f")
    
    if st.button("💾 LƯU KỊCH BẢN (Bắt buộc bấm)"):
        temp_dict = {} 
        for s in selected_months_str:
            parts = s.split('/')
            m = int(parts[0].replace('Tháng ', ''))
            y = int(parts[1])
            temp_dict[(y, m)] = (final_score, st.session_state.get('temp_reason', ''))
        st.session_state.param_dict = temp_dict
        st.success(f"✅ Đã lưu kịch bản: {final_score}%")

st.write("---")

# PHẦN 2: CHẠY DỰ BÁO
if uploaded_train and uploaded_input:
    if st.button("🚀 THỰC HIỆN DỰ BÁO", type="primary"):
        try:
            try: df_train_org = pd.read_excel(uploaded_train, sheet_name='Bang tinh 5 tppt')
            except: df_train_org = pd.read_excel(uploaded_train, sheet_name=0)
            df_input_org = pd.read_excel(uploaded_input)

            # --- DEBUG PANEL: HIỂN THỊ THÔNG SỐ ĐANG CHẠY ---
            st.info(f"🔍 DEBUG: Đang chạy với Seed = {selected_seed} | Số tháng điều chỉnh: {len(st.session_state.param_dict)}")
            
            with st.spinner(f"Đang tính toán (Seed={selected_seed})..."):
                df_final = chay_mo_phong(df_train_org, df_input_org, st.session_state.param_dict, selected_seed)
                
                df_actual = df_train_org[['Năm', 'Tháng', 'Tổng thương phẩm']].copy()
                df_final = pd.merge(df_final, df_actual, on=['Năm', 'Tháng'], how='left')
                df_final.rename(columns={'Tổng thương phẩm': 'Thuc_te'}, inplace=True)
                df_final['ThoiGian'] = pd.to_datetime(dict(year=df_final['Năm'], month=df_final['Tháng'], day=1))

                mask = df_final['Thuc_te'].notnull()
                df_final['Loi_NN(%)'] = np.where(mask, abs(df_final['Thuc_te'] - df_final['NN'])/df_final['Thuc_te']*100, np.nan)
                df_final['Loi_RF(%)'] = np.where(mask, abs(df_final['Thuc_te'] - df_final['RF'])/df_final['Thuc_te']*100, np.nan)
                df_final['Loi_XGB(%)'] = np.where(mask, abs(df_final['Thuc_te'] - df_final['XGB'])/df_final['Thuc_te']*100, np.nan)

                st.subheader("📊 Bảng Kết Quả Chi Tiết")
                
                df_final['Gemini (%)'] = df_final['Gemini_Pct'].apply(lambda x: f"{x:+.2f}%" if x != 0 else "-")
                
                cols_display = {
                    'Tháng': 'Tháng',
                    'Thuc_te': 'Thực Tế',
                    'Gemini (%)': 'Gemini (%)',
                    'Lý do AI': 'Lý do Điều Chỉnh',
                    'NN': 'Neural Network',      
                    'Loi_NN(%)': 'Sai số NN (%)',
                    'RF': 'Random Forest',       
                    'Loi_RF(%)': 'Sai số RF (%)',
                    'XGB': 'XGBoost',            
                    'Loi_XGB(%)': 'Sai số XGB (%)'
                }
                
                df_show = df_final[['Năm'] + list(cols_display.keys())].rename(columns=cols_display)
                
                def highlight_accuracy(val):
                    if isinstance(val, float) and val <= 1.5:
                        return 'background-color: #ccffcc; color: green; font-weight: bold' 
                    return ''

                st.dataframe(df_show.style.format({
                    'Thực Tế': '{:,.0f}',
                    'Neural Network': '{:,.0f}', 'Sai số NN (%)': '{:.2f}%',
                    'Random Forest': '{:,.0f}', 'Sai số RF (%)': '{:.2f}%',
                    'XGBoost': '{:,.0f}', 'Sai số XGB (%)': '{:.2f}%'
                }).applymap(highlight_accuracy, subset=['Sai số NN (%)', 'Sai số RF (%)', 'Sai số XGB (%)']), 
                use_container_width=True)

                st.subheader("📈 Biểu Đồ So Sánh")
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(df_final['ThoiGian'], df_final['NN'], 's-', color='#d62728', label='Neural Network', linewidth=2)
                ax.plot(df_final['ThoiGian'], df_final['RF'], 'x--', color='#1f77b4', label='Random Forest', linewidth=1.5, alpha=0.7)
                ax.plot(df_final['ThoiGian'], df_final['XGB'], '^-.', color='#2ca02c', label='XGBoost', linewidth=2, alpha=0.9)

                if df_final['Thuc_te'].notnull().any():
                    ax.plot(df_final['ThoiGian'], df_final['Thuc_te'], 'o-', color='black', linewidth=3, label='Thực Tế', zorder=10)

                ax.set_title(f"Kết Quả Dự Báo (Seed: {selected_seed})")
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_show.to_excel(writer, index=False, sheet_name='Ket_qua')
                st.download_button("📥 Tải Báo Cáo Excel", buffer.getvalue(), "Ket_qua_Final.xlsx")

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
