import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import io
import json

# Thử import các thư viện nâng cao
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự Báo Phụ Tải Thông Minh (Gemini)", layout="wide")
st.title("⚡ HỆ THỐNG DỰ BÁO PHỤ TẢI (TÍCH HỢP GEMINI)")

# ==============================================================================
# 1. HÀM GỌI GEMINI ĐỂ PHÂN TÍCH TIN TỨC
# ==============================================================================
def ask_gemini_to_rate_event(api_key, news_content):
    """Gửi nội dung báo chí cho Gemini và nhận về điểm số (-2 đến 2)"""
    if not api_key:
        return None, "Vui lòng nhập API Key trước."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Câu lệnh (Prompt) ra lệnh cho Gemini đóng vai chuyên gia điện lực
        prompt = f"""
        Bạn là chuyên gia phân tích phụ tải điện tại Long An, Việt Nam.
        Tôi sẽ cung cấp cho bạn một đoạn tin tức (về thời tiết, kinh tế, hoặc chính sách).
        Nhiệm vụ của bạn:
        1. Phân tích xem tin tức này ảnh hưởng TĂNG hay GIẢM đến nhu cầu sử dụng điện.
        2. Đánh giá mức độ ảnh hưởng theo thang điểm sau:
           -2: Giảm mạnh (Dịch bệnh phong tỏa, bão lũ lớn, suy thoái kinh tế nặng).
           -1: Giảm nhẹ (Mưa nhiều, giá điện tăng cao làm dân tiết kiệm).
            0: Không ảnh hưởng đáng kể.
            1: Tăng nhẹ (Nắng nóng thường, có thêm khu công nghiệp nhỏ).
            2: Tăng mạnh (Nắng nóng kỷ lục/El Nino, Kinh tế bùng nổ, KCN lớn hoạt động).
        
        Nội dung tin tức: "{news_content}"
        
        YÊU CẦU OUTPUT: Chỉ trả về duy nhất một con số nguyên (từ -2 đến 2). Không giải thích gì thêm.
        Ví dụ: 2
        """
        
        response = model.generate_content(prompt)
        # Lấy text và làm sạch
        score_text = response.text.strip()
        # Chuyển thành số
        score = int(score_text)
        return score, None
    except Exception as e:
        return 0, f"Lỗi Gemini: {str(e)}"

# ==============================================================================
# 2. HÀM XỬ LÝ SỐ LIỆU (CORE)
# ==============================================================================
def them_yeu_to_mua(df):
    def check_tet(row):
        try:
            nam = int(row['Năm'])
            thang = int(row['Tháng'])
            lich_tet = {2023: 1, 2024: 2, 2025: 1, 2026: 2, 2027: 2, 2028: 1}
            if nam in lich_tet and lich_tet[nam] == thang: return 1
            return 0
        except: return 0
    df['Co_Tet'] = df.apply(check_tet, axis=1)
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10, 11] else 0)
    return df

@st.cache_data
def train_and_predict(file_train, file_input, manual_events_dict):
    # Đọc dữ liệu
    try: df_train = pd.read_excel(file_train, sheet_name='Bang tinh 5 tppt')
    except: df_train = pd.read_excel(file_train, sheet_name=0)
    df_input = pd.read_excel(file_input)

    # 1. TẠO CỘT SỰ KIỆN TỪ DICTIONARY (Dữ liệu từ Gemini/Nhập tay)
    # manual_events_dict dạng: {(2025, 4): 2, (2025, 5): 2}
    
    def get_event_score(row):
        key = (int(row['Năm']), int(row['Tháng']))
        return manual_events_dict.get(key, 0) # Mặc định là 0 nếu không có tin tức

    # Áp dụng cho cả Train và Input
    # (Lưu ý: Với Train/Lịch sử, ta giả định là 0 hoặc bạn phải nhập tay quá khứ. 
    # Ở đây tập trung vào việc Gemini dự báo cho Input tương lai)
    df_train['Su_Kien_Tin_Tuc'] = df_train.apply(get_event_score, axis=1)
    df_input['Su_Kien_Tin_Tuc'] = df_input.apply(get_event_score, axis=1)

    # 2. Xử lý mùa vụ
    df_train = them_yeu_to_mua(df_train)
    df_input = them_yeu_to_mua(df_input)

    # 3. Features
    features = ['Tháng', 'Năm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Su_Kien_Tin_Tuc']
    # Thêm nhiệt độ nếu có
    if 'Nhiệt độ TB' in df_train.columns: features.append('Nhiệt độ TB')
    if 'Độ ẩm' in df_train.columns: features.append('Độ ẩm')
    
    target = 'Tổng thương phẩm'

    # 4. Huấn luyện
    data_clean = df_train.dropna(subset=features + [target]).copy()
    X = data_clean[features]
    y = data_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    h_size = (len(features)*3, len(features)*2, len(features))
    nn = MLPRegressor(hidden_layer_sizes=h_size, activation='relu', solver='lbfgs', max_iter=5000, random_state=0) 
    nn.fit(X_scaled, y)
    
    # ARIMA
    arima_fit = None
    if HAS_ARIMA:
        try:
            ts_data = data_clean.copy()
            ts_data['Date'] = pd.to_datetime(dict(year=ts_data['Năm'], month=ts_data['Tháng'], day=1))
            ts_series = ts_data.groupby('Date')[target].sum().sort_index().asfreq('MS')
            order = (12, 1, 1) if len(ts_series) > 24 else (5, 1, 0)
            arima_model = ARIMA(ts_series, order=order)
            arima_fit = arima_model.fit()
        except: pass

    # 5. Dự báo
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    if len(df_pred) == 0: return None, "File Input rỗng."

    df_pred['RF_Forecast'] = rf.predict(df_pred[features])
    df_pred['NN_Forecast'] = nn.predict(scaler.transform(df_pred[features]))
    
    if arima_fit:
        try:
            steps = len(df_pred)
            arima_vals = arima_fit.forecast(steps=steps)
            df_pred['ARIMA_Forecast'] = arima_vals.values
        except: df_pred['ARIMA_Forecast'] = 0
    else: df_pred['ARIMA_Forecast'] = 0

    # Merge kết quả
    df_actual = df_train[['Năm', 'Tháng', target]].copy()
    df_final = pd.merge(df_pred, df_actual, on=['Năm', 'Tháng'], how='left')
    df_final.rename(columns={target: 'Thuc_te'}, inplace=True)
    df_final['Date'] = pd.to_datetime(dict(year=df_final['Năm'], month=df_final['Tháng'], day=1))
    
    return df_final, None

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
# Sidebar nhập API Key
with st.sidebar:
    st.header("🤖 Cấu hình AI")
    api_key = st.text_input("Nhập Gemini API Key", type="password", help="Lấy tại aistudio.google.com")
    if not HAS_GEMINI:
        st.error("Chưa cài thư viện `google-generativeai`!")

# Main
col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. File Lịch Sử", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. File Input (Dự báo)", type=['xlsx', 'xls'])

# --- PHẦN TÍCH HỢP GEMINI ---
st.write("---")
st.subheader("📰 AI Phân Tích Tin Tức & Sự Kiện")

# Quản lý trạng thái các sự kiện đã thêm
if 'event_list' not in st.session_state:
    st.session_state.event_list = {} # Dạng {(2025, 4): 2, (2025, 5): 2}

c1, c2 = st.columns([2, 1])
with c1:
    news_text = st.text_area("Dán nội dung bài báo hoặc tin tức vào đây:", height=100, 
                             placeholder="Ví dụ: Theo dự báo khí tượng, tháng 4 và 5 năm 2025 sẽ có nắng nóng kỷ lục...")
with c2:
    st.write("Áp dụng cho thời gian:")
    col_y, col_m = st.columns(2)
    sel_year = col_y.number_input("Năm", 2023, 2030, 2025)
    sel_months = col_m.multiselect("Tháng", range(1, 13), default=[4, 5])
    
    if st.button("Hỏi Gemini & Áp dụng", type="primary"):
        if not news_text:
            st.warning("Vui lòng dán nội dung tin tức.")
        elif not api_key:
            st.warning("Vui lòng nhập API Key ở menu bên trái.")
        else:
            with st.spinner("Gemini đang đọc báo và suy luận..."):
                score, err = ask_gemini_to_rate_event(api_key, news_text)
                if err:
                    st.error(err)
                else:
                    # Cập nhật vào danh sách sự kiện
                    for m in sel_months:
                        st.session_state.event_list[(sel_year, m)] = score
                    
                    msg = "Tăng mạnh" if score == 2 else "Tăng nhẹ" if score == 1 else "Giảm" if score < 0 else "Không đổi"
                    st.success(f"Gemini đánh giá: {score} ({msg})")
                    st.caption("Đã tự động điền vào dữ liệu dự báo!")

# Hiển thị các sự kiện đang có
if st.session_state.event_list:
    st.info(f"Dữ liệu sự kiện đang áp dụng: {st.session_state.event_list}")

# --- NÚT CHẠY DỰ BÁO ---
st.write("---")
if uploaded_train and uploaded_input:
    if st.button("🚀 CHẠY DỰ BÁO (KẾT HỢP DỮ LIỆU AI)", type="primary"):
        with st.spinner('Đang chạy mô hình...'):
            # Truyền danh sách sự kiện vào hàm train
            df_result, err = train_and_predict(uploaded_train, uploaded_input, st.session_state.event_list)
            
        if err: st.error(err)
        else:
            # Bảng
            st.subheader("📊 Kết Quả")
            # Highlight cột Sự kiện để thấy tác động của AI
            cols = ['Năm', 'Tháng', 'Thuc_te', 'NN_Forecast', 'RF_Forecast', 'Su_Kien_Tin_Tuc']
            st.dataframe(df_result[cols].style.format("{:,.0f}"), use_container_width=True)

            # Biểu đồ
            st.subheader("📈 Biểu Đồ")
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df_result['Date'], df_result['NN_Forecast'], 's-', color='red', label='Neural Network', linewidth=2)
            
            # Vẽ các điểm có sự kiện đặc biệt
            mask_event = df_result['Su_Kien_Tin_Tuc'] != 0
            if mask_event.any():
                ax.scatter(df_result.loc[mask_event, 'Date'], df_result.loc[mask_event, 'NN_Forecast'], 
                           s=150, c='yellow', edgecolors='black', zorder=10, label='Tác động Sự kiện (AI)')

            mask_actual = df_result['Thuc_te'].notnull()
            if mask_actual.any():
                ax.plot(df_result.loc[mask_actual, 'Date'], df_result.loc[mask_actual, 'Thuc_te'], 'o-', color='black', label='Thực Tế', linewidth=2)
            
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
