import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import io

# --- KIỂM TRA THƯ VIỆN ---
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
st.set_page_config(page_title="Dự Báo & So Sánh Tối Ưu", layout="wide")
st.title("⚡ HỆ THỐNG DỰ BÁO PHỤ TẢI")
st.markdown("Hệ thống tự động so sánh **Neural Network**, **Random Forest** và **SARIMA** để tìm ra phương pháp chính xác nhất cho từng tháng.")

# ==============================================================================
# 1. HÀM GEMINI
# ==============================================================================
def ask_gemini_to_rate_event(api_key, news_content):
    if not api_key: return None, "Chưa nhập API Key."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Đánh giá tác động tin tức đến phụ tải điện (-2 đến 2). Tin: '{news_content}'. Trả về 1 số nguyên."
        response = model.generate_content(prompt)
        return int(response.text.strip()), None
    except Exception as e: return 0, str(e)

# ==============================================================================
# 2. HÀM XỬ LÝ SỐ LIỆU
# ==============================================================================
def them_yeu_to_mua(df):
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

@st.cache_data
def train_and_predict(file_train, file_input, manual_events_dict):
    # Đọc dữ liệu
    try: df_train = pd.read_excel(file_train, sheet_name='Bang tinh 5 tppt')
    except: df_train = pd.read_excel(file_train, sheet_name=0)
    df_input = pd.read_excel(file_input)

    # --- CƠ CHẾ ỔN ĐỊNH SỰ KIỆN ---
    use_event = False
    if manual_events_dict and any(v != 0 for v in manual_events_dict.values()):
        use_event = True
    
    def get_event(row): 
        return manual_events_dict.get((int(row['Năm']), int(row['Tháng'])), 0)
    
    if use_event:
        df_train['Su_Kien'] = df_train.apply(get_event, axis=1)
        df_input['Su_Kien'] = df_input.apply(get_event, axis=1)
    
    df_train = them_yeu_to_mua(df_train)
    df_input = them_yeu_to_mua(df_input)

    # Features
    if use_event:
        features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Su_Kien']
    else:
        features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua']

    valid_features = [f for f in features if f in df_train.columns and f in df_input.columns]
    target = 'Tổng thương phẩm'

    # Dữ liệu Train
    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 2. Neural Network (Chuẩn)
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=0)
    nn.fit(X_scaled, y)
    
# 3. SARIMA (PHIÊN BẢN AUTO-TUNING + ÉP KIỂU EVIEWS)
    arima_fit = None
    if HAS_ARIMA:
        try:
            ts_data = data_clean.copy()
            # 1. Chuẩn hóa thời gian
            ts_data['Date'] = pd.to_datetime(dict(year=ts_data['Năm'], month=ts_data['Tháng'], day=1))
            ts_data = ts_data.sort_values('Date')
            ts_series = ts_data.set_index('Date')[target].asfreq('MS')
            
            # 2. Logarit hóa (Giống EViews để ổn định dao động)
            ts_log = np.log(ts_series)

            # 3. AUTO-SEARCH (Mô phỏng EViews tìm tham số tốt nhất)
            # Danh sách các bộ tham số "vàng" cho dữ liệu điện lực
            # Cấu trúc: (p,d,q) x (P,D,Q,s)
            param_grid = [
                ((1, 1, 1), (0, 1, 1, 12)),  # Cấu hình 1: Tiêu chuẩn (Thường EViews chọn cái này)
                ((1, 1, 0), (0, 1, 0, 12)),  # Cấu hình 2: Đơn giản hơn
                ((0, 1, 1), (0, 1, 1, 12)),  # Cấu hình 3: Thiên về trung bình trượt
                ((2, 1, 0), (1, 1, 0, 12))   # Cấu hình 4: Phức tạp (nếu dữ liệu đủ tốt)
            ]
            
            best_aic = float("inf")
            best_model = None

            for order, seasonal_order in param_grid:
                try:
                    # enforce_stationarity=False: ĐÂY LÀ CHÌA KHÓA!
                    # Nó cho phép mô hình chạy giống EViews kể cả khi dữ liệu chưa dừng hẳn.
                    mod = ARIMA(ts_log, order=order, seasonal_order=seasonal_order, 
                                enforce_stationarity=False, 
                                enforce_invertibility=False)
                    res = mod.fit()
                    
                    # So sánh AIC (Chỉ số đo độ tốt, càng thấp càng tốt)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_model = res
                except:
                    continue
            
            # Chọn được mô hình tốt nhất thì gán vào biến chính
            if best_model is not None:
                arima_fit = best_model
            else:
                # Fallback cuối cùng nếu Auto thất bại: ARIMA(1,1,0) cơ bản
                arima_fit = ARIMA(ts_log, order=(1, 1, 0)).fit()

        except: pass

    # --- DỰ BÁO ---
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    if len(df_pred) == 0: return None, "File Input rỗng."

    df_pred[valid_features] = df_pred[valid_features].fillna(0)
    
    # Predict RF & NN
    df_pred['RF_Forecast'] = rf.predict(df_pred[valid_features])
    df_pred['NN_Forecast'] = nn.predict(scaler.transform(df_pred[valid_features]))

    # Predict ARIMA (Bung nén Logarit)
    if arima_fit:
        try:
            steps = len(df_pred)
            log_forecast = arima_fit.forecast(steps=steps)
            df_pred['ARIMA_Forecast'] = np.exp(log_forecast).values
        except: df_pred['ARIMA_Forecast'] = 0
    else: df_pred['ARIMA_Forecast'] = 0

    # MERGE & TÍNH SAI SỐ TOÀN DIỆN
    df_actual = df_train[['Năm', 'Tháng', target]].copy()
    df_final = pd.merge(df_pred, df_actual, on=['Năm', 'Tháng'], how='left')
    df_final.rename(columns={target: 'Thuc_te'}, inplace=True)
    df_final['Date'] = pd.to_datetime(dict(year=df_final['Năm'], month=df_final['Tháng'], day=1))
    
    return df_final, None

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
with st.sidebar:
    st.header("Cấu hình")
    # Điền sẵn Key của bạn vào đây để đỡ phải nhập lại
    # Ví dụ: value="AIzaSy..."
    api_key = st.text_input("Gemini API Key", value="", type="password")

col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. File Dự Báo (Input)", type=['xlsx', 'xls'])

# Phần Tin tức
st.write("---")
if 'event_list' not in st.session_state: st.session_state.event_list = {}
c1, c2 = st.columns([2, 1])
with c1: news = st.text_area("Dán tin tức (Gemini phân tích):", height=80)
with c2:
    if st.button("Phân tích"):
        s, e = ask_gemini_to_rate_event(api_key, news)
        if e: st.error(e)
        else: 
            st.session_state.event_list[(2025, 4)] = s
            st.session_state.event_list[(2025, 5)] = s
            st.success(f"Đánh giá: {s}")

# CHẠY DỰ BÁO
st.write("---")
if uploaded_train and uploaded_input:
    if st.button("🚀 CHẠY DỰ BÁO & SO SÁNH", type="primary"):
        df_result, err = train_and_predict(uploaded_train, uploaded_input, st.session_state.event_list)
        
        if err: st.error(err)
        else:
            # --- TÍNH TOÁN SAI SỐ CHI TIẾT ---
            # Chỉ tính với các dòng có số liệu thực tế
            mask = df_result['Thuc_te'].notnull()
            
            # Tính % Lệch cho từng phương pháp
            df_result['Lệch NN (%)'] = np.where(mask, abs(df_result['Thuc_te'] - df_result['NN_Forecast'])/df_result['Thuc_te']*100, np.nan)
            df_result['Lệch RF (%)'] = np.where(mask, abs(df_result['Thuc_te'] - df_result['RF_Forecast'])/df_result['Thuc_te']*100, np.nan)
            df_result['Lệch ARIMA (%)'] = np.where(mask, abs(df_result['Thuc_te'] - df_result['ARIMA_Forecast'])/df_result['Thuc_te']*100, np.nan)

            # --- TÌM PHƯƠNG PHÁP TỐT NHẤT ---
            def tim_best(row):
                if pd.isna(row['Thuc_te']): return ""
                errors = {
                    'Neural Net': row['Lệch NN (%)'],
                    'Random Forest': row['Lệch RF (%)'],
                    'ARIMA': row['Lệch ARIMA (%)'] if row['ARIMA_Forecast'] > 0 else 999
                }
                # Lấy tên phương pháp có sai số nhỏ nhất
                return min(errors, key=errors.get)

            df_result['Tốt nhất'] = df_result.apply(tim_best, axis=1)

            # --- HIỂN THỊ BẢNG ---
            st.subheader("📊 Bảng So Sánh Sai Số Chi Tiết")
            
            cols_show = ['Tháng', 'Thuc_te', 
                         'NN_Forecast', 'Lệch NN (%)', 
                         'RF_Forecast', 'Lệch RF (%)', 
                         'ARIMA_Forecast', 'Lệch ARIMA (%)', 
                         'Tốt nhất']
            
            # Format bảng
            st.dataframe(df_result[cols_show].style.format({
                'Thuc_te': '{:,.0f}', 
                'NN_Forecast': '{:,.0f}', 'Lệch NN (%)': '{:.2f}%',
                'RF_Forecast': '{:,.0f}', 'Lệch RF (%)': '{:.2f}%',
                'ARIMA_Forecast': '{:,.0f}', 'Lệch ARIMA (%)': '{:.2f}%'
            }).applymap(lambda x: 'background-color: #d4edda; color: green; font-weight: bold' if isinstance(x, str) and len(x)>0 else '', subset=['Tốt nhất']), 
            use_container_width=True)

            # --- BIỂU ĐỒ ---
            st.subheader("📈 Biểu Đồ So Sánh")
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Vẽ đường thực tế
            if mask.any():
                ax.plot(df_result.loc[mask, 'Date'], df_result.loc[mask, 'Thuc_te'], 'o-', color='black', linewidth=3, label='Thực Tế')
            
            # Vẽ 3 đường dự báo
            ax.plot(df_result['Date'], df_result['NN_Forecast'], 's-', color='red', label='Neural Network', alpha=0.8)
            ax.plot(df_result['Date'], df_result['RF_Forecast'], 'x--', color='blue', label='Random Forest', alpha=0.6)
            
            # Vẽ ARIMA nếu số liệu hợp lý
            if df_result['ARIMA_Forecast'].max() > 1000:
                ax.plot(df_result['Date'], df_result['ARIMA_Forecast'], '^-.', color='green', label='ARIMA (Seasonal)', alpha=0.7)
            
            ax.set_title("So Sánh Các Mô Hình Dự Báo")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
            # Tải về
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_result.drop(columns=['Date']).to_excel(writer, index=False)
            st.download_button("📥 Tải Báo Cáo Chi Tiết", buffer.getvalue(), "Ket_qua_so_sanh.xlsx")



