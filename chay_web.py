import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import io

# Kiểm tra thư viện Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# --- CẤU HÌNH TRANG (Tiếng Việt) ---
st.set_page_config(page_title="Hệ Thống Dự Báo Phụ Tải", layout="wide")
st.title("HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN")
st.markdown("---")

# ==============================================================================
# 1. MODULE XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)
# ==============================================================================
def xu_ly_du_lieu_dinh_tinh(api_key, text_input):
    """Chuyển đổi thông tin văn bản thành tham số số học."""
    if not api_key: return 0, "⚠️ Chưa nhập khóa API. Giá trị mặc định là 0."
    try:
        genai.configure(api_key=api_key)
        # Tự động chọn model khả dụng
        candidate_models = ["gemini-1.5-flash", "gemini-pro", "gemini-1.0-pro"]
        final_model = "gemini-pro"
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if "flash" in m.name: final_model = m.name; break
                    final_model = m.name
        except: pass

        model = genai.GenerativeModel(final_model)
        prompt = f"Hãy đóng vai chuyên gia năng lượng. Đánh giá mức độ ảnh hưởng của thông tin sau đến nhu cầu tiêu thụ điện trên thang điểm từ -2 (Giảm mạnh) đến 2 (Tăng mạnh). Nội dung: '{text_input}'. Chỉ trả về duy nhất một con số nguyên."
        response = model.generate_content(prompt)
        
        import re
        match = re.search(r'-?\d+', response.text)
        if match: 
            val = int(match.group())
            msg = f"✅ Đã phân tích xong. Mức độ tác động: {val}"
            if val > 0: msg += " (Dự báo Tăng)"
            elif val < 0: msg += " (Dự báo Giảm)"
            else: msg += " (Không ảnh hưởng đáng kể)"
            return val, msg
        return 0, "⚠️ Không xác định được mức độ tác động."
    except Exception as e: return 0, f"❌ Lỗi xử lý: {str(e)}"

# ==============================================================================
# 2. HÀM TÍNH TOÁN & DỰ BÁO (CORE)
# ==============================================================================
def feature_engineering(df):
    def check_tet(row):
        try:
            nam, thang = int(row['Năm']), int(row['Tháng'])
            # Lịch Tết Âm Lịch (Cần cập nhật chính xác theo từng năm)
            lich_tet = {2023: 1, 2024: 2, 2025: 1, 2026: 2, 2027: 2}
            return 1 if nam in lich_tet and lich_tet[nam] == thang else 0
        except: return 0
    df['Co_Tet'] = df.apply(check_tet, axis=1)
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3, 4, 5] else 0) # Mùa khô/nắng
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10, 11] else 0) # Mùa mưa
    return df

@st.cache_data
def chay_mo_phong(df_train_origin, df_input_origin, exogenous_params, scenario_label="Kịch bản Gốc"):
    """Chạy mô phỏng dự báo cho một kịch bản cụ thể."""
    df_train = df_train_origin.copy()
    df_input = df_input_origin.copy()

    # Gán biến ngoại sinh (Yếu tố tác động bên ngoài)
    def get_exogenous(row): return exogenous_params.get((int(row['Năm']), int(row['Tháng'])), 0)
    
    df_train['Bien_Ngoai_Sinh'] = df_train.apply(get_exogenous, axis=1)
    df_input['Bien_Ngoai_Sinh'] = df_input.apply(get_exogenous, axis=1)

    # Tạo đặc trưng dữ liệu (Feature Engineering)
    df_train = feature_engineering(df_train)
    df_input = feature_engineering(df_input)

    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_features = [f for f in features if f in df_train.columns and f in df_input.columns]
    target = 'Tổng thương phẩm'

    # Chuẩn bị dữ liệu huấn luyện
    data_clean = df_train.dropna(subset=valid_features + [target]).copy()
    X = data_clean[valid_features]
    y = data_clean[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Mô hình 1: Mạng Nơ-ron (Neural Network)
    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', max_iter=5000, random_state=0)
    nn.fit(X_scaled, y)
    
    # Mô hình 2: XGBoost (Gradient Boosting)
    xgb_model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, subsample=0.7, random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)

    # Thực hiện dự báo
    df_pred = df_input.copy().sort_values(['Năm', 'Tháng'])
    df_pred[valid_features] = df_pred[valid_features].fillna(0)
    
    suffix = "" if scenario_label == "Final" else f"_{scenario_label}"
    
    # Dự báo ra số liệu
    df_pred[f'NN{suffix}'] = nn.predict(scaler.transform(df_pred[valid_features]))
    df_pred[f'XGB{suffix}'] = xgb_model.predict(df_pred[valid_features])
    
    return df_pred[['Năm', 'Tháng', f'NN{suffix}', f'XGB{suffix}']]

# ==============================================================================
# GIAO DIỆN (UI) - VIỆT HÓA
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu Hình Tham Số")
    api_key = st.text_input("Nhập Khóa API (Google GenAI)", value="", type="password", help="Dùng để kích hoạt tính năng phân tích tin tức tự động.")

col1, col2 = st.columns(2)
with col1: uploaded_train = st.file_uploader("1. Tải lên Dữ liệu Lịch sử (Train)", type=['xlsx', 'xls'])
with col2: uploaded_input = st.file_uploader("2. Tải lên Dữ liệu Cần Dự báo (Input)", type=['xlsx', 'xls'])

st.write("---")
if 'param_dict' not in st.session_state: st.session_state.param_dict = {}

# PHẦN NHẬP LIỆU
st.subheader("📰 Phân Tích Thông Tin Đầu Vào (Kịch Bản Dự Báo)")
st.markdown("Nhập các thông tin định tính (dự báo thời tiết, tình hình kinh tế, chính sách...) để AI đánh giá tác động.")

c1, c2 = st.columns([2, 1])
with c1: 
    text_data = st.text_area("Nội dung thông tin:", height=100, placeholder="Ví dụ: Dự báo nắng nóng gay gắt kéo dài trong tháng 5 và tháng 6...")
with c2:
    st.write("Thao tác:")
    if st.button("🔍 Phân Tích & Áp Dụng", type="secondary"):
        with st.spinner("AI đang đọc hiểu nội dung..."):
            val, log = xu_ly_du_lieu_dinh_tinh(api_key, text_data)
        
        if "Lỗi" in log: st.warning(log)
        else: st.success(log)
        
        # Gán tham số demo cho tháng 5, 6 năm 2025
        st.session_state.param_dict[(2025, 5)] = val
        st.session_state.param_dict[(2025, 6)] = val

# Hiển thị trạng thái các tham số đã gán
if st.session_state.param_dict:
    st.info(f"⚡ Đã ghi nhận các yếu tố tác động ngoại sinh cho các tháng: {st.session_state.param_dict}")
else:
    st.caption("Chưa có thông tin tác động nào. Hệ thống sẽ chạy ở chế độ 'Kịch bản Cơ sở' (Chỉ dựa vào lịch sử).")

st.write("---")

if uploaded_train and uploaded_input:
    if st.button("🚀 THỰC HIỆN DỰ BÁO & SO SÁNH", type="primary"):
        # BẮT ĐẦU QUÁ TRÌNH TÍNH TOÁN
        try:
            # Đọc file
            try: df_train_org = pd.read_excel(uploaded_train, sheet_name='Bang tinh 5 tppt')
            except: df_train_org = pd.read_excel(uploaded_train, sheet_name=0)
            df_input_org = pd.read_excel(uploaded_input)

            with st.spinner("Đang chạy mô phỏng đa kịch bản..."):
                # 1. Chạy Kịch bản Cơ sở (Base Case) - Không có tác động bên ngoài
                df_base = chay_mo_phong(df_train_org, df_input_org, {}, "Goc")
                
                # 2. Chạy Kịch bản Điều chỉnh (Adjusted Case) - Có tác động từ thông tin đầu vào
                df_adj = chay_mo_phong(df_train_org, df_input_org, st.session_state.param_dict, "DieuChinh")
                
                # 3. Tổng hợp kết quả
                df_final = pd.merge(df_base, df_adj, on=['Năm', 'Tháng'])
                
                # Ghép với dữ liệu thực tế (nếu có để kiểm chứng)
                df_actual = df_train_org[['Năm', 'Tháng', 'Tổng thương phẩm']].copy()
                df_final = pd.merge(df_final, df_actual, on=['Năm', 'Tháng'], how='left')
                df_final.rename(columns={'Tổng thương phẩm': 'Thuc_te'}, inplace=True)
                df_final['ThoiGian'] = pd.to_datetime(dict(year=df_final['Năm'], month=df_final['Tháng'], day=1))

                # Tính toán chênh lệch (Delta)
                df_final['Chenh_Lech_NN'] = df_final['NN_DieuChinh'] - df_final['NN_Goc']
                
                # === HIỂN THỊ KẾT QUẢ ===
                
                # TAB 1: BẢNG TỔNG HỢP CHUNG (Theo yêu cầu của bạn)
                st.subheader("📊 Bảng Tổng Hợp So Sánh Các Kịch Bản")
                st.markdown("Bảng dưới đây so sánh chi tiết kết quả dự báo giữa trường hợp **Bình thường (Gốc)** và trường hợp **Có xét đến tác động thông tin (Điều chỉnh)**.")
                
                # Chọn các cột cần hiển thị và đặt tên tiếng Việt cho dễ hiểu
                cols_display = {
                    'Tháng': 'Tháng',
                    'Thuc_te': 'Thực Tế (kWh)',
                    'NN_Goc': 'Neural Net (Gốc)',
                    'NN_DieuChinh': 'Neural Net (Điều chỉnh)',
                    'Chenh_Lech_NN': 'Tác động (+/-)',
                    'XGB_Goc': 'XGBoost (Gốc)',
                    'XGB_DieuChinh': 'XGBoost (Điều chỉnh)'
                }
                
                df_show = df_final[['Năm'] + list(cols_display.keys())].rename(columns=cols_display)
                
                # Format hiển thị số liệu
                st.dataframe(df_show.style.format("{:,.0f}").applymap(
                    lambda x: 'background-color: #ffcccc' if x < 0 else 'background-color: #ccffcc' if x > 0 else '', 
                    subset=['Tác động (+/-)'] # Tô màu cột chênh lệch
                ), use_container_width=True)

                # BIỂU ĐỒ PHÂN TÍCH
                st.subheader("📈 Biểu Đồ Phân Tích Kịch Bản")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Vẽ đường Kịch bản Gốc (Nét đứt, màu xám)
                ax.plot(df_final['ThoiGian'], df_final['NN_Goc'], '--', color='gray', label='Kịch bản Gốc (Neural Net)', alpha=0.7)
                
                # Vẽ đường Kịch bản Điều chỉnh (Nét liền, màu đỏ nổi bật)
                ax.plot(df_final['ThoiGian'], df_final['NN_DieuChinh'], 's-', color='#d62728', label='Kịch bản Điều chỉnh (Neural Net)', linewidth=2)
                
                # Tô vùng chênh lệch để thấy rõ tác động
                ax.fill_between(df_final['ThoiGian'], df_final['NN_Goc'], df_final['NN_DieuChinh'], color='orange', alpha=0.15, label='Vùng tác động của thông tin')

                # Vẽ điểm thực tế (nếu có)
                if df_final['Thuc_te'].notnull().any():
                    ax.plot(df_final['ThoiGian'], df_final['Thuc_te'], 'k-', linewidth=2, label='Số liệu Thực tế', zorder=10)

                ax.set_title("So Sánh Tác Động Của Thông Tin Đến Nhu Cầu Phụ Tải")
                ax.set_ylabel("Sản lượng điện (kWh)")
                ax.set_xlabel("Thời gian")
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
                
                # Nút tải về
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_show.to_excel(writer, index=False, sheet_name='Ket_qua_Du_bao')
                st.download_button("📥 Tải Báo Cáo Excel", buffer.getvalue(), "Bao_cao_Du_bao_Phu_tai.xlsx")

        except Exception as e:
            st.error(f"❌ Có lỗi xảy ra trong quá trình tính toán: {e}")
