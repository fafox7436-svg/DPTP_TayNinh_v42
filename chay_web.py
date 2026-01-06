import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import time

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Dự Báo Phụ Tải (Kiểm Soát)", layout="wide")
st.title("⚡ HỆ THỐNG DỰ BÁO PHỤ TẢI")
st.markdown("---")

# ==============================================================================
# 1. CÁC HÀM XỬ LÝ CƠ BẢN
# ==============================================================================

def chuan_hoa_ten_cot(df):
    if df is None: return None
    df.columns = df.columns.astype(str).str.strip()
    col_map = {
        'month': 'Tháng', 'thang': 'Tháng', 'tháng': 'Tháng',
        'year': 'Năm', 'nam': 'Năm', 'năm': 'Năm',
        'tổng thương phẩm': 'Tổng thương phẩm', 'tong thuong pham': 'Tổng thương phẩm', 'sản lượng': 'Tổng thương phẩm',
        'nhiệt độ tb': 'Nhiệt độ TB', 'nhiet do tb': 'Nhiệt độ TB',
        'độ ẩm': 'Độ ẩm', 'do am': 'Độ ẩm',
        'số ngày': 'Số ngày', 'so ngay': 'Số ngày',
        'lượng mưa': 'Lượng mưa', 'luong mua': 'Lượng mưa'
    }
    new_cols = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in col_map:
            new_cols[col] = col_map[col_lower]
    return df.rename(columns=new_cols)

def doc_file_thong_minh(uploaded_file):
    try:
        xl = pd.ExcelFile(uploaded_file)
        for sheet in xl.sheet_names:
            preview = pd.read_excel(uploaded_file, sheet_name=sheet, header=None, nrows=10)
            for i, row in preview.iterrows():
                row_str = str(row.values).lower()
                if 'tháng' in row_str and 'năm' in row_str:
                    uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file, sheet_name=sheet, header=i)
                    return chuan_hoa_ten_cot(df)
        uploaded_file.seek(0)
        return chuan_hoa_ten_cot(pd.read_excel(uploaded_file, header=0))
    except: return None

# ==============================================================================
# 2. HÀM KIỂM TRA LỖI (NGHIÊM NGẶT)
# ==============================================================================
def kiem_tra_chat_luong_du_lieu(df, ten_file="Input"):
    """
    Hàm này soi lỗi từng dòng. Nếu thấy 0 hoặc rỗng ở cột quan trọng -> Báo lỗi ngay.
    """
    errors = []
    
    # 1. Kiểm tra cột bắt buộc
    required = ['Tháng', 'Năm']
    for col in required:
        if col not in df.columns:
            st.error(f"❌ File {ten_file} thiếu cột '{col}'")
            st.stop()
            
    # 2. Kiểm tra các cột biến số (Không được bằng 0 hoặc rỗng)
    # Lưu ý: Lượng mưa có thể bằng 0, nhưng Nhiệt độ/Độ ẩm/Số ngày thì KHÔNG THỂ bằng 0.
    check_cols = {
        'Nhiệt độ TB': 'Không được bằng 0 hoặc rỗng', 
        'Độ ẩm': 'Không được bằng 0 hoặc rỗng', 
        'Số ngày': 'Phải từ 28-31 ngày'
    }
    
    for col, msg in check_cols.items():
        if col in df.columns:
            # Tìm dòng bị NaN (Rỗng)
            if df[col].isnull().any():
                rows = df[df[col].isnull()].index.tolist()
                # +2 vì: index bắt đầu từ 0 + 1 dòng header + 1 để ra số dòng Excel thực tế
                excel_rows = [r + 2 for r in rows] 
                errors.append(f"❌ Cột '{col}' có ô Trống ở dòng Excel số: {excel_rows}")
            
            # Tìm dòng bằng 0
            if (df[col] == 0).any():
                rows = df[df[col] == 0].index.tolist()
                excel_rows = [r + 2 for r in rows]
                errors.append(f"❌ Cột '{col}' bằng 0 (Vô lý) ở dòng Excel số: {excel_rows}")
                
    # 3. Nếu có lỗi -> In ra và Dừng chương trình
    if errors:
        st.error(f"phát hiện lỗi trong file {ten_file}. Vui lòng sửa lại file Excel rồi upload lại:")
        for e in errors:
            st.write(e)
        st.stop() # Dừng tại đây, không chạy tiếp
        
    return True # Nếu sạch sẽ thì cho qua

def tao_dac_trung(df):
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3,4,5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6,7,8,9,10,11] else 0)
    def check_tet(row):
        try:
            return 1 if (row['Năm']==2024 and row['Tháng']==2) or (row['Năm']==2025 and row['Tháng']==1) else 0
        except: return 0
    df['Co_Tet'] = df.apply(check_tet, axis=1)
    df['Bien_Ngoai_Sinh'] = 0
    return df

# ==============================================================================
# 3. CORE DỰ BÁO
# ==============================================================================
def chay_mo_hinh(df_train_raw, df_input_raw):
    # 1. Kiểm tra nghiêm ngặt trước
    kiem_tra_chat_luong_du_lieu(df_train_raw, "Lịch Sử (Train)")
    kiem_tra_chat_luong_du_lieu(df_input_raw, "Dự Báo (Input)")
    
    # 2. Tạo đặc trưng (Không vá lỗi nữa, dữ liệu phải chuẩn mới chạy)
    df_train = tao_dac_trung(df_train_raw.copy())
    df_input = tao_dac_trung(df_input_raw.copy())
    
    features = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm', 'Co_Tet', 'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_cols = [c for c in features if c in df_train.columns and c in df_input.columns]
    target = 'Tổng thương phẩm'
    
    # 3. Train
    data_train = df_train.dropna(subset=valid_cols + [target])
    X_train = data_train[valid_cols]
    y_train = data_train[target]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    nn = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=5000, random_state=42)
    nn.fit(X_train_scaled, y_train)
    
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    
    xg = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xg.fit(X_train, y_train)
    
    # 4. Predict
    df_pred = df_input.copy()
    X_pred = df_pred[valid_cols] # Không fillna(0) bừa bãi nữa
    
    pred_nn = nn.predict(scaler.transform(X_pred))
    pred_rf = rf.predict(X_pred)
    pred_xg = xg.predict(X_pred)
    
    df_pred['NN'] = pred_nn
    df_pred['RF'] = pred_rf
    df_pred['XGB'] = pred_xg
    df_pred['Dự Báo Chốt'] = (pred_nn + pred_rf + pred_xg) / 3
    
    return df_pred

# ==============================================================================
# GIAO DIỆN
# ==============================================================================
c1, c2 = st.columns(2)
f_train = c1.file_uploader("1. File Lịch Sử (Train)", type=['xlsx', 'xls'])
f_input = c2.file_uploader("2. File Dự Báo (Input)", type=['xlsx', 'xls'])

st.write("### 🛠️ Điều chỉnh (Nếu cần)")
adj_pct = st.number_input("Tăng/Giảm (%):", value=0.0, step=0.1)

if f_train and f_input:
    if st.button("🚀 KIỂM TRA & DỰ BÁO", type="primary"):
        with st.spinner("Đang kiểm tra lỗi dữ liệu..."):
            df_train = doc_file_thong_minh(f_train)
            df_input = doc_file_thong_minh(f_input)
            
            if df_train is not None and df_input is not None:
                # Code sẽ tự dừng nếu hàm kiem_tra_chat_luong phát hiện lỗi
                res = chay_mo_hinh(df_train, df_input)
                
                # Nếu chạy đến đây nghĩa là dữ liệu SẠCH
                factor = 1.0 + (adj_pct / 100.0)
                res['Dự Báo Chốt'] *= factor
                res['NN'] *= factor
                res['RF'] *= factor
                res['XGB'] *= factor
                
                # Merge thực tế
                if 'Tổng thương phẩm' in df_train.columns:
                    actual = df_train[['Năm', 'Tháng', 'Tổng thương phẩm']]
                    res = pd.merge(res, actual, on=['Năm', 'Tháng'], how='left')
                    res.rename(columns={'Tổng thương phẩm': 'Thực Tế'}, inplace=True)
                
                st.success("✅ Dữ liệu chuẩn! Kết quả dự báo:")
                
                # Hiển thị
                final_cols = ['Năm', 'Tháng', 'Dự Báo Chốt', 'NN', 'RF', 'XGB']
                if 'Thực Tế' in res.columns: final_cols.insert(2, 'Thực Tế')
                
                st.dataframe(res[final_cols].style.format("{:,.0f}"), use_container_width=True)
                
                # Biểu đồ
                fig, ax = plt.subplots(figsize=(12, 6))
                res['Date'] = pd.to_datetime(dict(year=res['Năm'], month=res['Tháng'], day=1))
                ax.plot(res['Date'], res['Dự Báo Chốt'], 'o-', color='red', linewidth=3, label='Dự Báo Chốt')
                if 'Thực Tế' in res.columns:
                    mask = res['Thực Tế'].notnull()
                    ax.plot(res.loc[mask, 'Date'], res.loc[mask, 'Thực Tế'], 'ko', label='Thực Tế')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.error("Lỗi đọc file Excel. Kiểm tra lại định dạng.")
