# -*- coding: utf-8 -*-
"""Lõi dự báo khôi phục từ mã gốc do người sử dụng cung cấp.

Hai hàm tao_dac_trung và chay_mo_hinh_goc giữ nguyên phép tính, thứ tự
đặc trưng và tham số của NN, Random Forest, XGBoost trong mã gốc.
Chỉ khôi phục xuống dòng/thụt lề và tách phần tính toán khỏi giao diện.

Cách dùng trong chương trình vốn gọi chay_mo_hinh_goc:

    from loi_du_bao_goc import chay_mo_hinh_goc

    p_nn, p_rf, p_xg, p_trend = chay_mo_hinh_goc(
        df_train, df_input, USER_HOLIDAYS_MAP, K_DICT_DEFAULT, seed=seed_val
    )

Nếu cần giữ bộ nhớ đệm Streamlit của mã gốc, đặt trong tệp giao diện:

    chay_mo_hinh_goc = st.cache_data(show_spinner=False)(chay_mo_hinh_goc)

holidays_map: {(nam, thang): (so_ngay_tet, so_ngay_le)}.
k_dict: các khóa T2, T7, CN, Le, Tet, CatDien, lấy từ cấu hình đang dùng.
df_train và df_input là bảng gốc trước tiền xử lý của v2. Không truyền
đầu vào đã biến đổi thành sin/cos hoặc log của v2 vào các hàm này.

Trong bộ mã đầy đủ, forecasting.py gọi trực tiếp mô-đun này.
Khởi động giao diện bằng: python -m streamlit run chay_web.py.

Để tái lập kết quả đang chạy, cần cùng dữ liệu, thứ tự dòng, đơn vị,
lịch nghỉ, hệ số, seed và phiên bản các thư viện như môi trường gốc.
Không cố định thêm những tham số mà mã gốc để theo mặc định thư viện.
Các giả định lịch/cắt điện cũ được giữ để bảo toàn phép tính, không phải
khẳng định mọi giả định đó đã được kiểm chứng bằng dữ liệu thực tế.
"""

import calendar

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def tao_dac_trung(df, holidays_map):
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3,4,5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6,7,8,9,10,11] else 0)
    # Mặc định số ngày cắt điện = 0 nếu file excel không có cột này
    if 'Số ngày cắt điện' not in df.columns:
        df['Số ngày cắt điện'] = 0

    def get_calendar_info(row):
        y, m = int(row['Năm']), int(row['Tháng'])
        num_days = calendar.monthrange(y, m)[1]
        t2 = t7 = cn = 0
        for day in range(1, num_days + 1):
            wd = calendar.weekday(y, m, day)
            if wd == 0:
                t2 += 1
            elif wd == 5:
                t7 += 1
            elif wd == 6:
                cn += 1
        tet_am, le_nho = holidays_map.get((y, m), (0, 0))
        so_ngay_cat_dien = int(row.get('Số ngày cắt điện', 0))
        so_ngay_thuong = num_days - t2 - t7 - cn - tet_am - le_nho - so_ngay_cat_dien
        if so_ngay_thuong < 0:
            so_ngay_thuong = 0
        return pd.Series([so_ngay_thuong, t2, t7, cn, tet_am, le_nho, so_ngay_cat_dien])

    df[['So_Ngay_Thuong', 'So_Ngay_T2', 'So_Ngay_T7', 'So_Ngay_CN',
        'So_Ngay_Tet_Am', 'So_Ngay_Le_Nho', 'So_Ngay_Cat_Dien']] = df.apply(get_calendar_info, axis=1)
    df['Bien_Ngoai_Sinh'] = 0
    return df


def chay_mo_hinh_goc(df_train, df_input, holidays_map, k_dict, seed=42):
    df_train = tao_dac_trung(df_train.copy(), holidays_map)
    df_input = tao_dac_trung(df_input.copy(), holidays_map)
    start_year = df_train['Năm'].min()

    def create_time_index(row):
        return (row['Năm'] - start_year) * 12 + row['Tháng']

    df_train['Time_Index'] = df_train.apply(create_time_index, axis=1)
    df_input['Time_Index'] = df_input.apply(create_time_index, axis=1)

    # Tích hợp thêm Hệ số Cắt điện vào tính Tổng Ngày Tương Đương
    def calc_equiv_days(df):
        return (df['So_Ngay_Thuong'] * 1.0
                + df['So_Ngay_T2'] * k_dict['T2']
                + df['So_Ngay_T7'] * k_dict['T7']
                + df['So_Ngay_CN'] * k_dict['CN']
                + df['So_Ngay_Le_Nho'] * k_dict['Le']
                + df['So_Ngay_Tet_Am'] * k_dict['Tet']
                + df['So_Ngay_Cat_Dien'] * k_dict['CatDien'])

    df_train['Ngay_Tuong_Duong'] = calc_equiv_days(df_train)
    df_input['Ngay_Tuong_Duong'] = calc_equiv_days(df_input)

    # Chạy AI
    features = ['Tháng', 'Năm', 'Nhiệt độ TB', 'Độ ẩm', 'Mua_Nong', 'Mua_Mua', 'Bien_Ngoai_Sinh']
    valid_cols = [c for c in features if c in df_train.columns and c in df_input.columns]
    data_train = df_train.dropna(subset=valid_cols + ['Tổng thương phẩm'])
    X_train = data_train[valid_cols]
    y_train_base_daily = data_train['Tổng thương phẩm'] / data_train['Ngay_Tuong_Duong']
    y_train_log = np.log1p(y_train_base_daily)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred = df_input[valid_cols].fillna(0)

    nn = MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='relu',
                      solver='lbfgs', alpha=0.1, max_iter=5000, random_state=seed)
    nn.fit(X_train_scaled, y_train_log)

    trend_model = LinearRegression()
    trend_model.fit(data_train[['Time_Index']], y_train_log)
    trend_future = trend_model.predict(df_input[['Time_Index']])
    y_residual = y_train_log - trend_model.predict(data_train[['Time_Index']])

    rf = RandomForestRegressor(n_estimators=200, random_state=seed)
    rf.fit(X_train, y_residual)

    xg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05,
                          max_depth=4, reg_alpha=0.1, random_state=seed)
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
