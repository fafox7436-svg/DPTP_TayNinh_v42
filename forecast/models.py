"""Ba mô hình và xu hướng tuyến tính; giữ cấu trúc mô hình gốc."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .config import FEATURES
from .data import tao_dac_trung


def chay_mo_hinh_goc(df_train: pd.DataFrame, df_input: pd.DataFrame,
                      holidays_map: dict, k_dict: dict, seed: int = 42,
                      holiday_dates: pd.DataFrame | None = None,
                      outage_dates_map: dict | None = None):
    df_train = tao_dac_trung(df_train.copy(), holidays_map, holiday_dates,
                             outage_dates_map, k_dict)
    df_input = tao_dac_trung(df_input.copy(), holidays_map, holiday_dates,
                             outage_dates_map, k_dict)
    start_year = df_train['Năm'].min()

    def create_time_index(row):
        return (row['Năm'] - start_year) * 12 + row['Tháng']

    df_train['Time_Index'] = df_train.apply(create_time_index, axis=1)
    df_input['Time_Index'] = df_input.apply(create_time_index, axis=1)
    if 'Ngay_Tuong_Duong' not in df_train.columns or 'Ngay_Tuong_Duong' not in df_input.columns:
        raise ValueError('Không tính được ngày tương đương.')

    valid_cols = [c for c in FEATURES if c in df_train.columns and c in df_input.columns]
    data_train = df_train.dropna(subset=valid_cols + ['Tổng thương phẩm'])
    if data_train.empty:
        raise ValueError('Không có dữ liệu lịch sử hợp lệ để huấn luyện.')
    if (data_train['Ngay_Tuong_Duong'] <= 0).any():
        raise ValueError('Số ngày tương đương của lịch sử phải lớn hơn 0. Kiểm tra hệ số k.')
    if (df_input['Ngay_Tuong_Duong'] <= 0).any():
        raise ValueError('Số ngày tương đương dự báo phải lớn hơn 0. Kiểm tra hệ số k.')

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

    pred_nn_base = np.expm1(nn.predict(scaler.transform(X_pred)))
    pred_rf_base = np.expm1(rf.predict(X_pred) + trend_future)
    pred_xg_base = np.expm1(xg.predict(X_pred) + trend_future)
    pred_trend_base = np.expm1(trend_future)
    pred_nn = pred_nn_base * df_input['Ngay_Tuong_Duong']
    pred_rf = pred_rf_base * df_input['Ngay_Tuong_Duong']
    pred_xg = pred_xg_base * df_input['Ngay_Tuong_Duong']
    pred_trend = pred_trend_base * df_input['Ngay_Tuong_Duong']
    return pred_nn, pred_rf, pred_xg, pred_trend
