"""Dashboard dự báo phụ tải điện tỉnh Tây Ninh.

Bản nâng cấp theo phê duyệt:
- Backtest theo thời gian + MAE/MAPE/RMSE.
- Lịch từng ngày, xử lý lễ/cuối tuần/cắt điện trùng nhau.
- Xuất Excel nhiều sheet.
- Dashboard Streamlit + Plotly.
- Cấu trúc lịch trực quan (7 cột T2-CN) & công cụ đánh dấu ngày nghỉ/cắt điện theo từng ngày.
- Module tính toán khách quan bộ hệ số k trực tiếp từ nhiều file Điện nhận - Điện giao ngày.

Ba mô hình cốt lõi (Neural Network, Random Forest, XGBoost) vẫn giữ cấu trúc
và tham số theo mã nguồn gốc của tác giả Lê Minh Trí.
"""
from __future__ import annotations

import calendar
import datetime as dt
import html
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forecast.ai import danh_sach_model_gemini, xu_ly_du_lieu_dinh_tinh
from forecast.daily_ui import render_empirical_k_calculator
from forecast.calendar_state import dump_calendar, load_calendar
from forecast.calendar_utils import (
    CATEGORY_LABELS,
    build_month_calendar,
    calendar_detail_for_dataframe,
    coefficients_for_month,
    normalize_holiday_dates,
    outage_dates_for_month,
)
from forecast.config import DEFAULT_HOLIDAYS, DEFAULT_K
from forecast.data import ultra_scan_read_excel, kiem_tra_chat_luong, tao_dac_trung
from forecast.evaluation import backtest_expanding
from forecast.exporter import build_excel_report
from forecast.models import chay_mo_hinh_goc

warnings.filterwarnings('ignore', category=FutureWarning)

# Danh sách ngày nghỉ Lễ / Tết cụ thể mặc định (hỗ trợ nạp nhanh để xử lý trùng T7/CN)
DEFAULT_SPECIFIC_HOLIDAYS = [
    # --- Năm 2023 ---
    {'Ngày': '01/01/2023', 'Loại': 'Lễ', 'Ghi chú': 'Tết Dương lịch'},
    {'Ngày': '02/01/2023', 'Loại': 'Lễ', 'Ghi chú': 'Nghỉ bù Tết Dương lịch'},
    {'Ngày': '20/01/2023', 'Loại': 'Tết', 'Ghi chú': '29 Tết Quý Mão'},
    {'Ngày': '21/01/2023', 'Loại': 'Tết', 'Ghi chú': '30 Tết Quý Mão'},
    {'Ngày': '22/01/2023', 'Loại': 'Tết', 'Ghi chú': 'Mùng 1 Tết Quý Mão'},
    {'Ngày': '23/01/2023', 'Loại': 'Tết', 'Ghi chú': 'Mùng 2 Tết Quý Mão'},
    {'Ngày': '24/01/2023', 'Loại': 'Tết', 'Ghi chú': 'Mùng 3 Tết Quý Mão'},
    {'Ngày': '25/01/2023', 'Loại': 'Tết', 'Ghi chú': 'Mùng 4 Tết Quý Mão'},
    {'Ngày': '26/01/2023', 'Loại': 'Tết', 'Ghi chú': 'Mùng 5 Tết Quý Mão'},
    {'Ngày': '29/04/2023', 'Loại': 'Lễ', 'Ghi chú': 'Giỗ Tổ Hùng Vương'},
    {'Ngày': '30/04/2023', 'Loại': 'Lễ', 'Ghi chú': 'Ngày Chiến thắng'},
    {'Ngày': '01/05/2023', 'Loại': 'Lễ', 'Ghi chú': 'Quốc tế Lao động'},
    {'Ngày': '02/05/2023', 'Loại': 'Lễ', 'Ghi chú': 'Nghỉ bù 30/4'},
    {'Ngày': '03/05/2023', 'Loại': 'Lễ', 'Ghi chú': 'Nghỉ bù Giỗ Tổ'},
    {'Ngày': '01/09/2023', 'Loại': 'Lễ', 'Ghi chú': 'Nghỉ Quốc khánh'},
    {'Ngày': '02/09/2023', 'Loại': 'Lễ', 'Ghi chú': 'Quốc khánh'},
    {'Ngày': '03/09/2023', 'Loại': 'Lễ', 'Ghi chú': 'Nghỉ Quốc khánh'},
    {'Ngày': '04/09/2023', 'Loại': 'Lễ', 'Ghi chú': 'Nghỉ bù Quốc khánh'},
    # --- Năm 2024 ---
    {'Ngày': '01/01/2024', 'Loại': 'Lễ', 'Ghi chú': 'Tết Dương lịch'},
    {'Ngày': '08/02/2024', 'Loại': 'Tết', 'Ghi chú': '29 Tết Giáp Thìn'},
    {'Ngày': '09/02/2024', 'Loại': 'Tết', 'Ghi chú': '30 Tết Giáp Thìn'},
    {'Ngày': '10/02/2024', 'Loại': 'Tết', 'Ghi chú': 'Mùng 1 Tết Giáp Thìn'},
    {'Ngày': '11/02/2024', 'Loại': 'Tết', 'Ghi chú': 'Mùng 2 Tết Giáp Thìn'},
    {'Ngày': '12/02/2024', 'Loại': 'Tết', 'Ghi chú': 'Mùng 3 Tết Giáp Thìn'},
    {'Ngày': '13/02/2024', 'Loại': 'Tết', 'Ghi chú': 'Mùng 4 Tết Giáp Thìn'},
    {'Ngày': '14/02/2024', 'Loại': 'Tết', 'Ghi chú': 'Mùng 5 Tết Giáp Thìn'},
    {'Ngày': '18/04/2024', 'Loại': 'Lễ', 'Ghi chú': 'Giỗ Tổ Hùng Vương'},
    {'Ngày': '29/04/2024', 'Loại': 'Lễ', 'Ghi chú': 'Nghỉ hoán đổi 30/4'},
    {'Ngày': '30/04/2024', 'Loại': 'Lễ', 'Ghi chú': 'Ngày Chiến thắng'},
    {'Ngày': '01/05/2024', 'Loại': 'Lễ', 'Ghi chú': 'Quốc tế Lao động'},
    {'Ngày': '02/09/2024', 'Loại': 'Lễ', 'Ghi chú': 'Quốc khánh'},
    {'Ngày': '03/09/2024', 'Loại': 'Lễ', 'Ghi chú': 'Nghỉ Quốc khánh'},
    # --- Năm 2025 ---
    {'Ngày': '01/01/2025', 'Loại': 'Lễ', 'Ghi chú': 'Tết Dương lịch'},
    {'Ngày': '27/01/2025', 'Loại': 'Tết', 'Ghi chú': '28 Tết Ất Tỵ'},
    {'Ngày': '28/01/2025', 'Loại': 'Tết', 'Ghi chú': '29 Tết Ất Tỵ'},
    {'Ngày': '29/01/2025', 'Loại': 'Tết', 'Ghi chú': 'Mùng 1 Tết Ất Tỵ'},
    {'Ngày': '30/01/2025', 'Loại': 'Tết', 'Ghi chú': 'Mùng 2 Tết Ất Tỵ'},
    {'Ngày': '31/01/2025', 'Loại': 'Tết', 'Ghi chú': 'Mùng 3 Tết Ất Tỵ'},
    {'Ngày': '01/02/2025', 'Loại': 'Tết', 'Ghi chú': 'Mùng 4 Tết Ất Tỵ'},
    {'Ngày': '02/02/2025', 'Loại': 'Tết', 'Ghi chú': 'Mùng 5 Tết Ất Tỵ'},
    {'Ngày': '07/04/2025', 'Loại': 'Lễ', 'Ghi chú': 'Giỗ Tổ Hùng Vương'},
    {'Ngày': '30/04/2025', 'Loại': 'Lễ', 'Ghi chú': 'Ngày Chiến thắng'},
    {'Ngày': '01/05/2025', 'Loại': 'Lễ', 'Ghi chú': 'Quốc tế Lao động'},
    {'Ngày': '01/09/2025', 'Loại': 'Lễ', 'Ghi chú': 'Nghỉ Quốc khánh'},
    {'Ngày': '02/09/2025', 'Loại': 'Lễ', 'Ghi chú': 'Quốc khánh'},
    # --- Năm 2026 ---
    {'Ngày': '01/01/2026', 'Loại': 'Lễ', 'Ghi chú': 'Tết Dương lịch'},
    {'Ngày': '16/02/2026', 'Loại': 'Tết', 'Ghi chú': '29 Tết Bính Ngọ'},
    {'Ngày': '17/02/2026', 'Loại': 'Tết', 'Ghi chú': 'Mùng 1 Tết Bính Ngọ'},
    {'Ngày': '18/02/2026', 'Loại': 'Tết', 'Ghi chú': 'Mùng 2 Tết Bính Ngọ'},
    {'Ngày': '19/02/2026', 'Loại': 'Tết', 'Ghi chú': 'Mùng 3 Tết Bính Ngọ'},
    {'Ngày': '20/02/2026', 'Loại': 'Tết', 'Ghi chú': 'Mùng 4 Tết Bính Ngọ'},
    {'Ngày': '26/04/2026', 'Loại': 'Lễ', 'Ghi chú': 'Giỗ Tổ Hùng Vương'},
    {'Ngày': '27/04/2026', 'Loại': 'Lễ', 'Ghi chú': 'Nghỉ bù Giỗ Tổ'},
    {'Ngày': '30/04/2026', 'Loại': 'Lễ', 'Ghi chú': 'Ngày Chiến thắng'},
    {'Ngày': '01/05/2026', 'Loại': 'Lễ', 'Ghi chú': 'Quốc tế Lao động'},
    {'Ngày': '02/05/2026', 'Loại': 'Lễ', 'Ghi chú': 'Nghỉ lễ kéo dài 30/4-1/5'},
    {'Ngày': '01/09/2026', 'Loại': 'Lễ', 'Ghi chú': 'Nghỉ Quốc khánh'},
    {'Ngày': '02/09/2026', 'Loại': 'Lễ', 'Ghi chú': 'Quốc khánh'},
]

WEEKDAY_NAMES_VI = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']

st.set_page_config(page_title='Dự Báo Phụ Tải Tây Ninh', page_icon='⚡', layout='wide')
st.markdown('''
<style>
:root {--navy:#0b2f4f;--blue:#1769aa;--soft:#eef5fb;--muted:#66788a;}
.stApp {background:#f6f9fc;}
.block-container {padding-top:1.2rem;padding-bottom:4rem;max-width:1500px;}
h1,h2,h3 {color:var(--navy)!important;font-family:'Segoe UI',sans-serif;}
.hero {background:linear-gradient(115deg,#0b2f4f,#1769aa);padding:22px 26px;border-radius:18px;
color:white;margin-bottom:14px;box-shadow:0 8px 24px rgba(11,47,79,.15);}
.hero h2 {color:white!important;margin:0;font-size:1.65rem}.hero p{margin:.35rem 0 0;opacity:.9}
.card {background:white;border:1px solid #e4edf5;border-radius:14px;padding:15px 17px;
box-shadow:0 3px 12px rgba(25,60,90,.05);}
.small-note {color:#66788a;font-size:.9rem}
div[data-testid="stMetric"] {background:white;border:1px solid #e4edf5;padding:12px;border-radius:14px;}
div[data-testid="stTabs"] button {font-weight:650;}
[data-testid="stSidebar"] {background:#eef5fb;}

/* --- Cấu trúc Lịch Trực Quan Đánh Dấu Ngày Nghỉ --- */
.cal-legend {display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 12px 0;font-size:0.84rem;}
.cal-legend-item {display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:20px;font-weight:600;border:1px solid rgba(0,0,0,0.08);}
.cal-grid {display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin-top:6px;margin-bottom:14px;}
.cal-dow {background:#0b2f4f;color:white;text-align:center;padding:8px 4px;border-radius:8px;font-weight:700;font-size:0.85rem;}
.cal-cell {border-radius:10px;padding:8px 10px;min-height:88px;border:1px solid #dce6f0;display:flex;flex-direction:column;justify-content:space-between;transition:transform .12s;}
.cal-cell:hover {transform:translateY(-2px);box-shadow:0 4px 10px rgba(11,47,79,.10);}
.cal-empty {background:transparent;border:1px dashed #e2e8f0;min-height:88px;border-radius:10px;}
.cal-top {display:flex;justify-content:space-between;align-items:center;}
.cal-day-num {font-size:1.05rem;font-weight:800;}
.cal-badge {font-size:0.72rem;padding:2px 6px;border-radius:6px;font-weight:700;}
.cal-outage {background:#fef08a;color:#854d0e;border:1px solid #facc15;font-size:0.72rem;padding:1px 6px;border-radius:6px;font-weight:700;display:inline-block;margin-top:3px;}
.cal-note {font-size:0.74rem;opacity:0.85;margin-top:2px;line-height:1.15;}
.cal-k {font-size:0.76rem;font-weight:600;text-align:right;margin-top:4px;opacity:0.9;}

/* Màu sắc phân loại ngày trong lịch */
.cat-Thuong {background:#ffffff;color:#1e293b;border-left:4px solid #94a3b8;}
.cat-T2 {background:#f0f9ff;color:#0369a1;border-left:4px solid #38bdf8;}
.cat-T7 {background:#fffbeb;color:#b45309;border-left:4px solid #f59e0b;}
.cat-CN {background:#fff1f2;color:#be123c;border-left:4px solid #fb7185;}
.cat-Le {background:#fee2e2;color:#991b1b;border-left:4px solid #ef4444;}
.cat-Tet {background:#fae8ff;color:#86198f;border-left:4px solid #d946ef;}
</style>
''', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def du_bao_cache(df_train, df_input, holiday_map, k_dict, seed,
                 holiday_dates, outage_dates_map):
    return chay_mo_hinh_goc(
        df_train, df_input, holiday_map, k_dict, seed,
        holiday_dates=holiday_dates, outage_dates_map=outage_dates_map,
    )


@st.cache_data(show_spinner=False)
def backtest_cache(df_train, holiday_map, k_dict, holiday_dates,
                   outage_dates_map, n_months=12):
    return backtest_expanding(
        df_train, holiday_map, k_dict,
        holiday_dates=holiday_dates, outage_dates_map=outage_dates_map,
        n_months=n_months, seed=42, min_train_months=12,
    )


def doc_excel_hoac_bao_loi(file, ten_file, train=False):
    if file is None:
        return None
    try:
        df = ultra_scan_read_excel(file)
    except Exception as exc:
        st.error(f'Không đọc được file {ten_file}: {exc}')
        return None
    if df is None:
        st.error(f'Không đọc được file {ten_file}.')
        return None
    errors = kiem_tra_chat_luong(df, ten_file, la_train=train)
    if errors:
        st.error(f'Dữ liệu {ten_file} cần kiểm tra:')
        for error in errors:
            st.write('• ' + error)
        return None
    for column in ['Năm', 'Tháng', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm',
                   'Số ngày cắt điện', 'Tổng thương phẩm']:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column])
    return df


def tao_bang_ket_qua(df_input, df_train, holiday_map, holiday_dates,
                     outage_dates_map, k_dict, predicted):
    pred_nn, pred_rf, pred_xg, p_trend = predicted
    check = tao_dac_trung(df_input, holiday_map, holiday_dates,
                          outage_dates_map, k_dict)
    res = df_input[['Năm', 'Tháng', 'Số ngày']].copy()
    res['Số ngày cắt điện'] = df_input['Số ngày cắt điện'] if 'Số ngày cắt điện' in df_input.columns else 0
    for dst, src in [('Ngày Thường', 'So_Ngay_Thuong'), ('T2', 'So_Ngay_T2'),
                     ('T7', 'So_Ngay_T7'), ('CN', 'So_Ngay_CN'),
                     ('Lễ', 'So_Ngay_Le_Nho'), ('Tết', 'So_Ngay_Tet_Am'),
                     ('Cắt điện', 'So_Ngay_Cat_Dien')]:
        res[dst] = check[src]
    res['Ngày tương đương'] = check['Ngay_Tuong_Duong']
    res['Chế độ lễ'] = check['Che_Do_Le'].map({'exact': 'Ngày cụ thể', 'legacy': 'Tổng hợp cũ'})
    res['Chế độ cắt điện'] = check['Che_Do_Cat_Dien'].map({'exact': 'Ngày cụ thể', 'legacy': 'Tổng hợp cũ'})
    res['Neural Network'] = pred_nn
    res['Random Forest'] = pred_rf
    res['XGBoost'] = pred_xg
    res['Xu hướng'] = p_trend
    if 'Tổng thương phẩm' in df_train.columns:
        actual = df_train[['Năm', 'Tháng', 'Tổng thương phẩm']]
        res = pd.merge(res, actual, on=['Năm', 'Tháng'], how='left')
        res.rename(columns={'Tổng thương phẩm': 'Thực Tế'}, inplace=True)
    return res, p_trend


def plot_forecast(res: pd.DataFrame):
    plot = res.copy()
    plot['Ngày'] = pd.to_datetime(dict(year=plot['Năm'], month=plot['Tháng'], day=1))
    fig = go.Figure()
    for model in ['Neural Network', 'Random Forest', 'XGBoost']:
        fig.add_trace(go.Scatter(x=plot['Ngày'], y=plot[model], mode='lines+markers', name=model))
    if 'Thực Tế' in plot.columns and plot['Thực Tế'].notna().any():
        mask = plot['Thực Tế'].notna()
        fig.add_trace(go.Scatter(x=plot.loc[mask, 'Ngày'], y=plot.loc[mask, 'Thực Tế'],
                                 mode='markers+lines', name='Thực Tế',
                                 line=dict(width=3, color='#111827')))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=45, b=10),
                      hovermode='x unified', yaxis_title='Sản lượng (kWh)',
                      xaxis_title='', legend_title='Chuỗi')
    return fig


def month_key(y, m):
    return f'{int(m):02d}/{int(y)}'


def dataframe_outage_map(df_editor: pd.DataFrame) -> dict:
    result = {}
    if df_editor is None or df_editor.empty:
        return result
    for _, row in df_editor.iterrows():
        try:
            y, m = int(row['Năm']), int(row['Tháng'])
        except (ValueError, TypeError):
            continue
        value = row.get('Ngày cắt điện cụ thể', '')
        if pd.notna(value) and str(value).strip():
            result[(y, m)] = str(value).strip()
    return result


def finalized_dataframe() -> pd.DataFrame:
    data = st.session_state.get('finalized_results', {})
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(list(data.values())).sort_values(['Năm', 'Tháng']).reset_index(drop=True)


# ---------------- HỖ TRỢ CẤU TRÚC LỊCH & ĐÁNH DẤU NGÀY NGHỈ ----------------
def _extract_day_number(val, fallback_day: int) -> int:
    if isinstance(val, (int, float)) and not pd.isna(val):
        return int(val)
    if hasattr(val, 'day'):
        return int(val.day)
    try:
        return int(pd.to_datetime(str(val), dayfirst=True).day)
    except Exception:
        return fallback_day


def _parse_outage_days_set(outage_str: str, year: int, month: int) -> set[int]:
    days_set: set[int] = set()
    if not outage_str or pd.isna(outage_str):
        return days_set
    for part in str(outage_str).replace(';', ',').split(','):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            d = int(token)
            if 1 <= d <= 31:
                days_set.add(d)
            continue
        try:
            parsed = pd.to_datetime(token, dayfirst=True)
            if parsed.year == year and parsed.month == month:
                days_set.add(int(parsed.day))
        except Exception:
            continue
    return days_set


def render_visual_month_calendar(cal_df: pd.DataFrame, year: int, month: int,
                                 notes_map: dict[int, str] | None = None):
    """Hiển thị tờ lịch tháng trực quan 7 cột (Thứ Hai -> Chủ Nhật) có đánh dấu màu ngày nghỉ."""
    notes_map = notes_map or {}
    st.markdown('''
    <div class="cal-legend">
        <span class="cal-legend-item cat-Thuong">⬜ Ngày thường</span>
        <span class="cal-legend-item cat-T2">🟦 Thứ Hai (T2)</span>
        <span class="cal-legend-item cat-T7">🟨 Thứ Bảy (T7)</span>
        <span class="cal-legend-item cat-CN">🟥 Chủ Nhật (CN)</span>
        <span class="cal-legend-item cat-Le">🚩 Nghỉ Lễ</span>
        <span class="cal-legend-item cat-Tet">🌸 Nghỉ Tết Âm</span>
        <span class="cal-legend-item" style="background:#fef08a;color:#854d0e;border-color:#facc15;">⚡ Có cắt điện</span>
    </div>
    ''', unsafe_allow_html=True)

    first_weekday, num_days = calendar.monthrange(int(year), int(month))
    cells_html = ['<div class="cal-grid">']
    for dow_name in ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']:
        cells_html.append(f'<div class="cal-dow">{dow_name}</div>')

    for _ in range(first_weekday):
        cells_html.append('<div class="cal-empty"></div>')

    rows_by_day = {}
    for idx, (_, r) in enumerate(cal_df.iterrows()):
        d_num = _extract_day_number(r.get('Ngày'), idx + 1)
        rows_by_day[d_num] = r

    for d in range(1, num_days + 1):
        r = rows_by_day.get(d)
        if r is not None:
            cat = str(r.get('Category', 'Thuong'))
            label = CATEGORY_LABELS.get(cat, cat)
            is_outage = bool(r.get('Cat_Dien', False))
            k_cuoi = float(r.get('k_Cuoi', 1.0))
        else:
            wd = calendar.weekday(int(year), int(month), d)
            cat = 'T2' if wd == 0 else ('T7' if wd == 5 else ('CN' if wd == 6 else 'Thuong'))
            label = CATEGORY_LABELS.get(cat, cat)
            is_outage = False
            k_cuoi = 1.0

        note_text = html.escape(notes_map.get(d, ''))
        note_html = f'<div class="cal-note">📌 {note_text}</div>' if note_text else ''
        outage_html = '<div class="cal-outage">⚡ Cắt điện</div>' if is_outage else ''

        cell_html = (
            f'<div class="cal-cell cat-{cat}">'
            f'  <div>'
            f'    <div class="cal-top">'
            f'      <span class="cal-day-num">{d:02d}/{int(month):02d}</span>'
            f'      <span class="cal-badge">{html.escape(label)}</span>'
            f'    </div>'
            f'    {outage_html}'
            f'    {note_html}'
            f'  </div>'
            f'  <div class="cal-k">k = {k_cuoi:.3f}</div>'
            f'</div>'
        )
        cells_html.append(cell_html)

    trailing = (7 - ((first_weekday + num_days) % 7)) % 7
    for _ in range(trailing):
        cells_html.append('<div class="cal-empty"></div>')
    cells_html.append('</div>')
    st.markdown(''.join(cells_html), unsafe_allow_html=True)


def render_interactive_day_marker(available_ym: list[tuple[int, int]],
                                  holidays_map: dict,
                                  holiday_dates: pd.DataFrame,
                                  outage_dates_map: dict,
                                  k_default: dict,
                                  df_train: pd.DataFrame | None,
                                  df_input: pd.DataFrame | None):
    """Khung tương tác cho phép đánh dấu nhanh ngày nghỉ Lễ/Tết/Cắt điện trên lịch từng tháng."""
    st.subheader('🗓️ Lịch trực quan & Đánh dấu nhanh ngày nghỉ trong tháng')
    st.caption(
        'Chọn tháng để xem tờ lịch trực quan (Thứ Hai → Chủ Nhật) và đánh dấu trực tiếp ngày nào là '
        '**Nghỉ Lễ**, **Nghỉ Tết** hoặc **Cắt điện**. Khi bấm lưu, hệ thống sẽ tự động cập nhật vào bảng ngày cụ thể.'
    )

    c_sel1, c_sel2, c_sel3 = st.columns([2, 2, 2])
    with c_sel1:
        today = dt.date.today()
        sel_year = int(st.number_input('Năm xem lịch', min_value=2000, max_value=2100,
                                       value=today.year, step=1, key='visual_cal_year'))
        sel_month = int(st.selectbox('Tháng xem lịch', list(range(1, 13)),
                                     index=today.month - 1, key='visual_cal_month'))
        chosen_label = f'Tháng {sel_month:02d}/{sel_year}'

    fallback_outage_count = 0
    for df_src in [df_input, df_train]:
        if df_src is not None and 'Số ngày cắt điện' in df_src.columns:
            matched = df_src[(df_src['Năm'] == sel_year) & (df_src['Tháng'] == sel_month)]
            if not matched.empty:
                fallback_outage_count = int(matched.iloc[0].get('Số ngày cắt điện', 0) or 0)
                break

    exact_outage_str = outage_dates_map.get((sel_year, sel_month), '')

    seed_df = st.session_state.get('holiday_dates_seed', pd.DataFrame(columns=['Ngày', 'Loại', 'Ghi chú']))
    if 'Ghi chú' not in seed_df.columns:
        seed_df = seed_df.copy()
        seed_df['Ghi chú'] = ''

    month_holiday_type: dict[int, str] = {}
    month_holiday_note: dict[int, str] = {}
    for _, r in seed_df.iterrows():
        raw_d = r.get('Ngày', '')
        if pd.isna(raw_d) or not str(raw_d).strip():
            continue
        try:
            dt_val = pd.to_datetime(str(raw_d).strip(), dayfirst=True)
            if dt_val.year == sel_year and dt_val.month == sel_month:
                month_holiday_type[int(dt_val.day)] = str(r.get('Loại', 'Lễ')).strip()
                month_holiday_note[int(dt_val.day)] = str(r.get('Ghi chú', '') if pd.notna(r.get('Ghi chú', '')) else '').strip()
        except Exception:
            continue

    outage_days_set = _parse_outage_days_set(exact_outage_str, sel_year, sel_month)

    cal_preview_df = build_month_calendar(
        sel_year, sel_month, holidays_map, holiday_dates,
        exact_outage_str, fallback_outage_count, k_default
    )

    with c_sel2:
        total_equiv = float(cal_preview_df['k_Cuoi'].sum())
        st.metric('Tổng ngày tương đương tháng này', f'{total_equiv:.3f} ngày')
    with c_sel3:
        n_off = int(cal_preview_df['Category'].isin(['T7', 'CN', 'Le', 'Tet']).sum())
        st.metric('Số ngày T7, CN, lễ/Tết', f'{n_off} ngày')

    render_visual_month_calendar(cal_preview_df, sel_year, sel_month, notes_map=month_holiday_note)

    with st.expander(f'✍️ Đánh dấu / Chỉnh sửa từng ngày cho {chosen_label}', expanded=False):
        num_days = calendar.monthrange(sel_year, sel_month)[1]
        day_rows = []
        for d in range(1, num_days + 1):
            wd = calendar.weekday(sel_year, sel_month, d)
            auto_label = 'Thứ Bảy (Nghỉ cuối tuần)' if wd == 5 else (
                'Chủ Nhật (Nghỉ cuối tuần)' if wd == 6 else (
                    'Thứ Hai (Đặc thù đầu tuần)' if wd == 0 else 'Ngày làm việc thường'
                )
            )
            current_mark = month_holiday_type.get(d, 'Tự động theo thứ')
            if current_mark not in ['Tự động theo thứ', 'Lễ', 'Tết']:
                current_mark = 'Tự động theo thứ'
            day_rows.append({
                'Ngày': f'{d:02d}/{sel_month:02d}/{sel_year}',
                'Thứ': WEEKDAY_NAMES_VI[wd],
                'Mặc định theo lịch': auto_label,
                'Đánh dấu ngày nghỉ': current_mark,
                'Cắt điện': d in outage_days_set,
                'Ghi chú ngày nghỉ': month_holiday_note.get(d, ''),
            })

        day_marker_df = pd.DataFrame(day_rows)
        edited_month_days = st.data_editor(
            day_marker_df,
            hide_index=True,
            use_container_width=True,
            disabled=['Ngày', 'Thứ', 'Mặc định theo lịch'],
            key=f'interactive_day_marker_{sel_year}_{sel_month}_{st.session_state.get("cal_sync_ver", 0)}',
            column_config={
                'Đánh dấu ngày nghỉ': st.column_config.SelectboxColumn(
                    'Đánh dấu ngày nghỉ',
                    options=['Tự động theo thứ', 'Lễ', 'Tết'],
                    required=True,
                    help='Chọn Lễ hoặc Tết để đánh dấu ngày nghỉ cụ thể; chọn Tự động theo thứ để tính theo T2-CN bình thường.'
                ),
                'Cắt điện': st.column_config.CheckboxColumn(
                    '⚡ Cắt điện',
                    help='Tích chọn nếu ngày này có lịch cắt điện công tác/tiết giảm.'
                ),
                'Ghi chú ngày nghỉ': st.column_config.TextColumn(
                    'Ghi chú',
                    help='Ví dụ: Giỗ Tổ, nghỉ bù, hoạt động của khu công nghiệp...'
                ),
            },
            height=360,
        )

        if st.button(f'💾 Áp dụng đánh dấu của {chosen_label} vào cấu hình', type='primary', use_container_width=True):
            kept_rows = []
            for _, r in seed_df.iterrows():
                raw_d = r.get('Ngày', '')
                if pd.isna(raw_d) or not str(raw_d).strip():
                    continue
                try:
                    dt_val = pd.to_datetime(str(raw_d).strip(), dayfirst=True)
                    if dt_val.year == sel_year and dt_val.month == sel_month:
                        continue
                except Exception:
                    pass
                kept_rows.append({
                    'Ngày': str(raw_d).strip(),
                    'Loại': str(r.get('Loại', 'Lễ')).strip(),
                    'Ghi chú': str(r.get('Ghi chú', '') if pd.notna(r.get('Ghi chú', '')) else '').strip(),
                })

            new_outage_dates = []
            for _, r in edited_month_days.iterrows():
                mark = r['Đánh dấu ngày nghỉ']
                note = r.get('Ghi chú ngày nghỉ', '')
                note = '' if pd.isna(note) else str(note).strip()
                if mark in ['Lễ', 'Tết'] or note:
                    kept_rows.append({
                        'Ngày': r['Ngày'],
                        'Loại': mark if mark in ['Lễ', 'Tết'] else 'Ghi chú',
                        'Ghi chú': note,
                    })
                if bool(r.get('Cắt điện', False)):
                    new_outage_dates.append(r['Ngày'])

            st.session_state['holiday_dates_seed'] = pd.DataFrame(
                kept_rows, columns=['Ngày', 'Loại', 'Ghi chú']
            )
            st.session_state.setdefault('custom_outage_Overrides', {})
            st.session_state['custom_outage_Overrides'][(sel_year, sel_month)] = (
                ', '.join(new_outage_dates) if new_outage_dates else 'Không có'
            )
            st.session_state.setdefault('calendar_confirmed_months', set()).add((sel_year, sel_month))
            st.session_state['cal_sync_ver'] = st.session_state.get('cal_sync_ver', 0) + 1
            st.success(f'Đã cập nhật lịch nghỉ và ngày cắt điện cho {chosen_label}.')
            st.rerun()


def render_find_seed(df_train, df_input, holiday_map, k_default,
                     holiday_dates, outage_dates_map):
    st.subheader('🔬 Find Seed')
    st.warning('Find Seed là công cụ thử nghiệm. Seed được chọn theo giá trị mục tiêu không được dùng để tính MAE/MAPE/RMSE.')
    if df_train is None or df_input is None:
        st.caption('Cần dữ liệu lịch sử và dự báo hợp lệ.')
        return
    st.session_state.setdefault('scan_current_seed', 0)
    st.session_state.setdefault('scan_history', pd.DataFrame())
    dates = [f"Tháng {int(r['Tháng'])}/{int(r['Năm'])}" for _, r in df_input.iterrows()]
    c1, c2, c3 = st.columns(3)
    with c1:
        selected = st.selectbox('Tháng muốn soi', dates, index=len(dates)-1, key='seed_month')
        target_index = dates.index(selected)
    with c2:
        target_val = st.number_input('Giá trị mong muốn', value=740000000.0,
                                     step=1000000.0, format='%.0f', key='seed_target')
    with c3:
        model_choice = st.selectbox('Mô hình', ['Neural Network', 'Random Forest', 'XGBoost'], key='seed_model')
    c4, c5 = st.columns(2)
    with c4:
        acc = st.slider('Sai số cho phép (+/- %)', 0.1, 10.0, 1.0, key='seed_acc')
    with c5:
        batch_size = st.number_input('Số seed/lô', min_value=1, value=20, step=10, key='seed_batch')
    start = int(st.session_state.scan_current_seed)
    end = start + int(batch_size) - 1
    b1, b2 = st.columns([2, 1])
    if b1.button(f'▶️ Chạy Seed {start}–{end}', use_container_width=True):
        limit_min, limit_max = target_val * (1-acc/100), target_val * (1+acc/100)
        batch = []
        progress = st.progress(0)
        for i, seed in enumerate(range(start, end + 1)):
            try:
                p_nn, p_rf, p_xg, _ = chay_mo_hinh_goc(
                    df_train, df_input, holiday_map, k_default, seed,
                    holiday_dates=holiday_dates, outage_dates_map=outage_dates_map)
                vals = {'Neural Network': p_nn[target_index], 'Random Forest': p_rf[target_index], 'XGBoost': p_xg[target_index]}
                chosen = float(vals[model_choice])
                batch.append({'Seed': seed, 'Tháng': selected,
                              'Kết quả (NN)': float(p_nn[target_index]),
                              'Kết quả (RF)': float(p_rf[target_index]),
                              'Kết quả (XGB)': float(p_xg[target_index]),
                              'Độ lệch': chosen-target_val,
                              'Trạng thái': '✅ ĐẠT' if limit_min <= chosen <= limit_max else '❌'})
            except Exception as exc:
                batch.append({'Seed': seed, 'Tháng': selected, 'Trạng thái': f'⚠️ {str(exc)[:60]}'})
            progress.progress((i+1)/int(batch_size))
        st.session_state.scan_history = pd.concat([st.session_state.scan_history, pd.DataFrame(batch)], ignore_index=True)
        st.session_state.scan_current_seed = end + 1
        st.rerun()
    if b2.button('🗑️ Xóa lịch sử', use_container_width=True):
        st.session_state.scan_current_seed = 0
        st.session_state.scan_history = pd.DataFrame()
        st.rerun()
    if not st.session_state.scan_history.empty:
        show = st.session_state.scan_history.sort_values('Seed')
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={
                         'Kết quả (NN)': st.column_config.NumberColumn(format='%.0f'),
                         'Kết quả (RF)': st.column_config.NumberColumn(format='%.0f'),
                         'Kết quả (XGB)': st.column_config.NumberColumn(format='%.0f'),
                         'Độ lệch': st.column_config.NumberColumn(format='%.0f'),
                     })


# ----------------------------- Header -----------------------------
st.markdown('''<div class="hero"><h2>⚡ HỆ THỐNG DỰ BÁO PHỤ TẢI ĐIỆN TỈNH TÂY NINH</h2>
<p>Tác giả mã nguồn gốc: Lê Minh Trí · Dashboard dự báo, kiểm định và giải trình</p></div>''',
            unsafe_allow_html=True)

# Khởi tạo giá trị mặc định trong session_state cho 6 hệ số k để hỗ trợ cập nhật tự động từ file ngày
for _k_name, _k_val in DEFAULT_K.items():
    st.session_state.setdefault(f'k_input_{_k_name}', float(_k_val))
for _k_name, _k_val in st.session_state.pop('_pending_common_k', {}).items():
    st.session_state[f'k_input_{_k_name}'] = float(_k_val)

with st.sidebar:
    st.header('⚙️ Cấu hình mô hình')
    seed_val = st.number_input('Random Seed dự báo', value=42, step=1)
    st.caption('Backtest luôn dùng seed 42 để đánh giá khách quan và nhất quán.')
    st.divider()
    st.write('**Hệ số chung mặc định (k)**')
    st.caption('Để tính từ file điện nhận/giao: mở tab 🧮 Tính hệ số k, tải file, '
               'chọn tháng dự báo rồi bấm áp dụng. Hệ số riêng chỉ thay đổi tháng đã chọn.')
    c1, c2 = st.columns(2)
    with c1:
        k_t2 = st.number_input('T2', step=0.01, format='%.3f', key='k_input_T2')
        k_t7 = st.number_input('T7', step=0.01, format='%.3f', key='k_input_T7')
        k_cn = st.number_input('CN', step=0.01, format='%.3f', key='k_input_CN')
    with c2:
        k_le = st.number_input('Lễ', step=0.01, format='%.3f', key='k_input_Le')
        k_tet = st.number_input('Tết', step=0.01, format='%.3f', key='k_input_Tet')
        k_catdien = st.number_input('Cắt điện', step=0.01, format='%.3f', key='k_input_CatDien')
    k_default = {'T2': k_t2, 'T7': k_t7, 'CN': k_cn,
                 'Le': k_le, 'Tet': k_tet, 'CatDien': k_catdien}
    month_overrides = st.session_state.get('monthly_k_overrides', {})
    k_default['__monthly__'] = tuple(
        (int(y), int(m), tuple(sorted(changes.items())))
        for (y, m), changes in sorted(month_overrides.items())
    )
    if month_overrides:
        _k_month_text = ', '.join(f'{m:02d}/{y}' for y, m in sorted(month_overrides))
        st.info('Đã có hệ số riêng cho tháng: ' + _k_month_text + '. Xem chi tiết bên dưới.')
        with st.expander('Hệ số riêng từng tháng'):
            for (y, m), changes in sorted(month_overrides.items()):
                st.write(f'{m:02d}/{y}: ' + ', '.join(f'{k}={v:.4f}' for k, v in sorted(changes.items())))
            remove_ym = st.selectbox('Tháng muốn xóa hệ số riêng', sorted(month_overrides),
                                     format_func=lambda ym: f'{ym[1]:02d}/{ym[0]}')
            if st.button('Xóa hệ số riêng của tháng'):
                st.session_state['monthly_k_overrides'].pop(remove_ym, None)
                st.rerun()
    st.divider()
    st.write('**Lịch tổng hợp tương thích dữ liệu cũ**')
    holidays_edited = st.data_editor(pd.DataFrame(DEFAULT_HOLIDAYS), num_rows='dynamic',
                                     use_container_width=True, key='holiday_count_editor', height=260)
    holidays_map = {}
    for _, row in holidays_edited.dropna(subset=['Năm', 'Tháng']).fillna(0).iterrows():
        try:
            y, m = int(row['Năm']), int(row['Tháng'])
            holidays_map[(y, m)] = (max(0, int(row['Tết Âm'])), max(0, int(row['Lễ Nhỏ'])))
        except (TypeError, ValueError):
            pass
    st.divider()
    st.write('**AI phân tích tin tức (tùy chọn)**')
    provider = st.selectbox('Nhà cung cấp', ['Google Gemini', 'OpenAI ChatGPT'])
    api_key = st.text_input('API Key', type='password')
    available_models = []
    if provider == 'Google Gemini' and api_key:
        try:
            available_models = danh_sach_model_gemini(api_key)
        except Exception:
            st.warning('Không lấy được danh sách Gemini.')
    elif provider == 'OpenAI ChatGPT':
        available_models = ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']
    selected_model = st.selectbox('Model AI tin tức', available_models) if available_models else None
    st.divider()
    if st.button('🗑️ Reset kết quả & cache', use_container_width=True):
        st.cache_data.clear()
        for key in ['res_output', 'trend_val', 'result_signature', 'evaluation_metrics',
                    'evaluation_detail', 'finalized_results', 'scan_history', 'scan_current_seed']:
            st.session_state.pop(key, None)
        st.rerun()

# Upload nằm cố định phía trên để mọi tab dùng chung.
up1, up2 = st.columns(2)
with up1:
    f_train = st.file_uploader('📚 File Lịch Sử (Train)', type=['xlsx', 'xls'])
with up2:
    f_input = st.file_uploader('🔮 File Dự Báo (Input)', type=['xlsx', 'xls'])

df_train_main = doc_excel_hoac_bao_loi(f_train, 'Lịch Sử', train=True) if f_train else None
df_input_main = doc_excel_hoac_bao_loi(f_input, 'Dự Báo') if f_input else None

tabs = st.tabs(['🏠 Tổng quan', '📂 Dữ liệu & lịch', '🧮 Tính hệ số k', '🔮 Dự báo', '📊 Đánh giá mô hình',
                '🎛 Tinh chỉnh', '🤖 AI & công cụ', '📥 Xuất báo cáo'])

# ----------------------------- Dữ liệu & lịch -----------------------------
with tabs[1]:
    st.subheader('Dữ liệu đầu vào')
    c1, c2, c3 = st.columns(3)
    c1.metric('Tháng lịch sử', len(df_train_main) if df_train_main is not None else 0)
    c2.metric('Tháng cần dự báo', len(df_input_main) if df_input_main is not None else 0)
    total_missing = 0
    for df in [df_train_main, df_input_main]:
        if df is not None:
            total_missing += int(df.isna().sum().sum())
    c3.metric('Ô trống', total_missing)
    if df_train_main is not None:
        with st.expander('Xem dữ liệu lịch sử'):
            st.dataframe(df_train_main, hide_index=True, use_container_width=True)
    if df_input_main is not None:
        with st.expander('Xem dữ liệu dự báo'):
            st.dataframe(df_input_main, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader('📅 Lịch lễ theo ngày cụ thể')
    st.info('Lịch mẫu chỉ để tham khảo. Hãy đối chiếu và đánh dấu đủ các ngày ảnh hưởng thực tế; '
            'nếu một tháng có ngày Lễ/Tết cụ thể, danh sách cụ thể thay thế số lượng tổng hợp tháng đó. '
            'Dòng chỉ có ghi chú không làm thay đổi loại ngày.')

    if 'holiday_dates_seed' not in st.session_state:
        st.session_state['holiday_dates_seed'] = pd.DataFrame(DEFAULT_SPECIFIC_HOLIDAYS)

    btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 2])
    with btn_col1:
        if st.button('📥 Nạp lịch Lễ/Tết tham khảo (2023–2026)', use_container_width=True):
            st.session_state['holiday_dates_seed'] = pd.DataFrame(DEFAULT_SPECIFIC_HOLIDAYS)
            st.session_state['calendar_confirmed_months'] = set()
            st.session_state['cal_sync_ver'] = st.session_state.get('cal_sync_ver', 0) + 1
            st.rerun()
    with btn_col2:
        with st.popover('➕ Thêm nhanh khoảng ngày nghỉ Lễ/Tết', use_container_width=True):
            p_start = st.date_input('Từ ngày', value=dt.date(2026, 4, 30), key='range_hol_start')
            p_end = st.date_input('Đến ngày', value=dt.date(2026, 5, 1), key='range_hol_end')
            p_type = st.selectbox('Loại ngày nghỉ', ['Lễ', 'Tết'], key='range_hol_type')
            p_note = st.text_input('Ghi chú kỳ nghỉ', value='Nghỉ Lễ 30/4 - 1/5', key='range_hol_note')
            if st.button('✅ Thêm vào bảng ngày nghỉ', type='primary', use_container_width=True):
                if p_end < p_start:
                    st.error('Ngày kết thúc phải lớn hơn hoặc bằng ngày bắt đầu.')
                else:
                    cur_df = st.session_state.get(
                        'holiday_dates_seed',
                        pd.DataFrame(columns=['Ngày', 'Loại', 'Ghi chú'])
                    ).copy()
                    if 'Ghi chú' not in cur_df.columns:
                        cur_df['Ghi chú'] = ''
                    new_dates = pd.date_range(p_start, p_end, freq='D')
                    add_rows = [
                        {'Ngày': d.strftime('%d/%m/%Y'), 'Loại': p_type, 'Ghi chú': p_note}
                        for d in new_dates
                    ]
                    combined = pd.concat([cur_df, pd.DataFrame(add_rows)], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['Ngày'], keep='last').reset_index(drop=True)
                    st.session_state['holiday_dates_seed'] = combined
                    st.session_state['cal_sync_ver'] = st.session_state.get('cal_sync_ver', 0) + 1
                    st.rerun()
    with btn_col3:
        if st.button('🧹 Xóa trống bảng ngày cụ thể (dùng tổng hợp cũ)', use_container_width=True):
            st.session_state['holiday_dates_seed'] = pd.DataFrame(columns=['Ngày', 'Loại', 'Ghi chú'])
            st.session_state['calendar_confirmed_months'] = set()
            st.session_state['cal_sync_ver'] = st.session_state.get('cal_sync_ver', 0) + 1
            st.rerun()

    initial_holiday_dates = st.session_state.get(
        'holiday_dates_seed',
        pd.DataFrame(columns=['Ngày', 'Loại', 'Ghi chú'])
    )
    if 'Ghi chú' not in initial_holiday_dates.columns:
        initial_holiday_dates = initial_holiday_dates.copy()
        initial_holiday_dates['Ghi chú'] = ''

    holiday_dates_editor = st.data_editor(
        initial_holiday_dates, num_rows='dynamic', hide_index=True, use_container_width=True,
        key=f"holiday_dates_editor_{st.session_state.get('cal_sync_ver', 0)}",
        column_config={
            'Ngày': st.column_config.TextColumn('Ngày', help='Ví dụ 30/04/2026 hoặc 2026-04-30'),
            'Loại': st.column_config.SelectboxColumn('Loại', options=['Lễ', 'Tết', 'Ghi chú'], required=True),
            'Ghi chú': st.column_config.TextColumn('Ghi chú sự kiện', help='Tên ngày nghỉ lễ/Tết'),
        })
    st.session_state['holiday_dates_seed'] = holiday_dates_editor
    holiday_dates = normalize_holiday_dates(holiday_dates_editor[['Ngày', 'Loại']])
    for _idx, _entry in holiday_dates_editor.iterrows():
        _text = _entry.get('Ngày', '')
        if pd.notna(_text) and str(_text).strip() and pd.isna(pd.to_datetime(str(_text), dayfirst=True, errors='coerce')):
            st.warning(f"Ngày '{_text}' không hợp lệ và chưa được dùng; hãy sửa theo dạng dd/mm/yyyy.")
    # A month explicitly reviewed in the visual calendar must not silently
    # reintroduce legacy holiday counts after all marked holidays are removed.
    for confirmed_y, confirmed_m in st.session_state.get('calendar_confirmed_months', set()):
        holidays_map[(int(confirmed_y), int(confirmed_m))] = (0, 0)
    if not holiday_dates.empty:
        st.success(f'Đã nhận {len(holiday_dates)} ngày lễ/Tết cụ thể.')

    st.divider()
    st.subheader('⚡ Ngày cắt điện cụ thể')
    st.caption('Ví dụ: `12/11/2026, 19/11/2026`. Nếu để trống, chương trình dùng "Số ngày cắt điện" trong Excel và phân bổ tương thích cách tính cũ.')
    custom_outage_overrides = st.session_state.get('custom_outage_Overrides', {})
    months_rows = []
    for df in [df_train_main, df_input_main]:
        if df is not None:
            for _, r in df.iterrows():
                ym_pair = (int(r['Năm']), int(r['Tháng']))
                months_rows.append({
                    'Năm': ym_pair[0],
                    'Tháng': ym_pair[1],
                    'Ngày cắt điện cụ thể': custom_outage_overrides.get(ym_pair, ''),
                })
    for (oy, om), oval in sorted(custom_outage_overrides.items()):
        months_rows.append({'Năm': oy, 'Tháng': om, 'Ngày cắt điện cụ thể': oval})

    outage_seed = (
        pd.DataFrame(months_rows).drop_duplicates(['Năm', 'Tháng'])
        if months_rows
        else pd.DataFrame(columns=['Năm', 'Tháng', 'Ngày cắt điện cụ thể'])
    )
    outage_editor = st.data_editor(
        outage_seed, num_rows='dynamic', hide_index=True,
        use_container_width=True,
        key=f"outage_dates_editor_{st.session_state.get('cal_sync_ver', 0)}"
    )
    outage_dates_map = dataframe_outage_map(outage_editor)
    st.session_state['custom_outage_Overrides'] = dict(outage_dates_map)

    with st.expander('💾 Sao lưu / khôi phục lịch và ghi chú'):
        st.caption('Lịch và hệ số trong phiên làm việc có thể mất khi tải lại trang. '
                   'Hãy tải bản sao lưu lịch sau khi chỉnh sửa; bản sao lưu không chứa các file điện.')
        calendar_backup = dump_calendar(holiday_dates_editor, outage_dates_map,
                                        st.session_state.get('calendar_confirmed_months', set()),
                                        k_default, st.session_state.get('monthly_k_overrides', {}))
        st.download_button('Tải bản sao lưu lịch (.json)', data=calendar_backup,
                           file_name='lich_va_ghi_chu.json', mime='application/json')
        calendar_import = st.file_uploader('Chọn bản sao lưu lịch để khôi phục', type=['json'],
                                           key='calendar_backup_upload')
        if st.button('Khôi phục lịch từ tệp', disabled=calendar_import is None):
            try:
                restored_rows, restored_outages, restored_months, restored_common, restored_k = load_calendar(calendar_import.getvalue())
                st.session_state['holiday_dates_seed'] = restored_rows
                st.session_state['custom_outage_Overrides'] = restored_outages
                st.session_state['calendar_confirmed_months'] = restored_months
                st.session_state['_pending_common_k'] = restored_common
                st.session_state['monthly_k_overrides'] = restored_k
                st.session_state['cal_sync_ver'] = st.session_state.get('cal_sync_ver', 0) + 1
                st.rerun()
            except (ValueError, TypeError, KeyError) as exc:
                st.error(f'Không khôi phục được lịch: {exc}')

    st.divider()
    # Tập hợp danh sách các tháng để hiển thị trên Lịch trực quan & bảng đánh dấu
    all_ym_list = sorted({
        (int(r['Năm']), int(r['Tháng']))
        for df in [df_train_main, df_input_main] if df is not None
        for _, r in df.iterrows()
    } | set(holidays_map.keys()))
    render_interactive_day_marker(
        all_ym_list, holidays_map, holiday_dates,
        outage_dates_map, k_default, df_train_main, df_input_main
    )

    if df_input_main is not None:
        st.divider()
        st.subheader('Kiểm tra cơ cấu lịch tháng dự báo')
        features_preview = tao_dac_trung(df_input_main, holidays_map, holiday_dates,
                                         outage_dates_map, k_default)
        preview_cols = ['Năm', 'Tháng', 'So_Ngay_Thuong', 'So_Ngay_T2', 'So_Ngay_T7',
                        'So_Ngay_CN', 'So_Ngay_Le_Nho', 'So_Ngay_Tet_Am',
                        'So_Ngay_Cat_Dien', 'Ngay_Tuong_Duong', 'Che_Do_Le', 'Che_Do_Cat_Dien']
        st.dataframe(features_preview[preview_cols], hide_index=True, use_container_width=True)

# ----------------------- Phân tích hệ số k -----------------------
with tabs[2]:
    render_empirical_k_calculator(holiday_dates, outage_dates_map, k_default)

try:
    holiday_dates
except NameError:
    holiday_dates = normalize_holiday_dates(pd.DataFrame(columns=['Ngày', 'Loại']))
try:
    outage_dates_map
except NameError:
    outage_dates_map = {}

holiday_sig = holiday_dates.to_csv(index=False) if not holiday_dates.empty else ''
outage_sig = tuple(sorted((k, str(v)) for k, v in outage_dates_map.items()))
input_signature = (
    f_train.getvalue() if f_train else None,
    f_input.getvalue() if f_input else None,
    tuple(sorted(holidays_map.items())), tuple(sorted(k_default.items())), int(seed_val),
    holiday_sig, outage_sig,
)
if st.session_state.get('result_signature') not in (None, input_signature):
    for key in ['res_output', 'trend_val', 'evaluation_metrics', 'evaluation_detail', 'finalized_results']:
        st.session_state.pop(key, None)

# ----------------------------- Dự báo -----------------------------
with tabs[3]:
    st.subheader('Chạy dự báo định lượng')
    st.caption('Ba mô hình cốt lõi giữ nguyên: Neural Network, Random Forest và XGBoost. AI tin tức không tự động thay đổi kết quả này.')
    if df_train_main is None or df_input_main is None:
        st.info('Tải hai file Excel hợp lệ để bắt đầu.')
    else:
        if st.button('🚀 CHẠY DỰ BÁO', type='primary', use_container_width=True):
            try:
                with st.spinner('Đang chạy 3 mô hình...'):
                    prediction = du_bao_cache(df_train_main, df_input_main, holidays_map,
                                              k_default, int(seed_val), holiday_dates,
                                              outage_dates_map)
                    res, trend = tao_bang_ket_qua(df_input_main, df_train_main,
                                                  holidays_map, holiday_dates,
                                                  outage_dates_map, k_default, prediction)
                st.session_state.res_output = res
                st.session_state.trend_val = trend
                st.session_state.result_signature = input_signature
                st.success('Đã chạy xong dự báo.')
            except Exception as exc:
                st.error(f'Dự báo không thành công: {exc}')
        if 'res_output' in st.session_state:
            res = st.session_state.res_output
            st.plotly_chart(plot_forecast(res), use_container_width=True, key='chart_forecast_tab')
            display_cols = ['Năm', 'Tháng', 'Ngày tương đương', 'Neural Network', 'Random Forest', 'XGBoost', 'Thực Tế']
            display_cols = [c for c in display_cols if c in res.columns]
            st.dataframe(res[display_cols], hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format='%.0f')
                                        for c in ['Neural Network', 'Random Forest', 'XGBoost', 'Thực Tế'] if c in display_cols})

# ----------------------------- Đánh giá -----------------------------
with tabs[4]:
    st.subheader('Đánh giá độ chính xác bằng backtest theo thời gian')
    st.info('Kiểm định hiện tại là dự báo một bước: huấn luyện đến tháng ngay trước tháng kiểm định, '
            'dùng các biến đầu vào có trong file và seed 42. Chương trình chưa tự tái dựng số ước '
            'đã có vào ngày 10 hoặc tự tính lại k cho từng mốc quá khứ. Nếu nhiệt độ, độ ẩm hay '
            'hệ số k được biết sau mốc dự báo thì kết quả không phải kiểm định độc lập.')
    if df_train_main is None:
        st.caption('Cần file lịch sử.')
    elif len(df_train_main) <= 12:
        st.warning('Cần nhiều hơn 12 tháng lịch sử để thực hiện backtest theo cấu hình hiện tại.')
    else:
        n_backtest = st.slider('Số tháng backtest', min_value=3,
                               max_value=min(24, len(df_train_main)-12),
                               value=min(12, len(df_train_main)-12))
        if st.button('📊 CHẠY ĐÁNH GIÁ', use_container_width=True):
            try:
                with st.spinner('Đang walk-forward backtest...'):
                    metrics, detail = backtest_cache(df_train_main, holidays_map, k_default,
                                                     holiday_dates, outage_dates_map,
                                                     int(n_backtest))
                st.session_state.evaluation_metrics = metrics
                st.session_state.evaluation_detail = detail
                st.success('Đã đánh giá xong.')
            except Exception as exc:
                st.error(f'Không đánh giá được: {exc}')
        if 'evaluation_metrics' in st.session_state:
            metrics = st.session_state.evaluation_metrics
            best = metrics.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Mô hình MAPE thấp nhất', best['Mô hình'])
            m2.metric('MAPE', f"{best['MAPE (%)']:.2f}%")
            m3.metric('MAE', f"{best['MAE']:,.0f} kWh")
            m4.metric('RMSE', f"{best['RMSE']:,.0f} kWh")
            st.dataframe(metrics, hide_index=True, use_container_width=True,
                         column_config={'MAE': st.column_config.NumberColumn(format='%.0f'),
                                        'RMSE': st.column_config.NumberColumn(format='%.0f'),
                                        'MAPE (%)': st.column_config.NumberColumn(format='%.2f')})
            detail = st.session_state.evaluation_detail.copy()
            detail['Ngày'] = pd.to_datetime(dict(year=detail['Năm'], month=detail['Tháng'], day=1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=detail['Ngày'], y=detail['Thực Tế'], mode='lines+markers', name='Thực Tế', line=dict(width=3)))
            for model in ['Neural Network', 'Random Forest', 'XGBoost']:
                fig.add_trace(go.Scatter(x=detail['Ngày'], y=detail[model], mode='lines+markers', name=model))
            fig.update_layout(height=450, hovermode='x unified', yaxis_title='kWh', xaxis_title='', title='Backtest: Thực tế vs dự báo')
            st.plotly_chart(fig, use_container_width=True, key='chart_backtest_line')
            ape_cols = ['Ngày'] + [f'APE {m} (%)' for m in ['Neural Network', 'Random Forest', 'XGBoost']]
            ape = detail[ape_cols].melt('Ngày', var_name='Mô hình', value_name='APE (%)')
            ape['Mô hình'] = ape['Mô hình'].str.replace('APE ', '', regex=False).str.replace(' (%)', '', regex=False)
            fig2 = px.bar(ape, x='Ngày', y='APE (%)', color='Mô hình', barmode='group', title='Sai số tuyệt đối theo tháng (%)')
            fig2.update_layout(height=400, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig2, use_container_width=True, key='chart_backtest_bar')
            with st.expander('Chi tiết từng tháng'):
                st.dataframe(st.session_state.evaluation_detail, hide_index=True, use_container_width=True)

# ----------------------------- Tinh chỉnh -----------------------------
with tabs[5]:
    st.subheader('Tinh chỉnh & chốt sản lượng')
    if 'res_output' not in st.session_state or df_input_main is None:
        st.info('Chạy dự báo trước khi tinh chỉnh.')
    else:
        res = st.session_state.res_output
        model_for_base = st.selectbox('Mô hình làm cơ sở', ['XGBoost', 'Random Forest', 'Neural Network'])
        options = [month_key(r['Năm'], r['Tháng']) for _, r in df_input_main.iterrows()]
        selected = st.selectbox('Tháng cần tinh chỉnh', options)
        sm, sy = map(int, selected.split('/'))
        row_input = df_input_main[(df_input_main['Năm'] == sy) & (df_input_main['Tháng'] == sm)].iloc[0]
        row_res = res[(res['Năm'] == sy) & (res['Tháng'] == sm)].iloc[0]
        exact_global = outage_dates_map.get((sy, sm), '')
        current_count = int(row_input.get('Số ngày cắt điện', 0) or 0)
        active_month_k = coefficients_for_month(k_default, sy, sm)
        edit = pd.DataFrame([{
            'Tháng': selected,
            'Số ngày cắt điện (fallback)': current_count,
            'Ngày cắt điện cụ thể': exact_global,
            'k_T2': active_month_k['T2'], 'k_T7': active_month_k['T7'], 'k_CN': active_month_k['CN'],
            'k_Lễ': active_month_k['Le'], 'k_Tết': active_month_k['Tet'], 'k_Cắt điện': active_month_k['CatDien'],
        }])
        edited = st.data_editor(edit, hide_index=True, use_container_width=True,
                                disabled=['Tháng'], key=f'adjust_{selected}_{model_for_base}')
        e = edited.iloc[0]
        new_k = {'T2': float(e['k_T2']), 'T7': float(e['k_T7']), 'CN': float(e['k_CN']),
                 'Le': float(e['k_Lễ']), 'Tet': float(e['k_Tết']), 'CatDien': float(e['k_Cắt điện'])}
        old_exact = outage_dates_for_month(row_input, outage_dates_map)
        old_cal = build_month_calendar(sy, sm, holidays_map, holiday_dates,
                                       old_exact, current_count, k_default)
        eq_old = float(old_cal['k_Cuoi'].sum())
        q_model = float(row_res[model_for_base])
        base_x = q_model / eq_old if eq_old > 0 else 0
        new_exact = e['Ngày cắt điện cụ thể'] if pd.notna(e['Ngày cắt điện cụ thể']) else ''
        new_count = int(e['Số ngày cắt điện (fallback)']) if pd.notna(e['Số ngày cắt điện (fallback)']) else 0
        new_cal = build_month_calendar(sy, sm, holidays_map, holiday_dates,
                                       new_exact, new_count, new_k)
        eq_new = float(new_cal['k_Cuoi'].sum())
        final_total = base_x * eq_new
        a, b, c, d = st.columns(4)
        a.metric('Mô hình', model_for_base)
        b.metric('Tổng ban đầu', f'{q_model:,.0f} kWh')
        c.metric('Ngày tương đương', f'{eq_old:.3f} → {eq_new:.3f}')
        d.metric('Tổng sau điều chỉnh', f'{final_total:,.0f} kWh', f'{(final_total/q_model-1)*100:+.2f}%' if q_model else None)

        summary = pd.DataFrame([
            {'Loại ngày': CATEGORY_LABELS[key], 'Số ngày': int((new_cal['Category'] == key).sum()),
             'k loại ngày': 1.0 if key == 'Thuong' else new_k[{'T2': 'T2', 'T7': 'T7', 'CN': 'CN', 'Le': 'Le', 'Tet': 'Tet'}[key]],
             'Sản lượng/ngày trước cắt điện': base_x * (1.0 if key == 'Thuong' else new_k[{'T2': 'T2', 'T7': 'T7', 'CN': 'CN', 'Le': 'Le', 'Tet': 'Tet'}[key]])}
            for key in ['Thuong', 'T2', 'T7', 'CN', 'Le', 'Tet']
        ])
        st.dataframe(summary, hide_index=True, use_container_width=True,
                     column_config={'Sản lượng/ngày trước cắt điện': st.column_config.NumberColumn(format='%.0f')})

        with st.expander('🗓️ Xem tờ lịch trực quan & chi tiết từng ngày sau điều chỉnh', expanded=True):
            render_visual_month_calendar(new_cal, sy, sm)
            cal_show = new_cal[['Ngày', 'Category', 'Cat_Dien', 'k_Loai_Ngay', 'k_Cat_Dien', 'k_Cuoi', 'Nguon_Phan_Loai', 'Nguon_Cat_Dien']].copy()
            cal_show['Loại ngày'] = cal_show['Category'].map(CATEGORY_LABELS)
            st.dataframe(cal_show[['Ngày', 'Loại ngày', 'Cat_Dien', 'k_Loai_Ngay', 'k_Cat_Dien', 'k_Cuoi', 'Nguon_Phan_Loai', 'Nguon_Cat_Dien']],
                         hide_index=True, use_container_width=True)
        if st.button('✅ CHỐT THÁNG NÀY', type='primary', use_container_width=True):
            st.session_state.setdefault('finalized_results', {})
            st.session_state.finalized_results[selected] = {
                'Năm': sy, 'Tháng': sm, 'Mô hình chọn': model_for_base,
                'Tổng Ban Đầu': q_model, 'TỔNG SAU ĐIỀU CHỈNH': final_total,
                'Ngày tương đương ban đầu': eq_old, 'Ngày tương đương sau chỉnh': eq_new,
                'Số ngày cắt điện fallback': new_count,
                'Ngày cắt điện cụ thể': str(new_exact),
                'k_T2': new_k['T2'], 'k_T7': new_k['T7'], 'k_CN': new_k['CN'],
                'k_Lễ': new_k['Le'], 'k_Tết': new_k['Tet'], 'k_Cắt điện': new_k['CatDien'],
            }
            st.success(f'Đã lưu kết quả chốt {selected}.')
        final_df = finalized_dataframe()
        if not final_df.empty:
            st.write('**Các tháng đã chốt**')
            st.dataframe(final_df, hide_index=True, use_container_width=True)

# ----------------------------- AI & công cụ -----------------------------
with tabs[6]:
    st.subheader('🤖 Phân tích tin tức — chỉ tham khảo')
    text_data = st.text_area('Dán nội dung hoặc link bài báo', height=130,
                             placeholder='Ví dụ: Dự báo nắng nóng gay gắt tháng tới...')
    if st.button('AI PHÂN TÍCH', disabled=not selected_model, use_container_width=True):
        with st.spinner(f'Đang hỏi {provider}...'):
            val, log, reason = xu_ly_du_lieu_dinh_tinh(api_key, text_data, selected_model, provider)
        st.session_state.ai_suggestion_val = val
        st.session_state.ai_suggestion_reason = reason
        st.session_state.ai_log = log
    if 'ai_suggestion_val' in st.session_state:
        val = st.session_state.ai_suggestion_val
        st.metric('Ước lượng AI so cùng kỳ', f'{val:+.1f}%')
        st.info(st.session_state.ai_suggestion_reason)
        st.caption('Giá trị này không tự động cộng/trừ vào dự báo định lượng.')
    st.divider()
    with st.expander('🔬 Công cụ nâng cao: Find Seed', expanded=False):
        render_find_seed(df_train_main, df_input_main, holidays_map, k_default,
                         holiday_dates, outage_dates_map)

# ----------------------------- Tổng quan -----------------------------
with tabs[0]:
    if 'res_output' not in st.session_state:
        st.subheader('Tổng quan hệ thống')
        st.write('Tải dữ liệu, cấu hình lịch và chạy dự báo ở tab **🔮 Dự báo**. Dashboard sẽ tổng hợp kết quả tại đây.')
        st.markdown('''
        <div class="card"><b>Luồng làm việc đề xuất</b><br><br>
        1. Kiểm tra dữ liệu, đổ file Điện nhận ngày để tính hệ số k và kiểm tra lịch cụ thể → 2. Chạy dự báo → 3. Backtest MAE/MAPE/RMSE →
        4. Tinh chỉnh/chốt tháng → 5. Xuất Excel.
        </div>''', unsafe_allow_html=True)
    else:
        res = st.session_state.res_output
        default_model = 'XGBoost'
        first = res.iloc[0]
        selected_value = float(first[default_model])
        metrics = st.session_state.get('evaluation_metrics')
        best_name, best_mape = ('Chưa đánh giá', None)
        if isinstance(metrics, pd.DataFrame) and not metrics.empty:
            best_name = metrics.iloc[0]['Mô hình']
            best_mape = float(metrics.iloc[0]['MAPE (%)'])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Tháng dự báo đầu tiên', month_key(first['Năm'], first['Tháng']))
        m2.metric('XGBoost', f'{selected_value:,.0f} kWh')
        m3.metric('Mô hình backtest tốt nhất', best_name)
        m4.metric('MAPE tốt nhất', f'{best_mape:.2f}%' if best_mape is not None else '—')
        st.plotly_chart(plot_forecast(res), use_container_width=True, key='chart_overview_tab')
        c1, c2 = st.columns(2)
        with c1:
            st.write('**Cơ cấu lịch dự báo**')
            st.dataframe(res[['Năm', 'Tháng', 'Ngày Thường', 'T2', 'T7', 'CN', 'Lễ', 'Tết', 'Cắt điện', 'Ngày tương đương']],
                         hide_index=True, use_container_width=True)
        with c2:
            st.write('**Trạng thái dữ liệu lịch**')
            status = res[['Năm', 'Tháng', 'Chế độ lễ', 'Chế độ cắt điện']].copy()
            st.dataframe(status, hide_index=True, use_container_width=True)
            if (status['Chế độ lễ'] == 'Tổng hợp cũ').any() or (status['Chế độ cắt điện'] == 'Tổng hợp cũ').any():
                st.warning('Một số tháng vẫn dùng dữ liệu tổng hợp. Nhập ngày cụ thể nếu cần xử lý chính xác trường hợp trùng.')

# ----------------------------- Xuất báo cáo -----------------------------
with tabs[7]:
    st.subheader('📥 Xuất báo cáo Excel')
    if 'res_output' not in st.session_state or df_input_main is None:
        st.info('Chạy dự báo trước khi xuất báo cáo.')
    else:
        res = st.session_state.res_output
        metrics = st.session_state.get('evaluation_metrics', pd.DataFrame())
        detail = st.session_state.get('evaluation_detail', pd.DataFrame())
        final_df = finalized_dataframe()
        day_cols = ['Năm', 'Tháng', 'Số ngày', 'Ngày Thường', 'T2', 'T7', 'CN', 'Lễ', 'Tết', 'Cắt điện', 'Ngày tương đương']
        day_structure = res[[c for c in day_cols if c in res.columns]].copy()
        cal_detail = calendar_detail_for_dataframe(df_input_main, holidays_map, holiday_dates,
                                                   outage_dates_map, k_default)
        if not cal_detail.empty:
            note_map = {}
            for _, note_row in holiday_dates_editor.iterrows():
                parsed_note_date = pd.to_datetime(note_row.get('Ngày'), dayfirst=True, errors='coerce')
                if pd.notna(parsed_note_date):
                    note = note_row.get('Ghi chú', '')
                    note_map[parsed_note_date.normalize()] = '' if pd.isna(note) else str(note)
            cal_detail['Ghi chú'] = pd.to_datetime(cal_detail['Ngày']).dt.normalize().map(note_map).fillna('')
            export_cal_cols = ['Năm', 'Tháng', 'Ngày', 'Loại ngày', 'Ghi chú', 'Cat_Dien', 'k_Loai_Ngay', 'k_Cat_Dien', 'k_Cuoi', 'Nguon_Phan_Loai', 'Nguon_Cat_Dien']
            cal_detail = cal_detail[[c for c in export_cal_cols if c in cal_detail.columns]]
        params = {
            'Random Seed dự báo': int(seed_val), 'Seed backtest': 42,
            'k_T2': k_default['T2'], 'k_T7': k_default['T7'], 'k_CN': k_default['CN'],
            'k_Lễ': k_default['Le'], 'k_Tết': k_default['Tet'], 'k_Cắt điện': k_default['CatDien'],
            'Hệ số riêng tháng': str(k_default.get('__monthly__', ())),
            'Nguyên tắc cắt điện': 'k cuối = k loại ngày × k cắt điện',
            'AI tin tức tác động dự báo': 'Không',
        }
        excel_bytes = build_excel_report(res, metrics, detail, day_structure,
                                         final_df, params, cal_detail)
        st.download_button('⬇️ TẢI BÁO CÁO EXCEL', data=excel_bytes,
                           file_name='bao_cao_du_bao_phu_tai_tay_ninh.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           use_container_width=True, type='primary')
        st.caption('File gồm dự báo mô hình, đánh giá, sai số theo tháng, cơ cấu ngày, kết quả đã chốt, tham số và chi tiết lịch. API Key không được xuất.')
