"""Đọc Excel, chuẩn hóa cột và tạo đặc trưng lịch."""
from __future__ import annotations

import calendar
import io
import pandas as pd

from .calendar_utils import (build_month_calendar, outage_dates_for_month,
                             summarize_month_calendar)


def chuan_hoa_ten_cot(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    col_map = {
        "month": "Tháng", "thang": "Tháng", "tháng": "Tháng",
        "year": "Năm", "nam": "Năm", "năm": "Năm",
        "tổng thương phẩm": "Tổng thương phẩm", "tong thuong pham": "Tổng thương phẩm",
        "sản lượng": "Tổng thương phẩm",
        "nhiệt độ tb": "Nhiệt độ TB", "nhiet do tb": "Nhiệt độ TB",
        "độ ẩm": "Độ ẩm", "do am": "Độ ẩm",
        "số ngày": "Số ngày", "cắt điện": "Số ngày cắt điện",
        "cat dien": "Số ngày cắt điện", "số ngày cắt điện": "Số ngày cắt điện",
        "ngày cắt điện cụ thể": "Ngày cắt điện cụ thể",
        "ngay cat dien cu the": "Ngày cắt điện cụ thể",
    }
    return df.rename(columns={c: col_map[c.lower()] for c in df.columns if c.lower() in col_map})


def ultra_scan_read_excel(uploaded_file):
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, bytes):
        content = uploaded_file
    elif isinstance(uploaded_file, str) or hasattr(uploaded_file, '__fspath__'):
        with open(uploaded_file, 'rb') as f:
            content = f.read()
    else:
        content = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file.read()
    book = pd.ExcelFile(io.BytesIO(content))
    for sheet_name in book.sheet_names:
        preview = pd.read_excel(book, sheet_name=sheet_name, header=None, nrows=10)
        for i, row in preview.iterrows():
            text = str(row.values).lower()
            if 'tháng' in text and 'năm' in text:
                df = pd.read_excel(book, sheet_name=sheet_name, header=i)
                return chuan_hoa_ten_cot(df)
    return chuan_hoa_ten_cot(pd.read_excel(io.BytesIO(content), header=0))


def kiem_tra_chat_luong(df: pd.DataFrame, ten_file: str, la_train: bool = False) -> list[str]:
    errors = []
    required = ['Tháng', 'Năm', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm']
    if la_train:
        required.append('Tổng thương phẩm')
    for col in required:
        if col not in df.columns:
            errors.append(f"File {ten_file} thiếu cột '{col}'.")
    if errors:
        return errors
    if df.empty:
        return [f'File {ten_file} không có hàng dữ liệu.']
    for col in ['Năm', 'Tháng', 'Số ngày', 'Nhiệt độ TB', 'Độ ẩm',
                'Số ngày cắt điện', 'Tổng thương phẩm']:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors='coerce')
            if numeric.isna().any():
                errors.append(f"File {ten_file}: cột '{col}' có ô trống/không phải số.")
            if col in ['Nhiệt độ TB', 'Độ ẩm', 'Số ngày'] and (numeric == 0).any():
                errors.append(f"File {ten_file}: cột '{col}' bằng 0.")
    if errors:
        return errors
    for _, row in df.iterrows():
        try:
            year, month = int(row['Năm']), int(row['Tháng'])
            n = calendar.monthrange(year, month)[1]
            if int(row['Số ngày']) != n:
                errors.append(f"File {ten_file}: {month}/{year} có số ngày Excel "
                              f"{int(row['Số ngày'])}, lịch là {n}.")
            if 'Số ngày cắt điện' in df.columns and not (0 <= row['Số ngày cắt điện'] <= n):
                errors.append(f"File {ten_file}: số ngày cắt điện {month}/{year} không hợp lệ.")
        except (TypeError, ValueError, OverflowError):
            errors.append(f"File {ten_file}: Năm/Tháng không hợp lệ.")
    if la_train and (pd.to_numeric(df['Tổng thương phẩm'], errors='coerce') < 0).any():
        errors.append(f"File {ten_file}: Tổng thương phẩm âm.")
    if df.duplicated(subset=['Năm', 'Tháng']).any():
        errors.append(f"File {ten_file}: tháng/năm bị trùng.")
    return errors


def tao_dac_trung(df: pd.DataFrame, holidays_map: dict,
                  holiday_dates: pd.DataFrame | None = None,
                  outage_dates_map: dict | None = None,
                  k_dict: dict | None = None) -> pd.DataFrame:
    """Tạo season features + cơ cấu ngày; nếu có k_dict thì tính Ngày_Tương_Đương."""
    df = df.copy()
    df['Mua_Nong'] = df['Tháng'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    df['Mua_Mua'] = df['Tháng'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10, 11] else 0)
    if 'Số ngày cắt điện' not in df.columns:
        df['Số ngày cắt điện'] = 0

    rows = []
    for _, row in df.iterrows():
        y, m = int(row['Năm']), int(row['Tháng'])
        exact_out = outage_dates_for_month(row, outage_dates_map)
        outage_count = int(row.get('Số ngày cắt điện', 0) or 0)
        cal = build_month_calendar(
            y, m, holidays_map, holiday_dates=holiday_dates,
            outage_dates=exact_out, outage_count=outage_count, k_dict=k_dict,
        )
        summary = summarize_month_calendar(cal)
        summary['Che_Do_Le'] = cal.attrs.get('holiday_mode', 'legacy')
        summary['Che_Do_Cat_Dien'] = cal.attrs.get('outage_mode', 'legacy')
        rows.append(summary)

    summary_df = pd.DataFrame(rows, index=df.index)
    for col in summary_df.columns:
        df[col] = summary_df[col]
    df['Bien_Ngoai_Sinh'] = 0
    return df
