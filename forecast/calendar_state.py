"""Portable backup of calendar classifications, notes, and confirmed outages."""
from __future__ import annotations

import calendar
import json
import math
import pandas as pd


def dump_calendar(holiday_rows: pd.DataFrame, outages: dict, confirmed_months,
                  common_k=None, monthly_k=None) -> bytes:
    rows = []
    for _, row in holiday_rows.iterrows():
        day = pd.to_datetime(row.get('Ngày'), errors='coerce', dayfirst=True)
        kind = str(row.get('Loại', '')).strip()
        if pd.notna(day) and kind in ('Lễ', 'Tết', 'Ghi chú'):
            note = row.get('Ghi chú', '')
            rows.append({'Ngày': day.strftime('%d/%m/%Y'), 'Loại': kind,
                         'Ghi chú': '' if pd.isna(note) else str(note)})
    payload = {'format': 'lich-ngay-nghi-v1', 'ngay': rows,
               'cat_dien': [{'nam': int(y), 'thang': int(m), 'ngay': str(v)}
                            for (y, m), v in sorted(outages.items())],
               'thang_da_xac_nhan': [[int(y), int(m)] for y, m in sorted(confirmed_months)],
               'he_so_chung': {k: float(v) for k, v in (common_k or {}).items() if k != '__monthly__'},
               'he_so_thang': [{'nam': int(y), 'thang': int(m), 'he_so': dict(values)}
                                for (y, m), values in sorted((monthly_k or {}).items())]}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')


def load_calendar(content: bytes):
    if len(content) > 2_000_000:
        raise ValueError('Tệp lịch lớn hơn 2 MB.')
    try:
        obj = json.loads(content.decode('utf-8-sig'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('Tệp lịch JSON không hợp lệ.') from exc
    if not isinstance(obj, dict) or obj.get('format') != 'lich-ngay-nghi-v1':
        raise ValueError('Tệp không đúng định dạng sao lưu lịch.')
    rows, outages, months = obj.get('ngay', []), obj.get('cat_dien', []), obj.get('thang_da_xac_nhan', [])
    monthly = obj.get('he_so_thang', [])
    if not all(isinstance(x, list) for x in (rows, outages, months, monthly)) or max(map(len, (rows, outages, months, monthly))) > 10000:
        raise ValueError('Danh sách trong tệp lịch không hợp lệ.')
    checked = []
    for row in rows:
        if not isinstance(row, dict) or row.get('Loại') not in ('Lễ', 'Tết', 'Ghi chú'):
            raise ValueError('Loại ngày trong tệp lịch không hợp lệ.')
        d = pd.to_datetime(row.get('Ngày'), errors='coerce', dayfirst=True)
        if pd.isna(d):
            raise ValueError('Tệp lịch có ngày không hợp lệ.')
        checked.append({'Ngày': d.strftime('%d/%m/%Y'), 'Loại': row['Loại'],
                        'Ghi chú': str(row.get('Ghi chú', ''))[:500]})
    out = {}
    for row in outages:
        if not isinstance(row, dict):
            raise ValueError('Danh sách cắt điện không hợp lệ.')
        y, m = int(row['nam']), int(row['thang'])
        calendar.monthrange(y, m)
        out[(y, m)] = str(row.get('ngay', ''))[:1000]
    confirmed = set()
    for row in months:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError('Tháng đã xác nhận không hợp lệ.')
        y, m = int(row[0]), int(row[1])
        calendar.monthrange(y, m)
        confirmed.add((y, m))
    valid_keys = {'T2', 'T7', 'CN', 'Le', 'Tet', 'CatDien'}
    def check_values(values):
        if not isinstance(values, dict) or not set(values).issubset(valid_keys):
            raise ValueError('Mã hệ số trong tệp lịch không hợp lệ.')
        result = {k: float(v) for k, v in values.items()}
        if any(not math.isfinite(v) or v < 0 for v in result.values()):
            raise ValueError('Hệ số trong tệp lịch không hợp lệ.')
        return result
    common = check_values(obj.get('he_so_chung', {}))
    per_month = {}
    for row in monthly:
        if not isinstance(row, dict):
            raise ValueError('Hệ số theo tháng không hợp lệ.')
        y, m = int(row['nam']), int(row['thang'])
        calendar.monthrange(y, m)
        per_month[(y, m)] = check_values(row['he_so'])
    return pd.DataFrame(checked, columns=['Ngày', 'Loại', 'Ghi chú']), out, confirmed, common, per_month
