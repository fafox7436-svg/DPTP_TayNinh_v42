"""Import daily boundary-meter reports and estimate transparent day coefficients.

Received minus delivered is a boundary-flow proxy, not electricity sales.  No
assumption is made about the territorial membership of a meter.  In particular,
missing observations and overlapping uploads never silently become zeros.
"""
from __future__ import annotations

import calendar
import hashlib
import io
import json
import re
import unicodedata
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd
from openpyxl import load_workbook


def _plain(value: object) -> str:
    s = str(value or '').strip().lower().replace('đ', 'd')
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def infer_period(filename: str) -> tuple[int | None, int | None]:
    """Only infer from explicit month/year in the filename; never from a load shape."""
    name = _plain(filename)
    match = re.search(r'(?:thang|t)?\s*(0?[1-9]|1[0-2])[.\-_/\s]+(20\d{2})(?!\d)', name)
    if match:
        return int(match.group(2)), int(match.group(1))
    match = re.search(r'(20\d{2})[.\-_/\s]+(0?[1-9]|1[0-2])(?=\D|$)', name)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r'(?:thang|t)\s*(0?[1-9]|1[0-2])\s*(?:nam|year)\s*(20\d{2})(?!\d)', name)
    if match:
        return int(match.group(2)), int(match.group(1))
    return None, None


def _day(value: object) -> int | None:
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    s = str(value).strip()
    if re.fullmatch(r'\d{1,2}(?:\.0)?', s):
        n = int(float(s))
        if 1 <= n <= 31:
            return n
    return None


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return np.nan
    result = float(value)
    return result if np.isfinite(result) else np.nan


def _read_channel(sheet) -> dict:
    rows = list(sheet.iter_rows(values_only=True))
    header_at = None
    for i, row in enumerate(rows[:12]):
        words = [_plain(c) for c in row]
        if any('ma diem do' in c for c in words) and sum(_day(c) is not None for c in row) >= 15:
            header_at = i
            break
    if header_at is None:
        raise ValueError(f"Sheet '{sheet.title}' không có cột mã điểm đo và cột ngày 1–31.")
    header = rows[header_at]
    code_col = next(i for i, v in enumerate(header) if 'ma diem do' in _plain(v))
    name_col = next((i for i, v in enumerate(header) if 'ten diem do' in _plain(v)), None)
    day_cols = {}
    for i, value in enumerate(header):
        d = _day(value)
        if d is not None:
            if d in day_cols:
                raise ValueError(f"Sheet '{sheet.title}' trùng cột ngày {d}.")
            day_cols[d] = i
    records, seen = [], set()
    for row in rows[header_at + 1:]:
        code = row[code_col] if code_col < len(row) else None
        if code is None or not str(code).strip():
            continue
        code = str(code).strip()
        if code in seen:
            raise ValueError(f"Sheet '{sheet.title}' trùng mã điểm đo {code}.")
        seen.add(code)
        name = str(row[name_col] or '').strip() if name_col is not None and name_col < len(row) else ''
        vals = {d: _number(row[i]) if i < len(row) else np.nan for d, i in day_cols.items()}
        records.append({'code': code, 'name': name, 'values': vals})
    if not records:
        raise ValueError(f"Sheet '{sheet.title}' không có dòng điểm đo hợp lệ.")
    # Reconcile the final unlabelled daily total, but never double-count it.
    total = {}
    for row in reversed(rows[header_at + 1:]):
        code = row[code_col] if code_col < len(row) else None
        vals = {d: _number(row[i]) if i < len(row) else np.nan for d, i in day_cols.items()}
        if (code is None or not str(code).strip()) and sum(np.isfinite(v) for v in vals.values()) >= 15:
            total = vals
            break
    return {'sheet': sheet.title, 'meters': records, 'days': sorted(day_cols), 'reported_total': total}


def read_daily_workbook(content: bytes, filename: str) -> dict:
    if not content:
        raise ValueError('Tệp rỗng.')
    try:
        book = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    except Exception as exc:
        raise ValueError(f'Không đọc được Excel .xlsx: {exc}') from exc
    matched = {}
    for sheet in book:
        n = _plain(sheet.title)
        is_in = any(x in n for x in ('chieu nhan', 'dien nhan', 'san luong nhan')) or n.strip() == 'nhan'
        is_out = any(x in n for x in ('chieu giao', 'dien giao', 'san luong giao')) or n.strip() == 'giao'
        channel = 'nhan' if is_in and not is_out else ('giao' if is_out and not is_in else None)
        if channel:
            if channel in matched:
                raise ValueError(f'Tệp có nhiều sheet {channel}; cần chọn lại tệp.')
            matched[channel] = _read_channel(sheet)
    if not matched and len(book.sheetnames) == 1:
        n = _plain(filename)
        hints = [c for c in ['nhan', 'giao'] if c in n]
        if len(hints) == 1:
            matched[hints[0]] = _read_channel(book[book.sheetnames[0]])
    book.close()
    if not matched:
        raise ValueError("Không thấy sheet 'Chiều nhận' hoặc 'Chiều giao'.")
    year, month = infer_period(filename)
    return {'filename': filename, 'year': year, 'month': month, 'channels': matched,
            'sha256': hashlib.sha256(content).hexdigest()}


def combine_channels(items: list[dict]) -> dict:
    """Combine separate receive/deliver files, rejecting any repeated period/channel."""
    result = {}
    for item in items:
        try:
            y, m = int(item['year']), int(item['month'])
            calendar.monthrange(y, m)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ValueError(f"Chưa xác nhận năm/tháng cho {item.get('filename', '?')}.") from exc
        for direction, channel in item['channels'].items():
            key = (y, m, direction)
            if key in result:
                raise ValueError(f"Trùng dữ liệu {direction} tháng {m:02d}/{y}: "
                                 f"{result[key]['filename']} và {item['filename']}. "
                                 'Hãy giữ một tệp cho mỗi chiều/tháng.')
            result[key] = {'data': channel, 'filename': item['filename'], 'sha256': item['sha256']}
    return result


def _channel_signature(channel: dict) -> str:
    """Identify identical channel data even when the workbook names differ."""
    def number(value):
        return float(value) if pd.notna(value) and np.isfinite(value) else None
    meters = []
    for meter in sorted(channel['meters'], key=lambda x: x['code']):
        values = [[int(d), number(v)] for d, v in sorted(meter['values'].items())]
        meters.append([meter['code'], values])
    reported = [[int(d), number(v)] for d, v in sorted(channel['reported_total'].items())]
    canonical = json.dumps([channel['days'], meters, reported], separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def group_channel_candidates(items: list[dict]) -> tuple[dict, list[dict]]:
    """Group uploads by month/direction; isolate bad periods instead of stopping all."""
    groups, problems = {}, []
    for item in items:
        try:
            y, m = int(item['year']), int(item['month'])
            if not 2000 <= y <= 2100:
                raise ValueError('Năm nằm ngoài 2000–2100')
            calendar.monthrange(y, m)
        except (ValueError, TypeError, KeyError, OverflowError):
            problems.append({'Tệp': item.get('filename', '?'),
                             'Lý do': 'Chưa xác nhận tháng và năm hợp lệ'})
            continue
        for direction, channel in item['channels'].items():
            key = (y, m, direction)
            signature = _channel_signature(channel)
            versions = groups.setdefault(key, {})
            if signature not in versions:
                versions[signature] = {'data': channel, 'filename': item['filename'],
                                       'sha256': item['sha256'], 'aliases': [item['filename']]}
            else:
                versions[signature]['aliases'].append(item['filename'])
    return groups, problems


def resolve_channel_groups(groups: dict, decisions: dict | None = None) -> tuple[dict, list[dict]]:
    """Deduplicate identical readings; do not silently select conflicting versions."""
    decisions = decisions or {}
    channels, conflicts = {}, []
    for key, versions in sorted(groups.items()):
        if len(versions) == 1:
            channels[key] = next(iter(versions.values()))
        elif decisions.get(key) in versions:
            channels[key] = versions[decisions[key]]
        else:
            conflicts.append({'Năm': key[0], 'Tháng': key[1],
                              'Chiều': 'Nhận' if key[2] == 'nhan' else 'Giao',
                              'Số phiên bản khác nhau': len(versions),
                              'Trạng thái': 'Chưa chọn; bỏ qua chiều này'})
    return channels, conflicts


def common_meter_ids(channels: dict, direction: str) -> set[str]:
    """Codes present in every month being analyzed for this direction."""
    months = sorted({(y, m) for y, m, _ in channels})
    sets = []
    for y, m in months:
        channel = channels.get((y, m, direction))
        if channel is None:
            return set()
        sets.append({r['code'] for r in channel['data']['meters']})
    return set.intersection(*sets) if sets else set()


def meter_choices(channels: dict, direction: str) -> dict[str, str]:
    out = {}
    for (_, _, d), channel in channels.items():
        if d == direction:
            for meter in channel['data']['meters']:
                out[meter['code']] = meter['name']
    return out


def describe_scope(channels: dict) -> pd.DataFrame:
    rows = []
    prev = None
    for ym in sorted({(y, m) for y, m, _ in channels}):
        incoming = channels.get((*ym, 'nhan'))
        outgoing = channels.get((*ym, 'giao'))
        ids = {r['code'] for r in incoming['data']['meters']} if incoming else set()
        intersection = len(ids & prev) if prev is not None else None
        rows.append({'Năm': ym[0], 'Tháng': ym[1],
                     'Điểm nhận': len(ids),
                     'Điểm giao': len(outgoing['data']['meters']) if outgoing else 0,
                     'Điểm nhận chung kỳ trước': intersection,
                     'Điểm nhận mới': len(ids - prev) if prev is not None else None,
                     'Điểm nhận không còn': len(prev - ids) if prev is not None else None})
        if ids:
            prev = ids
    return pd.DataFrame(rows)


def meter_catalog(channels: dict) -> pd.DataFrame:
    rows = []
    for (y, m, direction), channel in sorted(channels.items()):
        for meter in channel['data']['meters']:
            rows.append({'Năm': y, 'Tháng': m,
                         'Chiều': 'Nhận' if direction == 'nhan' else 'Giao',
                         'Mã điểm đo': meter['code'], 'Tên điểm đo': meter['name'],
                         'Tên tệp': channel['filename'], 'SHA-256 tệp': channel['sha256']})
    return pd.DataFrame(rows)


def _aggregate(channel: dict | None, day: int, selection: set[str] | None) -> tuple[float, int, int, str]:
    if channel is None:
        return np.nan, 0, 0, 'Thiếu sheet chiều này'
    obj = channel['data']
    meters = [r for r in obj['meters'] if selection is None or r['code'] in selection]
    if not meters:
        return np.nan, 0, 0, 'Không có điểm đo đã chọn'
    # Explicitly flag selected IDs that do not exist in this month.
    absent = len(selection - {r['code'] for r in meters}) if selection is not None else 0
    values = np.array([m['values'].get(day, np.nan) for m in meters], dtype=float)
    missing = int(np.isnan(values).sum()) + absent
    negative = int((values < 0).sum())
    if missing or negative:
        return np.nan, len(meters), missing, f'Thiếu {missing} ô/điểm; {negative} giá trị âm'
    value = float(values.sum())
    status = ''
    reported = obj['reported_total'].get(day, np.nan)
    if selection is None and np.isfinite(reported) and not np.isclose(value, reported, rtol=1e-6, atol=1.0):
        status = 'Tổng chi tiết khác dòng tổng trong file'
    return value, len(meters), 0, status


def build_daily_frame(channels: dict, as_of: date, holidays: pd.DataFrame,
                      outages: set[date] | None = None, mode: str = 'net',
                      selected_in: set[str] | None = None,
                      selected_out: set[str] | None = None) -> pd.DataFrame:
    """Only days strictly before as_of; actual historical data vintages are unknown."""
    if mode not in {'net', 'nhan', 'giao'}:
        raise ValueError('Công thức không được hỗ trợ.')
    lookup = {}
    if holidays is not None:
        for _, r in holidays.iterrows():
            timestamp = pd.to_datetime(r.get('Ngày'), errors='coerce', dayfirst=True)
            if pd.isna(timestamp):
                continue
            label = r.get('Category', r.get('Loại', ''))
            p = _plain(label)
            if p in ('tet', 'le'):
                lookup[timestamp.date()] = 'Tet' if p == 'tet' else 'Le'
    outages = outages or set()
    records = []
    for y, m in sorted({(y, m) for y, m, _ in channels}):
        ni, gi = channels.get((y, m, 'nhan')), channels.get((y, m, 'giao'))
        for d in range(1, calendar.monthrange(y, m)[1] + 1):
            when = date(y, m, d)
            if when >= as_of:
                continue
            rv, rn, rmiss, rw = _aggregate(ni, d, selected_in)
            gv, gn, gmiss, gw = _aggregate(gi, d, selected_out)
            if mode == 'net':
                q, warnings = rv - gv, [rw, gw]
            elif mode == 'nhan':
                q, warnings = rv, [rw]
            else:
                q, warnings = gv, [gw]
            warnings = [w for w in warnings if w]
            valid = bool(np.isfinite(q) and q > 0 and not warnings)
            if np.isfinite(q) and q <= 0:
                warnings.append('Giá trị lựa chọn không dương')
            weekday = when.weekday()
            kind = lookup.get(when, 'T2' if weekday == 0 else 'T7' if weekday == 5 else 'CN' if weekday == 6 else 'Thuong')
            records.append({'Ngày': pd.Timestamp(when), 'Năm': y, 'Tháng': m,
                            'Loại': kind, 'Cắt điện xác nhận': when in outages,
                            'Điện nhận': rv, 'Điện giao': gv, 'Giá trị phân tích': q,
                            'Điểm nhận': rn, 'Điểm giao': gn,
                            'Thiếu nhận': rmiss, 'Thiếu giao': gmiss,
                            'Hợp lệ': valid, 'Cảnh báo': '; '.join(warnings),
                            'Nguồn nhận': ni['filename'] if ni else '',
                            'Nguồn giao': gi['filename'] if gi else ''})
    return pd.DataFrame(records)


def estimate_k(days: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Day/baseline ratios; baseline is comparable workdays in week, else month.

    Confirmed outages are excluded from day-type calibration.  Suspected outliers
    are annotated but not removed.  No interval is claimed from sparse events.
    """
    if days.empty:
        return days.copy(), pd.DataFrame()
    df = days.copy().sort_values('Ngày').reset_index(drop=True)
    df['Tuần'] = df['Ngày'].map(lambda d: d.date().isocalendar()[:2])
    df['Cơ sở kWh'] = np.nan
    df['Nguồn cơ sở'] = ''
    for (y, m), indexes in df.groupby(['Năm', 'Tháng']).groups.items():
        group = df.loc[indexes]
        base = group[(group['Loại'] == 'Thuong') & group['Hợp lệ'] & ~group['Cắt điện xác nhận']]
        month_median = float(base['Giá trị phân tích'].median()) if len(base) >= 3 else np.nan
        for index, row in group.iterrows():
            weekly = base[base['Tuần'] == row['Tuần']]['Giá trị phân tích']
            if len(weekly) >= 2:
                baseline, source = float(weekly.median()), 'Trung vị T3–T6 cùng tuần'
            else:
                baseline, source = month_median, 'Trung vị T3–T6 cùng tháng' if np.isfinite(month_median) else 'Thiếu ngày cơ sở'
            df.at[index, 'Cơ sở kWh'] = baseline
            df.at[index, 'Nguồn cơ sở'] = source
    df['Tỷ số ngày'] = np.where(df['Hợp lệ'] & (df['Cơ sở kWh'] > 0),
                                df['Giá trị phân tích'] / df['Cơ sở kWh'], np.nan)
    df['Nghi ngờ bất thường'] = False
    clean = df[df['Hợp lệ'] & ~df['Cắt điện xác nhận'] & df['Tỷ số ngày'].notna()]
    for kind, group in clean.groupby('Loại'):
        vals = group['Tỷ số ngày']
        if len(vals) >= 5:
            med = float(vals.median())
            mad = float((vals - med).abs().median())
            if mad > 0:
                df.loc[group.index, 'Nghi ngờ bất thường'] = (vals - med).abs() > 3 * 1.4826 * mad
    labels = {'T2': 'Thứ Hai', 'T7': 'Thứ Bảy', 'CN': 'Chủ nhật', 'Le': 'Lễ', 'Tet': 'Tết'}
    results = []
    for code, name in labels.items():
        sample = clean[clean['Loại'] == code]
        values = sample['Tỷ số ngày']
        months = sample[['Năm', 'Tháng']].drop_duplicates().shape[0]
        results.append({'Mã': code, 'Loại ngày': name, 'Số ngày': len(sample),
                        'Số tháng': months, 'Số năm': sample['Năm'].nunique(),
                        'k trung vị': float(values.median()) if len(values) else np.nan,
                        'k trung bình': float(values.mean()) if len(values) else np.nan,
                        'Q25': float(values.quantile(.25)) if len(values) else np.nan,
                        'Q75': float(values.quantile(.75)) if len(values) else np.nan,
                        'Số nghi ngờ': int(df.loc[sample.index, 'Nghi ngờ bất thường'].sum())})
    # k outage compares confirmed outage ratios to controls of the same type.
    # It is an association, not a causal estimate; no automatic outage inference.
    control_medians = clean.groupby('Loại')['Tỷ số ngày'].median()
    outage = df[df['Cắt điện xác nhận'] & df['Tỷ số ngày'].notna()].copy()
    ratios = []
    for _, row in outage.iterrows():
        control = clean[clean['Loại'] == row['Loại']]
        med = control_medians.get(row['Loại'], np.nan)
        if len(control) >= 2 and np.isfinite(med) and med > 0:
            ratios.append((row, float(row['Tỷ số ngày'] / med)))
    vals = pd.Series([r[1] for r in ratios], dtype=float)
    results.append({'Mã': 'CatDien', 'Loại ngày': 'Cắt điện xác nhận', 'Số ngày': len(vals),
                    'Số tháng': len({(r[0]['Năm'], r[0]['Tháng']) for r in ratios}),
                    'Số năm': len({r[0]['Năm'] for r in ratios}),
                    'k trung vị': float(vals.median()) if len(vals) else np.nan,
                    'k trung bình': float(vals.mean()) if len(vals) else np.nan,
                    'Q25': float(vals.quantile(.25)) if len(vals) else np.nan,
                    'Q75': float(vals.quantile(.75)) if len(vals) else np.nan,
                    'Số nghi ngờ': 0})
    return df, pd.DataFrame(results)


def _event_key(note: object, kind: str) -> str | None:
    """Conservative matches for labelled holidays; never infer an unlabeled date."""
    if pd.isna(note):
        return None
    text = re.sub(r'\s+', ' ', _plain(note)).strip()
    if not text:
        return None
    if kind == 'Tet':
        match = re.search(r'\bmung\s*(\d{1,2})\b', text)
        if match:
            return 'mung-' + match.group(1)
        match = re.search(r'\b(2[0-9]|30)\s*tet\b', text)
        if match:
            return 'truoc-tet-' + match.group(1)
        return None
    if kind == 'Le':
        known = [('duong lich', 'tet-duong-lich'), ('gio to', 'gio-to'),
                 ('chien thang', '30-4'), ('30/4', '30-4'),
                 ('quoc te lao dong', '1-5'), ('1/5', '1-5'),
                 ('quoc khanh', 'quoc-khanh'), ('2/9', 'quoc-khanh')]
        for token, key in known:
            if token in text:
                return key
    return None


def recommend_target_k(detail: pd.DataFrame, target_calendar: pd.DataFrame,
                       holiday_notes: pd.DataFrame | None = None,
                       overall: pd.DataFrame | None = None,
                       min_days: int = 6, min_years: int = 2) -> pd.DataFrame:
    """Suggest coefficients for the target calendar from prior eligible observations.

    Priority and thresholds are transparent heuristics, not evidence of optimality.
    Target holidays use labelled comparable holiday stages where available.
    """
    labels = {'T2': 'Thứ Hai', 'T7': 'Thứ Bảy', 'CN': 'Chủ nhật',
              'Le': 'Lễ', 'Tet': 'Tết', 'CatDien': 'Cắt điện'}
    if target_calendar.empty:
        return pd.DataFrame()
    target_dates = pd.to_datetime(target_calendar['Ngày'])
    target_month = int(target_dates.iloc[0].month)
    note_lookup = {}
    if holiday_notes is not None and not holiday_notes.empty:
        for _, r in holiday_notes.iterrows():
            d = pd.to_datetime(r.get('Ngày'), errors='coerce', dayfirst=True)
            if pd.notna(d):
                note_lookup[d.date()] = r.get('Ghi chú', '')
    hist = detail[detail['Hợp lệ'] & ~detail['Cắt điện xác nhận'] & detail['Tỷ số ngày'].notna()].copy()
    hist['Sự kiện'] = [
        _event_key(note_lookup.get(pd.Timestamp(day).date()), kind)
        for day, kind in zip(hist['Ngày'], hist['Loại'])
    ]
    data_rows = []
    for kind, label in labels.items():
        target_group = (target_calendar[target_calendar['Cat_Dien']]
                        if kind == 'CatDien' else target_calendar[target_calendar['Category'] == kind])
        n_target = len(target_group)
        result = {'Mã': kind, 'Loại ngày': label, 'Số ngày dự báo': n_target,
                  'k đề xuất': np.nan, 'Số ngày mẫu': 0, 'Số tháng mẫu': 0,
                  'Số năm mẫu': 0, 'Nguồn mẫu': 'Không xuất hiện trong lịch tháng dự báo',
                  'Q25': np.nan, 'Q75': np.nan, 'Lưu ý': ''}
        if n_target == 0:
            data_rows.append(result)
            continue
        if kind == 'CatDien':
            matching = overall[overall['Mã'] == 'CatDien'] if overall is not None and not overall.empty else pd.DataFrame()
            if not matching.empty and pd.notna(matching.iloc[0]['k trung vị']):
                row = matching.iloc[0]
                result.update({'k đề xuất': float(row['k trung vị']), 'Số ngày mẫu': int(row['Số ngày']),
                               'Số tháng mẫu': int(row['Số tháng']), 'Số năm mẫu': int(row['Số năm']),
                               'Q25': row['Q25'], 'Q75': row['Q75'],
                               'Nguồn mẫu': 'Cắt điện đã xác nhận; so với ngày cùng loại'})
            else:
                result['Nguồn mẫu'] = 'Chưa đủ ngày cắt điện và ngày đối chứng'
            if result['Số ngày mẫu'] < min_days or result['Số năm mẫu'] < min_years:
                result['Lưu ý'] = 'Ít dữ liệu; cần kiểm tra cắt điện và phạm vi ảnh hưởng.'
            data_rows.append(result)
            continue
        candidates = hist[hist['Loại'] == kind].copy()
        if candidates.empty:
            result['Nguồn mẫu'] = 'Không có ngày cùng loại hợp lệ trước mốc chốt'
            result['Lưu ý'] = 'Chưa đề xuất hệ số.'
            data_rows.append(result)
            continue
        estimate = None
        if kind in ('Tet', 'Le'):
            stage_keys = [_event_key(note_lookup.get(pd.Timestamp(d).date()), kind)
                          for d in target_group['Ngày']]
            keys = set(stage_keys)
            if None not in keys and keys and keys.issubset(set(candidates['Sự kiện'].dropna())):
                chosen = candidates[candidates['Sự kiện'].isin(keys)]
                by_event = chosen.groupby('Sự kiện')['Tỷ số ngày'].median()
                estimate = float(np.mean([by_event[k] for k in stage_keys]))
                basis = 'Cùng sự kiện/ngày Tết; trung vị theo từng nhãn'
            else:
                chosen = candidates
                basis = 'Gộp các ngày cùng loại Lễ/Tết; thiếu nhãn tương ứng'
                result['Lưu ý'] = 'Các kỳ nghỉ có thể khác nhau; cần đối chiếu ghi chú lịch.'
        else:
            same = candidates[candidates['Tháng'] == target_month]
            delta = (candidates['Tháng'] - target_month).abs()
            nearby = candidates[np.minimum(delta, 12 - delta) <= 1]
            if len(same) >= min_days and same['Năm'].nunique() >= min_years:
                chosen, basis = same, 'Cùng tháng dương lịch các năm trước'
            elif len(nearby) >= min_days and nearby[['Năm', 'Tháng']].drop_duplicates().shape[0] >= 2:
                chosen, basis = nearby, 'Tháng dự báo và tháng lân cận (±1)'
                result['Lưu ý'] = 'Cùng tháng chưa đủ mẫu; đã mở rộng tháng lân cận.'
            else:
                chosen, basis = candidates, 'Toàn bộ tháng có dữ liệu trước mốc chốt'
                result['Lưu ý'] = 'Chưa đủ cùng tháng/lân cận; dùng mẫu rộng để tham khảo.'
        values = chosen['Tỷ số ngày']
        if estimate is None:
            estimate = float(values.median())
        n_years = int(chosen['Năm'].nunique())
        result.update({'k đề xuất': estimate, 'Số ngày mẫu': len(chosen),
                       'Số tháng mẫu': chosen[['Năm', 'Tháng']].drop_duplicates().shape[0],
                       'Số năm mẫu': n_years, 'Nguồn mẫu': basis,
                       'Q25': float(values.quantile(.25)), 'Q75': float(values.quantile(.75))})
        if len(chosen) < min_days or n_years < min_years:
            result['Lưu ý'] = (result['Lưu ý'] + ' Mẫu ít hoặc chỉ một năm; độ ổn định chưa rõ.').strip()
        data_rows.append(result)
    return pd.DataFrame(data_rows)
