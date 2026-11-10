"""Streamlit interface for the auditable daily-coefficient analysis."""
from __future__ import annotations

import calendar
import datetime as dt
import pandas as pd
import streamlit as st

from .calendar_utils import parse_date_list, coefficients_for_month
from .daily_k import (read_daily_workbook, combine_channels, describe_scope, meter_catalog,
                      meter_choices, build_daily_frame, estimate_k)


@st.cache_data(show_spinner=False)
def _read_cached(content, filename):
    return read_daily_workbook(content, filename)


def _apply_month(values, year, month):
    # This callback runs before the next full script render.
    overrides = dict(st.session_state.get('monthly_k_overrides', {}))
    current = dict(overrides.get((year, month), {}))
    current.update({code: float(round(value, 4)) for code, value in values.items()})
    overrides[(year, month)] = current
    st.session_state['monthly_k_overrides'] = overrides
    st.session_state['applied_k_notice'] = f"{month:02d}/{year}: " + ', '.join(values)


def _outage_dates(mapping):
    result = set()
    for (y, m), text in (mapping or {}).items():
        for d in parse_date_list(text, year=int(y), month=int(m)):
            result.add(d.date())
    return result


def render_empirical_k_calculator(holiday_dates, outage_dates_map, k_default):
    st.subheader('🧮 Phân tích hệ số k từ điện nhận – điện giao ngày')
    st.info('Điện nhận trừ điện giao là chỉ tiêu tại ranh giới đo, dùng làm tín hiệu tham khảo; '
            'không tự động đồng nhất với điện thương phẩm của 14 huyện. '
            'Hãy kiểm tra phạm vi điểm đo và lịch nghỉ trước khi dùng kết quả.')
    upload = st.file_uploader('Chọn nhiều file điện nhận/điện giao (.xlsx)', type=['xlsx'],
                              accept_multiple_files=True, key='daily_k_upload')
    if not upload:
        st.caption('Mỗi file có thể chứa cả sheet “Chiều nhận”, “Chiều giao” hoặc chứa một trong hai sheet. '
                   'Bạn có thể bổ sung nhiều tháng và năm.')
        return
    parsed = []
    for f in upload:
        try:
            parsed.append(_read_cached(f.getvalue(), f.name))
        except Exception as exc:
            st.error(f'Không đọc được {f.name}: {exc}')
    if not parsed:
        return
    st.write('**1. Xác nhận năm và tháng của từng tệp**')
    meta = pd.DataFrame([{'Tệp': x['filename'], 'Năm': x['year'], 'Tháng': x['month'],
                          'Có nhận': 'nhan' in x['channels'], 'Có giao': 'giao' in x['channels']}
                         for x in parsed])
    for col in ['Năm', 'Tháng']:
        meta[col] = pd.array(meta[col], dtype='Int64')
    edit = st.data_editor(meta, hide_index=True, use_container_width=True,
                          disabled=['Tệp', 'Có nhận', 'Có giao'], key='daily_k_metadata',
                          column_config={
                              'Năm': st.column_config.NumberColumn('Năm', min_value=2000, max_value=2100, step=1),
                              'Tháng': st.column_config.NumberColumn('Tháng', min_value=1, max_value=12, step=1)})
    items = []
    for i, obj in enumerate(parsed):
        x = dict(obj)
        x['year'], x['month'] = edit.iloc[i]['Năm'], edit.iloc[i]['Tháng']
        items.append(x)
    try:
        all_channels = combine_channels(items)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.write('**2. Kiểm tra phạm vi điểm đo**')
    st.caption('So sánh số và mã điểm đo giữa các kỳ. Điểm đo chung không tự chứng minh '
               'cùng phạm vi cấp điện; cần đối chiếu sơ đồ ranh giới và nghiệp vụ.')
    scope = describe_scope(all_channels)
    st.dataframe(scope, hide_index=True, use_container_width=True)
    if (scope['Điểm nhận mới'].fillna(0) > 0).any() or (scope['Điểm nhận không còn'].fillna(0) > 0).any():
        st.warning('Danh sách điểm nhận thay đổi giữa các kỳ. Nên phân tích từng giai đoạn '
                   'hoặc chọn bộ điểm đo phù hợp trước khi so sánh.')
    with st.expander('Danh sách điểm đo và nguồn tệp'):
        catalog = meter_catalog(all_channels)
        st.dataframe(catalog, hide_index=True, use_container_width=True)
        st.download_button('Tải danh sách điểm đo (.csv)',
                           data=catalog.to_csv(index=False).encode('utf-8-sig'),
                           file_name='diem_do_va_nguon_tep.csv', mime='text/csv')
    st.write('**3. Chọn dữ liệu và thời điểm được phép sử dụng**')
    all_years = sorted({y for y, _, _ in all_channels})
    chosen_years = st.multiselect('Năm dữ liệu dùng tính k', all_years, default=[max(all_years)],
                                  key='daily_k_years')
    if not chosen_years:
        st.info('Chọn ít nhất một năm dữ liệu.')
        return
    channels = {key: v for key, v in all_channels.items() if key[0] in chosen_years}
    today = dt.date.today()
    next_month = today.replace(day=28) + dt.timedelta(days=4)
    c1, c2, c3 = st.columns(3)
    with c1:
        target_y = int(st.number_input('Năm dự báo', min_value=2000, max_value=2100,
                                       value=next_month.year, step=1, key='daily_k_target_year'))
    with c2:
        target_m = int(st.selectbox('Tháng dự báo', list(range(1, 13)),
                                    index=next_month.month - 1, key='daily_k_target_month'))
    first_target = dt.date(target_y, target_m, 1)
    prev = first_target - dt.timedelta(days=1)
    suggested_asof = dt.date(prev.year, prev.month, 10)
    with c3:
        as_of = st.date_input('Ngày chốt dự báo', value=suggested_asof,
                              key=f'daily_k_asof_{target_y}_{target_m}')
    st.caption('Chỉ dùng số liệu ngày trước ngày chốt. Cần lưu bản dữ liệu thực sự có vào ngày chốt '
               'để tránh sử dụng số liệu đã được chỉnh sửa về sau.')
    if as_of >= first_target:
        st.error('Ngày chốt phải trước tháng dự báo.')
        return
    if as_of > today:
        st.warning('Ngày chốt ở tương lai: đây là kịch bản, chưa phải kiểm định dự báo thực tế.')
    same_month = st.checkbox('Chỉ dùng các tháng cùng số tháng dự báo (nếu có đủ mẫu)',
                             value=False, key='daily_k_same_month')
    if same_month:
        channels = {key: v for key, v in channels.items() if key[1] == target_m}
        if not channels:
            st.warning('Không có dữ liệu cho tháng tương ứng trong các năm đã chọn.')
            return
    formula = st.selectbox('Chỉ tiêu phân tích',
                           ['Điện nhận − điện giao', 'Chỉ điện nhận', 'Chỉ điện giao'],
                           key='daily_k_formula')
    mode = {'Điện nhận − điện giao': 'net', 'Chỉ điện nhận': 'nhan', 'Chỉ điện giao': 'giao'}[formula]
    st.caption('Không tự trừ EDC hoặc điện mặt trời: cần xác nhận sơ đồ ranh giới và tránh trừ trùng.')
    point_mode = st.radio('Phạm vi điểm đo', ['Tất cả điểm trong từng tệp', 'Chọn điểm đo thủ công'],
                          horizontal=True, key='daily_k_scope')
    sel_in, sel_out = None, None
    if point_mode == 'Chọn điểm đo thủ công':
        received, delivered = meter_choices(channels, 'nhan'), meter_choices(channels, 'giao')
        if mode in ('net', 'nhan'):
            sel_in = set(st.multiselect('Điểm nhận được chọn', sorted(received),
                                        format_func=lambda x: f'{x} — {received[x]}', key='daily_k_meter_in'))
        if mode in ('net', 'giao'):
            sel_out = set(st.multiselect('Điểm giao được chọn', sorted(delivered),
                                         format_func=lambda x: f'{x} — {delivered[x]}', key='daily_k_meter_out'))
        if (sel_in is not None and not sel_in) or (sel_out is not None and not sel_out):
            st.info('Chọn ít nhất một điểm cho mỗi chiều đang dùng.')
            return
    days = build_daily_frame(channels, as_of, holiday_dates, _outage_dates(outage_dates_map),
                             mode, sel_in, sel_out)
    if days.empty:
        st.warning('Không còn ngày dữ liệu trước ngày chốt.')
        return
    detail, summary = estimate_k(days)
    detail['Ngày chốt dự báo'] = as_of.isoformat()
    detail['Tháng dự báo'] = f'{target_m:02d}/{target_y}'
    detail['Chỉ tiêu lựa chọn'] = formula
    detail['Điểm nhận được chọn'] = 'Tất cả trong file' if sel_in is None else ', '.join(sorted(sel_in))
    detail['Điểm giao được chọn'] = 'Tất cả trong file' if sel_out is None else ', '.join(sorted(sel_out))
    valid = int(detail['Hợp lệ'].sum())
    st.write('**4. Kết quả thăm dò**')
    st.caption('k ngày = giá trị ngày / trung vị ngày thường T3–T6 cùng tuần; '
               'nếu tuần thiếu ít nhất 2 ngày thường hợp lệ thì dùng trung vị tháng '
               '(cần ít nhất 3 ngày). Ngày cắt điện đã xác nhận được loại khỏi mẫu hệ số ngày.')
    st.metric('Ngày dữ liệu hợp lệ', f'{valid}/{len(detail)}')
    if valid < len(detail):
        st.warning(f'{len(detail) - valid} ngày thiếu dữ liệu, tổng không khớp hoặc giá trị không hợp lệ; '
                   'không được dùng để tính k. Xem bảng chi tiết bên dưới.')
    if 'Nghi ngờ bất thường' in detail and detail['Nghi ngờ bất thường'].any():
        st.caption('Ngày nghi ngờ bất thường chỉ được đánh dấu để kiểm tra, không tự loại bỏ '
                   'và không tự kết luận là cắt điện.')
    show = summary.copy()
    active_k = coefficients_for_month(k_default, target_y, target_m)
    show['k hiện dùng cho tháng dự báo'] = show['Mã'].map(active_k)
    show['Ngày chốt dự báo'] = as_of.isoformat()
    show['Tháng dự báo'] = f'{target_m:02d}/{target_y}'
    show['Chỉ tiêu lựa chọn'] = formula
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption('Q25–Q75 là khoảng giữa 50% tỷ số ngày trong mẫu, không phải khoảng tin cậy. '
               'Các ngày cùng kỳ lễ có thể liên quan nhau; số ngày không phải số sự kiện độc lập.')
    clean = detail[detail['Hợp lệ'] & ~detail['Cắt điện xác nhận'] & detail['Tỷ số ngày'].notna()]
    if not clean.empty:
        monthly = (clean[clean['Loại'].isin(['T2', 'T7', 'CN', 'Le', 'Tet'])]
                   .groupby(['Năm', 'Tháng', 'Loại'], as_index=False)
                   .agg(**{'Số ngày mẫu': ('Tỷ số ngày', 'size'),
                           'k trung vị của tháng': ('Tỷ số ngày', 'median')}))
        with st.expander('So sánh hệ số theo từng tháng dữ liệu'):
            st.dataframe(monthly, hide_index=True, use_container_width=True)
    samples = summary[(summary['Số ngày'] > 0) & summary['k trung vị'].notna()]
    if (samples['Số tháng'] < 3).any() or (samples['Số năm'] < 2).any():
        st.warning('Một số hệ số có ít tháng hoặc ít năm quan sát, nên chỉ dùng để tham khảo; '
                   'cần thêm dữ liệu và kiểm định ngoài mẫu.')
    with st.expander('🔎 Chi tiết từng ngày và lý do loại dữ liệu'):
        st.dataframe(detail, hide_index=True, use_container_width=True)
        st.download_button('Tải dữ liệu ngày đã phân tích (.csv)',
                           data=detail.to_csv(index=False).encode('utf-8-sig'),
                           file_name='du_lieu_ngay_phan_tich_k.csv', mime='text/csv')
    st.download_button('Tải bảng hệ số và số lượng mẫu (.csv)',
                       data=show.to_csv(index=False).encode('utf-8-sig'),
                       file_name='he_so_k_tham_khao.csv', mime='text/csv')
    eligible = {str(r['Mã']): float(r['k trung vị']) for _, r in samples.iterrows()
                if pd.notna(r['k trung vị']) and float(r['k trung vị']) > 0}
    if eligible:
        choose = st.multiselect('Chọn các hệ số muốn áp dụng', list(eligible), default=[], key='daily_k_apply_codes')
        acknowledge = st.checkbox('Tôi đã kiểm tra phạm vi điểm đo, các ngày nghỉ và hiểu đây là hệ số tham khảo.',
                                  key='daily_k_acknowledge')
        values = {k: eligible[k] for k in choose if k in eligible}
        st.caption(f'Chỉ cập nhật các hệ số đã chọn cho tháng {target_m:02d}/{target_y}; '
                   'các tháng khác tiếp tục dùng cấu hình riêng hoặc hệ số chung. '
                   'Chạy kiểm định theo thời gian trước khi kết luận cải thiện.')
        st.button('Áp dụng hệ số cho tháng dự báo', on_click=_apply_month,
                  args=(values, target_y, target_m),
                  disabled=not acknowledge or not values, type='primary')
    notice = st.session_state.pop('applied_k_notice', None)
    if notice:
        st.success('Đã cập nhật cấu hình: ' + notice)
