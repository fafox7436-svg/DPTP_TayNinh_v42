"""Streamlit interface for the auditable daily-coefficient analysis."""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import pandas as pd
import streamlit as st

from .calendar_utils import (parse_date_list, coefficients_for_month, build_month_calendar,
                             outage_dates_for_month)
from .daily_k import (read_daily_workbook, group_channel_candidates, resolve_channel_groups,
                      common_meter_ids, describe_scope, meter_catalog, meter_choices,
                      build_daily_frame, estimate_k, recommend_target_k)


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


def render_empirical_k_calculator(holiday_dates, outage_dates_map, k_default,
                                  holidays_map=None, holiday_notes=None, forecast_input=None):
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
    with st.spinner(f'Đang đọc {len(upload)} file điện ngày...'):
        for f in upload:
            try:
                parsed.append(_read_cached(f.getvalue(), f.name))
            except Exception as exc:
                st.error(f'Không đọc được file {f.name}; các file khác vẫn được xử lý: {exc}')
    if not parsed:
        return
    st.write('**1. Xác nhận năm và tháng của từng tệp**')
    meta = pd.DataFrame([{'Tệp': x['filename'], 'Năm': x['year'], 'Tháng': x['month'],
                          'Có nhận': 'nhan' in x['channels'], 'Có giao': 'giao' in x['channels']}
                         for x in parsed])
    for col in ['Năm', 'Tháng']:
        meta[col] = pd.array(meta[col], dtype='Int64')
    upload_key = hashlib.sha256('|'.join(sorted(x['sha256'] for x in parsed)).encode()).hexdigest()[:12]
    edit = st.data_editor(meta, hide_index=True, use_container_width=True,
                          disabled=['Tệp', 'Có nhận', 'Có giao'], key=f'daily_k_metadata_{upload_key}',
                          column_config={
                              'Năm': st.column_config.NumberColumn('Năm', min_value=2000, max_value=2100, step=1),
                              'Tháng': st.column_config.NumberColumn('Tháng', min_value=1, max_value=12, step=1)})
    items = []
    for i, obj in enumerate(parsed):
        x = dict(obj)
        x['year'], x['month'] = edit.iloc[i]['Năm'], edit.iloc[i]['Tháng']
        items.append(x)
    groups, bad_periods = group_channel_candidates(items)
    if bad_periods:
        st.warning('Một số file chưa xác nhận được tháng/năm; các file còn lại vẫn được tính. '
                   'Hãy sửa bảng phía trên để bổ sung các file này.')
        st.dataframe(pd.DataFrame(bad_periods), hide_index=True, use_container_width=True)
    decisions = {}
    duplicates = [key for key, versions in groups.items()
                  if any(len(value['aliases']) > 1 for value in versions.values())]
    if duplicates:
        st.info(f'Có {len(duplicates)} chiều/tháng bị tải trùng nội dung; mỗi bộ số liệu giống nhau chỉ được tính một lần.')
    for (y, m, direction), versions in sorted(groups.items()):
        if len(versions) <= 1:
            continue
        st.warning(f'Có {len(versions)} phiên bản khác nhau cho chiều {direction} {m:02d}/{y}; '
                   'hãy chọn một phiên bản hoặc bỏ qua kỳ này. Các tháng khác vẫn được tính.')
        options = ['Bỏ qua kỳ này'] + sorted(versions)
        def label(sig, versions=versions):
            return sig if sig == 'Bỏ qua kỳ này' else f"{versions[sig]['filename']} — bản {sig[:8]}"
        signature_key = hashlib.sha256('|'.join(options[1:]).encode()).hexdigest()[:10]
        chosen = st.selectbox(f'Chọn file cho {direction} {m:02d}/{y}', options,
                              format_func=label, key=f'k_version_{y}_{m}_{direction}_{signature_key}')
        if chosen != 'Bỏ qua kỳ này':
            decisions[(y, m, direction)] = chosen
    all_channels, conflicts = resolve_channel_groups(groups, decisions)
    if conflicts:
        st.warning('Các chiều chưa được chọn đang bỏ qua; kiểm tra bảng dưới đây.')
        st.dataframe(pd.DataFrame(conflicts), hide_index=True, use_container_width=True)
    if not all_channels:
        st.info('Chưa có tháng hợp lệ để tính. Hãy kiểm tra năm/tháng và phiên bản file.')
        return
    st.success(f"Đã nhận {len(parsed)} file; có {len({key[:2] for key in all_channels})} tháng có ít nhất một chiều dữ liệu.")
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
    year_key = '_'.join(str(y) for y in all_years)
    chosen_years = st.multiselect('Năm dữ liệu dùng tính k', all_years, default=all_years,
                                  key=f'daily_k_years_{year_key}')
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
    before_cutoff = {key: value for key, value in channels.items()
                     if dt.date(key[0], key[1], 1) < as_of}
    excluded_months = len({key[:2] for key in channels} - {key[:2] for key in before_cutoff})
    if excluded_months:
        st.caption(f'{excluded_months} tháng bắt đầu sau mốc chốt được giữ trong danh sách file '
                   'nhưng không dùng để tính hoặc xác định điểm đo chung.')
    channels = before_cutoff
    if not channels:
        st.warning('Không có tháng dữ liệu nào trước ngày chốt dự báo.')
        return
    same_month = st.checkbox('Chỉ dùng các tháng cùng số tháng dự báo (nếu có đủ mẫu)',
                             value=False, key='daily_k_same_month')
    if same_month:
        st.caption('Tết có thể chuyển giữa tháng 1 và 2; bộ lọc này có thể loại mất kỳ Tết ở tháng khác.')
        channels = {key: v for key, v in channels.items() if key[1] == target_m}
        if not channels:
            st.warning('Không có dữ liệu cho tháng tương ứng trong các năm đã chọn.')
            return
    formula = st.selectbox('Chỉ tiêu phân tích',
                           ['Điện nhận − điện giao', 'Chỉ điện nhận', 'Chỉ điện giao'],
                           key='daily_k_formula')
    mode = {'Điện nhận − điện giao': 'net', 'Chỉ điện nhận': 'nhan', 'Chỉ điện giao': 'giao'}[formula]
    st.caption('Không tự trừ EDC hoặc điện mặt trời: cần xác nhận sơ đồ ranh giới và tránh trừ trùng.')
    point_mode = st.radio('Phạm vi điểm đo', ['Tất cả điểm trong từng tệp',
                                               'Điểm chung của tất cả tháng đã chọn',
                                               'Chọn điểm đo thủ công'],
                          horizontal=True, key='daily_k_scope')
    sel_in, sel_out = None, None
    if point_mode == 'Điểm chung của tất cả tháng đã chọn':
        sel_in = common_meter_ids(channels, 'nhan') if mode in ('net', 'nhan') else None
        sel_out = common_meter_ids(channels, 'giao') if mode in ('net', 'giao') else None
        st.caption('Điểm chung giúp giữ danh sách ổn định qua các kỳ nhưng có thể không đại diện đầy đủ '
                   'cho phạm vi dự báo; cần kiểm tra sơ đồ ranh giới.')
        if (sel_in is not None and not sel_in) or (sel_out is not None and not sel_out):
            st.warning('Không có điểm chung, hoặc một tháng thiếu chiều cần thiết. '
                       'Hãy chọn các tháng có đủ hai chiều hoặc chọn điểm thủ công.')
            return
    elif point_mode == 'Chọn điểm đo thủ công':
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
    st.write(f'**4. Gợi ý hệ số cho tháng dự báo {target_m:02d}/{target_y}**')
    st.caption('k ngày = giá trị ngày / trung vị ngày thường T3–T6 cùng tuần; '
               'nếu tuần thiếu ít nhất 2 ngày thường hợp lệ thì dùng trung vị tháng '
               '(cần ít nhất 3 ngày). Ngày cắt điện đã xác nhận được loại khỏi mẫu hệ số ngày.')
    st.metric('Ngày dữ liệu hợp lệ', f'{valid}/{len(detail)}')
    used_months = detail[detail['Hợp lệ']][['Năm', 'Tháng']].drop_duplicates()
    st.caption(f"Đã dùng dữ liệu của {len(used_months)} tháng có ngày hợp lệ trong các năm: "
               + (', '.join(str(y) for y in sorted(used_months['Năm'].unique())) or 'không có')
               + '. Các ngày từ ngày chốt trở đi không được tính.')
    if valid < len(detail):
        st.warning(f'{len(detail) - valid} ngày thiếu dữ liệu, tổng không khớp hoặc giá trị không hợp lệ; '
                   'không được dùng để tính k. Xem bảng chi tiết bên dưới.')
    if 'Nghi ngờ bất thường' in detail and detail['Nghi ngờ bất thường'].any():
        st.caption('Ngày nghi ngờ bất thường chỉ được đánh dấu để kiểm tra, không tự loại bỏ '
                   'và không tự kết luận là cắt điện.')
    target_row = {'Năm': target_y, 'Tháng': target_m, 'Số ngày cắt điện': 0}
    if isinstance(forecast_input, pd.DataFrame) and not forecast_input.empty:
        chosen_input = forecast_input[(forecast_input['Năm'] == target_y) & (forecast_input['Tháng'] == target_m)]
        if not chosen_input.empty:
            target_row.update(chosen_input.iloc[0].to_dict())
    outage_count = target_row.get('Số ngày cắt điện', 0)
    outage_count = int(outage_count) if pd.notna(outage_count) else 0
    target_calendar = build_month_calendar(target_y, target_m, holidays_map or {}, holiday_dates,
                                            outage_dates_for_month(target_row, outage_dates_map),
                                            outage_count, k_default)
    target_counts = target_calendar['Category'].value_counts()
    st.caption(f"Lịch tháng dự báo đang có {int(target_counts.get('Tet', 0))} ngày Tết, "
               f"{int(target_counts.get('Le', 0))} ngày lễ và {int(target_calendar['Cat_Dien'].sum())} "
               'ngày cắt điện. Kiểm tra và sửa tại tab Dữ liệu & lịch nếu chưa đúng.')
    show = recommend_target_k(detail, target_calendar, holiday_notes, summary)
    if target_calendar.attrs.get('holiday_mode') == 'legacy' and (
            int(target_counts.get('Tet', 0)) or int(target_counts.get('Le', 0))):
        st.warning('Tháng dự báo đang dùng số ngày lễ/Tết tổng hợp; nếu muốn so sánh đúng từng ngày lễ '
                   'hãy nhập ngày và ghi chú cụ thể trong lịch.')
    active_k = coefficients_for_month(k_default, target_y, target_m)
    show['k hiện dùng cho tháng dự báo'] = show['Mã'].map(active_k)
    show['Ngày chốt dự báo'] = as_of.isoformat()
    show['Tháng dự báo'] = f'{target_m:02d}/{target_y}'
    show['Chỉ tiêu lựa chọn'] = formula
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption('Với T2/T7/CN, ưu tiên cùng tháng qua ít nhất 2 năm và 6 ngày mẫu; '
               'nếu thiếu thì mở rộng tháng lân cận hoặc các tháng đã chọn. Với Lễ/Tết, '
               'ưu tiên ghi chú tương ứng (ví dụ cùng Mùng 1), nếu thiếu thì xem toàn bộ '
               'ngày cùng loại. Đây là quy tắc gợi ý thăm dò, cần kiểm định dự báo ngoài mẫu.')
    st.caption('Q25–Q75 là khoảng giữa 50% tỷ số ngày trong mẫu, không phải khoảng tin cậy. '
               'Các ngày cùng kỳ lễ có thể liên quan nhau; số ngày không phải số sự kiện độc lập.')
    candidates = show[show['k đề xuất'].notna() & (show['k đề xuất'] > 0)]
    if not candidates.empty:
        hypothetical = dict(active_k)
        hypothetical.update({str(r['Mã']): float(r['k đề xuất']) for _, r in candidates.iterrows()})
        category_keys = {'T2': 'T2', 'T7': 'T7', 'CN': 'CN', 'Le': 'Le', 'Tet': 'Tet'}
        equiv_suggested = sum(
            (hypothetical.get(category_keys.get(r['Category']), 1.0) if r['Category'] != 'Thuong' else 1.0)
            * (hypothetical['CatDien'] if r['Cat_Dien'] else 1.0)
            for _, r in target_calendar.iterrows()
        )
        m_old, m_new = st.columns(2)
        m_old.metric('Ngày tương đương hiện dùng', f"{target_calendar['k_Cuoi'].sum():.3f}")
        m_new.metric('Nếu áp dụng các k có đề xuất', f'{equiv_suggested:.3f}')
    else:
        st.warning('Chưa đủ dữ liệu hợp lệ để đề xuất k cho tháng này. Hãy xem lý do loại ngày bên dưới.')
    clean = detail[detail['Hợp lệ'] & ~detail['Cắt điện xác nhận'] & detail['Tỷ số ngày'].notna()]
    if not clean.empty:
        monthly = (clean[clean['Loại'].isin(['T2', 'T7', 'CN', 'Le', 'Tet'])]
                   .groupby(['Năm', 'Tháng', 'Loại'], as_index=False)
                   .agg(**{'Số ngày mẫu': ('Tỷ số ngày', 'size'),
                           'k trung vị của tháng': ('Tỷ số ngày', 'median')}))
        with st.expander('So sánh hệ số theo từng tháng dữ liệu'):
            st.dataframe(monthly, hide_index=True, use_container_width=True)
    if (candidates['Số tháng mẫu'] < 3).any() or (candidates['Số năm mẫu'] < 2).any():
        st.warning('Một số hệ số có ít tháng hoặc ít năm quan sát, nên chỉ dùng để tham khảo; '
                   'cần thêm dữ liệu và kiểm định ngoài mẫu.')
    with st.expander('Thống kê chung của toàn bộ ngày hợp lệ'):
        st.dataframe(summary, hide_index=True, use_container_width=True)
    with st.expander('🔎 Chi tiết từng ngày và lý do loại dữ liệu'):
        st.dataframe(detail, hide_index=True, use_container_width=True)
        st.download_button('Tải dữ liệu ngày đã phân tích (.csv)',
                           data=detail.to_csv(index=False).encode('utf-8-sig'),
                           file_name='du_lieu_ngay_phan_tich_k.csv', mime='text/csv')
    st.download_button('Tải bảng hệ số và số lượng mẫu (.csv)',
                       data=show.to_csv(index=False).encode('utf-8-sig'),
                       file_name='he_so_k_tham_khao.csv', mime='text/csv')
    eligible = {str(r['Mã']): float(r['k đề xuất']) for _, r in candidates.iterrows()}
    if eligible:
        choice_sig = hashlib.sha256(repr(sorted(eligible.items())).encode()).hexdigest()[:10]
        widget_suffix = f'{target_y}_{target_m}_{choice_sig}'
        choose = st.multiselect('Chọn các hệ số muốn áp dụng', list(eligible), default=list(eligible),
                                key=f'daily_k_apply_codes_{widget_suffix}')
        acknowledge = st.checkbox('Tôi đã kiểm tra phạm vi điểm đo, các ngày nghỉ và hiểu đây là hệ số tham khảo.',
                                  key=f'daily_k_acknowledge_{widget_suffix}')
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
