"""Run with: python -m unittest discover -s tests -p 'test_daily_k.py'."""
from __future__ import annotations

import calendar
import io
import unittest
from datetime import date

import numpy as np
import pandas as pd
from openpyxl import Workbook

from forecast.calendar_state import dump_calendar, load_calendar
from forecast.calendar_utils import (build_month_calendar, coefficients_for_month,
                                     outage_dates_for_month, normalize_holiday_dates)
from forecast.config import DEFAULT_K
from forecast.data import tao_dac_trung
from forecast.daily_k import (read_daily_workbook, infer_period, combine_channels,
                              describe_scope, build_daily_frame, estimate_k)


def example_excel(missing=False, include_giao=True):
    wb = Workbook()
    for idx, name in enumerate(['Chiều nhận', 'Chiều giao'] if include_giao else ['Chiều nhận']):
        ws = wb.active if idx == 0 else wb.create_sheet()
        ws.title = name
        ws.append(['STT', None, 'Tên điểm đo', 'Mã điểm đo', 'Cấp điện áp', 'Đơn vị'] + list(map(str, range(1, 32))))
        ws.append(['Đơn vị: kiểm thử'])
        first = [100 if name == 'Chiều nhận' else 10] * 31
        second = [50 if name == 'Chiều nhận' else 5] * 31
        if missing and name == 'Chiều nhận':
            first[2] = None
        ws.append([None, 1, 'Điểm A', 'A', '110kV', 'Đơn vị'] + first)
        ws.append([None, 2, 'Điểm B', 'B', '110kV', 'Đơn vị'] + second)
        ws.append([None] * 6 + [150 if name == 'Chiều nhận' else 15] * 31)
    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()


class DailyDataTests(unittest.TestCase):
    def test_period_requires_explicit_month_and_year(self):
        self.assertEqual(infer_period('diennhanngay thang 01.2023xlsx.xlsx'), (2023, 1))
        self.assertEqual(infer_period('dien_2026-02.xlsx'), (2026, 2))
        self.assertEqual(infer_period('bao_cao.xlsx'), (None, None))

    def test_receive_and_delivery_use_detail_only(self):
        item = read_daily_workbook(example_excel(), 'thang 01.2026.xlsx')
        channels = combine_channels([item])
        frame = build_daily_frame(channels, date(2026, 2, 1), pd.DataFrame())
        self.assertEqual(len(frame), 31)
        self.assertTrue(frame['Hợp lệ'].all())
        self.assertEqual(frame.iloc[0]['Điện nhận'], 150)
        self.assertEqual(frame.iloc[0]['Điện giao'], 15)
        self.assertEqual(frame.iloc[0]['Giá trị phân tích'], 135)

    def test_missing_is_not_zero_and_cutoff_is_strict(self):
        item = read_daily_workbook(example_excel(missing=True), 'thang 01.2026.xlsx')
        frame = build_daily_frame(combine_channels([item]), date(2026, 1, 10), pd.DataFrame())
        self.assertEqual(len(frame), 9)
        self.assertTrue(pd.isna(frame.iloc[2]['Điện nhận']))
        self.assertFalse(frame.iloc[2]['Hợp lệ'])
        self.assertIn('Thiếu', frame.iloc[2]['Cảnh báo'])

    def test_split_files_and_duplicate_rejection(self):
        item = read_daily_workbook(example_excel(), 'thang 01.2026.xlsx')
        a, b = dict(item), dict(item)
        a['channels'] = {'nhan': item['channels']['nhan']}
        b['channels'] = {'giao': item['channels']['giao']}
        both = combine_channels([a, b])
        self.assertEqual(len(both), 2)
        with self.assertRaisesRegex(ValueError, 'Trùng dữ liệu'):
            combine_channels([item, item])

    def test_missing_direction_does_not_become_zero(self):
        item = read_daily_workbook(example_excel(include_giao=False), 'thang 01.2026.xlsx')
        channels = combine_channels([item])
        net = build_daily_frame(channels, date(2026, 2, 1), pd.DataFrame(), mode='net')
        received = build_daily_frame(channels, date(2026, 2, 1), pd.DataFrame(), mode='nhan')
        self.assertFalse(net['Hợp lệ'].any())
        self.assertTrue(received['Hợp lệ'].all())

    def test_selected_missing_meter_is_not_silently_removed(self):
        item = read_daily_workbook(example_excel(), 'thang 01.2026.xlsx')
        df = build_daily_frame(combine_channels([item]), date(2026, 2, 1), pd.DataFrame(),
                               selected_in={'A', 'MISSING'})
        self.assertFalse(df['Hợp lệ'].any())
        self.assertEqual(df.iloc[0]['Thiếu nhận'], 1)

    def test_scope_changes_detected(self):
        one = read_daily_workbook(example_excel(), 'thang 01.2026.xlsx')
        two = read_daily_workbook(example_excel(), 'thang 02.2026.xlsx')
        two['channels']['nhan']['meters'] = two['channels']['nhan']['meters'][:1]
        scope = describe_scope(combine_channels([one, two]))
        self.assertEqual(int(scope.iloc[1]['Điểm nhận không còn']), 1)

    def test_baseline_holiday_and_confirmed_outage(self):
        records = []
        for d in pd.date_range('2026-01-01', '2026-01-31'):
            wd = d.weekday()
            kind = 'T2' if wd == 0 else 'T7' if wd == 5 else 'CN' if wd == 6 else 'Thuong'
            value = {'T2': 90., 'T7': 80., 'CN': 70., 'Thuong': 100.}[kind]
            if d.day == 6:
                kind, value = 'Tet', 40.
            if d.day == 12:  # Confirmed outage on a Monday.
                value = 45.
            records.append({'Ngày': d, 'Năm': 2026, 'Tháng': 1, 'Loại': kind,
                            'Hợp lệ': True, 'Cắt điện xác nhận': d.day == 12,
                            'Giá trị phân tích': value})
        detail, summary = estimate_k(pd.DataFrame(records))
        result = summary.set_index('Mã')
        self.assertAlmostEqual(result.loc['T2', 'k trung vị'], .9)
        self.assertAlmostEqual(result.loc['T7', 'k trung vị'], .8)
        self.assertAlmostEqual(result.loc['CN', 'k trung vị'], .7)
        self.assertAlmostEqual(result.loc['Tet', 'k trung vị'], .4)
        self.assertAlmostEqual(result.loc['CatDien', 'k trung vị'], .5)
        self.assertEqual(int(result.loc['CatDien', 'Số ngày']), 1)
        self.assertFalse(detail.loc[detail['Ngày'].dt.day == 12, 'Nghi ngờ bất thường'].iloc[0])

    def test_normalized_calendar_categories_are_used(self):
        item = read_daily_workbook(example_excel(), 'thang 01.2026.xlsx')
        holiday = pd.DataFrame([{'Ngày': pd.Timestamp('2026-01-06'), 'Category': 'Tet'}])
        df = build_daily_frame(combine_channels([item]), date(2026, 2, 1), holiday)
        self.assertEqual(df.loc[df['Ngày'].dt.day == 6, 'Loại'].iloc[0], 'Tet')


class CalendarTests(unittest.TestCase):
    def test_already_normalized_holiday_is_not_lost(self):
        raw = pd.DataFrame([{'Ngày': '17/02/2026', 'Loại': 'Tết'}])
        normalized = normalize_holiday_dates(raw)
        twice = normalize_holiday_dates(normalized)
        self.assertEqual(twice.iloc[0]['Category'], 'Tet')
        cal = build_month_calendar(2026, 2, {}, normalized, k_dict=DEFAULT_K)
        day = cal[cal['Ngày'] == pd.Timestamp('2026-02-17')].iloc[0]
        self.assertEqual(day['Category'], 'Tet')
        self.assertAlmostEqual(day['k_Cuoi'], DEFAULT_K['Tet'])

    def test_month_override_only_applies_to_its_month(self):
        base = dict(DEFAULT_K)
        base['__monthly__'] = ((2026, 2, (('Tet', .21),)),)
        holiday = pd.DataFrame([{'Ngày': '17/02/2026', 'Loại': 'Tết'}])
        feb = build_month_calendar(2026, 2, {}, holiday, k_dict=base)
        specific = feb[feb['Ngày'] == pd.Timestamp('2026-02-17')].iloc[0]
        self.assertAlmostEqual(specific['k_Cuoi'], .21)
        self.assertAlmostEqual(coefficients_for_month(base, 2026, 3)['Tet'], DEFAULT_K['Tet'])

    def test_month_override_changes_model_features_for_target_only(self):
        frame = pd.DataFrame([{'Năm': 2026, 'Tháng': 2, 'Số ngày cắt điện': 0},
                              {'Năm': 2026, 'Tháng': 3, 'Số ngày cắt điện': 0}])
        holiday = normalize_holiday_dates(pd.DataFrame([{'Ngày': '17/02/2026', 'Loại': 'Tết'}]))
        base = dict(DEFAULT_K)
        override = dict(DEFAULT_K, __monthly__=((2026, 2, (('Tet', .21),)),))
        normal_features = tao_dac_trung(frame, {}, holiday, k_dict=base)
        changed_features = tao_dac_trung(frame, {}, holiday, k_dict=override)
        self.assertAlmostEqual(normal_features.iloc[0]['Ngay_Tuong_Duong'] -
                               changed_features.iloc[0]['Ngay_Tuong_Duong'], DEFAULT_K['Tet'] - .21)
        self.assertEqual(normal_features.iloc[1]['Ngay_Tuong_Duong'], changed_features.iloc[1]['Ngay_Tuong_Duong'])

    def test_explicit_zero_outages_overrides_count(self):
        row = {'Năm': 2026, 'Tháng': 1}
        explicit = outage_dates_for_month(row, {(2026, 1): 'Không có'})
        self.assertEqual(explicit, [])
        cal = build_month_calendar(2026, 1, {}, outage_dates=explicit, outage_count=2, k_dict=DEFAULT_K)
        self.assertEqual(int(cal['Cat_Dien'].sum()), 0)
        fallback = build_month_calendar(2026, 1, {}, outage_dates=outage_dates_for_month(row, {}),
                                        outage_count=2, k_dict=DEFAULT_K)
        self.assertEqual(int(fallback['Cat_Dien'].sum()), 2)

    def test_calendar_backup_roundtrip(self):
        records = pd.DataFrame([{'Ngày': '17/02/2026', 'Loại': 'Tết', 'Ghi chú': 'Ghi chú lễ'},
                                {'Ngày': '18/03/2026', 'Loại': 'Ghi chú', 'Ghi chú': 'Nhà máy nghỉ'}])
        content = dump_calendar(records, {(2026, 2): 'Không có'}, {(2026, 2)},
                                DEFAULT_K, {(2026, 2): {'Tet': .35}})
        restored, outages, months, common, specific = load_calendar(content)
        self.assertEqual(restored.iloc[1]['Ghi chú'], 'Nhà máy nghỉ')
        self.assertEqual(outages[(2026, 2)], 'Không có')
        self.assertIn((2026, 2), months)
        self.assertEqual(common['Tet'], DEFAULT_K['Tet'])
        self.assertEqual(specific[(2026, 2)]['Tet'], .35)

    def test_reject_invalid_backup(self):
        with self.assertRaises(ValueError):
            load_calendar(b'{"format":"unknown"}')


if __name__ == '__main__':
    unittest.main()
