"""Real Streamlit AppTest with synthetic uploads; skipped without the runtime."""
from __future__ import annotations

import importlib.util
import unittest


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ('streamlit', 'plotly', 'xgboost', 'requests', 'bs4'))

SCRIPT = '''
import io, runpy
import streamlit as st
from openpyxl import Workbook

def make_workbook(changed=False):
    book = Workbook()
    for index, name in enumerate(['Chiều nhận', 'Chiều giao']):
        sheet = book.active if index == 0 else book.create_sheet()
        sheet.title = name
        sheet.append(['STT', None, 'Tên điểm đo', 'Mã điểm đo', 'Cấp điện áp', 'Đơn vị'] + list(map(str, range(1, 32))))
        sheet.append(['Đơn vị: giả lập'])
        val = 100 if index == 0 else 10
        values = [val] * 31
        if changed:
            values[0] += 1
        sheet.append([None, 1, 'Điểm giả lập', 'A', '110kV', 'Đơn vị'] + values)
        sheet.append([None] * 6 + values)
    data = io.BytesIO()
    book.save(data)
    return data.getvalue()

class Upload:
    def __init__(self, name, data):
        self.name, self.data = name, data
    def getvalue(self):
        return self.data

normal, different = make_workbook(), make_workbook(True)
files = [Upload(f'diennhan thang {m:02d}.{y}.xlsx', normal)
         for y in range(2018, 2023) for m in range(1, 13)]
files.append(Upload('diennhan thang 01.2018.xlsx', normal))
files.append(Upload('diennhan thang 02.2018.xlsx', different))
files.append(Upload('khong_ro_thoi_gian.xlsx', normal))

def uploader(*args, **kwargs):
    return files if kwargs.get('key') == 'daily_k_upload' else None
st.file_uploader = uploader
runpy.run_path('app.py', run_name='__main__')
'''


@unittest.skipUnless(HAS_RUNTIME, 'Cần môi trường Streamlit/XGBoost thực để chạy AppTest.')
class StreamlitManyMonthsTests(unittest.TestCase):
    def test_many_uploads_show_and_apply_target_k(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string(SCRIPT).run(timeout=90)
        self.assertFalse(list(app.exception), [str(x) for x in app.exception])
        self.assertTrue(any('Đã nhận 63 file' in str(element.value) for element in app.success))
        suggestion_tables = []
        for element in app.dataframe:
            value = element.value
            if hasattr(value, 'columns') and 'k đề xuất' in value.columns:
                suggestion_tables.append(value)
        self.assertTrue(suggestion_tables, 'Không tìm thấy bảng k cho tháng dự báo.')
        self.assertTrue(suggestion_tables[0]['k đề xuất'].notna().any())
        acknowledge = next(x for x in app.checkbox if str(x.label).startswith('Tôi đã kiểm tra phạm vi'))
        acknowledge.check().run(timeout=90)
        apply_button = next(x for x in app.button if x.label == 'Áp dụng hệ số cho tháng dự báo')
        apply_button.click().run(timeout=90)
        self.assertFalse(list(app.exception), [str(x) for x in app.exception])
        self.assertTrue(app.session_state['monthly_k_overrides'])

    def test_real_models_use_target_month_k(self):
        import calendar
        import numpy as np
        import pandas as pd
        from forecast.config import DEFAULT_K
        from forecast.data import tao_dac_trung
        from forecast.models import chay_mo_hinh_goc

        history = pd.DataFrame([
            {'Năm': y, 'Tháng': m, 'Số ngày': calendar.monthrange(y, m)[1],
             'Nhiệt độ TB': 27 + (m % 5), 'Độ ẩm': 70 + (m % 8),
             'Tổng thương phẩm': 500_000_000 + (y - 2023) * 20_000_000 + m * 1_000_000}
            for y in (2023, 2024, 2025) for m in range(1, 13)
        ])
        future = pd.DataFrame([
            {'Năm': 2026, 'Tháng': m, 'Số ngày': calendar.monthrange(2026, m)[1],
             'Nhiệt độ TB': 27 + (m % 5), 'Độ ẩm': 70 + (m % 8)}
            for m in (2, 3)
        ])
        holiday = pd.DataFrame([{'Ngày': '17/02/2026', 'Loại': 'Tết'}])
        original = dict(DEFAULT_K)
        updated = dict(DEFAULT_K, __monthly__=((2026, 2, (('Tet', .25),)),))
        old_days = tao_dac_trung(future, {}, holiday, k_dict=original)['Ngay_Tuong_Duong'].to_numpy()
        new_days = tao_dac_trung(future, {}, holiday, k_dict=updated)['Ngay_Tuong_Duong'].to_numpy()
        original_pred = chay_mo_hinh_goc(history, future, {}, original, seed=42, holiday_dates=holiday)
        updated_pred = chay_mo_hinh_goc(history, future, {}, updated, seed=42, holiday_dates=holiday)
        for baseline, revised in zip(original_pred, updated_pred):
            baseline, revised = np.asarray(baseline, dtype=float), np.asarray(revised, dtype=float)
            self.assertTrue(np.all(np.isfinite(baseline)))
            self.assertGreater(baseline[0], 0)
            self.assertAlmostEqual(revised[0] / baseline[0], new_days[0] / old_days[0], places=7)
            self.assertAlmostEqual(revised[1], baseline[1], delta=0.01)


if __name__ == '__main__':
    unittest.main()
