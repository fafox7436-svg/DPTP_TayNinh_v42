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


if __name__ == '__main__':
    unittest.main()
