"""Xuất báo cáo dự báo ra Excel nhiều sheet."""
from __future__ import annotations

from io import BytesIO
from typing import Mapping

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def _safe_df(df: pd.DataFrame | None) -> pd.DataFrame:
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()


def build_excel_report(result_df: pd.DataFrame,
                       metrics_df: pd.DataFrame | None = None,
                       backtest_df: pd.DataFrame | None = None,
                       day_structure_df: pd.DataFrame | None = None,
                       finalized_df: pd.DataFrame | None = None,
                       params: Mapping | None = None,
                       calendar_detail_df: pd.DataFrame | None = None) -> bytes:
    output = BytesIO()
    params_df = pd.DataFrame([
        {"Tham số": str(k), "Giá trị": str(v)} for k, v in (params or {}).items()
    ])
    sheets = {
        "Tong_quan": _safe_df(finalized_df) if finalized_df is not None and not finalized_df.empty else _safe_df(result_df),
        "Du_bao_mo_hinh": _safe_df(result_df),
        "Danh_gia": _safe_df(metrics_df),
        "Sai_so_thang": _safe_df(backtest_df),
        "Co_cau_ngay": _safe_df(day_structure_df),
        "Tinh_chinh": _safe_df(finalized_df),
        "Tham_so": params_df,
        "Chi_tiet_lich": _safe_df(calendar_detail_df),
    }
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df.empty and name not in {"Danh_gia", "Sai_so_thang", "Tinh_chinh", "Chi_tiet_lich"}:
                continue
            df.to_excel(writer, sheet_name=name[:31], index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            header_fill = PatternFill("solid", fgColor="1F4E78")
            final_fill = PatternFill("solid", fgColor="E2F0D9")
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = Font(color="FFFFFF", bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center")
            for col_idx, col_cells in enumerate(ws.columns, 1):
                max_len = 0
                for cell in col_cells:
                    value = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, min(len(value), 45))
                    if isinstance(cell.value, (int, float)) and cell.row > 1:
                        cell.number_format = '#,##0.00' if isinstance(cell.value, float) and not float(cell.value).is_integer() else '#,##0'
                ws.column_dimensions[get_column_letter(col_idx)].width = max(10, max_len + 2)
            # Highlight các cột kết quả cuối nếu có.
            headers = {cell.value: cell.column for cell in ws[1]}
            for key in ["TỔNG SAU ĐIỀU CHỈNH", "Tổng Sau Điều Chỉnh", "Kết quả chốt"]:
                if key in headers:
                    col = headers[key]
                    for row in range(2, ws.max_row + 1):
                        ws.cell(row, col).fill = final_fill
                        ws.cell(row, col).font = Font(bold=True)
    return output.getvalue()
