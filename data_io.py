"""Read and validate monthly energy data without changing the source workbook."""
from __future__ import annotations

import calendar
import io
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


YEAR, MONTH, ENERGY = "Năm", "Tháng", "Tổng thương phẩm"
TEMP, HUMIDITY, DAYS, OUTAGE = "Nhiệt độ TB", "Độ ẩm", "Số ngày", "Số ngày cắt điện"
MAX_ROWS = 1200
MAX_BYTES = 20 * 1024 * 1024


class InputError(ValueError):
    """A recoverable, user-facing data issue."""


def normalize(value: object) -> str:
    s = str(value).strip().lower().replace("đ", "d")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(s.replace("_", " ").split())


ALIASES = {
    "nam": YEAR, "year": YEAR, "thang": MONTH, "month": MONTH,
    "tong thuong pham": ENERGY, "dien thuong pham": ENERGY,
    "san luong": ENERGY, "energy": ENERGY,
    "nhiet do tb": TEMP, "nhiet do trung binh": TEMP, "temperature": TEMP,
    "do am": HUMIDITY, "humidity": HUMIDITY,
    "so ngay": DAYS, "days": DAYS,
    "cat dien": OUTAGE, "so ngay cat dien": OUTAGE,
    "so ngay cat dien tuong duong": OUTAGE,
}


def canonical(value: object) -> str:
    return ALIASES.get(normalize(value), str(value).strip())


def inspect_workbook(data: bytes) -> list[tuple[str, int]]:
    if not data or len(data) > MAX_BYTES:
        raise InputError("Tệp trống hoặc lớn hơn 20 MB. Chỉ nhập bảng tổng hợp theo tháng.")
    try:
        book = pd.ExcelFile(io.BytesIO(data))
        candidates = []
        for sheet in book.sheet_names:
            preview = pd.read_excel(book, sheet_name=sheet, header=None, nrows=30)
            for index, row in preview.iterrows():
                if {YEAR, MONTH}.issubset({canonical(x) for x in row.dropna()}):
                    candidates.append((sheet, int(index)))
                    break
        if not candidates:
            raise InputError("Không tìm thấy dòng có cột Năm và Tháng trong 30 dòng đầu. Hãy dùng tệp mẫu.")
        return candidates
    except InputError:
        raise
    except Exception as exc:
        raise InputError("Không đọc được Excel. Kiểm tra tệp, mật khẩu bảo vệ và thư viện openpyxl/xlrd.") from exc


def read_table(data: bytes, sheet: str, header: int) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=header, nrows=MAX_ROWS + 1)
    except Exception as exc:
        raise InputError("Không đọc được bảng đã chọn.") from exc
    if len(df) > MAX_ROWS:
        raise InputError("Bảng vượt 1.200 dòng tháng; không được tự động cắt bớt dữ liệu.")
    df.columns = [canonical(c) for c in df.columns]
    if df.columns.duplicated().any():
        raise InputError("Tên cột bị trùng sau chuẩn hóa: " + ", ".join(df.columns[df.columns.duplicated()]))
    return df


@dataclass
class Validation:
    frame: pd.DataFrame
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_monthly(raw: pd.DataFrame, *, history: bool, decimal: str = ".") -> Validation:
    """Never silently drop invalid rows, duplicates, missing months or nonfinite values."""
    df = raw.copy().dropna(how="all")
    df.columns = [canonical(c) for c in df.columns]
    result = Validation(df)
    if df.columns.duplicated().any():
        result.errors.append("Có cột trùng tên sau chuẩn hóa.")
        return result
    required = [YEAR, MONTH] + ([ENERGY] if history else [])
    missing = [c for c in required if c not in df]
    if missing:
        result.errors.append("Thiếu cột: " + ", ".join(missing))
        return result
    if df.empty:
        result.errors.append("Bảng chưa có dữ liệu.")
        return result
    if len(df) > MAX_ROWS:
        result.errors.append("Bảng vượt 1.200 dòng tháng.")
        return result
    # Format choice applies to text cells only. Excel numeric cells are unchanged.
    for col in [YEAR, MONTH, ENERGY, TEMP, HUMIDITY, DAYS, OUTAGE]:
        if col not in df:
            continue
        def parse(v):
            if isinstance(v, str):
                v = v.strip().replace("\u00a0", "").replace(" ", "")
                v = v.replace(".", "").replace(",", ".") if decimal == "," else v.replace(",", "")
            return v
        series = pd.to_numeric(df[col].map(parse), errors="coerce")
        invalid = ~np.isfinite(series.astype(float))
        if invalid.any():
            rows = ", ".join(str(int(i) + 1) for i in np.flatnonzero(invalid)[:8])
            result.errors.append(f"Cột {col}: ô trống/không phải số hữu hạn tại dòng dữ liệu {rows}.")
        df[col] = series
    if result.errors:
        return result
    for col, lo, hi in [(YEAR, 1900, 2200), (MONTH, 1, 12)]:
        if ((df[col] % 1 != 0) | ~df[col].between(lo, hi)).any():
            result.errors.append(f"{col} phải là số nguyên trong khoảng {lo}–{hi}.")
    if result.errors:
        return result
    df[YEAR], df[MONTH] = df[YEAR].astype(int), df[MONTH].astype(int)
    df["Date"] = pd.to_datetime(dict(year=df[YEAR], month=df[MONTH], day=1))
    if df["Date"].duplicated().any():
        labels = df.loc[df["Date"].duplicated(False), "Date"].dt.strftime("%m/%Y").unique()
        result.errors.append("Trùng tháng: " + ", ".join(labels) + ". Cần tổng hợp thành một dòng/tháng.")
    actual_days = df["Date"].dt.days_in_month
    if DAYS in df and not (df[DAYS] == actual_days).all():
        result.errors.append("Cột Số ngày khác lịch thực tế. Sửa dữ liệu để thống nhất số ngày tính toán.")
    df[DAYS] = actual_days
    if OUTAGE not in df:
        df[OUTAGE] = 0.0
    if ((df[OUTAGE] < 0) | (df[OUTAGE] > df[DAYS])).any():
        result.errors.append("Số ngày cắt điện tương đương phải từ 0 đến số ngày của tháng.")
    if history and (df[ENERGY] <= 0).any():
        result.errors.append("Điện thương phẩm lịch sử phải lớn hơn 0; kiểm tra các tháng chưa đủ số liệu.")
    if TEMP in df and not df[TEMP].between(-50, 60).all():
        result.errors.append("Nhiệt độ TB phải dùng °C, trong khoảng kiểm tra −50 đến 60.")
    if HUMIDITY in df and not df[HUMIDITY].between(0, 100).all():
        result.errors.append("Độ ẩm phải dùng %, trong khoảng 0–100.")
    if HUMIDITY in df and df[HUMIDITY].le(1).all():
        result.warnings.append("Độ ẩm toàn bộ ≤ 1: kiểm tra xem dữ liệu đang dùng tỷ lệ 0–1 thay cho %.")
    df = df.sort_values("Date").reset_index(drop=True)
    if history:
        if len(df) < 24:
            result.errors.append("Cần tối thiểu 24 tháng lịch sử; nên có từ 36 tháng để kiểm chứng.")
        elif len(df) < 36:
            result.warnings.append("Lịch sử dưới 36 tháng; số kỳ kiểm chứng và độ ổn định mô hình còn hạn chế.")
        expected = pd.date_range(df["Date"].min(), df["Date"].max(), freq="MS")
        gaps = expected.difference(df["Date"])
        if len(gaps):
            result.errors.append("Thiếu tháng lịch sử: " + ", ".join(d.strftime("%m/%Y") for d in gaps[:12]))
    elif ENERGY in df:
        result.warnings.append("Cột điện thương phẩm trong tệp dự báo không được dùng làm đầu vào mô hình.")
    result.frame = df
    return result


def validate_pair(history: pd.DataFrame, future: pd.DataFrame) -> list[str]:
    errors = []
    if (future["Date"] <= history["Date"].max()).any():
        errors.append("Tháng dự báo phải sau tháng lịch sử cuối cùng. Dùng chức năng kiểm chứng để đánh giá lịch sử.")
    last = history["Date"].max()
    horizon = (future[YEAR] - last.year) * 12 + future[MONTH] - last.month
    if (horizon > 12).any():
        errors.append("Bản này hỗ trợ dự báo trong 12 tháng sau mốc lịch sử cuối cùng.")
    for col in [TEMP, HUMIDITY]:
        if (col in history) != (col in future):
            errors.append(f"Cột {col} phải có trong cả hai tệp hoặc bỏ ở cả hai; không tự điền 0.")
    return errors


def demonstration_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic demonstration; never claim these are EVN operational data."""
    dates = pd.date_range("2022-01-01", periods=54, freq="MS")
    temp = 27 + 2.8 * np.sin(2 * np.pi * (dates.month.to_numpy() - 1) / 12)
    frame = pd.DataFrame({YEAR: dates.year, MONTH: dates.month,
                          TEMP: temp.round(1), HUMIDITY: 77 - (temp - 27) * 2,
                          DAYS: dates.days_in_month, OUTAGE: 0.0})
    daily = 12_000_000 * np.exp(np.arange(len(dates)) * 0.003) * (1 + 0.035 * (temp - 27))
    frame[ENERGY] = np.round(daily * dates.days_in_month.to_numpy())
    return frame.iloc[:48].copy(), frame.iloc[48:].drop(columns=ENERGY).copy()
