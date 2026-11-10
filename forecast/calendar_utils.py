"""Lịch theo ngày và quy đổi ngày tương đương.

Nguyên tắc:
- Nếu có ngày lễ cụ thể cho một tháng, danh sách cụ thể là nguồn ưu tiên và
  thay thế số lượng lễ/Tết dạng tổng hợp của tháng đó.
- Nếu chưa có ngày cụ thể, chương trình dùng số lượng tổng hợp cũ và phân bổ
  vào các ngày T3-T6 trước để giữ tương thích gần nhất với công thức gốc.
- Cắt điện là hệ số bổ sung: k_cuoi = k_loai_ngay * k_cat_dien.
- Nếu chỉ có "Số ngày cắt điện", chương trình phân bổ vào ngày thường trước
  để giữ tương thích với cách tính cũ; muốn xử lý trùng chính xác cần nhập ngày.
"""
from __future__ import annotations

import calendar
import re
from collections.abc import Iterable
from datetime import date, datetime

import pandas as pd

CATEGORY_K_KEYS = {
    "Thuong": None,
    "T2": "T2",
    "T7": "T7",
    "CN": "CN",
    "Le": "Le",
    "Tet": "Tet",
}

CATEGORY_LABELS = {
    "Thuong": "Ngày thường (T3-T6)",
    "T2": "Thứ 2",
    "T7": "Thứ 7",
    "CN": "Chủ Nhật",
    "Le": "Ngày lễ",
    "Tet": "Tết Âm",
}


def _normalize_holiday_type(value: object) -> str | None:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return None
    if "tết" in text or "tet" in text:
        return "Tet"
    if "lễ" in text or "le" in text or "holiday" in text:
        return "Le"
    return None


def parse_date_value(value: object, *, year: int | None = None,
                     month: int | None = None) -> pd.Timestamp | None:
    """Đọc một ngày theo yyyy-mm-dd, dd/mm/yyyy hoặc dd/mm khi biết tháng/năm."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        ts = value.normalize()
    elif isinstance(value, (datetime, date)):
        ts = pd.Timestamp(value).normalize()
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "nat"}:
            return None
        # dd/mm không có năm: chỉ dùng khi caller đã biết tháng/năm.
        m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})", text)
        if m and year is not None:
            d, mo = map(int, m.groups())
            ts = pd.Timestamp(year=year, month=mo, day=d)
        else:
            parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
            if pd.isna(parsed):
                return None
            ts = pd.Timestamp(parsed).normalize()
    if year is not None and ts.year != int(year):
        return None
    if month is not None and ts.month != int(month):
        return None
    return ts


def parse_date_list(value: object, *, year: int | None = None,
                    month: int | None = None) -> list[pd.Timestamp]:
    """Đọc chuỗi nhiều ngày, ví dụ '12/11/2026, 19/11/2026'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        raw_items = list(value)
    else:
        raw_items = re.split(r"[,;\n]+", str(value))
    out: list[pd.Timestamp] = []
    for item in raw_items:
        ts = parse_date_value(item, year=year, month=month)
        if ts is not None and ts not in out:
            out.append(ts)
    return sorted(out)


def normalize_holiday_dates(holiday_dates: pd.DataFrame | None) -> pd.DataFrame:
    """Chuẩn hóa bảng có cột Ngày, Loại thành Ngày + Category."""
    if holiday_dates is None or len(holiday_dates) == 0:
        return pd.DataFrame({"Ngày": pd.Series(dtype="datetime64[ns]"), "Category": pd.Series(dtype="object")})
    df = holiday_dates.copy()
    if "Ngày" not in df.columns or not ({"Loại", "Category"} & set(df.columns)):
        return pd.DataFrame({"Ngày": pd.Series(dtype="datetime64[ns]"), "Category": pd.Series(dtype="object")})
    df["Ngày"] = df["Ngày"].apply(parse_date_value)
    category_source = "Loại" if "Loại" in df.columns else "Category"
    df["Category"] = df[category_source].apply(_normalize_holiday_type)
    df = df.dropna(subset=["Ngày", "Category"]).drop_duplicates(subset=["Ngày"], keep="last")
    return df[["Ngày", "Category"]].sort_values("Ngày").reset_index(drop=True)


def outage_dates_for_month(df_row: pd.Series | dict,
                            outage_dates_map: dict | None = None) -> list[pd.Timestamp] | None:
    y, m = int(df_row["Năm"]), int(df_row["Tháng"])
    if outage_dates_map and (y, m) in outage_dates_map:
        return parse_date_list(outage_dates_map[(y, m)], year=y, month=m)
    for col in ["Ngày cắt điện cụ thể", "Ngay cat dien cu the", "Ngày cắt điện"]:
        if col in df_row and pd.notna(df_row[col]):
            dates = parse_date_list(df_row[col], year=y, month=m)
            if dates:
                return dates
    # None means no explicit information; [] means an explicit, confirmed zero.
    return None


def _weekday_category(ts: pd.Timestamp) -> str:
    wd = ts.weekday()  # Mon=0
    if wd == 0:
        return "T2"
    if wd == 5:
        return "T7"
    if wd == 6:
        return "CN"
    return "Thuong"


def _assign_legacy_holidays(cal: pd.DataFrame, tet_count: int, le_count: int) -> pd.DataFrame:
    """Phân bổ count cũ vào ngày T3-T6 trước để không double-count cuối tuần."""
    cal = cal.copy()
    # Tết có ưu tiên cao hơn Lễ.
    for category, count in [("Tet", max(0, int(tet_count))), ("Le", max(0, int(le_count)))]:
        if count <= 0:
            continue
        # Ưu tiên Thuong, rồi T2, T7, CN; tránh ngày đã đổi thành Tet/Le.
        candidates = []
        for base in ["Thuong", "T2", "T7", "CN"]:
            candidates.extend(cal.index[cal["Category"] == base].tolist())
        for idx in candidates[:count]:
            cal.at[idx, "Category"] = category
            cal.at[idx, "Nguon_Phan_Loai"] = "Tổng hợp cũ"
    return cal


def coefficients_for_month(k_dict: dict | None, year: int, month: int) -> dict | None:
    """Apply explicitly selected year/month overrides, preserving other months."""
    if k_dict is None:
        return None
    result = {k: v for k, v in k_dict.items() if k != '__monthly__'}
    for y, m, changes in k_dict.get('__monthly__', ()):
        if (int(y), int(m)) == (int(year), int(month)):
            result.update(dict(changes))
    return result


def build_month_calendar(year: int, month: int, holidays_map: dict | None = None,
                         holiday_dates: pd.DataFrame | None = None,
                         outage_dates: object = None, outage_count: int = 0,
                         k_dict: dict | None = None) -> pd.DataFrame:
    """Tạo lịch từng ngày, xử lý ưu tiên Tết/Lễ và hệ số cắt điện."""
    year, month = int(year), int(month)
    k_dict = coefficients_for_month(k_dict, year, month)
    n_days = calendar.monthrange(year, month)[1]
    dates = pd.date_range(f"{year:04d}-{month:02d}-01", periods=n_days, freq="D")
    cal = pd.DataFrame({"Ngày": dates})
    cal["Thứ"] = cal["Ngày"].dt.day_name()
    cal["Category"] = cal["Ngày"].apply(_weekday_category)
    cal["Nguon_Phan_Loai"] = "Lịch tuần"

    normalized = normalize_holiday_dates(holiday_dates)
    exact_month = normalized[(normalized["Ngày"].dt.year == year) &
                             (normalized["Ngày"].dt.month == month)]
    if not exact_month.empty:
        # Danh sách ngày cụ thể là nguồn ưu tiên toàn tháng.
        mapping = dict(zip(exact_month["Ngày"], exact_month["Category"]))
        for idx, row in cal.iterrows():
            if row["Ngày"] in mapping:
                cal.at[idx, "Category"] = mapping[row["Ngày"]]
                cal.at[idx, "Nguon_Phan_Loai"] = "Ngày cụ thể"
        holiday_mode = "exact"
    else:
        tet, le = (holidays_map or {}).get((year, month), (0, 0))
        cal = _assign_legacy_holidays(cal, tet, le)
        holiday_mode = "legacy"

    exact_outages = parse_date_list(outage_dates, year=year, month=month)
    explicit_outages = (isinstance(outage_dates, (list, tuple, set)) or
                        (isinstance(outage_dates, str) and bool(outage_dates.strip())))
    cal["Cat_Dien"] = False
    cal["Nguon_Cat_Dien"] = "Không"
    if explicit_outages:
        mask = cal["Ngày"].isin(exact_outages)
        cal.loc[mask, "Cat_Dien"] = True
        cal.loc[mask, "Nguon_Cat_Dien"] = "Ngày cụ thể"
        outage_mode = "exact"
    else:
        count = max(0, min(int(outage_count or 0), n_days))
        if count:
            # Ưu tiên ngày thường để tương thích công thức cũ.
            candidates = []
            for category in ["Thuong", "T2", "T7", "CN", "Le", "Tet"]:
                candidates.extend(cal.index[cal["Category"] == category].tolist())
            for idx in candidates[:count]:
                cal.at[idx, "Cat_Dien"] = True
                cal.at[idx, "Nguon_Cat_Dien"] = "Tổng hợp cũ"
        outage_mode = "legacy"

    if k_dict is not None:
        def base_k(category: str) -> float:
            key = CATEGORY_K_KEYS[category]
            return 1.0 if key is None else float(k_dict[key])
        cal["k_Loai_Ngay"] = cal["Category"].apply(base_k)
        cal["k_Cat_Dien"] = cal["Cat_Dien"].map({True: float(k_dict["CatDien"]), False: 1.0})
        cal["k_Cuoi"] = cal["k_Loai_Ngay"] * cal["k_Cat_Dien"]
    cal.attrs["holiday_mode"] = holiday_mode
    cal.attrs["outage_mode"] = outage_mode
    return cal


def summarize_month_calendar(cal: pd.DataFrame) -> dict:
    counts = cal["Category"].value_counts().to_dict()
    result = {
        "So_Ngay_Thuong": int(counts.get("Thuong", 0)),
        "So_Ngay_T2": int(counts.get("T2", 0)),
        "So_Ngay_T7": int(counts.get("T7", 0)),
        "So_Ngay_CN": int(counts.get("CN", 0)),
        "So_Ngay_Tet_Am": int(counts.get("Tet", 0)),
        "So_Ngay_Le_Nho": int(counts.get("Le", 0)),
        "So_Ngay_Cat_Dien": int(cal["Cat_Dien"].sum()),
    }
    if "k_Cuoi" in cal.columns:
        result["Ngay_Tuong_Duong"] = float(cal["k_Cuoi"].sum())
    return result


def calendar_detail_for_dataframe(df: pd.DataFrame, holidays_map: dict,
                                  holiday_dates: pd.DataFrame | None,
                                  outage_dates_map: dict | None,
                                  k_dict: dict) -> pd.DataFrame:
    frames = []
    for _, row in df.iterrows():
        y, m = int(row["Năm"]), int(row["Tháng"])
        exact_out = outage_dates_for_month(row, outage_dates_map)
        count = int(row.get("Số ngày cắt điện", 0) or 0)
        cal = build_month_calendar(y, m, holidays_map, holiday_dates,
                                   exact_out, count, k_dict)
        cal.insert(0, "Năm", y)
        cal.insert(1, "Tháng", m)
        cal["Loại ngày"] = cal["Category"].map(CATEGORY_LABELS)
        frames.append(cal)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
