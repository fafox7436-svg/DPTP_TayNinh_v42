"""Cấu hình nguyên bản từ chương trình của tác giả Lê Minh Trí.

Các số ngày nghỉ là giả định đầu vào và có thể được sửa trên giao diện,
không phải lịch nghỉ được xác minh tự động.
"""

DEFAULT_HOLIDAYS = [
    {"Năm": 2023, "Tháng": 1, "Tết Âm": 7, "Lễ Nhỏ": 1},
    {"Năm": 2023, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2023, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 3},
    {"Năm": 2023, "Tháng": 9, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2024, "Tháng": 1, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2024, "Tháng": 2, "Tết Âm": 7, "Lễ Nhỏ": 0},
    {"Năm": 2024, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 3},
    {"Năm": 2024, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2024, "Tháng": 9, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2025, "Tháng": 1, "Tết Âm": 7, "Lễ Nhỏ": 1},
    {"Năm": 2025, "Tháng": 2, "Tết Âm": 2, "Lễ Nhỏ": 0},
    {"Năm": 2025, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2025, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2025, "Tháng": 9, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2026, "Tháng": 1, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2026, "Tháng": 2, "Tết Âm": 7, "Lễ Nhỏ": 0},
    {"Năm": 2026, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 3},
    {"Năm": 2026, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2026, "Tháng": 9, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2027, "Tháng": 1, "Tết Âm": 0, "Lễ Nhỏ": 1},
    {"Năm": 2027, "Tháng": 2, "Tết Âm": 7, "Lễ Nhỏ": 0},
    {"Năm": 2027, "Tháng": 4, "Tết Âm": 0, "Lễ Nhỏ": 2},
    {"Năm": 2027, "Tháng": 5, "Tết Âm": 0, "Lễ Nhỏ": 1},
]

DEFAULT_K = {"T2": 0.95, "T7": 0.94, "CN": 0.79,
             "Le": 0.56, "Tet": 0.40, "CatDien": 0.50}

FEATURES = ["Tháng", "Năm", "Nhiệt độ TB", "Độ ẩm", "Mua_Nong", "Mua_Mua", "Bien_Ngoai_Sinh"]
