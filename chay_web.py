"""Tương thích điểm khởi động Streamlit cũ với app.py chạy ở cấp mô-đun."""
from pathlib import Path
from runpy import run_path

run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
