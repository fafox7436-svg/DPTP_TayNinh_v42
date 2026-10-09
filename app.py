from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ai_assistant import AIError, assess, list_models
from data_io import (DAYS, ENERGY, HUMIDITY, MONTH, OUTAGE, TEMP, YEAR, InputError,
                     demonstration_data, inspect_workbook, normalize, read_table,
                     validate_monthly, validate_pair)
from forecasting import (DAY_TYPES, NAIVE, NN, RF, TREND, VERSION, XGB, Factors,
                         available_models, default_scenarios, finalize_scenarios,
                         fingerprint, predict_models, recommend, rolling_backtest,
                         validate_holidays)
from reporting import excel_bytes, input_template, restore_scenarios, scenario_json


st.set_page_config(page_title="Dự báo điện thương phẩm", page_icon="⚡", layout="wide")
st.markdown("""<style>
  .stApp {background: #f6f8fc;}
  .block-container {max-width: 1480px; padding-top: 2rem;}
  h1,h2,h3 {color: #123d64;}
  [data-testid="stMetric"] {background:white; border:1px solid #e0e7ef;
      border-radius:12px; padding:16px;}
  [data-testid="stSidebar"] {background: #edf2f8;}
  [data-testid="stTabs"] {margin-top: 1rem;}
  button {min-height: 2.6rem;}
  @media(max-width: 700px) {.block-container {padding: 1rem;}}
</style>""", unsafe_allow_html=True)


def now_label() -> str:
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(timespec="seconds")


def secret(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return ""


@st.cache_data(show_spinner=False)
def template_bytes():
    # A public blank template only. User workbooks and API keys are not globally cached.
    return input_template()


def upload_table(uploaded, label: str, widget_key: str) -> pd.DataFrame | None:
    if uploaded is None:
        return None
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    cache = st.session_state.setdefault("input_cache", {})
    try:
        candidates_key = (digest, "sheets")
        if candidates_key not in cache:
            if len(cache) > 16:
                cache.clear()
            cache[candidates_key] = inspect_workbook(data)
        candidates = cache[candidates_key]
        preferred = "lich" if widget_key == "history" else "du bao"
        index = next((i for i, (s, _) in enumerate(candidates) if preferred in normalize(s)), 0)
        chosen = st.selectbox(f"Bảng {label}", candidates, index=index,
                              format_func=lambda v: f"{v[0]} · tiêu đề dòng {v[1] + 1}",
                              key=f"sheet_{widget_key}_{digest}")
        table_key = (digest, *chosen)
        if table_key not in cache:
            cache[table_key] = read_table(data, *chosen)
        return cache[table_key].copy()
    except InputError as exc:
        st.error(str(exc))
        return None


def render_ai(area: str, history: pd.DataFrame | None, future: pd.DataFrame | None,
              unit: str, run: dict | None, fresh: bool):
    st.subheader("Phân tích thông tin tham khảo")
    st.caption("Chỉ nội dung dán, khu vực và tháng được gửi đến nhà cung cấp khi bấm Phân tích. "
               "Hai tệp Excel và khóa API không được đưa vào nội dung yêu cầu.")
    left, right = st.columns([1, 2])
    with left:
        provider = st.selectbox("Nhà cung cấp", ["OpenAI", "Google Gemini"], key="provider")
        env_key = "OPENAI_API_KEY" if provider == "OpenAI" else "GEMINI_API_KEY"
        server_key = secret(env_key)
        entered_key = st.text_input("API key dùng trong phiên", type="password", key=f"key_{provider}",
                                    help=f"Để trống nếu máy chạy ứng dụng đã cấu hình {env_key}.")
        api_key = entered_key or server_key
        if server_key and not entered_key:
            st.caption("Đang dùng khóa đã cấu hình trên máy chạy ứng dụng.")
        default_model = secret("OPENAI_MODEL" if provider == "OpenAI" else "GEMINI_MODEL")
        if provider == "OpenAI" and not default_model:
            default_model = "gpt-4.1-mini"
        model = st.text_input("Mã mô hình", value=default_model, key=f"model_{provider}",
                              placeholder="Nhập mã Gemini Flash đang dùng" if provider != "OpenAI" else "Mã mô hình hỗ trợ Responses")
        if st.button("Lấy danh sách mô hình", disabled=not api_key):
            try:
                with st.spinner("Đang kiểm tra danh sách được cấp quyền…"):
                    st.session_state[f"models_{provider}"] = list_models(provider, api_key)
            except AIError as exc:
                st.error(str(exc))
        if f"models_{provider}" in st.session_state:
            with st.expander("Mã mô hình từ tài khoản"):
                st.code("\n".join(st.session_state[f"models_{provider}"]))
                st.caption("Sao chép mã phù hợp vào ô trên. Có trong danh sách không đồng nghĩa hỗ trợ mọi chế độ.")
        if future is not None and not future.empty:
            date = st.selectbox("Tháng phân tích", future["Date"].tolist(),
                                format_func=lambda d: d.strftime("%m/%Y"))
            year, month = date.year, date.month
        else:
            year = int(st.number_input("Năm phân tích", 1900, 2200, datetime.now().year))
            month = int(st.number_input("Tháng phân tích", 1, 12, datetime.now().month))
        st.caption("OpenAI dùng khóa API riêng với Gemini. Bản này kết nối qua API key; "
                   "chưa tích hợp đăng nhập bằng tài khoản ChatGPT.")
    with right:
        article = st.text_area("Nội dung bài báo / thông tin cần phân tích", height=230, max_chars=16000,
                               key="article", placeholder="Dán nội dung có thời điểm, khu vực và nguồn rõ ràng…")
        source = st.text_input("Nguồn / đường dẫn để ghi nhận", key="article_source")
        st.caption("Đường dẫn dùng để ghi nguồn. Ứng dụng phân tích phần nội dung đã dán, không tự truy cập URL.")
        request_signature = fingerprint(provider, model, area, year, month, article, source)
        if st.button("Phân tích nội dung đã chọn", type="primary", disabled=not (api_key and model and article)):
            try:
                with st.spinner("Đang nhận phân tích từ AI…"):
                    value = assess(provider, api_key, model, article, area, int(year), int(month))
                st.session_state["assessment"] = {
                    "signature": request_signature, "value": value.model_dump(), "source": source,
                    "provider": provider, "model": model, "time": now_label(),
                    "year": year, "month": month,
                }
            except AIError as exc:
                st.error(str(exc))
        saved = st.session_state.get("assessment")
        if not saved:
            return
        if saved["signature"] != request_signature:
            st.info("Nội dung hoặc kỳ phân tích đã đổi. Bấm Phân tích để có kết quả phù hợp.")
            return
        value = saved["value"]
        st.write(value["summary"])
        st.caption(f"{saved['provider']} · {saved['model']} · {saved['time']}")
        if value["basis"]:
            st.write("Căn cứ:")
            for line in value["basis"]:
                st.write("• " + line)
        if value["missing_information"]:
            st.info("Thông tin cần bổ sung: " + "; ".join(value["missing_information"]))
        pct = value["yoy_percent"]
        if pct is None:
            st.warning("Chưa có căn cứ để đưa ra tỷ lệ tăng/giảm điện thương phẩm cùng kỳ.")
            return
        st.metric("Tỷ lệ cùng kỳ trích từ tư liệu", f"{pct:+.2f}%")
        st.write("Trích đoạn nguồn: " + value["evidence_quote"])
        st.caption("Cần đối chiếu phạm vi và kỳ số liệu của trích đoạn. Kết quả này chưa tự thay đổi dự báo.")
        if history is None:
            return
        base = history[(history[YEAR] == year - 1) & (history[MONTH] == month)]
        if base.empty:
            st.info("Chưa có điện thương phẩm cùng kỳ để quy đổi tỷ lệ.")
            return
        target = float(base.iloc[0][ENERGY]) * (1 + pct / 100)
        st.metric(f"Sản lượng tham khảo ({unit})", f"{target:,.2f}")
        if not run or not fresh:
            return
        month_label = f"{month:02d}/{year}"
        current = st.session_state["scenario_current"].copy()
        if month_label not in current["Tháng"].tolist():
            return
        if st.button("Dùng mức tham khảo cho phương án tháng này"):
            try:
                model_choice = st.session_state.get(f"base_model_{run['signature']}", run["recommended"])
                temporary = current.copy()
                row_index = temporary.index[temporary["Tháng"] == month_label][0]
                temporary.loc[row_index, "Điều chỉnh thêm (%)"] = 0.0
                temporary.loc[row_index, "Lý do"] = "Đối chiếu mức tham khảo AI"
                interim, _ = finalize_scenarios(run["predictions"], temporary, model_choice,
                                                run["factors"], run["holidays"])
                calendar_q = float(interim.loc[interim["Tháng"] == month_label, "Sản lượng chốt"].iloc[0])
                if calendar_q <= 0:
                    raise InputError("Không thể quy đổi từ phương án có sản lượng bằng 0.")
                adjustment = (target / calendar_q - 1) * 100
                if not -100 <= adjustment <= 100:
                    raise InputError("Tỷ lệ cần điều chỉnh vượt phạm vi −100% đến +100%; cần rà soát thủ công.")
                current.loc[row_index, "Điều chỉnh thêm (%)"] = adjustment
                current.loc[row_index, "Lý do"] = (f"Tham khảo {provider}/{model}: {pct:+.2f}% cùng kỳ. "
                                                   f"Nguồn: {source or 'nội dung đã dán'}; {saved['time']}. "
                                                   f"Căn cứ: {value['evidence_quote']}")
                st.session_state["scenario_base"] = current
                st.session_state["scenario_current"] = current.copy()
                st.session_state["editor_epoch"] += 1
                st.session_state["notice"] = "Đã đưa mức tham khảo vào phương án; xem lại tại bước 3."
                st.rerun()
            except InputError as exc:
                st.error(str(exc))


def render_scenarios(run: dict, fresh: bool, unit: str, operator: str):
    if not run:
        st.info("Chạy dự báo ở bước 2 để lập phương án.")
        return
    if not fresh:
        st.warning("Dữ liệu hoặc cấu hình đã đổi. Chạy lại dự báo trước khi chốt hay xuất kết quả.")
        return
    signature = run["signature"]
    st.subheader("Tinh chỉnh và giải trình phương án")
    choices = [m for m in run["models"] if np.isfinite(run["predictions"][m]).all()]
    if not choices:
        st.error("Không có mô hình hợp lệ để lập phương án.")
        return
    preferred = run["recommended"] if run["recommended"] in choices else choices[0]
    model = st.selectbox("Mô hình làm cơ sở", choices, index=choices.index(preferred), key=f"base_model_{signature}")
    st.caption("Hệ số áp dụng chung khi đổi mô hình. Sản lượng được tính lại từ mô hình vừa chọn. "
               "Mỗi thay đổi cần lý do; các tháng được lưu cùng một bảng trong phiên.")
    with st.expander("Mở lại phương án đã tải xuống"):
        uploaded = st.file_uploader("Tệp phương án JSON", type=["json"], key=f"restore_{signature}")
        if st.button("Mở phương án", disabled=uploaded is None):
            try:
                saved_model, frame = restore_scenarios(uploaded.getvalue(), signature,
                                                       default_scenarios(run["predictions"], run["factors"]), choices)
                finalize_scenarios(run["predictions"], frame, saved_model, run["factors"], run["holidays"])
                st.session_state["scenario_base"] = frame
                st.session_state["scenario_current"] = frame.copy()
                st.session_state["pending_model"] = saved_model
                st.session_state["editor_epoch"] += 1
                st.session_state["notice"] = "Đã mở phương án đã lưu."
                st.rerun()
            except InputError as exc:
                st.error(str(exc))
    factors_columns = {
        "monday": "k thứ Hai", "saturday": "k thứ Bảy", "sunday": "k Chủ nhật",
        "holiday": "k lễ", "tet": "k Tết", "outage": "k cắt điện",
    }
    config = {key: st.column_config.NumberColumn(label, min_value=0.0,
              max_value=1.0 if key == "outage" else 3.0, step=0.01, required=True)
              for key, label in factors_columns.items()}
    config.update({OUTAGE: st.column_config.NumberColumn("Ngày cắt điện tương đương", min_value=0.0, max_value=31.0, step=0.1, required=True),
                   "Điều chỉnh thêm (%)": st.column_config.NumberColumn(min_value=-100.0, max_value=100.0, step=0.1, required=True),
                   "Lý do": st.column_config.TextColumn(width="large")})
    edited = st.data_editor(st.session_state["scenario_base"], column_config=config,
                            disabled=["Tháng"], hide_index=True, width="stretch",
                            key=f"scenario_editor_{signature}_{st.session_state['editor_epoch']}")
    st.session_state["scenario_current"] = edited.copy()
    try:
        final, allocation = finalize_scenarios(run["predictions"], edited, model, run["factors"], run["holidays"])
    except InputError as exc:
        st.warning(str(exc))
        return
    original_total, final_total = final["Dự báo gốc"].sum(), final["Sản lượng chốt"].sum()
    a, b, c = st.columns(3)
    a.metric(f"Tổng dự báo gốc ({unit})", f"{original_total:,.2f}")
    b.metric(f"Tổng phương án ({unit})", f"{final_total:,.2f}",
             f"{(final_total / original_total - 1) * 100:+.2f}%" if original_total else None)
    c.metric("Số tháng", len(final))
    st.dataframe(final, hide_index=True, width="stretch")
    st.caption("Sản lượng chốt = Dự báo gốc + Tác động cơ cấu ngày + Tác động điều chỉnh thêm. "
               "Điều chỉnh thêm áp dụng sau khi thay cơ cấu ngày; không phải tỷ lệ cùng kỳ.")
    chart = final.melt(id_vars="Tháng", value_vars=["Dự báo gốc", "Sản lượng chốt"], var_name="Phương án", value_name=unit)
    st.plotly_chart(px.bar(chart, x="Tháng", y=unit, color="Phương án", barmode="group",
                           color_discrete_sequence=["#8baac5", "#047c83"]), width="stretch")
    with st.expander("Phân bổ ngày và đối chiếu tổng"):
        st.dataframe(allocation, hide_index=True, width="stretch")
        st.caption("Sản lượng/ngày là mức bình quân giả định. Tổng nhóm ngày = số ngày × sản lượng/ngày. "
                   "Ảnh hưởng cắt điện được phân bổ đều theo tỷ trọng ngày; đây chưa phải dự báo từng ngày vận hành.")
    if st.button("Ghi nhận phiên bản phương án"):
        st.session_state.setdefault("audit", []).append({
            "Thời điểm": now_label(), "Người lập": operator, "Mô hình": model,
            "Tổng sản lượng": float(final_total),
            "Chi tiết": edited.to_json(orient="records", force_ascii=False),
        })
        st.success("Đã ghi nhận trong phiên; tải Excel để lưu hồ sơ.")
    metadata = {"Phiên bản": VERSION, "Lần chạy": run["time"], "Mã lần chạy": signature,
                "Khu vực": run["area"], "Đơn vị": unit, "Người lập": operator,
                "Dữ liệu minh họa": run["demo"], "Mô hình chốt": model,
                "Hệ số khi huấn luyện": json.dumps(asdict(run["factors"]), ensure_ascii=False),
                "Seed": run["seed"], "Chân trời kiểm chứng": run["horizon"],
                "Mô hình có WAPE thấp nhất": run["recommended"] or "Chưa đủ kỳ để lựa chọn",
                "Giả định lịch nghỉ": run["calendar_note"],
                "Lưu ý kiểm chứng": "Có điều kiện theo thời tiết đã biết của tháng kiểm chứng; không đo sai số dự báo thời tiết.",
                "Thư viện": json.dumps(run["versions"]),
                "Cảnh báo": "; ".join(run["notes"])}
    sheets = {"Phuong_an": final, "Phan_bo_ngay": allocation, "Du_bao_mo_hinh": run["predictions"],
              "Sai_so": run["scores"], "Kiem_chung": run["backtest"], "He_so_phuong_an": edited,
              "Cau_hinh": pd.DataFrame({"Thuộc tính": metadata.keys(), "Giá trị": [str(v) for v in metadata.values()]}),
              "Lich_nghi": pd.DataFrame(list(run["holidays"].items()), columns=["Ngày", "Loại"]),
              "Lich_su": run["history"], "Dau_vao": run["future"],
              "Nhat_ky": pd.DataFrame(st.session_state.get("audit", []))}
    a, b = st.columns(2)
    with a:
        st.download_button("Tải hồ sơ Excel", excel_bytes(sheets),
                            file_name=f"du_bao_dien_{signature[:8]}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width="stretch")
    with b:
        st.download_button("Tải phương án để mở lại (JSON)", scenario_json(signature, model, edited),
                            file_name=f"phuong_an_{signature[:8]}.json", mime="application/json",
                            width="stretch")
    st.caption("Excel lưu kết quả và hồ sơ tính toán. JSON lưu hệ số để mở lại sau khi nạp đúng dữ liệu và cấu hình. "
               "Ứng dụng không tự lưu dữ liệu vào cơ sở dữ liệu; đóng phiên có thể mất chỉnh sửa chưa tải xuống.")


def main():
    with st.sidebar:
        st.title("⚡ Dự báo điện")
        area = st.text_input("Khu vực", value="Tây Ninh")
        operator = st.text_input("Người lập phương án", value="")
        unit = st.selectbox("Đơn vị số liệu trong Excel", ["kWh", "triệu kWh"])
        demo = st.checkbox("Dùng dữ liệu minh họa", value=False, key="use_demo")
        decimal_label = st.selectbox("Dấu thập phân của ô dạng chữ", ["Dấu chấm: 1,234.56", "Dấu phẩy: 1.234,56"])
        decimal = "." if decimal_label.startswith("Dấu chấm") else ","
        st.caption("Ô số Excel giữ nguyên. Chọn đúng định dạng đối với số lưu dưới dạng chữ.")
        with st.expander("Hệ số ngày khi huấn luyện"):
            st.caption("Giá trị khởi đầu từ mã gốc; chưa phải hệ số đã hiệu chỉnh theo dữ liệu thực tế.")
            defaults = asdict(Factors())
            labels = {"monday": "Thứ Hai", "saturday": "Thứ Bảy", "sunday": "Chủ nhật",
                      "holiday": "Ngày lễ", "tet": "Tết", "outage": "Cắt điện – tỷ lệ còn lại"}
            values = {key: st.number_input(labels[key], min_value=0.0,
                      max_value=1.0 if key == "outage" else 3.0, value=value, step=0.01,
                      key=f"factor_{key}") for key, value in defaults.items()}
            factors = Factors(**values)
            st.caption("Ngày thường (thứ Ba–thứ Sáu, không phải lễ/Tết) có hệ số 1.")
        with st.expander("Thiết lập kiểm chứng"):
            horizon = int(st.number_input("Chân trời dự báo kiểm chứng (tháng)", 1, 12, 1, key="bt_horizon"))
            periods = int(st.number_input("Số kỳ kiểm chứng gần nhất", 3, 24, 6, key="bt_periods"))
            seed = int(st.number_input("Seed cố định để tái lập", 0, 2147483647, 42))
            st.caption("So sánh theo thời gian: mô hình chỉ học từ các tháng trước mốc dự báo.")
        st.download_button("Tải tệp Excel mẫu", template_bytes(), file_name="mau_du_lieu_du_bao.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width="stretch")
        st.caption(f"Phiên bản {VERSION} · phát triển từ mã nguồn Lê Minh Trí")

    st.title("Dự báo điện thương phẩm theo tháng")
    st.write(f"{area} · Nhập dữ liệu → kiểm chứng mô hình → lập phương án → xuất hồ sơ")
    st.caption("Đại lượng dự báo là điện năng; không phải công suất phụ tải đỉnh (MW).")
    if demo:
        st.warning("Đang dùng dữ liệu tổng hợp giả lập để thử thao tác. Không phải số liệu của EVN hoặc của tỉnh.")
    notice = st.session_state.pop("notice", None)
    if notice:
        st.success(notice)
    data_tab, forecast_tab, scenario_tab, ai_tab = st.tabs([
        "1. Dữ liệu và lịch nghỉ", "2. Kiểm chứng và dự báo", "3. Chốt phương án", "4. Trợ lý AI"])
    history, future, signature = None, None, None
    input_errors, notes, holidays, calendar_note = [], [], {}, ""
    reviewed = False
    names = available_models()
    versions = {p: version(p) for p in ["pandas", "numpy", "scikit-learn"]}
    if XGB in names:
        versions["xgboost"] = version("xgboost")
    with data_tab:
        st.subheader("Dữ liệu tổng hợp theo tháng")
        left, right = st.columns(2)
        if demo:
            hist_raw, future_raw = demonstration_data()
            # Demo values are originally kWh. Keep units truthful if the user changes the selector.
            if unit == "triệu kWh":
                hist_raw[ENERGY] /= 1_000_000
        else:
            with left:
                fh = st.file_uploader("Tệp lịch sử", type=["xlsx", "xls"], key="file_history")
                hist_raw = upload_table(fh, "lịch sử", "history")
            with right:
                ff = st.file_uploader("Tệp dự báo", type=["xlsx", "xls"], key="file_future")
                future_raw = upload_table(ff, "dự báo", "future")
        if hist_raw is not None:
            checked = validate_monthly(hist_raw, history=True, decimal=decimal)
            input_errors += ["Lịch sử: " + e for e in checked.errors]
            notes += checked.warnings
            if not checked.errors:
                history = checked.frame
                left.metric("Số tháng lịch sử", len(history))
                with st.expander("Xem toàn bộ dữ liệu lịch sử"):
                    st.dataframe(history, hide_index=True, width="stretch")
        if future_raw is not None:
            checked = validate_monthly(future_raw, history=False, decimal=decimal)
            input_errors += ["Dự báo: " + e for e in checked.errors]
            notes += checked.warnings
            if not checked.errors:
                future = checked.frame
                right.metric("Số tháng cần dự báo", len(future))
                with st.expander("Xem toàn bộ đầu vào dự báo", expanded=True):
                    st.dataframe(future, hide_index=True, width="stretch")
        if history is not None and future is not None:
            input_errors += validate_pair(history, future)
            for col in [TEMP, HUMIDITY]:
                if col in history and col in future and not future[col].between(history[col].min(), history[col].max()).all():
                    notes.append(f"{col} của một số tháng dự báo nằm ngoài khoảng lịch sử; đây là trường hợp ngoại suy.")
        for message in input_errors:
            st.error(message)
        for message in list(dict.fromkeys(notes)):
            st.warning(message)
        if hist_raw is None or future_raw is None:
            st.info("Tải hai bảng theo tháng hoặc bật Dùng dữ liệu minh họa ở thanh bên để thử chương trình.")
        st.divider()
        st.subheader("Ngày lễ, Tết và nghỉ bù áp dụng")
        st.caption("Khai báo ngày cụ thể cho cả lịch sử và tương lai. Một ngày chỉ thuộc một loại; lễ/Tết thay thế thứ trong tuần.")
        holiday_upload = st.file_uploader("Nhập lịch nghỉ (không bắt buộc)", type=["xlsx", "csv"], key="holiday_file")
        holiday_raw = pd.DataFrame({"Ngày": pd.Series(dtype="datetime64[ns]"), "Loại": pd.Series(dtype="str")})
        holiday_import_ok = True
        holiday_key = "manual"
        if holiday_upload:
            try:
                content = holiday_upload.getvalue()
                if len(content) > 2_000_000:
                    raise InputError("Tệp lịch nghỉ vượt 2 MB.")
                holiday_key = hashlib.sha256(content).hexdigest()
                holiday_raw = (pd.read_csv(io.BytesIO(content)) if holiday_upload.name.lower().endswith(".csv")
                               else pd.read_excel(io.BytesIO(content), sheet_name="Lich_nghi"))
                holiday_raw.columns = holiday_raw.columns.astype(str).str.strip()
                if not {"Ngày", "Loại"}.issubset(holiday_raw):
                    raise InputError("Tệp lịch nghỉ cần cột Ngày và Loại; Excel phải có sheet Lich_nghi.")
                holiday_raw = holiday_raw[["Ngày", "Loại"]].dropna(how="all")
                validate_holidays(holiday_raw)
                holiday_raw["Ngày"] = pd.to_datetime(holiday_raw["Ngày"])
            except Exception:
                st.error("Không đọc được lịch nghỉ. Dùng sheet Lich_nghi với cột Ngày (YYYY-MM-DD hoặc ô Date) và Loại (Lễ/Tết).")
                holiday_import_ok = False
        edited_holidays = st.data_editor(holiday_raw, num_rows="dynamic", hide_index=True,
                                         column_config={"Ngày": st.column_config.DateColumn(format="DD/MM/YYYY", required=True),
                                                        "Loại": st.column_config.SelectboxColumn(options=["Lễ", "Tết"], required=True)},
                                         key=f"holiday_editor_{holiday_key}", width="stretch")
        try:
            holidays = validate_holidays(edited_holidays)
        except InputError as exc:
            input_errors.append(str(exc))
            st.error(str(exc))
        if not holiday_import_ok:
            input_errors.append("Tệp lịch nghỉ chưa hợp lệ.")
        if holidays:
            calendar_note = "Đã khai báo ngày cụ thể; các ngày không khai báo được tính theo thứ trong tuần."
            st.caption(f"Đã khai báo {len(holidays)} ngày. Phần mềm không tự xác định lịch này có đầy đủ hay không.")
        else:
            calendar_note = "Chưa khai báo lễ/Tết: chỉ tính theo thứ trong tuần."
            st.info(calendar_note)
        if demo:
            reviewed = True
        else:
            reviewed = st.checkbox("Tôi đã rà soát lịch nghỉ cho cả hai tệp, hoặc chấp nhận giả định chưa xét lễ/Tết.",
                                    key="calendar_reviewed")
        with st.expander("Hiểu đúng thông tin cắt điện"):
            st.write("Số ngày cắt điện tương đương phản ánh phần phạm vi dự báo bị ảnh hưởng và thời lượng, "
                     "không phải số vụ cắt điện. Hệ số cắt điện là tỷ lệ sản lượng còn lại trong phần thời gian đó.")
            st.write("Nếu có điện năng không cung cấp dự kiến, nên dùng dữ liệu này để hiệu chỉnh phương án. "
                     "Không suy diễn một lần cắt điện cục bộ thành một ngày cắt điện toàn tỉnh.")

    ready = history is not None and future is not None and not input_errors and reviewed and bool(area.strip())
    if ready:
        signature = fingerprint(history, future, holidays, factors,
                                {"seed": seed, "horizon": horizon, "periods": periods,
                                 "unit": unit, "area": area, "demo": demo, "models": names, "versions": versions})
    with forecast_tab:
        st.subheader("Đánh giá trước khi chọn mô hình")
        st.write("Mỗi kỳ kiểm chứng sử dụng ít nhất 24 tháng trước mốc dự báo để huấn luyện. "
                 "Mô hình có WAPE thấp nhất trên cùng các kỳ hợp lệ được gợi ý làm cơ sở.")
        st.caption("MAE/RMSE có cùng đơn vị với sản lượng; MAPE/WAPE tính bằng %. "
                   "Chỉ số thấp trên lịch sử không bảo đảm độ chính xác tương lai.")
        if not ready:
            st.info("Hoàn tất dữ liệu và xác nhận giả định lịch nghỉ tại bước 1.")
        if XGB not in names:
            st.warning("Chưa cài XGBoost; ứng dụng vẫn chạy các mô hình còn lại. Cài requirements.txt để có đủ mô hình.")
        if st.button("Kiểm chứng và chạy dự báo", type="primary", disabled=not ready, key="run_forecast"):
            try:
                bar = st.progress(0.0, text="Đang kiểm chứng theo thời gian…")
                backtest, scores, bt_notes = rolling_backtest(history, holidays, factors, seed, horizon, periods,
                                                             names, progress=lambda p: bar.progress(p * 0.85))
                predictions, pred_notes = predict_models(history, future, holidays, factors, seed, names)
                best = recommend(scores)
                usable = [m for m in names if np.isfinite(predictions[m]).all()]
                if not usable:
                    raise InputError("Không mô hình nào tạo được dự báo hợp lệ; cần rà soát dữ liệu.")
                run = {"signature": signature, "time": now_label(), "history": history.copy(),
                       "future": future.copy(), "holidays": holidays.copy(), "factors": factors,
                       "predictions": predictions, "backtest": backtest, "scores": scores,
                       "recommended": best, "models": names, "seed": seed, "horizon": horizon,
                       "area": area, "unit": unit, "demo": demo, "calendar_note": calendar_note,
                       "versions": versions, "notes": list(dict.fromkeys(notes + bt_notes + pred_notes))}
                if st.session_state.get("run", {}).get("signature") != signature:
                    st.session_state["scenario_base"] = default_scenarios(predictions, factors)
                    st.session_state["scenario_current"] = st.session_state["scenario_base"].copy()
                    st.session_state["editor_epoch"] = 0
                    st.session_state["audit"] = []
                st.session_state["run"] = run
                bar.progress(1.0, text="Đã hoàn thành.")
                st.success("Đã dự báo xong. Xem sai số dưới đây và lập phương án tại bước 3.")
            except InputError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Không hoàn tất tính toán ({type(exc).__name__}). Kiểm tra dữ liệu, cấu hình và thư viện.")
        run = st.session_state.get("run")
        fresh = bool(run and signature and run["signature"] == signature)
        if run and not fresh:
            st.warning("Kết quả lần trước đã cũ so với dữ liệu/cấu hình hiện tại. Cần chạy lại trước khi sử dụng.")
        elif fresh:
            for message in run["notes"]:
                st.warning(message)
            if run["recommended"]:
                st.success(f"WAPE thấp nhất trên các kỳ kiểm chứng: {run['recommended']}.")
            else:
                st.warning("Chưa đủ ít nhất 3 kỳ hợp lệ để gợi ý mô hình. Có thể xem dự báo, nhưng cần thẩm định thủ công.")
            st.dataframe(run["scores"], hide_index=True, width="stretch")
            st.caption(f"Chân trời kiểm chứng: {run['horizon']} tháng. Thời tiết của tháng kiểm chứng lấy từ số liệu đã biết; "
                       "các chỉ số này chưa bao gồm sai số dự báo thời tiết tại thời điểm lập kế hoạch.")
            maximum_horizon = int(((future[YEAR] - history['Date'].max().year) * 12 + future[MONTH] - history['Date'].max().month).max())
            if maximum_horizon != run["horizon"]:
                st.info(f"Đầu ra có tháng xa tới {maximum_horizon} tháng; sai số đang đo ở chân trời {run['horizon']} tháng. "
                        "Có thể đổi chân trời kiểm chứng ở thanh bên nếu lịch sử đủ dài.")
            plot_frame = run["predictions"].melt(id_vars="Date", value_vars=run["models"],
                                                 var_name="Mô hình", value_name=unit)
            st.plotly_chart(px.line(plot_frame, x="Date", y=unit, color="Mô hình", markers=True), width="stretch")
            st.dataframe(run["predictions"], hide_index=True, width="stretch")
            with st.expander("Các kỳ kiểm chứng và mốc huấn luyện"):
                st.dataframe(run["backtest"], hide_index=True, width="stretch")
            st.caption("Xu hướng và Cùng kỳ là các mô hình đối chiếu. Chênh lệch giữa các mô hình không phải khoảng tin cậy thống kê.")
    with scenario_tab:
        if run and "pending_model" in st.session_state:
            st.session_state[f"base_model_{run['signature']}"] = st.session_state.pop("pending_model")
        render_scenarios(run, fresh, unit, operator)
    with ai_tab:
        render_ai(area, history, future, unit, run, fresh)


if __name__ == "__main__":
    main()
