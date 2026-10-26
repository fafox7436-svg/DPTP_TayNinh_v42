"""Nối giao diện nâng cấp với hai hàm tính toán gốc, không đổi tham số học máy."""
from __future__ import annotations
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from data_io import DAYS, ENERGY, MONTH, OUTAGE, YEAR, InputError
from loi_du_bao_goc import chay_mo_hinh_goc, tao_dac_trung

VERSION = "2.1.0-loi-goc"
TREND, NAIVE = "Xu hướng", "Cùng kỳ (chuẩn hóa ngày)"
RF, XGB, NN = "Random Forest", "XGBoost", "Neural Network"
DAY_TYPES = ["Ngày thường", "Thứ Hai", "Thứ Bảy", "Chủ nhật", "Lễ", "Tết", "Cắt điện"]
LEGACY_COLUMNS = ["So_Ngay_Thuong", "So_Ngay_T2", "So_Ngay_T7", "So_Ngay_CN", "So_Ngay_Le_Nho", "So_Ngay_Tet_Am", "So_Ngay_Cat_Dien"]
HOLIDAY_COLUMNS = [YEAR, MONTH, "Tết Âm", "Lễ Nhỏ"]

@dataclass(frozen=True)
class Factors:
    monday: float = 0.95
    saturday: float = 0.94
    sunday: float = 0.79
    holiday: float = 0.56
    tet: float = 0.40
    outage: float = 0.50

    def checked(self):
        if not all(np.isfinite(v) and v >= 0 for v in asdict(self).values()):
            raise InputError("Hệ số ngày phải là số hữu hạn, không âm.")
        return self

    def legacy(self):
        self.checked()
        return dict(zip(["T2", "T7", "CN", "Le", "Tet", "CatDien"], asdict(self).values()))

    def day_weights(self):
        self.checked()
        return np.array([1.0, self.monday, self.saturday, self.sunday, self.holiday, self.tet, self.outage])

def validate_holidays(raw):
    if raw.empty:
        return {}
    if not set(HOLIDAY_COLUMNS).issubset(raw.columns):
        raise InputError("Lịch nghỉ cần bốn cột Năm, Tháng, Tết Âm, Lễ Nhỏ.")
    output = {}
    for _, row in raw.dropna(subset=[YEAR, MONTH]).fillna(0).iterrows():
        try:
            vals = [float(row[k]) for k in HOLIDAY_COLUMNS]
            if not all(np.isfinite(v) for v in vals):
                raise ValueError
            year, month, tet, holiday = (int(v) for v in vals)
        except (ValueError, TypeError, OverflowError) as exc:
            raise InputError("Lịch nghỉ có ô không phải số hữu hạn.") from exc
        if not 1900 <= year <= 2200 or not 1 <= month <= 12 or min(tet, holiday) < 0:
            raise InputError("Kiểm tra năm, tháng và số ngày lễ/Tết không âm.")
        output[(year, month)] = (tet, holiday)  # Dòng cuối thắng, đúng mã gốc.
    return output

def holidays_frame(holidays):
    return pd.DataFrame([(y, m, tet, le) for (y, m), (tet, le) in holidays.items()], columns=HOLIDAY_COLUMNS)

def month_calendar(year, month, holidays, outage_days=0.0):
    row = tao_dac_trung(pd.DataFrame({YEAR: [year], MONTH: [month], OUTAGE: [outage_days]}), holidays).iloc[0]
    return {label: int(row[col]) for label, col in zip(DAY_TYPES, LEGACY_COLUMNS)}

def equivalent_days(count, factors):
    k = factors.legacy()
    return float(count["Ngày thường"] * 1.0 + count["Thứ Hai"] * k['T2']
        + count["Thứ Bảy"] * k['T7'] + count["Chủ nhật"] * k['CN']
        + count["Lễ"] * k['Le'] + count["Tết"] * k['Tet'] + count["Cắt điện"] * k['CatDien'])

def add_calendar(df, holidays, factors):
    legacy = tao_dac_trung(df.copy(), holidays)
    out = df.copy()
    if "Date" not in out:
        out["Date"] = pd.to_datetime(dict(year=out[YEAR], month=out[MONTH], day=1))
    if DAYS not in out:
        out[DAYS] = out["Date"].dt.days_in_month
    out[OUTAGE] = legacy["So_Ngay_Cat_Dien"]
    for label, col in zip(DAY_TYPES, LEGACY_COLUMNS):
        out[label] = legacy[col]
    k = factors.legacy()
    out["Ngày tương đương"] = (legacy["So_Ngay_Thuong"] * 1.0
        + legacy["So_Ngay_T2"] * k['T2'] + legacy["So_Ngay_T7"] * k['T7']
        + legacy["So_Ngay_CN"] * k['CN'] + legacy["So_Ngay_Le_Nho"] * k['Le']
        + legacy["So_Ngay_Tet_Am"] * k['Tet'] + legacy["So_Ngay_Cat_Dien"] * k['CatDien'])
    return out

def fingerprint(*parts):
    digest = hashlib.sha256(VERSION.encode())
    for value in parts:
        if isinstance(value, pd.DataFrame):
            encoded = value.to_json(orient="split", date_format="iso", double_precision=15)
        elif isinstance(value, Factors):
            encoded = json.dumps(asdict(value), sort_keys=True)
        elif isinstance(value, dict) and any(isinstance(k, tuple) for k in value):
            encoded = json.dumps([(list(k), v) for k, v in sorted(value.items())], ensure_ascii=False)
        else:
            encoded = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
        encoded = encoded.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big")); digest.update(encoded)
    return digest.hexdigest()

def available_models():
    return [NN, RF, XGB, TREND, NAIVE]

def predict_models(history, future, holidays, factors, seed=42, models=None):
    if history.empty or future.empty:
        raise InputError("Cần dữ liệu lịch sử và đầu vào dự báo.")
    selected = models or available_models()
    if any(m not in available_models() for m in selected):
        raise InputError("Tên mô hình không hợp lệ.")
    train, pred = add_calendar(history, holidays, factors), add_calendar(future, holidays, factors)
    if (train["Ngày tương đương"] <= 0).any():
        raise InputError("Ngày tương đương lịch sử phải lớn hơn 0; kiểm tra lịch và hệ số.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        nn, rf, xgb, trend = chay_mo_hinh_goc(history, future, holidays, factors.legacy(), seed)
    mapping = {NN: nn, RF: rf, XGB: xgb, TREND: trend}
    output = pred[["Date", YEAR, MONTH, DAYS, OUTAGE, *DAY_TYPES, "Ngày tương đương"]].copy()
    notes = []
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        notes.append("NN báo chưa hội tụ đầy đủ với giới hạn của mã gốc; cấu hình không bị tự thay đổi.")
    for name in selected:
        if name == NAIVE:
            values = []
            for _, row in pred.iterrows():
                previous = train[(train[YEAR] == row[YEAR] - 1) & (train[MONTH] == row[MONTH])]
                values.append(float(previous.iloc[0][ENERGY]) / float(previous.iloc[0]["Ngày tương đương"])
                    * float(row["Ngày tương đương"]) if len(previous) == 1 else np.nan)
            output[name] = values
        else:
            output[name] = mapping[name].to_numpy()
        if not np.isfinite(output[name]).all():
            notes.append(f"{name}: có giá trị không hữu hạn hoặc thiếu dữ liệu đối chiếu.")
    return output, notes

def rolling_backtest(history, holidays, factors, seed=42, horizon=1, periods=6,
                     models=None, progress: Callable | None = None):
    if not 1 <= horizon <= 12 or not 1 <= periods <= 24:
        raise InputError("Chân trời kiểm chứng 1–12 tháng; số kỳ 1–24.")
    names = models or available_models()
    if history["Date"].duplicated().any():
        return pd.DataFrame(), pd.DataFrame(), ["Chưa kiểm chứng vì lịch sử có tháng trùng; dự báo gốc vẫn giữ thứ tự dữ liệu."]
    targets = []
    for date in sorted(history["Date"].tolist()):
        cutoff = date - pd.DateOffset(months=horizon)
        train = history[history["Date"] <= cutoff]
        if len(train) >= 24 and train["Date"].max() == cutoff:
            targets.append((date, cutoff))
    targets = targets[-periods:]
    rows, notes = [], []
    for i, (date, cutoff) in enumerate(targets):
        train = history[history["Date"] <= cutoff]
        test = history[history["Date"] == date]
        prediction, messages = predict_models(train, test, holidays, factors, seed, names)
        row = {"Date": date, "Mốc huấn luyện": cutoff, "Chân trời (tháng)": horizon, "Thực tế": float(test.iloc[0][ENERGY])}
        row.update({name: float(prediction.iloc[0][name]) for name in names})
        rows.append(row); notes.extend(messages)
        if progress:
            progress((i + 1) / len(targets))
    detail, scores = pd.DataFrame(rows), []
    for name in names:
        if detail.empty:
            break
        valid = np.isfinite(detail[name]) & np.isfinite(detail["Thực tế"])
        truth = detail.loc[valid, "Thực tế"].to_numpy()
        error = detail.loc[valid, name].to_numpy() - truth
        n, nonzero = int(valid.sum()), truth != 0
        scores.append({"Mô hình": name, "Số kỳ": n, "Tổng kỳ": len(targets),
            "MAE": float(np.mean(abs(error))) if n else np.nan,
            "RMSE": float(np.sqrt(np.mean(error ** 2))) if n else np.nan,
            "MAPE (%)": float(np.mean(abs(error[nonzero] / truth[nonzero])) * 100) if nonzero.any() else np.nan,
            "WAPE (%)": float(np.sum(abs(error)) / np.sum(abs(truth)) * 100) if n and np.sum(abs(truth)) else np.nan,
            "Đủ kỳ so sánh": n == len(targets) and n >= 3})
    score_frame = pd.DataFrame(scores)
    if not score_frame.empty:
        score_frame = score_frame.sort_values(["Đủ kỳ so sánh", "WAPE (%)"], ascending=[False, True]).reset_index(drop=True)
    return detail, score_frame, list(dict.fromkeys(notes))

def recommend(scores):
    if scores.empty:
        return None
    eligible = scores[scores["Đủ kỳ so sánh"] & np.isfinite(scores["WAPE (%)"])]
    return None if eligible.empty else str(eligible.sort_values("WAPE (%)").iloc[0]["Mô hình"])

def default_scenarios(predictions, factors):
    frame = predictions[["Date", OUTAGE]].copy()
    frame["Tháng"] = frame.pop("Date").dt.strftime("%m/%Y")
    for key, value in asdict(factors).items():
        frame[key] = value
    frame["Điều chỉnh thêm (%)"], frame["Lý do"] = 0.0, ""
    return frame[["Tháng", OUTAGE, *asdict(factors), "Điều chỉnh thêm (%)", "Lý do"]]

def finalize_scenarios(predictions, scenarios, model, baseline_factors, holidays):
    if model not in predictions:
        raise InputError("Không có kết quả của mô hình đã chọn.")
    expected = predictions["Date"].dt.strftime("%m/%Y").tolist()
    if len(scenarios) != len(expected) or scenarios["Tháng"].tolist() != expected:
        raise InputError("Danh sách tháng của phương án không khớp lần dự báo hiện tại.")
    rows, allocation = [], []
    for (_, p), (_, s) in zip(predictions.iterrows(), scenarios.iterrows()):
        try:
            values = {k: float(s[k]) if pd.notna(s[k]) else v for k, v in asdict(baseline_factors).items()}
            factors = Factors(**values).checked()
            outage = int(s[OUTAGE]) if pd.notna(s[OUTAGE]) else 0
            adjustment = float(s["Điều chỉnh thêm (%)"])
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise InputError("Kiểm tra hệ số, số ngày cắt điện và tỷ lệ điều chỉnh.") from exc
        if outage < 0 or not np.isfinite(adjustment) or not -100 <= adjustment <= 100:
            raise InputError("Ngày cắt điện không âm; điều chỉnh thêm từ −100% đến +100%.")
        reason = "" if pd.isna(s["Lý do"]) else str(s["Lý do"]).strip()
        changed = factors != baseline_factors or outage != int(p[OUTAGE]) or adjustment != 0
        if changed and not reason:
            raise InputError(f"Tháng {s['Tháng']}: cần ghi lý do khi thay đổi phương án.")
        old = {k: int(p[k]) for k in DAY_TYPES}
        old["Ngày thường"] = max(int(p[DAYS]) - sum(old[k] for k in DAY_TYPES[1:]), 0)
        eq_old = equivalent_days(old, baseline_factors)
        new = old.copy(); new["Cắt điện"] = outage
        new["Ngày thường"] = max(int(p[DAYS]) - sum(new[k] for k in DAY_TYPES[1:]), 0)
        eq_new = equivalent_days(new, factors)
        q_model = float(p[model])
        if not np.isfinite(q_model):
            raise InputError(f"Tháng {s['Tháng']}: dự báo không hữu hạn.")
        daily = q_model / eq_old if eq_old else 0.0
        q_calendar = daily * eq_new
        q_final = q_calendar * (1 + adjustment / 100)
        rows.append({"Tháng": s["Tháng"], "Mô hình": model, "Dự báo gốc": q_model,
            "Ngày tương đương gốc": eq_old, "Ngày tương đương mới": eq_new,
            "Tác động cơ cấu ngày": q_calendar - q_model,
            "Tác động điều chỉnh thêm": q_final - q_calendar,
            "Sản lượng chốt": q_final, "Lý do": reason})
        for kind, weight in zip(DAY_TYPES, factors.day_weights()):
            adjusted_daily = daily * weight * (1 + adjustment / 100)
            allocation.append({"Tháng": s["Tháng"], "Loại ngày": kind, "Số ngày": new[kind],
                "Sản lượng/ngày sau điều chỉnh": adjusted_daily, "Tổng nhóm ngày": adjusted_daily * new[kind]})
    return pd.DataFrame(rows), pd.DataFrame(allocation)
