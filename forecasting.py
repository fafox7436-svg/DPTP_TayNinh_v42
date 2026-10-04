"""Pure forecasting, chronological validation and scenario calculations."""
from __future__ import annotations

import calendar
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from data_io import DAYS, ENERGY, HUMIDITY, MONTH, OUTAGE, TEMP, YEAR, InputError


VERSION = "2.0.0"
TREND = "Xu hướng"
NAIVE = "Cùng kỳ (chuẩn hóa ngày)"
RF, XGB, NN = "Random Forest", "XGBoost", "Neural Network"
DAY_TYPES = ["Ngày thường", "Thứ Hai", "Thứ Bảy", "Chủ nhật", "Lễ", "Tết"]
DAY_KEYS = ["normal", "monday", "saturday", "sunday", "holiday", "tet"]


@dataclass(frozen=True)
class Factors:
    monday: float = 0.95
    saturday: float = 0.94
    sunday: float = 0.79
    holiday: float = 0.56
    tet: float = 0.40
    outage: float = 0.50

    def checked(self) -> "Factors":
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise InputError("Hệ số phải là số hữu hạn.")
        if any(not 0 <= v <= 3 for k, v in values.items() if k != "outage"):
            raise InputError("Hệ số loại ngày phải trong khoảng 0–3.")
        if not 0 <= self.outage <= 1:
            raise InputError("Hệ số cắt điện phải trong khoảng 0–1 (tỷ lệ sản lượng còn lại).")
        return self

    def day_weights(self) -> np.ndarray:
        self.checked()
        return np.array([1.0, self.monday, self.saturday, self.sunday, self.holiday, self.tet])


def validate_holidays(raw: pd.DataFrame) -> dict[str, str]:
    """Exact dates. A holiday replaces its weekday category, never adds a second day."""
    if raw.empty:
        return {}
    if not {"Ngày", "Loại"}.issubset(raw.columns):
        raise InputError("Bảng lịch nghỉ phải có hai cột Ngày và Loại.")
    output = {}
    for index, row in raw.dropna(how="all").iterrows():
        if pd.isna(row["Ngày"]) or pd.isna(row["Loại"]):
            raise InputError(f"Lịch nghỉ dòng {index + 1}: phải có ngày và loại ngày.")
        try:
            date = pd.Timestamp(row["Ngày"])
        except (ValueError, TypeError) as exc:
            raise InputError(f"Lịch nghỉ dòng {index + 1}: ngày không hợp lệ.") from exc
        if not 1900 <= date.year <= 2200 or date.tz is not None:
            raise InputError("Ngày nghỉ phải trong khoảng năm 1900–2200, không kèm múi giờ.")
        key = date.strftime("%Y-%m-%d")
        kind = str(row["Loại"]).strip()
        if kind not in ["Lễ", "Tết"]:
            raise InputError("Loại ngày nghỉ chỉ nhận Lễ hoặc Tết.")
        if key in output:
            raise InputError(f"Ngày nghỉ {key} bị khai báo trùng. Chỉ chọn một loại cho mỗi ngày.")
        output[key] = kind
    return dict(sorted(output.items()))


def month_calendar(year: int, month: int, holidays: dict[str, str]) -> dict[str, int]:
    count = dict.fromkeys(DAY_TYPES, 0)
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        key = f"{year:04d}-{month:02d}-{day:02d}"
        kind = holidays.get(key)
        if kind is None:
            weekday = calendar.weekday(year, month, day)
            kind = {0: "Thứ Hai", 5: "Thứ Bảy", 6: "Chủ nhật"}.get(weekday, "Ngày thường")
        if kind not in count:
            raise InputError(f"Loại ngày không hợp lệ tại {key}.")
        count[kind] += 1
    return count


def equivalent_days(count: dict[str, int], factors: Factors, outage_days: float = 0.0) -> float:
    n_days = sum(count.values())
    if not np.isfinite(outage_days) or not 0 <= outage_days <= n_days:
        raise InputError("Số ngày cắt điện tương đương nằm ngoài số ngày trong tháng.")
    # Monthly approximation: uniform exposure across day types. This does not
    # pretend that a local/partial outage affected the entire province for a day.
    calendar_equivalent = np.array([count[k] for k in DAY_TYPES]) @ factors.day_weights()
    availability = 1.0 - outage_days / n_days * (1.0 - factors.outage)
    return float(calendar_equivalent * availability)


def add_calendar(df: pd.DataFrame, holidays: dict[str, str], factors: Factors) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    counts = [month_calendar(int(r[YEAR]), int(r[MONTH]), holidays) for _, r in out.iterrows()]
    for col in DAY_TYPES:
        out[col] = [c[col] for c in counts]
    out["Ngày tương đương"] = [equivalent_days(c, factors, float(r.get(OUTAGE, 0)))
                                  for c, (_, r) in zip(counts, out.iterrows())]
    if (out["Ngày tương đương"] <= 0).any():
        raise InputError("Ngày tương đương phải lớn hơn 0 để huấn luyện và dự báo.")
    return out


def fingerprint(*parts) -> str:
    digest = hashlib.sha256(VERSION.encode())
    for value in parts:
        if isinstance(value, pd.DataFrame):
            encoded = value.to_json(orient="split", date_format="iso", double_precision=15)
        elif isinstance(value, Factors):
            encoded = json.dumps(asdict(value), sort_keys=True)
        else:
            encoded = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
        # Length delimiters prevent ambiguity between adjoining values.
        encoded = encoded.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def available_models() -> list[str]:
    models = [TREND, NAIVE, RF, NN]
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return models
    return models + [XGB]


def features(df: pd.DataFrame, start_year: int, weather: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "time": (df[YEAR] - start_year) * 12 + df[MONTH],
        "month_sin": np.sin(2 * np.pi * df[MONTH] / 12),
        "month_cos": np.cos(2 * np.pi * df[MONTH] / 12),
        **{c: df[c] for c in weather},
    }, index=df.index)


def predict_models(history: pd.DataFrame, future: pd.DataFrame, holidays: dict[str, str],
                   factors: Factors, seed: int = 42,
                   models: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if history.empty or future.empty or history["Date"].max() >= future["Date"].min():
        raise InputError("Mọi quan sát huấn luyện phải đứng trước tháng cần dự báo.")
    selected = models or available_models()
    train = add_calendar(history, holidays, factors)
    pred = add_calendar(future, holidays, factors)
    weather = [c for c in [TEMP, HUMIDITY] if c in train and c in pred]
    x = features(train, int(train[YEAR].min()), weather)
    xp = features(pred, int(train[YEAR].min()), weather)
    # Positive energy is validated at input. log/exp keeps kWh and million kWh
    # mathematically equivalent, unlike log1p for small numerical units.
    target = np.log(train[ENERGY].to_numpy() / train["Ngày tương đương"].to_numpy())
    trend = LinearRegression().fit(x[["time"]], target)
    future_trend = trend.predict(xp[["time"]])
    residual = target - trend.predict(x[["time"]])
    output = pred[["Date", YEAR, MONTH, DAYS, OUTAGE, *DAY_TYPES, "Ngày tương đương"]].copy()
    notes = []

    def monthly(log_daily):
        with np.errstate(over="raise", invalid="raise"):
            daily = np.exp(log_daily)
        if not np.isfinite(daily).all() or (daily < 0).any():
            raise ValueError("Nonfinite or negative prediction")
        return daily * pred["Ngày tương đương"].to_numpy()

    for name in selected:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                if name == TREND:
                    value = monthly(future_trend)
                elif name == NAIVE:
                    base = pd.Series(train[ENERGY].to_numpy() / train["Ngày tương đương"].to_numpy(),
                                     index=train["Date"])
                    value = np.array([base.get(d - pd.DateOffset(years=1), np.nan)
                                      for d in pred["Date"]]) * pred["Ngày tương đương"].to_numpy()
                elif name == RF:
                    model = RandomForestRegressor(n_estimators=200, max_depth=5,
                                                  min_samples_leaf=3, random_state=seed, n_jobs=1)
                    model.fit(x, residual)
                    value = monthly(model.predict(xp) + future_trend)
                elif name == XGB:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(n_estimators=150, learning_rate=0.04, max_depth=2,
                                         min_child_weight=3, reg_alpha=0.1, reg_lambda=2,
                                         subsample=1.0, colsample_bytree=1.0,
                                         random_state=seed, n_jobs=1, objective="reg:squarederror")
                    model.fit(x, residual)
                    value = monthly(model.predict(xp) + future_trend)
                elif name == NN:
                    scaler = StandardScaler().fit(x)
                    y_scaler = StandardScaler().fit(target.reshape(-1, 1))
                    model = MLPRegressor(hidden_layer_sizes=(12,), solver="lbfgs", alpha=1.0,
                                         max_iter=1500, max_fun=30000, random_state=seed)
                    model.fit(scaler.transform(x), y_scaler.transform(target.reshape(-1, 1)).ravel())
                    log_daily = y_scaler.inverse_transform(model.predict(scaler.transform(xp)).reshape(-1, 1)).ravel()
                    value = monthly(log_daily)
                else:
                    raise InputError(f"Mô hình chưa được hỗ trợ: {name}")
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                notes.append(f"{name}: thuật toán chưa hội tụ đầy đủ; cần xem lại kết quả kiểm chứng.")
            output[name] = value
            if not np.isfinite(value).all():
                notes.append(f"{name}: thiếu giá trị dự báo, không đủ điều kiện lựa chọn tự động.")
        except (ValueError, FloatingPointError, ImportError, RuntimeError) as exc:
            output[name] = np.nan
            notes.append(f"{name}: không tính được ({type(exc).__name__}); không thay lỗi bằng số 0.")
    return output, list(dict.fromkeys(notes))


def rolling_backtest(history: pd.DataFrame, holidays: dict[str, str], factors: Factors,
                     seed: int = 42, horizon: int = 1, periods: int = 6,
                     models: list[str] | None = None,
                     progress: Callable[[float], None] | None = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """At each origin refit only on earlier months; score a fixed lead horizon."""
    if not 1 <= horizon <= 12 or not 1 <= periods <= 24:
        raise InputError("Chân trời kiểm chứng phải từ 1–12 tháng; số kỳ từ 1–24.")
    names = models or available_models()
    targets = list(range(24 + horizon - 1, len(history)))[-periods:]
    rows, notes = [], []
    for index, target_i in enumerate(targets):
        stop = target_i - horizon + 1
        train = history.iloc[:stop]
        test = history.iloc[[target_i]]
        prediction, messages = predict_models(train, test, holidays, factors, seed, names)
        row = {"Date": test.iloc[0]["Date"], "Mốc huấn luyện": train["Date"].max(),
               "Chân trời (tháng)": horizon, "Thực tế": float(test.iloc[0][ENERGY])}
        row.update({name: float(prediction.iloc[0][name]) for name in names})
        rows.append(row)
        notes.extend(messages)
        if progress:
            progress((index + 1) / len(targets))
    detail = pd.DataFrame(rows)
    scores = []
    for name in names:
        if detail.empty:
            break
        valid = np.isfinite(detail[name]) & np.isfinite(detail["Thực tế"])
        truth = detail.loc[valid, "Thực tế"].to_numpy()
        error = detail.loc[valid, name].to_numpy() - truth
        n = int(valid.sum())
        scores.append({"Mô hình": name, "Số kỳ": n, "Tổng kỳ": len(targets),
                       "MAE": float(np.mean(abs(error))) if n else np.nan,
                       "RMSE": float(np.sqrt(np.mean(error ** 2))) if n else np.nan,
                       "MAPE (%)": float(np.mean(abs(error / truth)) * 100) if n else np.nan,
                       "WAPE (%)": float(np.sum(abs(error)) / np.sum(abs(truth)) * 100) if n else np.nan,
                       "Đủ kỳ so sánh": n == len(targets) and n >= 3})
    score_frame = pd.DataFrame(scores)
    if not score_frame.empty:
        score_frame = score_frame.sort_values(["Đủ kỳ so sánh", "WAPE (%)"], ascending=[False, True]).reset_index(drop=True)
    return detail, score_frame, list(dict.fromkeys(notes))


def recommend(scores: pd.DataFrame) -> str | None:
    if scores.empty:
        return None
    eligible = scores[scores["Đủ kỳ so sánh"] & np.isfinite(scores["WAPE (%)"])]
    if eligible.empty:
        return None
    return str(eligible.sort_values("WAPE (%)").iloc[0]["Mô hình"])


def default_scenarios(predictions: pd.DataFrame, factors: Factors) -> pd.DataFrame:
    frame = predictions[["Date", OUTAGE]].copy()
    frame["Tháng"] = frame.pop("Date").dt.strftime("%m/%Y")
    for key, value in asdict(factors).items():
        frame[key] = value
    frame["Điều chỉnh thêm (%)"] = 0.0
    frame["Lý do"] = ""
    return frame[["Tháng", OUTAGE, *asdict(factors), "Điều chỉnh thêm (%)", "Lý do"]]


def finalize_scenarios(predictions: pd.DataFrame, scenarios: pd.DataFrame, model: str,
                       baseline_factors: Factors, holidays: dict[str, str]
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide by the frozen equivalence used at fit time, never today's sidebar settings."""
    if model not in predictions:
        raise InputError("Không có kết quả của mô hình đã chọn.")
    expected = predictions["Date"].dt.strftime("%m/%Y").tolist()
    if len(scenarios) != len(expected) or scenarios["Tháng"].tolist() != expected:
        raise InputError("Danh sách tháng của phương án không khớp lần dự báo hiện tại.")
    rows, allocation = [], []
    for (_, p), (_, s) in zip(predictions.iterrows(), scenarios.iterrows()):
        try:
            factors = Factors(**{k: float(s[k]) for k in asdict(baseline_factors)}).checked()
            adjustment, outage = float(s["Điều chỉnh thêm (%)"]), float(s[OUTAGE])
        except (TypeError, ValueError, KeyError) as exc:
            raise InputError("Hệ số, ngày cắt điện và tỷ lệ điều chỉnh không được để trống.") from exc
        if not np.isfinite(adjustment) or not -100 <= adjustment <= 100:
            raise InputError("Điều chỉnh thêm phải từ −100% đến +100%.")
        reason = "" if pd.isna(s["Lý do"]) else str(s["Lý do"]).strip()
        changed = (factors != baseline_factors or abs(outage - p[OUTAGE]) > 1e-9 or adjustment != 0)
        if changed and not reason:
            raise InputError(f"Tháng {s['Tháng']}: cần ghi lý do khi thay đổi phương án.")
        count = {k: int(p[k]) for k in DAY_TYPES}
        eq_new = equivalent_days(count, factors, outage)
        q_model, eq_old = float(p[model]), float(p["Ngày tương đương"])
        if not np.isfinite(q_model) or q_model < 0 or eq_old <= 0:
            raise InputError(f"Tháng {s['Tháng']}: kết quả mô hình không hợp lệ.")
        daily = q_model / eq_old
        q_calendar = daily * eq_new
        q_final = q_calendar * (1 + adjustment / 100)
        rows.append({"Tháng": s["Tháng"], "Mô hình": model, "Dự báo gốc": q_model,
                     "Ngày tương đương gốc": eq_old, "Ngày tương đương mới": eq_new,
                     "Tác động cơ cấu ngày": q_calendar - q_model,
                     "Tác động điều chỉnh thêm": q_final - q_calendar,
                     "Sản lượng chốt": q_final, "Lý do": reason})
        availability = 1 - outage / sum(count.values()) * (1 - factors.outage)
        for kind, weight in zip(DAY_TYPES, factors.day_weights()):
            adjusted_daily = daily * weight * availability * (1 + adjustment / 100)
            allocation.append({"Tháng": s["Tháng"], "Loại ngày": kind, "Số ngày": count[kind],
                               "Sản lượng/ngày sau điều chỉnh": adjusted_daily,
                               "Tổng nhóm ngày": adjusted_daily * count[kind]})
    return pd.DataFrame(rows), pd.DataFrame(allocation)
