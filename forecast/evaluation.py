"""Backtest theo thời gian và các chỉ số MAE, MAPE, RMSE."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .models import chay_mo_hinh_goc

MODEL_NAMES = ["Neural Network", "Random Forest", "XGBoost"]


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(math.sqrt(np.mean((actual - pred) ** 2)))
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100) if mask.any() else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def backtest_expanding(df_history: pd.DataFrame, holidays_map: dict, k_dict: dict,
                       holiday_dates: pd.DataFrame | None = None,
                       outage_dates_map: dict | None = None,
                       n_months: int = 12, seed: int = 42,
                       min_train_months: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kiểm định walk-forward: mỗi tháng chỉ dùng dữ liệu trước tháng đó để train."""
    required = {"Năm", "Tháng", "Tổng thương phẩm"}
    missing = required.difference(df_history.columns)
    if missing:
        raise ValueError(f"Thiếu cột để backtest: {', '.join(sorted(missing))}")
    hist = df_history.copy().sort_values(["Năm", "Tháng"]).reset_index(drop=True)
    hist = hist[pd.to_numeric(hist["Tổng thương phẩm"], errors="coerce").notna()].reset_index(drop=True)
    if len(hist) <= min_train_months:
        raise ValueError(f"Cần nhiều hơn {min_train_months} tháng lịch sử để backtest.")

    first_test = max(min_train_months, len(hist) - int(n_months))
    rows = []
    for idx in range(first_test, len(hist)):
        train = hist.iloc[:idx].copy()
        target = hist.iloc[[idx]].copy()
        pred_nn, pred_rf, pred_xg, _ = chay_mo_hinh_goc(
            train, target, holidays_map, k_dict, seed=seed,
            holiday_dates=holiday_dates, outage_dates_map=outage_dates_map,
        )
        actual = float(target["Tổng thương phẩm"].iloc[0])
        row = {
            "Năm": int(target["Năm"].iloc[0]),
            "Tháng": int(target["Tháng"].iloc[0]),
            "Thực Tế": actual,
            "Neural Network": float(pred_nn.iloc[0] if hasattr(pred_nn, "iloc") else pred_nn[0]),
            "Random Forest": float(pred_rf.iloc[0] if hasattr(pred_rf, "iloc") else pred_rf[0]),
            "XGBoost": float(pred_xg.iloc[0] if hasattr(pred_xg, "iloc") else pred_xg[0]),
        }
        for model in MODEL_NAMES:
            row[f"Sai số {model}"] = row[model] - actual
            row[f"APE {model} (%)"] = abs(row[model] - actual) / abs(actual) * 100 if actual != 0 else np.nan
        rows.append(row)

    detail = pd.DataFrame(rows)
    metric_rows = []
    for model in MODEL_NAMES:
        vals = _metrics(detail["Thực Tế"].to_numpy(), detail[model].to_numpy())
        metric_rows.append({"Mô hình": model, **vals, "Số tháng": len(detail)})
    metrics = pd.DataFrame(metric_rows).sort_values(["MAPE (%)", "RMSE", "MAE"]).reset_index(drop=True)
    metrics.insert(1, "Xếp hạng", range(1, len(metrics) + 1))
    return metrics, detail
