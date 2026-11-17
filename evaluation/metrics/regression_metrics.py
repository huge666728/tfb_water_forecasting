# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "mase",
    "wape",
    "msmape",
    "mae_norm",
    "mse_norm",
    "rmse_norm",
    "mape_norm",
    "smape_norm",
    "mase_norm",
    "wape_norm",
    "msmape_norm",
    "rse",
    "r2",
    "rse_norm",
    "r2_norm",

]


def _error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Simple error"""
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Percentage error"""
    return (actual - predicted) / actual


def mse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Mean Squared Error"""
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Root Mean Squared Error"""
    return np.sqrt(mse(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Mean Absolute Error"""

    return np.mean(np.abs(_error(actual, predicted)))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    hist_data: np.ndarray,
    # seasonality: int = 2,
    seasonality: int = 24,
    **kwargs
):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    if seasonality == 2:
        return -1
    scale = len(predicted) / (len(hist_data) - seasonality)

    dif = 0
    for i in range((seasonality + 1), len(hist_data)):
        dif = dif + abs(hist_data[i] - hist_data[i - seasonality])

    scale = scale * dif

    return (sum(abs(actual - predicted)) / scale)[0]


def mape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    """
    return np.mean(np.abs(_percentage_error(actual, predicted))) * 100


def smape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Symmetric Mean Absolute Percentage Error
    """
    return (
        np.mean(
            2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
        )
        * 100
    )


def wape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Masked weighted absolute percentage error (WAPE)

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: masked mean absolute error
    """
    loss = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    return loss


def msmape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 0.1, **kwargs):
    """
    Function to calculate series wise smape values

    Parameters
    forecasts - a matrix containing forecasts for a set of series
                no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon
    test_set - a matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
    """

    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(predicted - actual) / denom) * 100
    return msmape_per_series


def _error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Simple error"""
    return scaler.transform(actual) - scaler.transform(predicted)


def _percentage_error_norm(
    actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs
):
    """Percentage error"""
    return (scaler.transform(actual) - scaler.transform(predicted)) / scaler.transform(
        actual
    )


def mse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Mean Squared Error"""
    return np.mean(np.square(_error_norm(actual, predicted, scaler)))


def rmse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Root Mean Squared Error"""
    return np.sqrt(mse_norm(actual, predicted, scaler))


def mae_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Mean Absolute Error"""

    return np.mean(np.abs(_error_norm(actual, predicted, scaler)))


def mase_norm(
    actual: np.ndarray,
    predicted: np.ndarray,
    scaler: object,
    hist_data: np.ndarray,
    # seasonality: int = 2,
    seasonality: int = 24,
    **kwargs
):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    hist_data = scaler.transform(hist_data)
    if seasonality == 2:
        return -1
    scale = len(predicted) / (len(hist_data) - seasonality)

    dif = 0
    for i in range((seasonality + 1), len(hist_data)):
        dif = dif + abs(hist_data[i] - hist_data[i - seasonality])

    scale = scale * dif

    return (sum(abs(actual - predicted)) / scale)[0]


def mape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    """
    return np.mean(np.abs(_percentage_error_norm(actual, predicted, scaler))) * 100


def smape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """
    Symmetric Mean Absolute Percentage Error
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    return (
        np.mean(
            2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
        )
        * 100
    )


def wape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Masked weighted absolute percentage error (WAPE)

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: masked mean absolute error
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    loss = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    return loss


def msmape_norm(
    actual: np.ndarray,
    predicted: np.ndarray,
    scaler: object,
    epsilon: float = 0.1,
    **kwargs
):
    """
    Function to calculate series wise smape values

    Parameters
    forecasts - a matrix containing forecasts for a set of series
                no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon
    test_set - a matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(predicted - actual) / denom) * 100
    return msmape_per_series

def rse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Relative Squared Error
    RSE = sqrt(MSE / variance(actual))
    """
    mse_val = mse(actual, predicted)
    variance = np.var(actual)
    return np.sqrt(mse_val / variance) if variance != 0 else float('inf')

def r2(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    R-squared (Coefficient of Determination)
    R2 = 1 - (SS_residual / SS_total)
    """
    ss_residual = np.sum((actual - predicted) ** 2)
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_residual / ss_total) if ss_total != 0 else float('inf')




def rse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """
    Relative Squared Error (normalized version)
    """
    actual_orig = scaler.transform(actual)
    predicted_orig = scaler.transform(predicted)
    return rse(actual_orig, predicted_orig)

def r2_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """
    R-squared (Coefficient of Determination) - normalized version

    """
    actual_orig = scaler.transform(actual)
    predicted_orig = scaler.transform(predicted)
    return r2(actual_orig, predicted_orig)

# # -*- coding: utf-8 -*-
# import numpy as np
#
# EPS = 1e-12
#
# __all__ = [
#     "mae", "mse", "rmse", "mape", "smape", "mase", "wape", "msmape",
#     "mae_norm", "mse_norm", "rmse_norm", "mape_norm", "smape_norm",
#     "mase_norm", "wape_norm", "msmape_norm",
#     "rse", "r2", "rse_norm", "r2_norm",
#     # extra (可选)
#     "r2_avg", "r2_vw", "r2_avg_norm", "r2_vw_norm",
#     "rse_sqrt", "rse_sqrt_norm"
# ]
#
# # ----------------------------
# # helpers
# # ----------------------------
# def _to_2d(x: np.ndarray) -> np.ndarray:
#     """支持 [B,H,C] / [N,C] / [N] -> 统一变成 [N,C]"""
#     x = np.asarray(x)
#     if x.ndim == 3:
#         b, h, c = x.shape
#         return x.reshape(-1, c)
#     if x.ndim == 2:
#         return x
#     if x.ndim == 1:
#         return x.reshape(-1, 1)
#     raise ValueError(f"Unsupported shape: {x.shape}")
#
# def _align_and_mask(actual: np.ndarray, predicted: np.ndarray):
#     """对齐形状 + 去掉 NaN/Inf（行级 mask：该行所有特征都得是 finite）"""
#     a = _to_2d(actual).astype(np.float64)
#     p = _to_2d(predicted).astype(np.float64)
#     if a.shape != p.shape:
#         raise ValueError(f"Shape mismatch: actual {a.shape}, predicted {p.shape}")
#     m = np.isfinite(a) & np.isfinite(p)
#     row_mask = np.all(m, axis=1)
#     return a[row_mask], p[row_mask]
#
# def _apply_scaler_2d(scaler, x2d: np.ndarray) -> np.ndarray:
#     """scaler.transform 仅吃 [N,C]，这里保证输入合法"""
#     return scaler.transform(x2d)
#
# def _error(actual: np.ndarray, predicted: np.ndarray):
#     return actual - predicted
#
# def _percentage_error(actual: np.ndarray, predicted: np.ndarray, eps: float = EPS):
#     denom = np.maximum(np.abs(actual), eps)
#     return (actual - predicted) / denom
#
# # ----------------------------
# # basic errors (micro-average over all elements)
# # ----------------------------
# def mse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     return float(np.mean((a - p) ** 2))
#
# def rmse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     return float(np.sqrt(mse(actual, predicted)))
#
# def mae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     return float(np.mean(np.abs(a - p)))
#
# def mape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     return float(np.mean(np.abs(_percentage_error(a, p))) * 100)
#
# def smape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     denom = np.maximum(np.abs(a) + np.abs(p), EPS)
#     return float(np.mean(2.0 * np.abs(a - p) / denom) * 100)
#
# def wape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     denom = np.maximum(np.sum(np.abs(a)), EPS)
#     return float(np.sum(np.abs(a - p)) / denom * 100)
#
# def msmape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 0.1, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     # 强制 float comparator，避免整数吞小数
#     comparator = np.full(a.shape, 0.5 + epsilon, dtype=np.float64)
#     denom = np.maximum(comparator, np.abs(p) + np.abs(a) + epsilon)
#     return float(np.mean(2 * np.abs(p - a) / denom) * 100)
#
# # mase 这版本来就不适配 [B,H,C]，先保留但不建议用
# def mase(actual: np.ndarray, predicted: np.ndarray, hist_data: np.ndarray, seasonality: int = 2, **kwargs):
#     return -1
#
# # ----------------------------
# # R2 / RSE (按特征聚合，口径对齐你“指标1”)
# # ----------------------------
# EPS = 1e-8
#
# def _r2_by_feature_vw(a2d: np.ndarray, p2d: np.ndarray, eps: float = EPS):
#     """
#     a2d, p2d: [N, C]
#     返回：
#       r2_vw, r2_avg, rse_vw
#     规则：
#       - 正常：按特征算 SST/SSE，valid 特征参与聚合
#       - 极端：如果所有特征 SST 都很小（近似常数），不返回 NaN：
#             SSE 很小 -> (R2=1, RSE=0)
#             否则   -> (R2=0, RSE=1)
#     """
#     mu = np.mean(a2d, axis=0)                       # [C]
#     ss_res = np.sum((a2d - p2d) ** 2, axis=0)       # [C]
#     ss_tot = np.sum((a2d - mu) ** 2, axis=0)        # [C]
#
#     valid = ss_tot > eps
#
#     # ---- 正常情况：至少有一个特征在窗口内有波动 ----
#     if np.any(valid):
#         r2_each = np.full_like(ss_tot, np.nan, dtype=np.float64)
#         r2_each[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
#
#         rse_vw = float(np.sum(ss_res[valid]) / np.sum(ss_tot[valid]))
#         r2_vw  = float(1.0 - rse_vw)
#         r2_avg = float(np.nanmean(r2_each))
#         return r2_vw, r2_avg, rse_vw
#
#     # ---- 退化情况：这个窗口里所有特征几乎是常数，R2数学上不定义 ----
#     ss_res_all = float(np.sum(ss_res))
#     if ss_res_all <= eps:
#         # 预测几乎完美
#         return 1.0, 1.0, 0.0
#     else:
#         # 常数序列但预测有偏差：用 0 当“不过 baseline”
#         return 0.0, 0.0, 1.0
#
#
# def r2_vw(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     r2v, _, _ = _r2_by_feature_vw(a, p)
#     return r2v
#
# def r2_avg(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     _, r2a, _ = _r2_by_feature_vw(a, p)
#     return r2a
#
# def r2(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     """兼容入口：默认返回方差加权 R2_vw（推荐）"""
#     return r2_vw(actual, predicted)
#
# def rse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     """与指标1一致：RSE = SSE/SST（方差加权）"""
#     a, p = _align_and_mask(actual, predicted)
#     _, _, rse_vw = _r2_by_feature_vw(a, p)
#     return rse_vw
#
# # 如果你还想保留“旧版 sqrt(MSE/Var)”这种RSE：
# def rse_sqrt(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     mse_val = np.mean((a - p) ** 2)
#     var = np.var(a)
#     return float(np.sqrt(mse_val / var)) if var > EPS else np.inf
#
# # ----------------------------
# # normalized metrics: 在 scaler.transform 空间里计算（和你指标1口径一致）
# # ----------------------------
# def mse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     a = _apply_scaler_2d(scaler, a)
#     p = _apply_scaler_2d(scaler, p)
#     return float(np.mean((a - p) ** 2))
#
# def rmse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
#     return float(np.sqrt(mse_norm(actual, predicted, scaler)))
#
# def mae_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     a = _apply_scaler_2d(scaler, a)
#     p = _apply_scaler_2d(scaler, p)
#     return float(np.mean(np.abs(a - p)))
#
# def mape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     a = _apply_scaler_2d(scaler, a)
#     p = _apply_scaler_2d(scaler, p)
#     return float(np.mean(np.abs(_percentage_error(a, p))) * 100)
#
# def smape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     a = _apply_scaler_2d(scaler, a)
#     p = _apply_scaler_2d(scaler, p)
#     denom = np.maximum(np.abs(a) + np.abs(p), EPS)
#     return float(np.mean(2.0 * np.abs(a - p) / denom) * 100)
#
# def wape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     a = _apply_scaler_2d(scaler, a)
#     p = _apply_scaler_2d(scaler, p)
#     denom = np.maximum(np.sum(np.abs(a)), EPS)
#     return float(np.sum(np.abs(a - p)) / denom * 100)
#
# def msmape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, epsilon: float = 0.1, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     a = _apply_scaler_2d(scaler, a)
#     p = _apply_scaler_2d(scaler, p)
#     comparator = np.full(a.shape, 0.5 + epsilon, dtype=np.float64)
#     denom = np.maximum(comparator, np.abs(p) + np.abs(a) + epsilon)
#     return float(np.mean(2 * np.abs(p - a) / denom) * 100)
#
# # mase_norm 同理：旧实现不适配 3D，这里先不启用
# def mase_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, hist_data: np.ndarray, seasonality: int = 2, **kwargs):
#     return -1
#
# def r2_vw_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     a = _apply_scaler_2d(scaler, a)
#     p = _apply_scaler_2d(scaler, p)
#     r2v, _, _ = _r2_by_feature_vw(a, p)
#     return r2v
#
# def r2_avg_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
#     a, p = _align_and_mask(actual, predicted)
#     a = _apply_scaler_2d(scaler, a)
#     p = _apply_scaler_2d(scaler, p)
#     _, r2a, _ = _r2_by_feature_vw(a, p)
#     return r2a
#
# def r2_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
#     """兼容入口：默认返回方差加权 R2_vw_norm（推荐）"""
#     return r2_vw_norm(actual, predicted, scaler)
#
# def rse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
#     """与指标1一致：RSE_norm = SSE/SST（方差加权），所以 R2_norm = 1 - RSE_norm"""
#     a, p = _align_and_mask(actual, predicted)
#     a = _apply_scaler_2d(scaler, a)
#     p = _apply_scaler_2d(scaler, p)
#     _, _, rse_vw = _r2_by_feature_vw(a, p)
#     return rse_vw
#
# def rse_sqrt_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
#     """保留旧口径 sqrt(MSE/Var) 的 norm 版本（可选）"""
#     a, p = _align_and_mask(actual, predicted)
#     a = _apply_scaler_2d(scaler, a)
#     p = _apply_scaler_2d(scaler, p)
#     mse_val = np.mean((a - p) ** 2)
#     var = np.var(a)
#     return float(np.sqrt(mse_val / var)) if var > EPS else np.inf
