# -*- coding: utf-8 -*-
import itertools
import time
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.forecasting import ForecastingStrategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.models.model_base import BatchMaker, ModelBase
from ts_benchmark.utils.data_processing import split_time
from ts_benchmark.utils.data_processing import split_channel


class RollingForecastEvalBatchMaker:
    def __init__(
        self,
        series: pd.DataFrame,
        index_list: List[int],
        covariates: Optional[dict] = None,
    ):
        self.series = series
        self.index_list = index_list
        self.current_sample_count = 0
        self.covariates = covariates

    def make_batch_predict(self, batch_size: int, win_size: int) -> dict:
        """
        Return a batch of data with index and column to be used for batch prediction.

        :param batch_size: The size of batch.
        :param win_size: The length of data used for prediction.
        :return: a batch of data and its time stamps.
        """
        index_list = self.index_list[
            self.current_sample_count : self.current_sample_count + batch_size
        ]
        series = self.series.values
        predict_batch = self._make_batch_data(
            series, np.array(index_list) - win_size, win_size
        )

        indexes = self.series.index
        time_stamps_batch = self._make_batch_data(
            indexes, np.array(index_list) - win_size, win_size
        )
        covariates_batch = self._make_batch_covariates(
            np.array(index_list) - win_size, win_size
        )
        self.current_sample_count += len(index_list)
        return {
            "input": predict_batch,
            "time_stamps": time_stamps_batch,
            "covariates": covariates_batch,
        }

    def make_batch_eval(self, horizon: int) -> dict:
        """
        Return all data to be used for batch evaluation.

        :param horizon: The size of horizon.
        :return: All data to be used for batch evaluation.
        """
        series = self.series.values
        test_batch = self._make_batch_data(series, np.array(self.index_list), horizon)
        covariates_batch = self._make_batch_covariates(
            np.array(self.index_list), horizon
        )
        return {
            "target": test_batch,
            "covariates": covariates_batch,
        }

    def _make_batch_covariates(self, index_list: np.ndarray, win_size: int) -> Dict:
        """
        Create a batch of covariates

        :param index_list: An array of starting indices for each window.
        :param win_size: The size of each window.
        :return: A batch of covariates.
        """
        covariates = {} if self.covariates is None else self.covariates
        covariates_batch = {}
        if covariates.get("exog") is not None:
            covariates_batch["exog"] = self._make_batch_data(
                self.covariates["exog"], index_list, win_size
            )
        return covariates_batch

    @staticmethod
    def _make_batch_data(
        data: Any, index_list: np.ndarray, win_size: int
    ) -> np.ndarray:
        """
        Create a batch of data

        :param data: Array_like. Array to create the batch.
        :param index_list: An array of starting indices for each window.
        :param win_size: The size of each window.
        :return: A batch of data.
        """
        windows = sliding_window_view(data, window_shape=(win_size, *data.shape[1:]))
        data_batch = windows[index_list]
        data_batch = np.squeeze(data_batch, axis=tuple(range(1, np.ndim(data))))
        return data_batch

    def has_more_batches(self) -> bool:
        """
        Check if there are more batches to process.

        :return: True if there are more batches, False otherwise.
        """
        return self.current_sample_count < len(self.index_list)


class RollingForecastPredictBatchMaker(BatchMaker):
    def __init__(self, batch_maker: RollingForecastEvalBatchMaker):
        self._batch_maker = batch_maker

    def make_batch(self, batch_size: int, win_size: int) -> dict:
        """
        Return a batch of data to be used for batch prediction.

        :param batch_size: The size of batch.
        :param win_size: The length of data used for prediction.
        :return: A batch of data.
        """
        return self._batch_maker.make_batch_predict(batch_size, win_size)

    def has_more_batches(self) -> bool:
        """
        Check if there are more batches to process.

        :return: True if there are more batches, False otherwise.
        """
        return self._batch_maker.has_more_batches()


class RollingForecast(ForecastingStrategy):
    """
    Rolling forecast strategy class

    This strategy defines a forecasting task that fits once on the training set and
    forecasts on the testing set in a rolling window style.

    The required strategy configs include:

    - horizon (int): The length of each prediction;
    - tv_ratio (float): The ratio of the train-validation series when performing
      train-test split;
    - train_ratio_in_tv (float): The ratio of the training series when performing
      train-validation split;
    - stride (int): Rolling stride, i.e. the interval between two windows;
    - num_rollings (int): The maximum number of steps to forecast;

    The accepted metrics include all regression metrics.

    The return fields other than the specified metrics are (in order):

    - FieldNames.FILE_NAME: The name of the series;
    - FieldNames.FIT_TIME: The training time;
    - FieldNames.INFERENCE_TIME: The inference time;
    - FieldNames.ACTUAL_DATA: The true test data, encoded as a string.
    - FieldNames.INFERENCE_DATA: The predicted data, encoded as a string.
    - FieldNames.LOG_INFO: Any log returned by the evaluator.
    """

    REQUIRED_CONFIGS = [
        "horizon",
        "tv_ratio",
        "train_ratio_in_tv",
        "stride",
        "num_rollings",
        "save_true_pred",
        "target_channel",
    ]

    @staticmethod
    def _get_index(
        train_length: int, test_length: int, horizon: int, stride: int
    ) -> List[int]:
        """
        Get the index list of the rolling windows.

        :param train_length: Training data length.
        :param test_length: Test data length.
        :param horizon: Prediction length.
        :param stride: Rolling stride.
        :return: Index list of the rolling windows.
        """
        data_len = train_length + test_length
        index_list = list(range(train_length, data_len - horizon + 1, stride)) + (
            [data_len - horizon] if (test_length - horizon) % stride != 0 else []
        )
        return index_list

    def _get_split_lens(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        tv_ratio: float,
    ) -> Tuple[int, int]:
        """
        Gets the size of the train-validation series and the test series

        :param series: Target series.
        :param meta_info: Meta-information of the target series.
        :param tv_ratio: The ratio of the train-validation series when performing
            train-test split;
        :return: The length of the train-validation series, and the length of the test series.
        """
        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        train_length = int(tv_ratio * data_len)
        test_length = data_len - train_length
        if train_length <= 0 or test_length <= 0:
            raise ValueError(
                "The length of training or testing data is less than or equal to 0"
            )
        return train_length, test_length

    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model_factory: ModelFactory,
        series_name: str,
    ) -> List:
        """
        The entry function of execution pipeline of forecasting tasks

        :param series: Target series to evaluate.
        :param meta_info: The corresponding meta-info.
        :param model_factory: The factory to create models.
        :param series_name: the name of the target series.
        :return: The evaluation results.
        """
        model = model_factory()
        if model.batch_forecast.__annotations__.get("not_implemented_batch"):
            return self._eval_sample(series, meta_info, model, series_name)
        else:
            return self._eval_batch(series, meta_info, model, series_name)

    def _eval_sample(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
    ) -> List:
        """
        The sample execution pipeline of forecasting tasks.

        :param series: Target series to evaluate.
        :param meta_info: The corresponding meta-info.
        :param model: The model used for prediction.
        :param series_name: the name of the target series.
        :return: The evaluation results.
        """
        target_channel = self._get_scalar_config_value("target_channel", series_name)
        stride = self._get_scalar_config_value("stride", series_name)
        horizon = self._get_scalar_config_value("horizon", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)
        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )
        tv_ratio = self._get_scalar_config_value("tv_ratio", series_name)

        train_length, test_length = self._get_split_lens(series, meta_info, tv_ratio)
        train_valid_data, test_data = split_time(series, train_length)

        target_train_valid_data, exog_data = split_channel(
            train_valid_data, target_channel
        )
        covariates_train = {}
        covariates_train["exog"] = exog_data

        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(
            target_train_valid_data,
            covariates=covariates_train,
            train_ratio_in_tv=train_ratio_in_tv,
        )
        end_fit_time = time.time()

        eval_scaler = self._get_eval_scaler(target_train_valid_data, train_ratio_in_tv)

        index_list = self._get_index(train_length, test_length, horizon, stride)
        total_inference_time = 0
        all_test_results = []
        all_rolling_actual = []
        all_rolling_predict = []
        all_rolling_input = []  # 新增：存储所有回望窗口数据

        # 获取模型的回望窗口长度（一般就是 seq_len）
        # seq_len = getattr(model.config, "seq_len", None)
        # if seq_len is None:
        #     raise AttributeError("model.config.seq_len not found, cannot save input windows in _eval_sample.")

        ###11.21修改，解决darts库没有seq_len参数
        seq_len = getattr(model, "seq_len", None)
        if seq_len is None:
            seq_len = getattr(getattr(model, "config", None), "seq_len", None)
        if seq_len is None:
            # 对于 Darts 模型，使用 lags 作为 seq_len
            seq_len = getattr(model, "lags", None)
            if isinstance(seq_len, dict):
                seq_len = max(seq_len.get('target', [0]))
            elif isinstance(seq_len, (list, tuple)):
                seq_len = max(seq_len) if seq_len else None

        if seq_len is None:
            # 如果还是找不到，使用默认值或跳过保存输入窗口
            seq_len = 192  # 默认值，或者根据你的需求调整
            # 或者直接跳过：return results  # 不保存输入窗口


        for i, index in itertools.islice(enumerate(index_list), num_rollings):
            train, rest = split_time(series, index)
            test, _ = split_channel(split_time(rest, horizon)[0], target_channel)
            target_train, exog_train = split_channel(train, target_channel)
            covariates_forecast = {}
            covariates_forecast["exog"] = exog_train

            # 新增：只拿“最后 seq_len 长度”的回望窗口，保证每次长度一致
            if len(target_train) < seq_len:
                # 太短你可以选择跳过或报错；我这边直接跳过+continue
                continue
            input_window = target_train.iloc[-seq_len:]    # [seq_len, 特征数]
            all_rolling_input.append(input_window)

            start_inference_time = time.time()
            predict = model.forecast(
                horizon, target_train, covariates=covariates_forecast
            )
            end_inference_time = time.time()
            total_inference_time += end_inference_time - start_inference_time

            single_series_result = self.evaluator.evaluate(
                test.to_numpy(), predict, eval_scaler, target_train_valid_data.values
            )
            inference_data = pd.DataFrame(
                predict, columns=test.columns, index=test.index
            )

            all_rolling_actual.append(test)
            all_rolling_predict.append(inference_data)
            all_test_results.append(single_series_result)

        average_inference_time = float(total_inference_time) / min(
            len(index_list), num_rollings
        )
        single_series_results = np.mean(np.stack(all_test_results), axis=0).tolist()

        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data_encoded = (
            self._encode_data(all_rolling_actual) if save_true_pred else np.nan
        )
        inference_data_encoded = (
            self._encode_data(all_rolling_predict) if save_true_pred else np.nan
        )
        # 新增：编码回望窗口数据
        input_data_encoded = (
            self._encode_data(all_rolling_input) if save_true_pred else np.nan
        )
        single_series_results += [
            series_name,
            end_fit_time - start_fit_time,
            average_inference_time,
            actual_data_encoded,
            inference_data_encoded,
            input_data_encoded,  # 新增
            "",
        ]
        return single_series_results

    def _eval_batch(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
    ) -> List:
        target_channel = self._get_scalar_config_value("target_channel", series_name)
        stride = self._get_scalar_config_value("stride", series_name)
        horizon = self._get_scalar_config_value("horizon", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)
        train_ratio_in_tv = self._get_scalar_config_value("train_ratio_in_tv", series_name)
        tv_ratio = self._get_scalar_config_value("tv_ratio", series_name)

        train_length, test_length = self._get_split_lens(series, meta_info, tv_ratio)
        train_valid_data, _ = split_time(series, train_length)

        target_train_valid_data, exog_train_valid = split_channel(train_valid_data, target_channel)
        target_all, exog_all = split_channel(series, target_channel)

        # 拟合
        start_fit = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(
            target_train_valid_data,
            covariates={"exog": exog_train_valid} if exog_train_valid is not None else {},
            train_ratio_in_tv=train_ratio_in_tv,
        )
        fit_time = time.time() - start_fit

        eval_scaler = self._get_eval_scaler(target_train_valid_data, train_ratio_in_tv)

        # 滚动起点索引
        index_list = self._get_index(train_length, test_length, horizon, stride)[:num_rollings]

        # 强制获取 seq_len
        if not hasattr(model.config, "seq_len") or model.config.seq_len is None:
            raise ValueError("model.config.seq_len 必须存在才能保存输入窗口！")

        seq_len = model.config.seq_len

        # 构造 batch maker
        batch_maker = RollingForecastEvalBatchMaker(
            target_all,
            index_list,
            {"exog": exog_all} if exog_all is not None else {}
        )

        # 批量预测
        all_preds = []
        total_inf_time = 0
        pred_maker = RollingForecastPredictBatchMaker(batch_maker)

        while pred_maker.has_more_batches():
            st = time.time()
            batch_pred = model.batch_forecast(horizon, pred_maker)
            total_inf_time += time.time() - st
            all_preds.append(batch_pred)

        all_preds = np.concatenate(all_preds, axis=0)
        targets = batch_maker.make_batch_eval(horizon)["target"]

        # ================== 关键：在这里定义 input_windows，绝对不会漏 ==================
        input_windows = RollingForecastEvalBatchMaker._make_batch_data(
            data=target_all.values,
            index_list=np.array(index_list) - seq_len,
            win_size=seq_len,
        )
        # shape: (N, seq_len, features)
        # ============================================================================

        # 评估
        metric_list = []
        for y_t, y_p in zip(targets, all_preds):
            mets = self.evaluator.evaluate(y_t, y_p, eval_scaler, target_train_valid_data.values)
            metric_list.append(mets)
        avg_metrics = np.mean(metric_list, axis=0).tolist()

        avg_inf_time = total_inf_time / len(index_list)

        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)

        actual_enc = self._encode_data(targets) if save_true_pred else np.nan
        pred_enc   = self._encode_data(all_preds) if save_true_pred else np.nan
        input_enc  = self._encode_data(input_windows) if save_true_pred else np.nan

        return avg_metrics + [
            series_name,
            fit_time,
            avg_inf_time,
            actual_enc,
            pred_enc,
            input_enc,   # 这就是你的回望窗口真实值
            "",
        ]


    @staticmethod
    def accepted_metrics() -> List[str]:
        return regression_metrics.__all__

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.INPUT_DATA,  # 新增：回望窗口数据字段
            FieldNames.LOG_INFO,
        ]
