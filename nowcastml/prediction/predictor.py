import pandas as pd
import torch

import nowcastml.mlops.mlflow_utils as mlflow_utils
import nowcastml.prediction.plot as plot
import nowcastml.prediction.transform as trf
from _utils.types import DateLike
import temporal_fusion_transformers as tft


class Predictor:
    """Helper class to cache predictions for countries that might be required repeatedly"""

    def __init__(self, model, start_date: DateLike, end_date: DateLike):
        self.model = model
        self.start_date = start_date
        self.end_date = end_date
        self.predictions: dict[str, torch.Tensor] = {}

    def predict(self, country: str) -> torch.Tensor:
        """Generate or retrieve cached predictions for a given country.

        If predictions for the given country are already cached, those are returned.
        Otherwise, a new prediction sample is created, predicted upon, and then cached.

        Args:
            country: The ISO2 country code for which the predictions are to be made.

        Returns:
            The prediction result from the model.
        """
        if country in self.predictions:
            return self.predictions[country]
        data = trf.create_predicition_sample(self.start_date, self.end_date, country)
        self.predictions[country] = self.model.predict(data)
        return self.predictions[country]

    def build_n_months_ahead_predictions(
        self, n_months_ahead: int, country: str, add_suffix: bool = False
    ) -> pd.DataFrame:
        """Builds

        Args:
            n_months_ahead: How many months are predicted
            country: ISO2 country code of the target country
            add_suffix: Defaults to False.

        Returns:
            pd.DataFrame: _description_
        """
        # Collect data to build DataFrame from
        results = {"index": [], "data": []}
        y_pred = self.predict(country)[:, :n_months_ahead, :]
        for imonth, month in enumerate(
            trf.loop_over_month_starts(self.start_date, self.end_date)
        ):

            # Step 1: Transform prediction tensor into a DataFrame with n_months_ahead columns
            # with the index starting at start date and the predicted quantiles transformed
            # into columns
            month_predictions = y_pred[imonth]
            monthly_index = trf.create_monthly_index(month, n_months_ahead)

            pred_df = self._prediction_to_data_frame(month_predictions, monthly_index)

            # Step 2: Combine the n_months_ahead predictions with 12 - n_months_ahead true
            # inflations
            n_true_months = 12 - n_months_ahead
            start_idx = tft.df_target_1m_pct.index.get_loc(month)
            prev_months_truth = tft.df_target_1m_pct.iloc[
                start_idx - n_true_months : start_idx
            ][country].rename("inflation")
            df = pd.concat((pd.DataFrame(prev_months_truth), pred_df))

            # Step 3: We have no true quantiles in the past -> broadcast the truth to the quantile
            # columns
            for q in tft.quantiles:
                df[f"quantile_{q:.2f}"].fillna(df["inflation"], axis=0, inplace=True)
            df["inflation"].fillna(df["quantile_0.50"], inplace=True)

            # Step 4: Use a rolling window product to calulate the year-on-year inflation prediction
            df_roll = trf.yoy_rolling(df)
            # inflation prediction for this month is the last row
            results["index"].append(month)
            results["data"].append(df_roll.iloc[-1])

        df_result = pd.DataFrame(index=results["index"], data=results["data"])
        if add_suffix:
            df_result.columns = [
                f"{col}_{country}_{n_months_ahead}" for col in df_result.columns
            ]
        return df_result

    def build_monthly_prediction_df(self, country: str) -> pd.DataFrame:
        """_summary_

        Args:
            country (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        predictions = self.predict(country)
        index = trf.create_monthly_index(self.start_date, len(predictions))
        pred_df = self._prediction_to_data_frame(predictions[:, 0, :], index)

        return pred_df

    def _prediction_to_data_frame(
        self, y_pred: torch.tensor, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        pred_df = pd.DataFrame(
            data=y_pred,
            columns=[f"quantile_{q:.2f}" for q in tft.quantiles],
            index=index,
        )
        return pred_df


class AutoRegPredictor:
    def __init__(self, start_date: DateLike, end_date: DateLike):
        self.start_date = start_date
        self.end_date = end_date
        # self.model = ar_model

    # generate AR forecasts
    def generate_ar_forecast(
        ar_model, country, start_date: str, end_date: str, n_months_ahead: int
    ):
        df_target = tft.df_target_1m_pct

        ar_predictions = []
        for month in trf.loop_over_month_starts(start_date, end_date):
            # Get index of current target month and fit model
            idx = df_target.index.get_loc(month)
            prediction = ar_model.predict(
                month, month + pd.DateOffset(months=n_months_ahead)
            )
            # Join this with the 12 - n_months_ahead previous months of truth
            prev_months_truth = df_target[idx - (12 - n_months_ahead) : idx][country]
            all_months = pd.concat((prev_months_truth, prediction))
            x = trf.yoy_rolling(all_months)
            ar_predictions.append((month, x.iloc[-1]))
        return (
            pd.DataFrame(ar_predictions)
            .set_index(0)
            .rename(columns={1: "prediction AR"})
        )


def build_model_benchmark_df(
    predictor: Predictor,
    parent_id: str,
    start_date: str,
    end_date: str,
    countries: list[str],
    months_ahead_vals: list[int],
):
    """_summary_

    Args:
        predictor (Predictor): _description_
        parent_id (str): _description_
        start_date (str): _description_
        end_date (str): _description_
        countries (list[str]): _description_
        months_ahead_vals (list[int]): _description_

    Returns:
        _type_: _description_
    """
    dfs = []
    for country in countries:
        ar_model = mlflow_utils.get_ar_model(parent_id, country)
        for n_months_ahead in months_ahead_vals:
            dfs.append(
                plot.generate_ar_forecast(
                    ar_model, country, start_date, end_date, n_months_ahead
                )
            )

            dfs.append(
                predictor.build_n_months_ahead_predictions(
                    n_months_ahead, country, add_suffix=True
                )
            )
        truths = tft.df_target_12m_pct.loc[start_date:end_date, country].rename(
            f"truth_{country}"
        )
        dfs.append(truths)
    return pd.concat(dfs, axis=1)
