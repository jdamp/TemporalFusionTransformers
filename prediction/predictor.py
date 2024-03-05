import pandas as pd
import torch

import prediction.transform as trf
import temporal_fusion_transformers as tft


class Predictor:
    """Helper class to cache predictions for countries that might be required repeatedly"""

    def __init__(self, model, start_date: str, end_date: str):
        self.model = model
        self.start_date = start_date
        self.end_date = end_date
        self.predictions: dict[str, torch.Tensor] = {}

    def predict(self, country: str) -> torch.Tensor:
        """Generate or retrieve cached predictions for a given country.

        If predictions for the given country are already cached, those are returned.
        Otherwise, a new prediction sample is created, predicted upon, and then cached.

        Args:
            country (str): The ISO country code for which the predictions are to be made.

        Returns:
            Any: The prediction result from the model.
        """
        if country in self.predictions:
            return self.predictions[country]
        data = trf.create_predicition_sample(self.start_date, self.end_date, country)
        self.predictions[country] = self.model.predict(data)
        return self.predictions[country]

    def build_n_months_ahead_predictions(self, n_months_ahead: int, country: str) -> pd.DataFrame:
        # Collect data to build DataFrame from
        results = {"index": [], "data": []}
        y_pred = self.predict(country)[:, :n_months_ahead, :]
        for imonth, month in enumerate(trf.loop_over_month_starts(self.start_date, self.end_date)):

            # Step 1: Transform prediction tensor into a DataFrame with n_months_ahead columns
            # with the index starting at start date and the predicted quantiles transformed
            # into columns
            pred_df = pd.DataFrame(
                data=y_pred[imonth],
                columns=[f"quantile_{q:.2f}" for q in tft.quantiles],
                index=trf.create_monthly_index(month, n_months_ahead),
            )

            # Step 2: Combine the n_months_ahead predictions with 12 - n_months_ahead true
            # inflations
            n_true_months = 12 - n_months_ahead
            start_idx = tft.df_target_1m_pct.index.get_loc(month)
            prev_months_truth = tft.df_target_1m_pct.iloc[start_idx - n_true_months : start_idx][
                country
            ].rename("inflation")
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
        return pd.DataFrame(index=results["index"], data=results["data"])
