"""Module containing data transformation functions used to build nowcasting predictions"""

from typing import Generator

import keras
import numpy as np
import pandas as pd
import torch

import temporal_fusion_transformers as tft


def create_predicition_sample(start_date: str, end_date: str, country: str) -> list[torch.Tensor]:
    """Creates a sample that can be directly passed to the model to create predictions for
    a specific country between 'start_date' and 'end_date'.

    Args:
        start_date (str): The start_date for the sample
        end_date (str): The end date for the sample
        country (str): The country for which the prediction will be performed

    Returns:
        list[torch.Tensor]: List of inputs to the TFT model
    """
    X_cont_hist = []
    X_cat_hist = []
    X_fut = []
    X_cat_stat = []
    for month in loop_over_month_starts(start_date, end_date):
        x, _ = tft.sample_nowcasting_data(
            df_daily_input=tft.df_input_scl,
            df_target=tft.df_target_1m_pct,
            sampled_day=month,
            min_context=365,
            context_length=365,
            country=country,
            skip_y=True,
        )
        [indiv_cont_hist, indiv_cat_hist, indiv_fut, indiv_cat_stat] = x
        X_cont_hist.append(indiv_cont_hist)
        X_cat_hist.append(indiv_cat_hist)
        X_fut.append(indiv_fut)
        X_cat_stat.append(indiv_cat_stat)

    X_cont_hist = keras.ops.stack(X_cont_hist, axis=0)
    X_cat_hist = keras.ops.stack(X_cat_hist, axis=0)
    X_fut = keras.ops.stack(X_fut, axis=0)
    X_cat_stat = keras.ops.stack(X_cat_stat, axis=0)
    X = [X_cont_hist, X_cat_hist, X_fut, X_cat_stat]
    return X


def create_monthly_index(start_date: str, n: int) -> pd.DatetimeIndex:
    """
    Creates a Pandas DatetimeIndex starting from a given date with n monthly starts.

    Args:
    start_date (str): The start date in 'YYYY-MM-DD' format, expected to be the first of a month.
    n (int): The number of months to include in the index.

    Returns:
    pd.DatetimeIndex: A DatetimeIndex with n monthly starts beginning from start_date.
    """
    start = pd.to_datetime(start_date)
    end = start + pd.DateOffset(months=n - 1)
    return pd.date_range(start=start, end=end, freq="MS")


def yoy_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate year-over-year (YoY) rolling growth rates from monthly percentage growth rates
    in a DataFrame, calculated as a 12-month rolling product of the input monthly growth rates.

    Args:
        df (pd.DataFrame): A DataFrame with columns representing monthly percentage growth rates.

    Returns:
        pd.DataFrame: A DataFrame with columns representing YoY percentage growth rates,
    """
    # convert percentages into growth factors (0% -> 1, 5% -> 1.05, ...)
    monthly_growths = df / 100.0 + 1
    yoy_growths = monthly_growths.rolling(window=12).apply(np.prod, raw=True)
    # convert back to percentages
    return (yoy_growths - 1) * 100


def loop_over_month_starts(start_date: str, end_date: str) -> Generator[pd.Timestamp, None, None]:
    """
    Generate all monthly start dates between two date strings.

    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Yields:
        pd.Timestamp: The next monthly start date in the range from start_date to end_date.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    current = start

    while current <= end:
        if current.day == 1:
            yield current
        current += pd.DateOffset(months=1)
        current = current.replace(day=1)


def transform_rmse_box_plot(run_df: pd.DataFrame) -> pd.DataFrame:
    """Restructures the DataFrame so that it's in a long format, suitable for creating box plots
    of the n-month-ahead RMSE.
    It extracts the number of months and country codes from the column names.

    Args:
        run_df (pd.DataFrame): DataFrame with run information on mlflow

    Returns:
        pd.DataFrame: DataFrame with Country, "Months Ahead" and "RMSE" column
    """

    data = []
    for column in run_df.columns:
        match = re.match(r"metrics\.rmse_yoy_([A-Z]{2})_(\d+)_months_ahead", column)
        if match:
            country_code, months_ahead = match.groups()
            for value in run_df[column]:
                data.append(
                    {"Country": country_code, "Months Ahead": int(months_ahead), "RMSE": value}
                )
    return pd.DataFrame(data)
