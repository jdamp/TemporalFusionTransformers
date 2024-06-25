"""Module containing utilities for loading multi-frequency TFT data"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Optional

import keras
import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from torch import Tensor
from .utils import get_max_days_in_n_months, get_max_quarters_in_n_months

DATA_KEYS = ["X_cont_hist", "X_cat_hist", "X_cat_stat", "X_fut"]


@dataclass
class MultiFrequencySamples:
    """Dataclass that encapsulates different categories of multi-frequency data for the TFT.

    In the dictionary fields X_cont_hist and X_cat_hist, each key corresponds to a frequency.
    The associated values are lists of Tensors, where each list entry corresponds to a single data
    sample.

    Attributes:
        X_cont_hist: Continuous, historical data for multiple frequencies.
        X_cat_hist: Categorical, historical data for multiple frequences.
        X_fut: A list of tensors representing known future data.
        X_cat_stat: A list of tensors representing categorical static data.
    """

    X_cont_hist: dict[str, Tensor]
    X_cat_hist: dict[str, Tensor]
    X_fut: Tensor
    X_cat_stat: Tensor

    def print(self):
        for freq in self.X_cont_hist:
            print(f"Frequency: {freq}")
            print(f"{self.X_cont_hist[freq].shape=}")
            print(f"{self.X_cat_hist[freq].shape=}")
        print(f"{self.X_fut.shape=}")
        print(f"{self.X_cat_stat}")

    def concatenate(self, other: "MultiFrequencySamples") -> "MultiFrequencySamples":
        # Concatenate list fields
        new_X_fut = keras.ops.concatenate((self.X_fut, other.X_fut))
        new_X_cat_stat = keras.ops.concatenate((self.X_cat_stat, other.X_cat_stat))

        new_X_cont_hist = {}
        new_X_cat_hist = {}

        for freq in self.X_cont_hist:
            new_X_cont_hist[freq] = keras.ops.concatenate(
                (self.X_cont_hist[freq], other.X_cont_hist[freq])
            )

        for freq in self.X_cat_hist:
            new_X_cat_hist[freq] = keras.ops.concatenate(
                (self.X_cat_hist[freq], other.X_cat_hist[freq])
            )

        return MultiFrequencySamples(
            X_cont_hist=new_X_cont_hist,
            X_cat_hist=new_X_cat_hist,
            X_fut=new_X_fut,
            X_cat_stat=new_X_cat_stat,
        )

    @classmethod
    def concatenate_datasets(
        cls, datasets: list["MultiFrequencySamples"]
    ) -> "MultiFrequencySamples":
        if not datasets:
            raise ValueError("Received empty list")

        result = datasets[0]
        for dataset in datasets[1:]:
            result = result.concatenate(dataset)
        return result


def concatenate(datasets: MultiFrequencySamples):
    return MultiFrequencySamples.concatenate_datasets(datasets)


def sample_nowcasting_data(
    dfs_input: dict[
        str, pd.DataFrame
    ],  # dict mapping frequency codes to DataFrames with the time series of the input variables
    df_target: pd.DataFrame,  # DataFrame with the time series of the target variable
    country_enc_dict: dict[str, int],
    country_dec_dict: dict[int, str],
    cont_cols,
    cat_cols,
    min_context: int = 3,  # minimum context length in number of monthly observations
    context_length: int = 12,  # context length in number of monthly observations (leads to padding if not reached in sampling)
    num_months: int = 12,  # number of months to nowcast(1 is current month)
    sampled_day: (
        None | str
    ) = None,  # None (default) randomly chooses a date; otherwise, YYYY-MM-DD date selected a date
    country: (
        None | str
    ) = None,  # None (default) randomly chooses a country; otherwise, 2-digit ISO code selectes a country
    skip_y: bool = False,
):
    earliest_date, sampled_day = sample_dates(
        dfs_input["d"], min_context, num_months, context_length, sampled_day
    )

    X_cat_stat = (
        np.random.randint(low=1, high=len(country_enc_dict) + 1, size=(1,))
        if country is None
        else keras.ops.reshape(np.array(country_enc_dict[country]), (1,))
    )

    dfs_hist = {}
    for freq in dfs_input:
        # if freq == "m":
        #    earliest_date = earliest_date.replace(day=1)
        # if freq == "q":
        #    earliest_date = earliest_date.to_period("Q").to_timestamp()

        dfs_hist[freq] = dfs_input[freq].loc[earliest_date:sampled_day]

    # Pad with zeroes if less data than context_length available
    dfs_padded = pad_to_context_length(dfs_hist, context_length)

    # create the future known data: month of the year
    # note: any other known future information of interest should be included here as well
    # eg: mon pol committee meeting dates, months of major sports events, etc
    # anything that could influence inflation dynamics
    target_month = (
        (sampled_day + relativedelta(months=num_months)).replace(day=1)
        if num_months == 1
        else [
            (sampled_day + relativedelta(months=d)).replace(day=1)
            for d in range(num_months)
        ]
    )
    X_fut = date_features(
        pd.DataFrame(index=target_month).index, is_monthly=True
    ).values

    X_cont_hist = {}
    X_cat_hist = {}
    for freq, df_pad in dfs_padded.items():
        X_cont_hist[freq] = keras.ops.expand_dims(
            Tensor(df_pad[cont_cols[freq]].values), axis=0
        )
        X_cat_hist[freq] = keras.ops.expand_dims(
            Tensor(df_pad[cat_cols[freq]].values), axis=0
        )

    X_fut = keras.ops.expand_dims(Tensor(X_fut), axis=0)
    X_cat_stat = keras.ops.expand_dims(Tensor(X_cat_stat), axis=0)

    sample = MultiFrequencySamples(X_cont_hist, X_cat_hist, X_fut, X_cat_stat)

    # For predictions at recent dates no labels for all future twelve months are available -
    # provide the option to skip creating the target variable in that case
    if skip_y:
        return sample, None
    # create the target variables
    y = Tensor(df_target.loc[target_month, country_dec_dict[int(X_cat_stat[0])]].values)

    return sample, y, dfs_padded


def sample_dates(
    df_daily_input: pd.DataFrame,
    min_context_months: int = 3,
    num_months: int = 12,
    context_length_months: int = 12,
    sampled_day: Optional[str] = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Samples a date range for a TFT input data sample

    A random context length is sampled from min_context up to context_length.
    Between the sampled dates and the end date of the training data, dates equivalent to
    'num_months' months must be left out.

    Parameters:
        df_daily_input: Nowcasting DataFrame with daily frequency
        min_context: Minimum context length in number of daily observations. Defaults to 90.
        num_months:  Number of months to nowcast (1 is current month). Defaults to 12.
        context_length: Context length in number of daily observations. Leads to padding if not
            reached in sampling. Defaults to 365.
        sampled_day: If None, randomly chooses a date; otherwise, a date in the format YYYY-MM-DD
            can be specified. Defaults to None.

    Returns:
        Start and end date timestamp of the sampling range
    """
    # first step: determine the valid dates for sampling
    # from the left, they should allow at least `context_length` up until the sampled date
    # from the right, they should allow enough room to retrieve all of the target inflation months
    # only the dates "in the middle" can be sampled
    all_dates = df_daily_input.index
    first_date = all_dates[0]
    earliest_poss_date = first_date + relativedelta(months=min_context_months)
    delta_latest_month = num_months - 1
    latest_poss_date = (
        all_dates.max()
        - relativedelta(months=delta_latest_month)
        + pd.offsets.MonthEnd(0)
    )
    dates_for_sampling = df_daily_input.loc[earliest_poss_date:latest_poss_date].index
    ## sample a random date, context length and country
    if sampled_day is None:
        sampled_day = pd.to_datetime(np.random.choice(dates_for_sampling))
        sampled_context_length = (
            np.random.randint(low=min_context_months, high=context_length_months)
            if min_context_months < context_length_months
            else context_length_months
        )
    else:
        # sampled_context_length is the longest possible since setting a date means the data
        # will not be used for training models but for prediction/evaluation
        sampled_day = pd.to_datetime(sampled_day)
        sampled_context_length = context_length_months

    earliest_date = sampled_day - relativedelta(months=sampled_context_length)
    return earliest_date, sampled_day


def pad_to_context_length(dfs: dict[str, pd.DataFrame], context_length: int):
    """Pads daily, monthly, and quarterly DataFrames to ensure they meet a specified context length

    Uses the monthly DataFrame to generate the proper indices for daily and quarterly padding.

    Args:
        dfs: Dictionary mapping frequency keys to DataFrames
        context_length: The target number of rows each DataFrame should have.

    Returns:
        The padded daily, monthly, and quarterly DataFrames.
    """
    # Determine max number of days in any 'context_length windows
    n_days = get_max_days_in_n_months(context_length)
    # Pad the daily frequency DataFrame
    df_daily = dfs["d"]
    last_day = df_daily.index[-1]
    daily_index = pd.date_range(end=last_day, periods=n_days, freq="D")
    if df_daily.shape[0] < n_days:
        df_pad_daily = pd.DataFrame(
            np.zeros((n_days - df_daily.shape[0], df_daily.shape[1])),
            columns=df_daily.columns,
        )
        df_pad_daily.index = daily_index[: len(df_pad_daily)]
        df_daily = pd.concat([df_pad_daily, df_daily])

    # Monthly DataFrame
    df_monthly = dfs["m"]
    last_month = df_monthly.index[-1]
    monthly_index = pd.date_range(end=last_month, periods=context_length, freq="MS")
    if df_monthly.shape[0] < context_length:
        df_pad_monthly = pd.DataFrame(
            np.zeros((context_length - df_monthly.shape[0], df_monthly.shape[1])),
            columns=df_monthly.columns,
        )
        df_pad_monthly.index = monthly_index[: len(df_pad_monthly)]
        df_monthly = pd.concat([df_pad_monthly, df_monthly])

    # Pad the quarterly frequency DataFrame to match the dates in the quarterly_index
    n_quarters = get_max_quarters_in_n_months(context_length)
    df_quarterly = dfs["q"]
    quarterly_index = pd.date_range(end=last_month, periods=n_quarters, freq="QS")
    if df_quarterly.shape[0] < n_quarters:
        df_pad_quarterly = pd.DataFrame(
            np.zeros(
                (len(quarterly_index) - df_quarterly.shape[0], df_quarterly.shape[1])
            ),
            columns=df_quarterly.columns,
        )
        df_pad_quarterly.index = quarterly_index[: len(df_pad_quarterly)]
        df_quarterly = pd.concat([df_pad_quarterly, df_quarterly])
        # df_full_quarterly.index = quarterly_index

    return {"d": df_daily, "m": df_monthly, "q": df_quarterly}


def date_features(
    date_range,  # Range of dates for which to create date features
    is_monthly: bool = False,  # Is the date measured at the monthly frequency?
) -> pd.DataFrame:  # Categorical date features
    "Categorical features for each day in a range of dates"
    if is_monthly:
        return pd.DataFrame({"Month of Year": date_range.month})
    else:
        return pd.DataFrame(
            {
                "Day of Week": date_range.dayofweek
                + 1,  # This is the only date feature with zeros, which are masked out
                "Day of Month": date_range.day,
                "Day of Year": date_range.dayofyear,
                "Week of Month": (date_range.day - 1) // 7 + 1,
                "Week of Year": pd.Index(date_range.isocalendar().week).astype("int32"),
                "Month of Year": date_range.month,
            }
        )


def pad_and_stack_tensors(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Pads and stacks a list of 2D tensors with varying sizes in the first dimension.

    This function finds the maximum size of the first dimension among all input tensors,
    pads the smaller tensors with zeros at the beginning of the first dimension to match
    this size, and then stacks all tensors into a single 3D tensor.

    Args:
        tensor_list: A list of 2D tensors to be padded and stacked.
            All tensors must have the same size in the second dimension.

    Returns:
        torch.Tensor: A 3D tensor containing all input tensors stacked together after padding.
            The shape will be (batch_size, max_months, num_features)

    Raises:
        ValueError: If the input list is empty or if the tensors have different sizes
                    in the second dimension.

    Example:
        >>> t1 = torch.rand(3, 5)
        >>> t2 = torch.rand(4, 5)
        >>> result = pad_and_stack_tensors([t1, t2])
        >>> print(result.shape)
        torch.Size([2, 4, 5])
    """
    if not tensor_list:
        raise ValueError("Input list is empty")

    if any(tensor.dim() != 2 for tensor in tensor_list):
        raise ValueError("All input tensors must be 2-dimensional")

    second_dim = tensor_list[0].size(1)
    if any(tensor.size(1) != second_dim for tensor in tensor_list):
        raise ValueError(
            "All input tensors must have the same size in the second dimension"
        )

    max_size = max(tensor.size(0) for tensor in tensor_list)

    padded_tensors = [
        F.pad(tensor, (0, 0, max_size - tensor.size(0), 0)) for tensor in tensor_list
    ]
    return torch.stack(padded_tensors)
