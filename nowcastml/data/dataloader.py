from dataclasses import dataclass
from dateutil.relativedelta import relativedelta
import keras
import numpy as np
import pandas as pd
from typing import Optional


def sample_nowcasting_data(
    dfs_input: dict[
        str, pd.DataFrame
    ],  # dict mapping frequency codes to DataFrames with the time series of the input variables
    df_target: pd.DataFrame,  # DataFrame with the time series of the target variable
    country_enc_dict: dict[str, int],
    country_dec_dict: dict[int, str],
    cont_cols,
    cat_cols,
    min_context: int = 90,  # minimum context length in number of daily observations
    context_length: int = 365,  # context length in number of daily observations (leads to padding if not reached in sampling)
    num_months: int = 12,  # number of months (1 is current month)
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
        dfs_hist[freq] = dfs_input[freq].loc[earliest_date:sampled_day]

    dfs_padded = pad_to_context_length(dfs_hist, context_length)

    X_cont_hist = {}
    X_cat_hist = {}
    for freq, df_pad in dfs_padded.items():
        # Pad with zeroes if less data than context_length available
        # TODO: padding for monthly and quarterly data
        X_cont_hist[freq] = df_pad[cont_cols[freq]].values
        X_cat_hist[freq] = df_pad[cat_cols[freq]].values

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

    # For predictions at recent dates no labels for all future twelve months are available -
    # provide the option to skip creating the target variable in that case
    if skip_y:
        return [X_cont_hist, X_cat_hist, X_fut, X_cat_stat], None
    # create the target variables
    y = df_target.loc[target_month, country_dec_dict[int(X_cat_stat[0])]].values

    return [X_cont_hist, X_cat_hist, X_fut, X_cat_stat], y


def sample_dates(
    df_daily_input: pd.DataFrame,
    min_context: int = 90,
    num_months: int = 12,
    context_length: int = 365,
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
    earliest_poss_date = all_dates[min_context]
    delta_latest_month = num_months - 1
    latest_poss_date = (
        all_dates.max()
        - relativedelta(months=delta_latest_month)
        + pd.offsets.MonthEnd(0)
    )
    dates_for_sampling = df_daily_input.loc[earliest_poss_date:latest_poss_date].index
    # sample a random date, context length and country
    if sampled_day is None:
        sampled_day = pd.to_datetime(np.random.choice(dates_for_sampling))
        sampled_context_length = (
            np.random.randint(low=min_context, high=context_length)
            if min_context < context_length
            else context_length
        )
    else:
        # sampled_context_length is the longest possible since setting a date means the data
        # will not be used for training models but for prediction/evaluation
        sampled_day = pd.to_datetime(sampled_day)
        sampled_context_length = context_length

    earliest_date = sampled_day - relativedelta(days=sampled_context_length - 1)
    return earliest_date, sampled_day


def pad_to_context_length(dfs: dict[str, pd.DataFrame], context_length: int):
    """Pads daily, monthly, and quarterly DataFrames to ensure they meet a specified context length

    Uses the padded daily DataFrame to generate the proper indices for monthly and quarterly padding.

    Args:
        dfs: Dictionary mapping frequency keys to DataFrames
        context_length: The target number of rows each DataFrame should have.

    Returns:
        The padded daily, monthly, and quarterly DataFrames.
    """
    # Pad the daily frequency DataFrame
    df_daily = dfs["d"]
    if df_daily.shape[0] < context_length:
        df_pad_daily = pd.DataFrame(
            np.zeros((context_length - df_daily.shape[0], df_daily.shape[1])),
            columns=df_daily.columns,
        )
        df_pad_daily.index = pd.date_range(
            end=df_daily.index.min() - pd.Timedelta(days=1),
            periods=(context_length - df_daily.shape[0]),
            freq="D",
        )
        df_daily = pd.concat([df_pad_daily, df_daily])
    # Resample the daily DataFrame to to get the new indices fro the lower frequencies
    monthly_index = df_daily.resample("MS").asfreq().index
    quarterly_index = df_daily.resample("QS").asfreq().index

    # Pad the monthly frequency DataFrame to match the dates in the monthly_index
    df_monthly = dfs["m"]
    if df_monthly.shape[0] < len(monthly_index):
        df_pad_monthly = pd.DataFrame(
            np.zeros((len(monthly_index) - df_monthly.shape[0], df_monthly.shape[1])),
            columns=df_monthly.columns,
        )
        df_pad_monthly.index = monthly_index[: len(df_pad_monthly)]
        df_monthly = pd.concat([df_pad_monthly, df_monthly])
        df_monthly.index = monthly_index

    # Pad the quarterly frequency DataFrame to match the dates in the quarterly_index
    df_quarterly = dfs["q"]
    if df_quarterly.shape[0] < len(quarterly_index):
        df_pad_quarterly = pd.DataFrame(
            np.zeros(
                (len(quarterly_index) - df_quarterly.shape[0], df_quarterly.shape[1])
            ),
            columns=df_quarterly.columns,
        )
        df_pad_quarterly.index = quarterly_index[: len(df_pad_quarterly)]
        df_quarterly = pd.concat([df_pad_quarterly, df_quarterly])
        df_quarterly.index = quarterly_index

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
