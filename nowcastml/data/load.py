"""Contains utilities to load the multi-frequency Nowcasting data"""

import pandas as pd

_DEFAULT_FREQUENCIES = ["d", "m", "q"]


def load_multi_freq_data(
    path: str,
    countries: list[str],
    variables: dict[str, list[str]],
    start_date: str,
) -> dict[str, pd.DataFrame]:
    """Load multi-frequency nowcasting data from a CSV file into a pandas DataFrame.

    Parameters:
        path: The path to the CSV file.
        countries: List of countries to use in the dataset.
        variables: Dictionary mapping frequency keys to a list of variables for each frequency
        start_date: Start date for the dataset


    Returns:
        Dictionary mapping frequency keys to DataFrame objects
    """
    df_all = pd.read_csv(path)
    df_all["index"] = pd.to_datetime(df_all["index"], format="mixed")
    df_all.set_index("index", drop=True, inplace=True)
    # Ignore errors and continue in case no "Unnamed: 0" column exists
    df_all.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

    df_filtered = _filter_data(
        df_all,
        countries=countries,
        frequencies=_DEFAULT_FREQUENCIES,
        start_date=start_date,
        variables=variables,
    )

    return df_filtered


def _filter_data(
    data: pd.DataFrame,
    countries: list[str],
    frequencies: list[str],
    start_date: str,
    variables: dict[str, list[str]],
    drop_na: bool = True,
) -> dict[str, pd.DataFrame]:
    filter_country = data["country"].isin(countries)
    filter_dates = data.index >= start_date

    df_input = data.loc[filter_country & filter_dates]

    dfs = {}
    for freq in frequencies:
        filter_freq = df_input["frequency"] == freq
        # List of columns we want to keep for this particular frequrency
        columns = variables[freq] + ["country"]
        df_freq = df_input.loc[filter_freq, columns]
        df_out = set_country_index_prefix(df_freq)
        if freq == "d":
            df_out = _resample_to_daily(df_out)
        if drop_na:
            df_out.dropna(how="all", inplace=True)
        dfs[freq] = df_out
    return dfs


def _resample_to_daily(df_business_daily: pd.DataFrame) -> pd.DataFrame:
    """Resamples the daily data with business day frequency to a daily frequency

    Args:
        df_daily (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Assert that our DataFrame actually has a business day frequency
    days = df_business_daily.index
    expected = pd.date_range(start=days.min(), end=days.max(), freq="B")
    if not days.equals(expected):
        raise ValueError("Input DataFrame differs from a business day frequency")
    df_daily = df_business_daily.resample("D").ffill()
    return df_daily


def set_country_index_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Set country as a prefix in the index of the DataFrame.

    Parameters:
    df (pd.DataFrame): The original DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with country as a prefix in the index.
    """
    df_multi = df.reset_index().set_index(["index", "country"])
    df_unstack = df_multi.unstack("country")
    df_unstack.columns = ["__".join(col).strip() for col in df_unstack.columns.values]
    return df_unstack


def build_target_frames(
    df: pd.DataFrame, target_var: str = "CPIh"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds the target frames from the variable 'target_var'

    Will contain columns named according to the countries.

    Args:
        df: Transformed nowcasting dataset
        target_var: Variable to use. Defaults to "CPIh".

    Returns:
        Target DataFrame containing monthly and yearly percentage change of the target variable
    """
    df_target = df.filter(regex=f"{target_var}__.*")
    new_columns = [col.split("__")[1] for col in df_target.columns]
    df_target.columns = new_columns
    df_target_12m_pct = 100 * df_target.pct_change(12)
    df_target_1m_pct = 100 * df_target.pct_change(1)
    return df_target_1m_pct, df_target_12m_pct
