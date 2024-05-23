import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, AutoRegResultsWrapper


def fit_ar_model(
    df_target: pd.DataFrame, country: str, start_date: str, end_date: str, lags: int
) -> tuple[AutoReg, AutoRegResultsWrapper]:
    """Fits an AutoRegressive model on the target dataframe for a specific country and date range.

    Args:
        df_target: Target dataframe containing the time series data.
        country: Country for which the model is to be fit.
        start_date: Start date of the period for fitting the model.
        end_date: End date of the period for fitting the model.
        lags: The number of lags to be used in the model.

    Returns:
        A tuple containing the fitted AutoReg model and the results.
    """
    # Silence warnings
    df_target.index.freq = "MS"

    # Drop the first row of nans from calculating pct_changes
    df_target = df_target.dropna()

    # Fit the model
    model = AutoReg(df_target.loc[start_date:end_date, country], lags)
    result = model.fit()

    return model, result
