import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import temporal_fusion_transformers as tft

def predictions(model, country, sampled_day):
    """Create an array of predictions"""
    x, _ = tft.prepare_data_samples(
        n_samples=1,
        df_daily_input=tft.df_input_scl,
        df_target=tft.df_target_1m_pct,
        sampled_day=sampled_day,
        min_context=365,
        context_length=365,
        country=country,
    )
    # Prediction shape: n_samples, months, quantiles
    y_pred = model.predict(x)
    return y_pred

def get_monthly_target_df(model, country, target_date):
    y_pred = predictions(model, country, target_date)
    start_idx = tft.df_target_1m_pct.index.get_loc(target_date.replace(day=1))

    next_12_months_truth = tft.df_target_1m_pct.iloc[start_idx: start_idx+12][country].rename("truth")

    pred_df = pd.DataFrame(
        data=y_pred[0],
        columns=[f"quantile_{q:.2f}" for q in tft.quantiles],
        index = next_12_months_truth.index
    )

    df_next_12_months = pd.DataFrame(next_12_months_truth).join(pred_df)
    previous_12_months_truth = tft.df_target_1m_pct.iloc[start_idx-12: start_idx][country].rename("truth")
    # We have no true quantiles in the past -> broadcast the truth to the quantile columns
    df = pd.concat((pd.DataFrame(previous_12_months_truth), df_next_12_months))
    for q in tft.quantiles:
        df[f"quantile_{q:.2f}"].fillna(df["truth"], axis=0, inplace=True)
    df_roll = ((1+df/100.).rolling(window=12).apply(np.prod, raw=True) - 1)*100
    df_roll.columns = [f"{col}_yoy" for col in df_roll.columns]
    df_all = pd.concat((df, df_roll), axis=1)
    df_all["truth_yoy"] = tft.df_target_12m_pct.iloc[start_idx-3: start_idx+12][country]
    return df_all


def plot_mom_change(model, country, target_date, ax=None):
    monthly_target = get_monthly_target_df(model, country, target_date) 
    if ax is None:
        fig, ax = plt.subplots()
    # Plot a 15 months window comparing truth and prediction, mark the cutoff date
    monthly_target = monthly_target.iloc[9:]
    monthly_target["truth"].plot(ax=ax, color="r", label="Truth")
    monthly_target[f"quantile_0.50"].plot(ax=ax, color="b", label="Prediction")
    ax.fill_between(monthly_target.index, monthly_target["quantile_0.05"], monthly_target["quantile_0.95"], color="b", alpha=0.15)
    ax.fill_between(monthly_target.index, monthly_target["quantile_0.25"], monthly_target["quantile_0.75"], color="b", alpha=0.35)
    ax.legend()
    ax.set(ylabel="Month-on-month inflation change [%]", title="US Inflation example")
    print(target_date)
    ax.axvline(x=target_date-pd.DateOffset(months=1), color="gray", ls="--")
    return fig


def plot_yoy_change(model, country, target_date, ax=None):
    monthly_target = get_monthly_target_df(model, country, target_date) 
    if ax is None:
        fig, ax = plt.subplots()
    # Plot a 15 months window comparing truth and prediction, mark the cutoff date
    monthly_target = monthly_target.iloc[10:]
    monthly_target["truth_yoy"].plot(ax=ax, color="r", label="Truth")
    monthly_target["quantile_0.50_yoy"].plot(ax=ax, color="b", label="Prediction")
    ax.fill_between(monthly_target.index, monthly_target["quantile_0.05_yoy"], monthly_target["quantile_0.95_yoy"], color="b", alpha=0.15)
    ax.fill_between(monthly_target.index, monthly_target["quantile_0.25_yoy"], monthly_target["quantile_0.75_yoy"], color="b", alpha=0.35)

    ax.legend()
    ax.set(ylabel="Year-on-Year inflation change [%]", title="US Inflation example")
    ax.axvline(x=target_date-pd.DateOffset(months=1), color="gray", ls="--")
    return fig
