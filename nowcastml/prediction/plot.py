import re

from typing import Optional

from statsmodels.tsa.ar_model import AutoReg

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import prediction.transform as trf

import temporal_fusion_transformers as tft


def predictions(model, country, start_date, end_date):
    """Create an array of predictions"""
    X = trf.create_predicition_sample(start_date, end_date, country)
    # Prediction shape: n_samples, months, quantiles
    y_pred = model.predict(X)
    return y_pred


# generate AR forecasts
def generate_ar_forecast(ar_model, country, start_date: str, end_date: str, n_months_ahead: int):
    df_target = tft.df_target_1m_pct

    ar_predictions = []
    for month in trf.loop_over_month_starts(start_date, end_date):
        # Get index of current target month and fit model
        idx = df_target.index.get_loc(month)
        prediction = ar_model.predict(month, month + pd.DateOffset(months=n_months_ahead))
        # Join this with the 12 - n_months_ahead previous months of truth
        prev_months_truth = df_target[idx - (12 - n_months_ahead) : idx][country]
        all_months = pd.concat((prev_months_truth, prediction))
        x = trf.yoy_rolling(all_months)
        ar_predictions.append((month, x.iloc[-1]))
    return (
        pd.DataFrame(ar_predictions)
        .set_index(0)
        .rename(index={0: "date"}, columns={1: f"AutoReg_{country}_{n_months_ahead}"})
    )


def build_n_months_prediction_df(
    model,
    n_months_ahead: int,
    country: str,
    start_date: str,
    end_date: str,
):
    res = {"index": [], "data": []}
    y_pred = predictions(model, country, start_date, end_date)
    # Select only the n months ahead from the prediction (and the first and only batch)
    y_pred_ahead = y_pred[:, :n_months_ahead, :]
    for imonth, month in enumerate(trf.loop_over_month_starts(start_date, end_date)):
        pred_df = pd.DataFrame(
            data=y_pred_ahead[imonth],
            columns=[f"quantile_{q:.2f}" for q in tft.quantiles],
            index=trf.create_monthly_index(month, n_months_ahead),
        )
        # Join this with the 12 - n_months_ahead previous months of truth
        start_idx = tft.df_target_1m_pct.index.get_loc(month)
        prev_months_truth = tft.df_target_1m_pct.iloc[
            start_idx - (12 - n_months_ahead) : start_idx
        ][country].rename("inflation")
        # We have no true quantiles in the past -> broadcast the truth to the quantile columns
        df = pd.concat((pd.DataFrame(prev_months_truth), pred_df))
        for q in tft.quantiles:
            df[f"quantile_{q:.2f}"].fillna(df["inflation"], axis=0, inplace=True)
        df["inflation"].fillna(df["quantile_0.50"], inplace=True)
        df_roll = trf.yoy_rolling(df)
        # inflation prediction for this month is the last row
        res["index"].append(month)
        res["data"].append(df_roll.iloc[-1])
    return pd.DataFrame(index=res["index"], data=res["data"])


def build_monthly_prediction_df(
    tft_model: keras.Model,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    country: str,
):
    x = trf.create_predicition_sample(start_date=start_date, end_date=end_date, country=country)
    predictions = tft_model.predict(x)
    pred_df = pd.DataFrame(
        data=predictions[:, 0, :],
        columns=[f"quantile_{q:.2f}" for q in tft.quantiles],
        index=trf.create_monthly_index(start_date, len(predictions)),
    )
    return pred_df


def yoy_plot(
    tft_model: keras.Model,
    ar_model: AutoReg,
    start_date: str,
    end_date: str,
    country: str,
    n_months_ahead: int,
    quantiles: bool,
):
    ar_forecasts = generate_ar_forecast(ar_model, country, start_date, end_date, n_months_ahead)
    truths = tft.df_target_12m_pct.loc[start_date:end_date, country].rename("truth")
    tft_pred = build_n_months_prediction_df(
        tft_model, n_months_ahead, country, start_date, end_date
    )
    fig, ax = plt.subplots()
    ax.set_title(f"{n_months_ahead} months ahead yoy growth, {country}")
    tft_pred["inflation"].plot(ax=ax, label="prediction")
    if quantiles:
        ax.fill_between(
            tft_pred.index,
            tft_pred["quantile_0.05"],
            tft_pred["quantile_0.95"],
            color="b",
            alpha=0.15,
        )
        ax.fill_between(
            tft_pred.index,
            tft_pred["quantile_0.25"],
            tft_pred["quantile_0.75"],
            color="b",
            alpha=0.35,
        )
    ar_forecasts.plot(ax=ax, label="prediction AR")
    truths.plot(ax=ax, label="Truth")
    ax.legend()
    ax.axvline(pd.Timestamp("2018-01-01"), ls="--", color="gray")


def get_monthly_target_df(model, country, target_date):
    y_pred = predictions(model, country, target_date)
    start_idx = tft.df_target_1m_pct.index.get_loc(target_date.replace(day=1))

    next_12_months_truth = tft.df_target_1m_pct.iloc[start_idx : start_idx + 12][country].rename(
        "truth"
    )

    pred_df = pd.DataFrame(
        data=y_pred[0],
        columns=[f"quantile_{q:.2f}" for q in tft.quantiles],
        index=next_12_months_truth.index,
    )

    df_next_12_months = pd.DataFrame(next_12_months_truth).join(pred_df)
    previous_12_months_truth = tft.df_target_1m_pct.iloc[start_idx - 12 : start_idx][
        country
    ].rename("truth")
    # We have no true quantiles in the past -> broadcast the truth to the quantile columns
    df = pd.concat((pd.DataFrame(previous_12_months_truth), df_next_12_months))
    for q in tft.quantiles:
        df[f"quantile_{q:.2f}"].fillna(df["truth"], axis=0, inplace=True)
    df_roll = ((1 + df / 100.0).rolling(window=12).apply(np.prod, raw=True) - 1) * 100
    df_roll.columns = [f"{col}_yoy" for col in df_roll.columns]
    df_all = pd.concat((df, df_roll), axis=1)
    df_all["truth_yoy"] = tft.df_target_12m_pct.iloc[start_idx - 3 : start_idx + 12][country]
    return df_all


def plot_mom_change(model, country, target_date, ax=None):
    monthly_target = get_monthly_target_df(model, country, target_date)
    if ax is None:
        fig, ax = plt.subplots()
    # Plot a 15 months window comparing truth and prediction, mark the cutoff date
    monthly_target = monthly_target.iloc[9:]
    monthly_target["truth"].plot(ax=ax, color="r", label="Truth")
    monthly_target[f"quantile_0.50"].plot(ax=ax, color="b", label="Prediction")
    ax.fill_between(
        monthly_target.index,
        monthly_target["quantile_0.05"],
        monthly_target["quantile_0.95"],
        color="b",
        alpha=0.15,
    )
    ax.fill_between(
        monthly_target.index,
        monthly_target["quantile_0.25"],
        monthly_target["quantile_0.75"],
        color="b",
        alpha=0.35,
    )
    ax.legend()
    ax.set(ylabel="Month-on-month inflation change [%]", title="US Inflation example")
    print(target_date)
    ax.axvline(x=target_date - pd.DateOffset(months=1), color="gray", ls="--")
    return fig


def plot_yoy_change(model, country, target_date, ax=None):
    monthly_target = get_monthly_target_df(model, country, target_date)
    if ax is None:
        fig, ax = plt.subplots()
    # Plot a 15 months window comparing truth and prediction, mark the cutoff date
    monthly_target = monthly_target.iloc[10:]
    monthly_target["truth_yoy"].plot(ax=ax, color="r", label="Truth")
    monthly_target["quantile_0.50_yoy"].plot(ax=ax, color="b", label="Prediction")
    ax.fill_between(
        monthly_target.index,
        monthly_target["quantile_0.05_yoy"],
        monthly_target["quantile_0.95_yoy"],
        color="b",
        alpha=0.15,
    )
    ax.fill_between(
        monthly_target.index,
        monthly_target["quantile_0.25_yoy"],
        monthly_target["quantile_0.75_yoy"],
        color="b",
        alpha=0.35,
    )

    ax.legend()
    ax.set(ylabel="Year-on-Year inflation change [%]", title="US Inflation example")
    ax.axvline(x=target_date - pd.DateOffset(months=1), color="gray", ls="--")
    return fig


def plot_scatter_model_params(runs: pd.DataFrame, x: str, y: str, z: str, marker_size: int):
    fig = px.scatter(runs, x=x, y=y, color=z, hover_data=["run_id"])


def add_date_line(date: pd.Timestamp, label: str, fig: go.Figure):
    fig.add_shape(
        type="line",
        x0=date,
        y0=0,
        x1=date,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="gray", width=3, dash="dot"),
    )
    fig.add_annotation(
        x=date,
        y=1,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        yshift=25,
        font={"size": 12},
    )


def plot_prediction_benchmark(
    df_benchmark: pd.DataFrame, country: str, n_month_ahead_vals: list[int]
):
    hovertemplate = "Model: %{meta}<br>Inflation: %{y:.3f}%<extra></extra>"
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    fig = go.Figure()
    for i, n_months_ahead in enumerate(n_month_ahead_vals):
        name_ar = f"{n_months_ahead} months ahead, AutoReg"

        fig.add_trace(
            go.Scatter(
                x=df_benchmark.index,
                y=df_benchmark[f"AutoReg_{country}_{n_months_ahead}"],
                mode="lines",
                name=name_ar,
                meta=name_ar,
                hovertemplate=hovertemplate,
                line=dict(color=colors[i], dash="dash"),
            )
        )

        name_tft = f"{n_months_ahead} months ahead, TFT"
        fig.add_trace(
            go.Scatter(
                x=df_benchmark.index,
                y=df_benchmark[f"quantile_0.50_{country}_{n_months_ahead}"],
                mode="lines",
                name=name_tft,
                meta=name_tft,
                hovertemplate=hovertemplate,
                line=dict(color=colors[i]),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df_benchmark.index,
            y=df_benchmark[f"truth_{country}"],
            mode="lines",
            name="truth",
            meta="truth",
            hovertemplate=hovertemplate,
            line=dict(color="#111111"),
        )
    )

    # TODO: I can be retrieved from mlflow!
    add_date_line("2018-01-01", "Training cutoff", fig)
    add_date_line("2020-01-01", "Validation cutoff", fig)

    fig.update_layout(
        width=1500,
        height=500,
        hovermode="x unified",
        title=f"YoY inflation predictions for {country}",
        xaxis_title="Date",
        yaxis_title="YoY inflation [%]",
    )
    return fig
