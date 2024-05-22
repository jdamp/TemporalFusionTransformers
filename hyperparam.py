from dataclasses import dataclass
from functools import partial
from typing import Literal, Union, Callable

import keras
import mlflow
import mlops.mlflow_utils as mlflow_utils
import numpy as np
import optuna
import pandas as pd
from prediction.plot import (
    build_n_months_prediction_df,
    build_monthly_prediction_df
)
from prediction.transform import loop_over_month_starts

from _utils.types import DateLike
import temporal_fusion_transformers as tft
from model_tft import build_tft, get_train_val_data, get_default_callbacks


@dataclass
class OptunaParamConfig:
    """Dataclass containing all information required to configure a parameter of an Optuna trial"""
    name: str
    lower: Union[int, float]
    upper: Union[int, float]
    dtype: type
    log: bool = False
    

def objective_builder(
    n_samples: int,
    batch_size: int,
    start_date_train: DateLike,
    end_date_train: DateLike,
    start_date_val: DateLike,
    end_date_val: DateLike,
    max_epochs: int,
    min_context: int,
    context_length: int,
    d_model: OptunaParamConfig,
    dropout_rate: OptunaParamConfig,
    n_head: OptunaParamConfig,
    learning_rate: OptunaParamConfig,
) -> optuna.Trial:  
    return partial(
        objective,
        n_samples=n_samples,
        batch_size=batch_size,
        start_date_train=start_date_train,
        end_date_train=end_date_train,
        start_date_val=start_date_val,
        end_date_val=end_date_val,
        max_epochs=max_epochs,
        min_context=min_context,
        context_length=context_length,
        d_model_range=d_model,
        dropout_rate_range=dropout_rate,
        n_head_range=n_head,
        learning_rate_range=learning_rate,
    )


def objective(
    trial: optuna.Trial,
    **kwargs
):

    experiment_id = mlflow_utils.get_or_create_experiment("TFT")

    with mlflow.start_run(nested=True, experiment_id=experiment_id) as tft_run:
        params = {}
        for param, config in kwargs.items():
            if isinstance(config, OptunaParamConfig):
                if config.dtype == int:
                    method = trial.suggest_int
                else: 
                    method = trial.suggest_float
                params[param] = method(config.name, config.lower, config.upper, log=config.log)
            else:
                params[param] = config


        # Retrieve data using hyperparameters
        train_data, val_data = get_train_val_data(
            n_samples=params["n_samples"],
            n_samples_val=int(0.1 * params["n_samples"]),
            df_daily_input=tft.df_input_scl,
            df_target=tft.df_target_1m_pct,
            start_date_train=params["start_date_train"],
            end_date_train=params["end_date_train"],
            start_date_val=params["start_date_val"],
            end_date_val=params["end_date_val"],
            batch_size=2,
            min_context=params["min_context"],
            context_length=params["context_length"],
        )
        tft_model = build_tft(
            d_model=params["d_model"],
            dropout_rate=params["dropout_rate"],
            learning_rate=params["learning_rate"],
            n_head=params["n_head"],
            partial=False,
        )
        # hacky: Do one forward pass to force model building
        x, y = next(iter(train_data))
        _ = tft_model(x)

        callbacks = get_default_callbacks(tft_run)

        hist = tft_model.fit(
            train_data,
            validation_data=val_data,
            callbacks=callbacks,
            epochs=epochs,
        )
        # Calculate metrics: RMSE for every country, both MoM and YoY
        for country in tft.countries:
            rmse_mom = get_rmse_mom(
                model=tft_model,
                start_date=start_date_val,
                end_date=end_date_val,
                country=country,
            )

            mlflow.log_metric(f"rmse_mom_{country}", rmse_mom)

            for n_months_ahead in range(1, 13):
                rmse_yoy = get_rmse_yoy(
                    model=tft_model,
                    start_date=start_date_val,
                    end_date=end_date_val,
                    country=country,
                    n_months_ahead=n_months_ahead,
                )

                mlflow.log_metric(f"rmse_yoy_{country}_{n_months_ahead}_months_ahead", rmse_yoy)

        # Metric: mean val loss over all countries
        errors = []
        months = loop_over_month_starts(start_date_val, end_date_val)
        for country in tft.countries:
            for month in months:
                X, y = tft.prepare_data_samples(
                    n_samples=1,
                    df_daily_input=tft.df_input_scl,
                    df_target=tft.df_target_1m_pct,
                    sampled_day=month,
                    min_context=365,
                    context_length=365,
                    country=country,
                )
                y_pred = keras.ops.array(tft_model.predict(X))
                errors.append(tft.quantile_loss(y, y_pred).cpu())
        error = np.mean(errors)

        mlflow.log_metric("mean_val_loss", error)

        # Log parameters to MLflow
        mlflow.log_params(params)

        # Log model to mlflow
        mlflow_utils.log_keras_model(tft_model, "model")

    return error


def champion_callback(study: optuna.Study, frozen_trial: optuna.Trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.


    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def rmse(truth, prediction):
    return np.sqrt(((truth - prediction) ** 2).mean())


def get_rmse_yoy(
    model,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    country: str,
    n_months_ahead: int,
):
    truths = tft.df_target_12m_pct.loc[start_date:end_date, country]
    predictions = build_n_months_prediction_df(
        model, n_months_ahead, country, start_date, end_date
    ).loc[:, "quantile_0.50"]
    return rmse(truths, predictions)


def get_rmse_mom(
    model,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    country,
):
    truths = tft.df_target_1m_pct.loc[start_date:end_date, country]
    predictions = build_monthly_prediction_df(model, start_date, end_date, country).loc[
        :, "quantile_0.50"
    ]
    return rmse(truths, predictions)
