import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["MLFLOW_TRACKING_URI"] = (
    "https://mlflow-server-med-jupyter-central-dev.apps.dev.ocp.bisinfo.org"
)
import keras
import mlflow
import numpy as np
import pandas as pd
import temporal_fusion_transformers as tft

from model_autoreg import fit_ar_model
from model_tft import build_tft, get_train_val_data, get_default_callbacks
import prediction.plot as plot
import optuna
from hyperparam import objective_builder, champion_callback, OptunaParamConfig
import mlops.mlflow_utils as mlflow_utils


def main():
    experiment_id = mlflow_utils.get_or_create_experiment("jdamp-test")
    n_models = 1
    start_date_train = pd.Timestamp("1980-01-01")
    end_date_train = pd.Timestamp("2018-01-01")
    start_date_val = pd.Timestamp("2018-01-01")
    end_date_val = pd.Timestamp("2020-01-01")
    start_date_plot = pd.Timestamp("2000-01-01")
    end_date_plot = pd.Timestamp("2023-01-01")

    # Define hyperparameters
    min_context = 90  # trial.suggest_int("min_context", 30, 120)
    # Context length needs to be at least min_context
    context_length = 365  # trial.suggest_int("context_length", min_context, 365)

    objective = objective_builder(
        n_samples=10,
        batch_size=2,
        start_date_train=start_date_train,
        end_date_train=end_date_train,
        start_date_val=start_date_val,
        end_date_val=end_date_val,
        max_epochs=60,
        min_context=min_context,
        context_length=context_length,
        d_model=OptunaParamConfig("d_model", 8, 128, int),
        dropout_rate=OptunaParamConfig("dropout_rate", 0, 0.2, float),
        n_head=OptunaParamConfig("n_head", 1, 6, int),
        learning_rate=OptunaParamConfig("learning_rate", 1e-5, 5e-2, float),
    )
    autoreg_models = {}
    with mlflow.start_run(run_name="Parent run", experiment_id=experiment_id) as run:
        for country in tft.countries:
            with mlflow.start_run(
                nested="True",
                run_name=f"Autoreg_{country}",
                experiment_id=experiment_id,
            ):
                autoreg_data = tft.df_target_1m_pct
                autoreg_models[country] = fit_ar_model(
                    autoreg_data, country, start_date_train, end_date_train, lags=12
                )

        # TFT hyperparameter tuning runs
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_models, callbacks=[champion_callback])
        mlflow.log_params(study.best_params)
        mlflow.log_param("start_date_train", start_date_train)
        mlflow.log_param("end_date_train", end_date_train)
        mlflow.log_metric("best_val_loss", study.best_value)

        # Get best model
        # Retrieve all runs matching the query
        # Initialize variables to track the minimum RMSE and corresponding run
        min_error = float("inf")
        best_run_id = None

        runs = mlflow.MlflowClient().search_runs(
            experiment_ids=experiment_id,
            filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'",
        )
        for run in runs:
            # Skip Autoreg runs
            if "Autoreg" in run.info.run_name:
                continue
            error = run.data.metrics["mean_val_loss"]
            if error < min_error:
                min_error = error
                best_run_id = run.info.run_id
        model_path = f"runs:/{best_run_id}/model"
        best_tft_model = mlflow_utils.load_keras_model(model_path)
        for n_months_ahead in [1, 2, 6, 12]:
            for country in tft.countries:
                plot.yoy_plot(
                    best_tft_model,
                    autoreg_models[country],
                    start_date_plot,
                    end_date_plot,
                    country,
                    n_months_ahead,
                    True,
                )


if __name__ == "__main__":
    mlflow.pytorch.autolog()
    mlflow.statsmodels.autolog()
    main()
