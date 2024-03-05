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
import plot
import optuna
from hyperparam import objective, champion_callback
import mlflow_utils


def main():
    experiment_id = mlflow_utils.get_or_create_experiment("TFT")
    start_date_train = pd.Timestamp("1980-01-01")
    end_date_train = pd.Timestamp("2018-01-01")
    start_date_plot = pd.Timestamp("2000-01-01")
    end_date_plot = pd.Timestamp("2023-01-01")
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
        study.optimize(objective, n_trials=50, callbacks=[champion_callback])
        mlflow.log_params(study.best_params)
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
