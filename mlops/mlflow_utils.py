import keras
import mlflow
import numpy as np
import pandas as pd

from functools import wraps
from keras.models import Model
from mlflow import pyfunc
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from typing import Callable, Optional

import temporal_fusion_transformers as tft


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def ensure_tft_experiment(func: Callable) -> Callable:
    """A decorator to ensure that the 'experiment_id' argument is set for the decorated function.

    If 'experiment_id' is None when the decorated function is called, 'experiment_id' will be
    set using the TFT experiment id.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapper function that adds the 'experiment_id' handling logic.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if "experiment_id" in kwargs and kwargs["experiment_id"] is None:
            kwargs["experiment_id"] = get_or_create_experiment("TFT")
        elif "experiment_id" not in kwargs:
            # If 'experiment_id' is not provided at all, set it
            kwargs["experiment_id"] = get_or_create_experiment("TFT")
        return func(*args, **kwargs)

    return wrapper


class KerasModelWrapper(pyfunc.PythonModel):
    """Wrapper class to store and load keras models in mflflow < 2.11"""

    def __init__(self, model: Model):
        self.model = model

    def load_context(self, context):
        """Loads the model from the artifact path"""
        self.model = keras.models.load_model(context.artifacts["model"])
        self._hack_build()
        self.model.load_weights(context.artifacts["weights"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)

    def _hack_build(self):
        """The TFT model layers are only built once the model is called. Therefore, the model needs
        to be called at least once before the weights can be loaded."""
        x, _ = tft.prepare_data_samples(
            n_samples=1,
            df_daily_input=tft.df_input_scl,
            df_target=tft.df_target_1m_pct,
            sampled_day="2000-01-01",
            min_context=365,
            context_length=365,
            country="US",
        )
        self.model(x)


def log_keras_model(model: Model, artifact_path: str):
    """Log Keras model to MLflow as an artifact with custom serialization.

    Args:
        model (Model): The model to log.
        artifact_path (str): The artifact path in MLflow.
    """
    model.save_weights("tft.weights.h5")
    model.save("model.keras")
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=KerasModelWrapper(None),
        artifacts={"model": "model.keras", "weights": "tft.weights.h5"},
    )


def load_keras_model(model_uri: str) -> KerasModelWrapper:
    """Load a Keras model from MLflow.

    Args:
        model_uri (str): The URI to the logged model in MLflow.

    Returns:
        KerasModelWrapper: The loaded model.
    """
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model


def load_keras_model_for_run(run_id: str) -> KerasModelWrapper:
    """Loads the "model" artifact for the run specified by "run_id".

    Args:
        run_id (str): id of the run

    Returns:
        KerasModelWrapper: The lpaded model from the run
    """
    uri = f"runs:/{run_id}/model"
    return load_keras_model(uri)


def add_tag_to_active_run(key: str, value: str):
    """
    Add a tag to the active MLflow run.

    Args:
    key (str): The key of the tag.
    value (str): The value of the tag.
    """
    if mlflow.active_run() is None:
        raise RuntimeError("No active run. Ensure you are within an active run context.")

    run_id = mlflow.active_run().info.run_id
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, key, value)


@ensure_tft_experiment
def get_most_recent_parent_run_id(experiment_id: Optional[str] = None) -> str:
    """Retrieves the run ID of the most recent parent run in the specified experiment from mlflow.

    Args:
        experiment_id: ID of the experiment. If None, defaults to the TFT experiment.

    Returns:
        The ID of the most recent parent run
    """
    parent_id = mlflow.search_runs(
        experiment_ids=experiment_id,
        order_by=["created DESC"],
        filter_string="`Run Name`='Parent run'",
        max_results=1,
    ).run_id.values[0]
    return parent_id


@ensure_tft_experiment
def get_tft_child_run_df(parent_run_id: str, experiment_id: Optional[str] = None) -> pd.DataFrame:
    """Retrieves a DataFrame containing information on run info, metrics and parameters for all
    child runs of TFT models of a given parent run.

    Args:
        parent_run_id: The ID of the parent run
        experiment_id: ID of the experiment. If None, defaults to the TFT experiment.

    Returns:
        A DataFrame containing information from mlflow for the child runs
    """
    runs = mlflow.search_runs(
        experiment_id, filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
    )
    # We only want finished runs with metrics available and not the Autoregressive models
    runs = runs[
        (runs.status == "FINISHED") & (~runs["tags.mlflow.runName"].str.contains("Autoreg"))
    ]
    # Some of the columns will contain 'None' strings, for example some default metrics
    # Replace these by nan values and drop columns that only contain nan
    runs = runs.replace({"None": np.nan, "False": False, "True": True}).dropna(axis=1)
    # Some of the columns might be of object dtype, convert
    for col in runs.columns:
        # Try to convert to numeric
        tmp = pd.to_numeric(runs[col], errors="coerce")
        if tmp.notnull().all():
            runs[col] = tmp
            continue

        # Numeric failed, try to convert to datetime
        tmp = pd.to_datetime(runs[col], errors="coerce")
        if tmp.notnull().all():
            runs[col] = tmp
        # Should likely stay as object type

    return runs


@ensure_tft_experiment
def get_best_model(
    parent_run_id: str,
    metric: str = "metrics.mean_val_loss",
    experiment_id: Optional[str] = None,
) -> KerasModelWrapper:
    """Retrieves the best TFT model according to the specified metric for all child runs of the
    run with id "parent_run_id".

    Args:
        parent_run_id: Id of the mlflow parent run.
        metric: Metric to compare the models. Defaults to "metrics.mean_val_loss".
        experiment_id: Id of the MLflow experiment. Defaults to the TFT experiment.


    Returns:
        The retrieved model
    """
    child_runs = get_tft_child_run_df(parent_run_id, experiment_id=experiment_id)
    best_run_id = child_runs.sort_values(by=metric, ascending=True).iloc[0]["run_id"]
    return load_keras_model_for_run(best_run_id)


@ensure_tft_experiment
def get_ar_model(
    parent_run_id: str, country: str, experiment_id: Optional[str] = None
) -> AutoRegResultsWrapper:
    """Retrieves a fitted autoregressive model from mlflow for the parent run "parent_run_id"
    and the country with the code "country".

    Args:
        parent_run_id (str): _description_
        country (str): _description_
        experiment_id (Optional[str], optional): _description_. Defaults to None.
    """
    runs = mlflow.search_runs(
        experiment_id,
        filter_string=(
            f"tags.mlflow.parentRunId = '{parent_run_id}' "
            f"AND tags.mlflow.runName ='Autoreg_{country}'"
        ),
    )
    if not len(runs) == 1:
        raise ValueError(f"Found zero or multiple runs for Autoreg_{country}! ({len(runs)=})")
    run_id = runs.iloc[0].run_id
    model = mlflow.statsmodels.load_model(f"runs:/{run_id}/model")
    return model
