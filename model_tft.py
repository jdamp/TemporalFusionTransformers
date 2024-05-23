import keras
import mlflow
from mlflow.entities.run import Run
from typing import Union

from torch.utils.data import DataLoader
import pandas as pd
import temporal_fusion_transformers as tft


def build_tft(
    d_model: int, dropout_rate: float, learning_rate: float, n_head: int, partial: bool
) -> tft.TFT:
    """_summary_

    Args:
        d_model (int): Internal model dimension, used for the embedding dimensions
        dropout_rate (float): Dropout rate used in the internal dropout layers
        learning_rate (float): Learning rate for the adam optimizer

    Returns:
        tft.TFT: Temporal fusion transformer
    """
    model = tft.TFT(
        d_model=d_model,
        output_size=12,  # forecast 12 months - fine to hardcode this.
        dropout_rate=dropout_rate,
        quantiles=tft.quantiles,
        name="tft",
        skip_attention=partial,
        n_head=n_head,
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tft.quantile_loss)
    return model


def get_default_callbacks(run: Run) -> list[keras.callbacks.Callback]:
    """Gets a list of callbacks for the Keras model, which are evaluated at specific point in
    the training process, e.g. epoch start/end or batch start/end

    Args:
        run (Run): The currently active mlflow run. Used to configure the MlFlow callback

    Returns:
        list[keras.callbacks.Callback]: List of callbacks
    """
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.001
    )
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    mlflow_log = mlflow.keras.MLflowCallback(run, log_every_epoch=True)
    return [reduce_lr, early_stop, mlflow_log]


def get_train_val_data(
    n_samples: int,
    n_samples_val: int,
    df_daily_input: pd.DataFrame,
    df_target: pd.DataFrame,
    start_date_train: pd.Timestamp,
    end_date_train: pd.Timestamp,
    start_date_val: pd.Timestamp,
    end_date_val: pd.Timestamp,
    batch_size: int,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Create and return DataLoader for training and validation data

    Args:
        n_samples (int): Number of training samples
        n_samples_val (int): Number of validation samples
        df_daily_input (pd.DataFrame): DataFrame with input data in daily frequency
        df_target (pd.DataFrame): DataFrame with inflation target in monthly frequency
        start_date_train (pd.Timestamp): Training period start date
        end_date_train (pd.Timestamp): Training period end date
        start_date_val (pd.Timestamp): Validation period start date
        end_date_val (pd.Timestamp): Validation period end date
        batch_size (int): batch size to be used for the training and validation


    Returns:
        tuple[DataLoader, DataLoader]: _description_
    """
    train_data_loader = DataLoader(
        tft.NowcastingData(
            n_samples=n_samples,
            df_daily_input=df_daily_input,
            df_target=df_target,
            start_date=start_date_train,
            end_date=end_date_train,
            **kwargs,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    val_data_loader = DataLoader(
        tft.NowcastingData(
            n_samples=n_samples_val,
            df_daily_input=df_daily_input,
            df_target=df_target,
            start_date=start_date_val,
            end_date=end_date_val,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_data_loader, val_data_loader
