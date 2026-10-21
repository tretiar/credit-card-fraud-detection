"""Dataset discovery and deterministic feature transformations."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .config import DATASET_CANDIDATES, SCALE_COLUMNS, SECONDS_IN_DAY


def find_dataset_path(candidate_paths: tuple[Path, ...] = DATASET_CANDIDATES) -> Path:
    """Return the first local dataset path that exists."""
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        "Could not find creditcard.csv. Place it in data/ or the project root."
    )


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add linear and cyclical representations of transaction time."""
    transformed = data.copy()
    seconds_of_day = transformed["Time"] % SECONDS_IN_DAY

    transformed["SecondsOfDay"] = seconds_of_day
    transformed["Hour_sin"] = np.sin(2 * np.pi * seconds_of_day / SECONDS_IN_DAY)
    transformed["Hour_cos"] = np.cos(2 * np.pi * seconds_of_day / SECONDS_IN_DAY)
    return transformed


def fit_feature_scaler(data: pd.DataFrame) -> RobustScaler:
    """Fit a RobustScaler using training features only."""
    scaler = RobustScaler()
    scaler.fit(add_time_features(data)[SCALE_COLUMNS])
    return scaler


def transform_features(data: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """Create time features and scale the columns learned during training."""
    transformed = add_time_features(data)
    transformed.loc[:, SCALE_COLUMNS] = scaler.transform(transformed[SCALE_COLUMNS])
    return transformed


def prepare_train_eval_data(
    X_fit_raw: pd.DataFrame,
    X_eval_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit preprocessing on training data and transform evaluation data."""
    scaler = fit_feature_scaler(X_fit_raw)
    return transform_features(X_fit_raw, scaler), transform_features(X_eval_raw, scaler)
