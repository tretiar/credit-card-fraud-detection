"""Unit tests for deterministic feature engineering."""

import pandas as pd

from src.features import add_time_features


def test_add_time_features_adds_expected_columns():
    data = pd.DataFrame({"Time": [0, 43_200], "Amount": [10.0, 20.0]})

    transformed = add_time_features(data)

    assert {"SecondsOfDay", "Hour_sin", "Hour_cos"}.issubset(transformed.columns)
    assert transformed.loc[0, "SecondsOfDay"] == 0
    assert transformed.loc[1, "SecondsOfDay"] == 43_200
