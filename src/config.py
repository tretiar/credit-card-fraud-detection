"""Shared project constants."""

from pathlib import Path


RANDOM_STATE = 42
TARGET_COLUMN = "Class"
SECONDS_IN_DAY = 24 * 60 * 60
SCALE_COLUMNS = ["Amount", "Time"]
TIME_FEATURES = ["SecondsOfDay", "Hour_sin", "Hour_cos"]

# The notebook works both when launched from the repository root and from notebooks/.
DATASET_CANDIDATES = (
    Path("data/creditcard.csv"),
    Path("creditcard.csv"),
    Path("../data/creditcard.csv"),
)
