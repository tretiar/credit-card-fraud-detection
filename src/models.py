"""Model factories and scoring logic."""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def create_rf_smote_pipeline(
    random_state: int = RANDOM_STATE,
    smote_params: dict | None = None,
    rf_params: dict | None = None,
) -> ImbPipeline:
    """Build an imbalanced-learn pipeline containing SMOTE and Random Forest."""
    smote_config = {"random_state": random_state, **(smote_params or {})}
    rf_config = {"random_state": random_state, "n_jobs": -1, **(rf_params or {})}

    return ImbPipeline(
        steps=[
            ("smote", SMOTE(**smote_config)),
            ("random_forest", RandomForestClassifier(**rf_config)),
        ]
    )


def create_isolation_forest_model(
    random_state: int = RANDOM_STATE,
    contamination: float | str = "auto",
    iforest_params: dict | None = None,
) -> IsolationForest:
    """Build an Isolation Forest model for anomaly scoring."""
    config = {
        "random_state": random_state,
        "contamination": contamination,
        "n_jobs": -1,
        **(iforest_params or {}),
    }
    return IsolationForest(**config)


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """Return the negative-to-positive class ratio for XGBoost."""
    return float((y == 0).sum() / (y == 1).sum())


def create_xgboost_model(
    scale_pos_weight: float,
    random_state: int = RANDOM_STATE,
    xgb_params: dict | None = None,
) -> XGBClassifier:
    """Build an XGBoost classifier for imbalanced binary classification."""
    config = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
        **(xgb_params or {}),
    }
    return XGBClassifier(**config)


def fit_and_score_selected_model(
    model_info: dict,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
):
    """Fit one selected configuration and return continuous fraud scores."""
    category = model_info["category"]
    config = model_info["config"]

    if category == "Random Forest + SMOTE":
        model = create_rf_smote_pipeline(
            rf_params=config["rf_params"],
            smote_params=config["smote_params"],
        )
        model.fit(X_fit, y_fit)
        scores = model.predict_proba(X_eval)[:, 1]

    elif category == "Isolation Forest":
        model = create_isolation_forest_model(
            contamination=y_fit.mean(),
            iforest_params=config["params"],
        )
        model.fit(X_fit.loc[y_fit == 0])
        scores = -model.decision_function(X_eval)

    elif category == "XGBoost":
        model = create_xgboost_model(
            scale_pos_weight=calculate_scale_pos_weight(y_fit),
            xgb_params=config["params"],
        )
        model.fit(X_fit, y_fit)
        scores = model.predict_proba(X_eval)[:, 1]

    else:
        raise ValueError(f"Unknown model category: {category}")

    return model, scores
