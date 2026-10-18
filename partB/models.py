import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import (
    categorical_features,
    features,
    numerical_features,
    settings,
    train_df,
    y_train,
)

# ==========================================
# Models
# ==========================================


def build_preprocessor(numerical, categorical):
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def ols_builder(drop):
    """Builder for an OLS variant using input.features minus `drop`."""
    unknown = set(drop) - set(features)
    if unknown:
        raise ValueError(f"drop lists columns not in input.features: {sorted(unknown)}")
    numerical = [c for c in numerical_features if c not in drop]
    categorical = [c for c in categorical_features if c not in drop]

    def build():
        return Pipeline(
            [
                ("preprocessor", build_preprocessor(numerical, categorical)),
                ("ols", LinearRegression()),
            ]
        )

    return build, numerical + categorical


def build_trivial():
    # always predict the training mean
    return DummyRegressor(strategy="mean")


def build_simple():
    # OLS on a single feature (mileage)
    return Pipeline([("scaler", StandardScaler()), ("ols", LinearRegression())])


def build_strong():
    # random forest on all features
    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "random_forest",
                RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            ),
        ]
    )


OLS_MODELS = {name: ols_builder((config or {}).get("drop") or []) for name, config in settings["models"].items()}

BASELINE_MODELS = {
    "baseline-trivial": (build_trivial, features),
    "baseline-simple": (build_simple, settings["baseline"]["simple-features"]),
    "baseline-strong": (build_strong, features),
}

MODELS = {**OLS_MODELS, **BASELINE_MODELS}

OUT_DIR = Path(settings["output"]["dir"])


def model_path(name):
    return OUT_DIR / f"{name}.joblib"


def report_path(name):
    return OUT_DIR / f"{name}.json"


# ==========================================
# Training
# ==========================================

CV = KFold(n_splits=5, shuffle=True, random_state=42)

SCORING = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}

# GroupKFold: rows from the same group (e.g. dealer) stay on one side of every split.
# Rows with no group value are private sellers / unknown, so each forms its own group.
GROUP_COLUMN = settings["validation"]["group-column"]
GROUP_CV = GroupKFold(n_splits=5)
GROUPS = train_df[GROUP_COLUMN].astype(object).where(
    train_df[GROUP_COLUMN].notna(),
    [f"_row{i}" for i in range(len(train_df))],
)


def run_cv(build, X, cv, groups=None):
    results = cross_validate(build(), X, y_train, cv=cv, groups=groups, scoring=SCORING, n_jobs=1, error_score="raise")
    return (
        (-results["test_MAE"]).tolist(),
        (-results["test_RMSE"]).tolist(),
        results["test_R2"].tolist(),
    )


def fit_weights(name, overwrite):
    """Fit the final model on all training data and save it to <name>.joblib."""
    if not overwrite and model_path(name).exists():
        print(f"{name}: {model_path(name)} exists, skipped")
        return

    build, columns = MODELS[name]
    X = train_df[columns]

    model = build().fit(X, y_train)
    assert np.isfinite(model.predict(X)).all(), f"Invalid predictions in {name}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path(name))
    print(f"{name}: weights -> {model_path(name)}")


def run_analysis(name, overwrite):
    """Cross-validate the model and save the scores to <name>.json."""
    if not overwrite and report_path(name).exists():
        print(f"{name}: {report_path(name)} exists, skipped")
        return

    build, columns = MODELS[name]
    X = train_df[columns]

    cv_mae, cv_rmse, cv_r2 = run_cv(build, X, CV)
    group_mae, group_rmse, group_r2 = run_cv(build, X, GROUP_CV, GROUPS)
    report = {
        "columns": columns,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "cv_r2": cv_r2,
        "group_column": GROUP_COLUMN,
        "group_cv_mae": group_mae,
        "group_cv_rmse": group_rmse,
        "group_cv_r2": group_r2,
    }

    if name in OLS_MODELS:
        train_sizes, train_scores, val_scores = learning_curve(
            build(),
            X,
            y_train,
            cv=CV,
            train_sizes=np.linspace(0.2, 1.0, 5),
            scoring="neg_mean_squared_error",
            n_jobs=1,
            error_score="raise",
        )
        report["learning_curve"] = {
            "train_sizes": train_sizes.tolist(),
            "train_mse": (-train_scores.mean(axis=1)).tolist(),
            "val_mse": (-val_scores.mean(axis=1)).tolist(),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path(name), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        f"{name}: CV MAE {np.mean(report['cv_mae']):.2f}"
        f" | RMSE {np.mean(report['cv_rmse']):.2f}"
        f" | R2 {np.mean(report['cv_r2']):.4f}"
        f" | GroupKFold MAE {np.mean(report['group_cv_mae']):.2f}"
        f" -> {report_path(name)}"
    )


if __name__ == "__main__":
    overwrite = settings["overwrite"]

    # 1. OLS variants
    for name in OLS_MODELS:
        fit_weights(name, overwrite["model"])
        run_analysis(name, overwrite["analysis"])

    # 2. Baselines
    for name in BASELINE_MODELS:
        fit_weights(name, overwrite["baseline"])
        run_analysis(name, overwrite["analysis"])
