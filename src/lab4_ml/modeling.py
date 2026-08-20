"""Modelos y esquemas de validacion para clasificacion de CYA alta."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import SAFE_PREDICTORS


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    parameters: pd.DataFrame
    feature_importance: pd.DataFrame
    random_predictions: pd.DataFrame
    spatial_predictions: pd.DataFrame
    best_model: str


def model_spaces(random_state: int = 2026) -> dict[str, tuple[Pipeline, dict]]:
    """Define los tres algoritmos y una busqueda pequena, reproducible."""

    return {
        "Regresion logistica": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2_000,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=120,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    )
                ]
            ),
            {
                "model__max_depth": [12, None],
                "model__min_samples_leaf": [2, 8],
            },
        ),
        "Gradient Boosting": (
            Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            class_weight="balanced",
                            max_iter=160,
                            random_state=random_state,
                        ),
                    )
                ]
            ),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [15, 31],
            },
        ),
    }


def _metric_row(y_true, probability, *, model: str, validation: str, fold: str) -> dict:
    predicted = (np.asarray(probability) >= 0.5).astype(int)
    y_true = np.asarray(y_true)
    try:
        auc = roc_auc_score(y_true, probability)
    except ValueError:
        auc = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "modelo": model,
        "validacion": validation,
        "fold": fold,
        "n": len(y_true),
        "prevalencia_alta": float(y_true.mean()),
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _fit_and_score(name, estimator, train, test, validation, fold):
    fitted = clone(estimator).fit(train[SAFE_PREDICTORS], train["cya_alta"])
    probability = fitted.predict_proba(test[SAFE_PREDICTORS])[:, 1]
    return fitted, probability, _metric_row(
        test["cya_alta"], probability, model=name, validation=validation, fold=fold
    )


def run_experiments(
    data: pd.DataFrame,
    *,
    random_state: int = 2026,
    tuning_cap: int = 50_000,
) -> ExperimentResult:
    """Ajusta y compara validacion aleatoria, espacial, temporal y entre lagos."""

    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.30,
        random_state=random_state,
        stratify=data["cya_alta"],
    )
    random_train = data.iloc[train_idx].copy()
    random_test = data.iloc[test_idx].copy()

    if len(random_train) > tuning_cap:
        tune_idx, _ = train_test_split(
            np.arange(len(random_train)),
            train_size=tuning_cap,
            random_state=random_state,
            stratify=random_train["cya_alta"],
        )
        tune_data = random_train.iloc[tune_idx]
    else:
        tune_data = random_train

    tuned: dict[str, Pipeline] = {}
    parameter_rows: list[dict] = []
    metrics: list[dict] = []
    random_prediction_frames: list[pd.DataFrame] = []
    spatial_prediction_frames: list[pd.DataFrame] = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    for name, (estimator, grid) in model_spaces(random_state).items():
        search = GridSearchCV(
            estimator,
            grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(tune_data[SAFE_PREDICTORS], tune_data["cya_alta"])
        best = clone(search.best_estimator_)
        tuned[name] = best
        parameter_rows.append(
            {
                "modelo": name,
                "roc_auc_cv_ajuste": search.best_score_,
                "parametros": repr(search.best_params_),
                "n_busqueda": len(tune_data),
            }
        )
        fitted, probability, row = _fit_and_score(
            name, best, random_train, random_test, "aleatoria_70_30", "test"
        )
        tuned[name] = fitted
        metrics.append(row)
        pred = random_test[
            ["lago", "fecha", "x_utm", "y_utm", "longitud", "latitud", "bloque_1km", "CYA", "cya_alta"]
        ].copy()
        pred["modelo"] = name
        pred["probabilidad_alta"] = probability
        pred["prediccion"] = (probability >= 0.5).astype("int8")
        random_prediction_frames.append(pred)

    # Validacion espacial: todos los pixeles del mismo bloque de 1 km quedan juntos.
    unique_groups = data["bloque_1km"].nunique()
    n_splits = min(5, unique_groups)
    spatial_cv = GroupKFold(n_splits=n_splits)
    for fold_number, (sp_train_idx, sp_test_idx) in enumerate(
        spatial_cv.split(data, data["cya_alta"], groups=data["bloque_1km"]), start=1
    ):
        sp_train = data.iloc[sp_train_idx]
        sp_test = data.iloc[sp_test_idx]
        for name, fitted_random in tuned.items():
            _, probability, row = _fit_and_score(
                name,
                fitted_random,
                sp_train,
                sp_test,
                "espacial_groupkfold",
                str(fold_number),
            )
            metrics.append(row)
            pred = sp_test[["lago", "fecha", "x_utm", "y_utm", "bloque_1km", "cya_alta"]].copy()
            pred["modelo"] = name
            pred["fold_espacial"] = fold_number
            pred["probabilidad_alta"] = probability
            pred["prediccion"] = (probability >= 0.5).astype("int8")
            spatial_prediction_frames.append(pred)

    # Validacion temporal: la fecha mas reciente de cada lago queda fuera.
    last_dates = data.groupby("lago")["fecha"].max()
    is_temporal_test = data["fecha"].eq(data["lago"].map(last_dates))
    temporal_train = data.loc[~is_temporal_test]
    temporal_test = data.loc[is_temporal_test]
    for name, fitted_random in tuned.items():
        _, _, row = _fit_and_score(
            name, fitted_random, temporal_train, temporal_test, "temporal_ultima_fecha", "test"
        )
        metrics.append(row)

    # Transferencia entre lagos en ambas direcciones.
    for source, destination in (("atitlan", "amatitlan"), ("amatitlan", "atitlan")):
        lake_train = data.loc[data["lago"] == source]
        lake_test = data.loc[data["lago"] == destination]
        for name, fitted_random in tuned.items():
            _, _, row = _fit_and_score(
                name,
                fitted_random,
                lake_train,
                lake_test,
                "entre_lagos",
                f"{source}_a_{destination}",
            )
            metrics.append(row)

    metrics_df = pd.DataFrame(metrics)
    random_metrics = metrics_df.loc[metrics_df["validacion"] == "aleatoria_70_30"]
    best_name = random_metrics.sort_values(["roc_auc", "recall"], ascending=False).iloc[0]["modelo"]
    best_model = tuned[best_name]
    importance_sample = random_test.sample(
        n=min(20_000, len(random_test)), random_state=random_state
    )
    permutation = permutation_importance(
        best_model,
        importance_sample[SAFE_PREDICTORS],
        importance_sample["cya_alta"],
        scoring="roc_auc",
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "variable": SAFE_PREDICTORS,
            "importancia_media": permutation.importances_mean,
            "importancia_sd": permutation.importances_std,
            "modelo": best_name,
        }
    ).sort_values("importancia_media", ascending=False)

    return ExperimentResult(
        metrics=metrics_df,
        parameters=pd.DataFrame(parameter_rows),
        feature_importance=importance,
        random_predictions=pd.concat(random_prediction_frames, ignore_index=True),
        spatial_predictions=pd.concat(spatial_prediction_frames, ignore_index=True),
        best_model=best_name,
    )
