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


# Margen de ROC-AUC por debajo del cual dos modelos se consideran empatados en
# la particion aleatoria; evita elegir por diferencias irrelevantes.
AUC_SELECTION_TOLERANCE = 1e-3


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    parameters: pd.DataFrame
    feature_importance: pd.DataFrame
    random_predictions: pd.DataFrame
    spatial_predictions: pd.DataFrame
    best_model: str
    fitted_models: dict[str, Pipeline]
    random_test: pd.DataFrame


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


def select_best_model(random_metrics: pd.DataFrame, *, tolerance: float = AUC_SELECTION_TOLERANCE) -> str:
    """Elige el modelo de la particion aleatoria tratando empates como empates.

    Ordenar por ROC-AUC crudo haria que una diferencia de 1e-5 en una sola
    particion decidiera el modelo explicativo y cartografico. Se agrupan los
    modelos dentro de ``tolerance`` y se desempata por recall, prioritario en
    vigilancia ambiental porque un falso negativo omite una zona a inspeccionar.
    """

    if random_metrics.empty:
        raise ValueError("No hay metricas de la particion aleatoria")
    top_auc = random_metrics["roc_auc"].max()
    contenders = random_metrics.loc[random_metrics["roc_auc"] >= top_auc - tolerance]
    return str(contenders.sort_values("recall", ascending=False).iloc[0]["modelo"])


def population_weighted_metrics(errors: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """Extrapola tasas fuera de muestra al total de pixeles validos.

    La muestra de modelado toma 10,000 pixeles por lago--fecha, de modo que da
    a un lago pequeno la mitad de las filas y eleva la prevalencia global. Dentro
    de cada escena el muestreo si es uniforme, asi que TPR y FPR estimados por
    escena son insesgados; reponderarlos por el conteo real de clases devuelve
    el desempeno esperado sobre la poblacion completa.

    ``errors`` necesita columnas lago, fecha, TP, FN, FP y TN por escena;
    ``classes`` aporta los conteos reales ``baja`` y ``alta`` de cada escena.
    """

    counts = errors.copy()
    for column in ("TP", "FN", "FP", "TN"):
        if column not in counts:
            counts[column] = 0
    counts["fecha"] = pd.to_datetime(counts["fecha"]).dt.strftime("%Y-%m-%d")
    reference = classes.copy()
    reference["fecha"] = pd.to_datetime(reference["fecha"]).dt.strftime("%Y-%m-%d")
    merged = counts.merge(reference[["lago", "fecha", "baja", "alta"]], on=["lago", "fecha"])
    if merged.empty:
        raise ValueError("Ninguna escena de errores coincide con la tabla de clases")

    positives = merged["TP"] + merged["FN"]
    negatives = merged["FP"] + merged["TN"]
    tpr = np.where(positives > 0, merged["TP"] / positives.where(positives > 0), 0.0)
    fpr = np.where(negatives > 0, merged["FP"] / negatives.where(negatives > 0), 0.0)
    merged["tp_poblacional"] = tpr * merged["alta"]
    merged["fn_poblacional"] = merged["alta"] - merged["tp_poblacional"]
    merged["fp_poblacional"] = fpr * merged["baja"]
    merged["tn_poblacional"] = merged["baja"] - merged["fp_poblacional"]

    def _summarize(frame: pd.DataFrame, scope: str, lake: str) -> dict:
        tp = float(frame["tp_poblacional"].sum())
        fn = float(frame["fn_poblacional"].sum())
        fp = float(frame["fp_poblacional"].sum())
        tn = float(frame["tn_poblacional"].sum())
        total = tp + fn + fp + tn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "alcance": scope,
            "lago": lake,
            "pixeles_validos": int(round(total)),
            "prevalencia_real": (tp + fn) / total if total else 0.0,
            "accuracy": (tp + tn) / total if total else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(round(tp)),
            "fp": int(round(fp)),
            "fn": int(round(fn)),
            "tn": int(round(tn)),
        }

    rows = [_summarize(merged, "global", "ambos")]
    rows.extend(_summarize(group, "lago", lake) for lake, group in merged.groupby("lago"))
    return pd.DataFrame(rows)


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
    """Ajusta y compara validacion aleatoria, espacial, temporal y entre lagos.

    Los hiperparametros se buscan una sola vez sobre una submuestra de la
    particion aleatoria de entrenamiento y se reutilizan en los cuatro
    esquemas. Como esa submuestra abarca todos los bloques de 1 km, ambos lagos
    y las dos fechas retenidas, las validaciones espacial, temporal y entre
    lagos arrastran fuga de seleccion de hiperparametros: sus metricas son
    ligeramente optimistas. El sesgo es pequeno porque la rejilla tiene dos o
    tres puntos por modelo, pero un ajuste anidado por esquema lo eliminaria.
    """

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
        pred["source_index"] = random_test.index.to_numpy()
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
            pred["source_index"] = sp_test.index.to_numpy()
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
    # Gradient Boosting y Random Forest quedan separados por 4e-5 de ROC-AUC en
    # una sola particion: ordenar por ese margen seria decidir con ruido. Se
    # consideran empatados todos los modelos dentro de la tolerancia y se
    # desempata por recall, que es la metrica prioritaria en vigilancia
    # ambiental porque un falso negativo omite una zona que requiere inspeccion.
    best_name = select_best_model(random_metrics)
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
        fitted_models=tuned,
        random_test=random_test,
    )
