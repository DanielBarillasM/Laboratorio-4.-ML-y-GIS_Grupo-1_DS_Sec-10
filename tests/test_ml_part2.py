"""Pruebas de invariantes criticas de la Parte 2."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from lab4_ml.data import (
    ALL_BANDS,
    CYA_THRESHOLD,
    LEAKAGE_EXCLUDED,
    SAFE_PREDICTORS,
    classify_errors,
    load_raster_observation,
    values_to_surface,
)
from lab4_ml.modeling import (
    _metric_row,
    population_weighted_metrics,
    select_best_model,
)


def _load_final_script():
    """Carga scripts/run_ml_final.py para verificar sus constantes de reporte."""

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ml_final.py"
    spec = importlib.util.spec_from_file_location("run_ml_final", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_threshold_matches_who_moderate_alert_scale():
    assert CYA_THRESHOLD == 100.0


def test_predictors_do_not_include_target_inputs():
    assert set(SAFE_PREDICTORS).isdisjoint(LEAKAGE_EXCLUDED)
    assert "CYA" not in SAFE_PREDICTORS


def test_metric_row_confusion_counts():
    row = _metric_row(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.9, 0.2]),
        model="prueba",
        validation="prueba",
        fold="1",
    )
    assert (row["tn"], row["fp"], row["fn"], row["tp"]) == (1, 1, 1, 1)
    assert row["accuracy"] == 0.5


def test_error_categories_are_correct():
    result = classify_errors([0, 0, 1, 1], [0, 1, 0, 1])
    assert result.tolist() == ["TN", "FP", "FN", "TP"]


def test_full_raster_features_preserve_nodata_and_blocks(tmp_path):
    path = tmp_path / "openEO_2026-07-22Z.tif"
    cube = np.full((len(ALL_BANDS), 2, 3), 0.2, dtype="float32")
    cube[ALL_BANDS.index("CYA")] = np.array([[20, 120, 50], [110, 10, 80]], dtype="float32")
    cube[:, 1, 2] = np.nan
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 3,
        "count": len(ALL_BANDS),
        "dtype": "float32",
        "crs": "EPSG:32615",
        "transform": from_origin(500_000, 1_600_000, 20, 20),
        "nodata": np.nan,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(cube)
        for index, name in enumerate(ALL_BANDS, start=1):
            dst.set_band_description(index, name)

    observation = load_raster_observation(path, "atitlan")
    assert len(observation.data) == 5
    assert observation.data["cya_alta"].sum() == 2
    assert observation.data["bloque_1km"].str.startswith("atitlan_").all()
    surface = values_to_surface(observation, np.arange(5))
    assert surface.shape == (2, 3)
    assert np.isnan(surface[1, 2])
    assert np.isfinite(surface).sum() == 5


def test_category_labels_stay_in_sync_with_edges():
    """El informe y la barra de color deben citar los cortes reales del codigo."""

    script = _load_final_script()
    low, high = script.CATEGORY_EDGES
    assert low < high
    assert f"{low:.2f}" in script.CATEGORY_LABELS[0]
    assert f"{high:.2f}" in script.CATEGORY_LABELS[2]
    assert f"{low:.2f}" in script.CATEGORY_LABELS[1]
    assert f"{high:.2f}" in script.CATEGORY_LABELS[1]


def test_select_best_model_treats_a_hairline_auc_gap_as_a_tie():
    metrics = pd.DataFrame(
        [
            {"modelo": "Random Forest", "roc_auc": 0.9986780, "recall": 0.9735},
            {"modelo": "Gradient Boosting", "roc_auc": 0.9987186, "recall": 0.9865},
        ]
    )
    # Gradient Boosting gana por recall, no por 4e-5 de ROC-AUC.
    assert select_best_model(metrics) == "Gradient Boosting"
    inverted = metrics.assign(recall=[0.9990, 0.9865])
    assert select_best_model(inverted) == "Random Forest"


def test_select_best_model_ignores_high_recall_when_auc_is_clearly_worse():
    metrics = pd.DataFrame(
        [
            {"modelo": "Gradient Boosting", "roc_auc": 0.9987, "recall": 0.9865},
            {"modelo": "Regresion logistica", "roc_auc": 0.9500, "recall": 0.9999},
        ]
    )
    assert select_best_model(metrics) == "Gradient Boosting"


def test_population_metrics_reweight_scene_rates_by_true_counts():
    """Una escena muestreada al 50 % debe reponderarse a su prevalencia real."""

    errors = pd.DataFrame(
        [{"lago": "atitlan", "fecha": "2026-07-22", "TP": 90, "FN": 10, "FP": 20, "TN": 80}]
    )
    classes = pd.DataFrame(
        [{"lago": "atitlan", "fecha": "2026-07-22", "baja": 99_000, "alta": 1_000}]
    )
    result = population_weighted_metrics(errors, classes)
    row = result.loc[result["alcance"] == "global"].iloc[0]

    # TPR = 0.90 y FPR = 0.20 medidos en la muestra equilibrada.
    assert row["tp"] == 900 and row["fn"] == 100
    assert row["fp"] == 19_800 and row["tn"] == 79_200
    assert row["prevalencia_real"] == pytest.approx(0.01)
    # La precision cae de 0.818 en la muestra a 0.043 en la poblacion real.
    assert row["precision"] == pytest.approx(900 / (900 + 19_800))
    assert row["recall"] == pytest.approx(0.90)


def test_population_metrics_reject_scenes_without_reference_counts():
    errors = pd.DataFrame(
        [{"lago": "atitlan", "fecha": "2026-07-22", "TP": 1, "FN": 0, "FP": 0, "TN": 1}]
    )
    classes = pd.DataFrame(
        [{"lago": "amatitlan", "fecha": "2025-01-28", "baja": 10, "alta": 1}]
    )
    with pytest.raises(ValueError):
        population_weighted_metrics(errors, classes)
