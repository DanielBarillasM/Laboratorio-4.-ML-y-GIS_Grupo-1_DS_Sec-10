"""Pruebas de invariantes criticas de la Parte 2."""

import numpy as np
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
from lab4_ml.modeling import _metric_row


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
