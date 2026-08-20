"""Pruebas de invariantes criticas de la Parte 2."""

import numpy as np

from lab4_ml.data import CYA_THRESHOLD, LEAKAGE_EXCLUDED, SAFE_PREDICTORS
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
