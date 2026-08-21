"""Genera todos los resultados finales del Laboratorio 4, ML y GIS, Parte 2."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_geom
import seaborn as sns
import shap
from scipy.stats import spearmanr
from sklearn.base import clone


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.config import load_lake_geometry  # noqa: E402
from lab4_ml.data import (  # noqa: E402
    SAFE_PREDICTORS,
    classify_errors,
    load_raster_observation,
    values_to_surface,
)
from lab4_ml.modeling import (  # noqa: E402
    ExperimentResult,
    population_weighted_metrics,
    run_experiments,
)
from sklearn.pipeline import Pipeline  # noqa: E402


COLORS = {"atitlan": "#0f9d91", "amatitlan": "#f59e42"}
LAKE_NAMES = {"atitlan": "Atitlán", "amatitlan": "Amatitlán"}
ERROR_COLORS = {"TN": "#cbd5e1", "TP": "#0f9d91", "FP": "#f59e0b", "FN": "#dc2626"}

# Unica fuente de verdad de las categorias operativas: el informe, la barra de
# color y el CSV se derivan de aqui para que no puedan volver a desincronizarse.
CATEGORY_EDGES = (1 / 3, 2 / 3)
CATEGORY_LABELS = (
    f"Baja < {CATEGORY_EDGES[0]:.2f}",
    f"Media {CATEGORY_EDGES[0]:.2f}-{CATEGORY_EDGES[1]:.2f}",
    f"Alta >= {CATEGORY_EDGES[1]:.2f}",
)
# Por debajo de esta fraccion del |SHAP| maximo, el signo de la correlacion es
# ruido de colinealidad entre bandas vecinas y no se reporta como direccion.
SHAP_SIGNAL_FLOOR = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "ml_stack",
        help="Directorio que contiene atitlan/ y amatitlan/ con 11 TIFF cada uno.",
    )
    return parser.parse_args()


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7fafc",
            "axes.facecolor": "#ffffff",
            "axes.titleweight": "bold",
            "axes.titlesize": 12.5,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
        }
    )


def _projected_lake_rings(lake: str) -> list[np.ndarray]:
    feature = load_lake_geometry(lake)["features"][0]
    geom = transform_geom("EPSG:4326", "EPSG:32615", feature["geometry"], precision=2)
    polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return [np.asarray(polygon[0], dtype=float) for polygon in polygons]


def _draw_outline(ax, lake: str, *, color="#102a43", linewidth=1.2) -> None:
    for ring in _projected_lake_rings(lake):
        ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=linewidth, zorder=5)


def save_schema_and_counts(data: pd.DataFrame, classes: pd.DataFrame, tables: Path) -> None:
    schema = pd.DataFrame(
        {
            "variable": data.columns,
            "tipo": [str(data[column].dtype) for column in data.columns],
            "faltantes": [int(data[column].isna().sum()) for column in data.columns],
            "faltantes_pct": [100.0 * float(data[column].isna().mean()) for column in data.columns],
            "valores_unicos": [int(data[column].nunique(dropna=True)) for column in data.columns],
        }
    )
    schema.to_csv(tables / "ml_esquema_faltantes.csv", index=False)

    global_low = int(classes["baja"].sum())
    global_high = int(classes["alta"].sum())
    rows = [
        {
            "alcance": "global",
            "lago": "ambos",
            "fecha": "todas",
            "baja": global_low,
            "alta": global_high,
            "total": global_low + global_high,
            "alta_pct": 100 * global_high / (global_low + global_high),
        }
    ]
    for lake, group in classes.groupby("lago"):
        low, high = int(group["baja"].sum()), int(group["alta"].sum())
        rows.append(
            {
                "alcance": "lago",
                "lago": lake,
                "fecha": "todas",
                "baja": low,
                "alta": high,
                "total": low + high,
                "alta_pct": 100 * high / (low + high),
            }
        )
    for row in classes.itertuples(index=False):
        rows.append(
            {
                "alcance": "lago_fecha",
                "lago": row.lago,
                "fecha": row.fecha,
                "baja": int(row.baja),
                "alta": int(row.alta),
                "total": int(row.baja + row.alta),
                "alta_pct": float(row.alta_pct),
            }
        )
    pd.DataFrame(rows).to_csv(tables / "ml_distribucion_respuesta.csv", index=False)


def save_eda(data: pd.DataFrame, classes: pd.DataFrame, figures: Path) -> None:
    summary = classes.groupby("lago")[["baja", "alta"]].sum().reset_index()
    long = summary.melt(id_vars="lago", var_name="clase", value_name="observaciones")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(
        data=long,
        x="lago",
        y="observaciones",
        hue="clase",
        ax=ax,
        palette={"baja": "#64748b", "alta": "#e45756"},
    )
    ax.set(
        title="Distribución completa de la respuesta por lago",
        xlabel="Lago",
        ylabel="Píxeles válidos",
    )
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    fig.savefig(figures / "ml_distribucion_clases.png", dpi=190)
    plt.close(fig)

    sample = data.sample(n=min(35_000, len(data)), random_state=2026)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for lake, color in COLORS.items():
        subset = sample.loc[sample["lago"] == lake]
        sns.kdeplot(
            subset["CYA"].clip(upper=subset["CYA"].quantile(0.99)),
            ax=axes[0],
            label=LAKE_NAMES[lake],
            color=color,
            fill=False,
        )
    axes[0].axvline(100, color="#dc2626", linestyle="--", label="Umbral = 100")
    axes[0].set(title="CYA (recorte p99)", xlabel=r"CYA ($10^3$ células/mL)")
    axes[0].legend(fontsize=8)
    sns.boxplot(data=sample, x="lago", y="NDVI", hue="lago", palette=COLORS, ax=axes[1], legend=False)
    axes[1].set(title="NDVI por lago", xlabel="Lago")
    sns.boxplot(data=sample, x="lago", y="NDWI", hue="lago", palette=COLORS, ax=axes[2], legend=False)
    axes[2].set(title="NDWI por lago", xlabel="Lago")
    fig.tight_layout()
    fig.savefig(figures / "ml_eda_indices_cya.png", dpi=190)
    plt.close(fig)

    classes = classes.copy()
    classes["fecha"] = pd.to_datetime(classes["fecha"])
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=False)
    for ax, lake in zip(axes, ("atitlan", "amatitlan")):
        subset = classes.loc[classes["lago"] == lake].sort_values("fecha")
        ax.plot(subset["fecha"], subset["alta_pct"], marker="o", linewidth=2.2, color=COLORS[lake])
        ax.fill_between(subset["fecha"], 0, subset["alta_pct"], alpha=0.13, color=COLORS[lake])
        ax.set(title=f"{LAKE_NAMES[lake]} — proporción de CYA alta", ylabel="CYA alta (%)", xlabel="Fecha")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(bottom=0)
    fig.suptitle("Distribución temporal de la respuesta binaria", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "ml_clases_por_fecha.png", dpi=190)
    plt.close(fig)


def save_block_grid(data: pd.DataFrame, figures: Path, tables: Path) -> None:
    blocks = (
        data.groupby(["lago", "bloque_1km", "bloque_x", "bloque_y"], as_index=False)
        .agg(observaciones=("cya_alta", "size"), altas=("cya_alta", "sum"))
    )
    blocks["alta_pct"] = 100 * blocks["altas"] / blocks["observaciones"]
    blocks.to_csv(tables / "ml_bloques_1km.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    for ax, lake in zip(axes, ("atitlan", "amatitlan")):
        lake_blocks = blocks.loc[blocks["lago"] == lake]
        rectangles = [
            Rectangle((row.bloque_x * 1000, row.bloque_y * 1000), 1000, 1000)
            for row in lake_blocks.itertuples(index=False)
        ]
        collection = PatchCollection(rectangles, cmap="viridis", edgecolor="white", linewidth=0.6)
        collection.set_array(lake_blocks["observaciones"].to_numpy())
        ax.add_collection(collection)
        _draw_outline(ax, lake)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set(
            title=f"{LAKE_NAMES[lake]} — {len(lake_blocks)} bloques",
            xlabel="Este UTM (m)",
            ylabel="Norte UTM (m)",
        )
        fig.colorbar(collection, ax=ax, label="Observaciones muestreadas")
    fig.suptitle("Cuadrícula espacial de 1 km usada por GroupKFold", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "ml_bloques_espaciales_1km.png", dpi=190)
    plt.close(fig)


def save_metric_products(result: ExperimentResult, figures: Path, tables: Path) -> pd.DataFrame:
    result.metrics.to_csv(tables / "ml_metricas_detalle.csv", index=False)
    result.parameters.to_csv(tables / "ml_hiperparametros.csv", index=False)
    result.feature_importance.to_csv(tables / "ml_importancia_global.csv", index=False)
    aggregate = (
        result.metrics.groupby(["modelo", "validacion"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            roc_auc=("roc_auc", "mean"),
            folds=("fold", "nunique"),
            n_evaluaciones=("n", "sum"),
        )
    )
    aggregate.to_csv(tables / "ml_metricas_resumen.csv", index=False)

    plot = aggregate.melt(
        id_vars=["modelo", "validacion"],
        value_vars=["precision", "recall", "f1", "roc_auc"],
        var_name="metrica",
        value_name="valor",
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    validations = ["aleatoria_70_30", "espacial_groupkfold", "temporal_ultima_fecha", "entre_lagos"]
    for ax, validation in zip(axes.flat, validations):
        subset = plot.loc[plot["validacion"] == validation]
        sns.barplot(data=subset, x="metrica", y="valor", hue="modelo", ax=ax)
        ax.set(title=validation.replace("_", " ").title(), xlabel="", ylabel="Valor", ylim=(0, 1.03))
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Desempeño según esquema de validación", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "ml_comparacion_validaciones.png", dpi=190)
    plt.close(fig)

    subset = result.metrics.loc[result.metrics["validacion"] == "aleatoria_70_30"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (_, row) in zip(axes, subset.sort_values("modelo").iterrows()):
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", cbar=False, ax=ax)
        ax.set(
            title=row.modelo,
            xlabel="Predicción",
            ylabel="Real",
            xticklabels=["Baja", "Alta"],
            yticklabels=["Baja", "Alta"],
        )
    fig.tight_layout()
    fig.savefig(figures / "ml_matrices_confusion_aleatoria.png", dpi=190)
    plt.close(fig)

    ordered = result.feature_importance.sort_values("importancia_media")
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.barh(
        ordered["variable"],
        ordered["importancia_media"],
        xerr=ordered["importancia_sd"],
        color="#2563eb",
        alpha=0.85,
    )
    ax.set(
        title=f"Importancia global por permutación — {ordered['modelo'].iloc[0]}",
        xlabel="Caída media de ROC-AUC",
        ylabel="Variable",
    )
    fig.tight_layout()
    fig.savefig(figures / "ml_importancia_global.png", dpi=190)
    plt.close(fig)
    return aggregate


def _class_one_explanation(explanation: shap.Explanation) -> shap.Explanation:
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    if values.ndim == 3:
        values = values[:, :, 1]
        if base.ndim == 2:
            base = base[:, 1]
    return shap.Explanation(
        values=values,
        base_values=base,
        data=np.asarray(explanation.data),
        feature_names=list(explanation.feature_names),
    )


def save_shap_products(
    result: ExperimentResult,
    figures: Path,
    tables: Path,
    *,
    random_state: int = 2026,
) -> None:
    best_pipeline = result.fitted_models[result.best_model]
    estimator = best_pipeline.named_steps["model"]
    test = result.random_test
    sampled = (
        test.groupby("cya_alta", group_keys=False)
        .apply(lambda group: group.sample(n=min(2_000, len(group)), random_state=random_state), include_groups=False)
        .sort_index()
    )
    x_sample = sampled[SAFE_PREDICTORS]
    try:
        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(x_sample)
        method = "TreeExplainer"
    except Exception:
        background = result.random_test[SAFE_PREDICTORS].sample(n=300, random_state=random_state)
        explainer = shap.Explainer(best_pipeline.predict_proba, background, feature_names=SAFE_PREDICTORS)
        explanation = explainer(x_sample, max_evals=2 * len(SAFE_PREDICTORS) + 1)
        method = "PermutationExplainer"
    explanation = _class_one_explanation(explanation)
    values = np.asarray(explanation.values)

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=len(SAFE_PREDICTORS), show=False)
    plt.title(f"SHAP global — {result.best_model}")
    plt.tight_layout()
    plt.savefig(figures / "ml_shap_beeswarm.png", dpi=190, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(explanation, max_display=len(SAFE_PREDICTORS), show=False)
    plt.title(f"Importancia SHAP media — {result.best_model}")
    plt.tight_layout()
    plt.savefig(figures / "ml_shap_bar.png", dpi=190, bbox_inches="tight")
    plt.close()

    probabilities = best_pipeline.predict_proba(x_sample)[:, 1]
    high_position = int(np.argmax(probabilities))
    low_position = int(np.argmin(probabilities))
    for position, label in ((high_position, "alta"), (low_position, "baja")):
        plt.figure()
        shap.plots.waterfall(explanation[position], max_display=11, show=False)
        plt.title(f"Explicación local — probabilidad {label} ({probabilities[position]:.3f})")
        plt.tight_layout()
        plt.savefig(figures / f"ml_shap_local_{label}.png", dpi=190, bbox_inches="tight")
        plt.close()

    # La muestra explicada esta balanceada a proposito (misma cantidad de alta y
    # baja), de modo que el valor base y las magnitudes corresponden a un prior
    # 50/50 y no a la prevalencia del test ni a la poblacional. Se registra para
    # que la tabla no se lea como si fuera la distribucion real.
    sampled_labels = test.loc[sampled.index, "cya_alta"]
    magnitudes = np.abs(values).mean(axis=0)
    largest = float(magnitudes.max())
    summary_rows = []
    for index, variable in enumerate(SAFE_PREDICTORS):
        feature = x_sample[variable].to_numpy()
        shap_values = values[:, index]
        correlation = float(spearmanr(feature, shap_values).statistic)
        share = float(magnitudes[index]) / largest if largest > 0 else 0.0
        if share < SHAP_SIGNAL_FLOOR:
            # Bandas vecinas (B07, B08, B8A) son casi colineales: cuando el
            # efecto es marginal frente al dominante, el signo es un artefacto
            # del reparto de cortes y no una direccion interpretable.
            direction = "no_concluyente"
        elif correlation > 0.1:
            direction = "aumenta"
        elif correlation < -0.1:
            direction = "reduce"
        else:
            direction = "no_monotona"
        summary_rows.append(
            {
                "variable": variable,
                "shap_abs_medio": float(magnitudes[index]),
                "senal_relativa": share,
                "correlacion_rango_valor_shap": correlation,
                "direccion_general": direction,
                "metodo": method,
                "modelo": result.best_model,
                "n_explicado": len(x_sample),
                "n_alta": int(sampled_labels.sum()),
                "n_baja": int((1 - sampled_labels).sum()),
            }
        )
    pd.DataFrame(summary_rows).sort_values("shap_abs_medio", ascending=False).to_csv(
        tables / "ml_shap_resumen.csv", index=False
    )


def _predict_in_chunks(model, frame: pd.DataFrame, chunk_size: int = 100_000) -> np.ndarray:
    probability = np.empty(len(frame), dtype="float32")
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        probability[start:stop] = model.predict_proba(frame.iloc[start:stop][SAFE_PREDICTORS])[:, 1]
    return probability


def _write_probability_raster(path: Path, observation, probability_surface: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = deepcopy(observation.profile)
    profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(probability_surface), probability_surface, -9999.0).astype("float32"), 1)
        dst.set_band_description(1, "probabilidad_cya_alta")


def save_predictive_maps(
    template: Pipeline,
    data: pd.DataFrame,
    raw_dir: Path,
    processed: Path,
    figures: Path,
    tables: Path,
) -> None:
    """Cartografia cada escena con un modelo que nunca la vio.

    Se mapea la ultima fecha de cada lago, que es tambien la fecha retenida en
    la validacion temporal. Reajustar sobre las 220,000 filas dejaria dentro de
    entrenamiento hasta el 29 % de los pixeles validos de esa misma escena, de
    modo que el mapa dejaria de ser comparable con las metricas fuera de
    muestra. Por eso cada escena se predice con un modelo reentrenado sin su
    fecha, conservando los hiperparametros seleccionados.
    """

    rows = []
    output_dir = processed / "prediction_maps"
    category_cmap = ListedColormap(["#dbeafe", "#fbbf24", "#dc2626"])
    category_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], category_cmap.N)
    for lake in ("atitlan", "amatitlan"):
        path = sorted((raw_dir / lake).glob("*.tif"))[-1]
        observation = load_raster_observation(path, lake)
        train = data.loc[data["fecha"] != observation.date]
        if len(train) == len(data):
            raise ValueError(
                f"{lake}: la fecha {observation.date.date()} no esta en el dataset; "
                "no se puede garantizar un mapa fuera de muestra"
            )
        model = clone(template).fit(train[SAFE_PREDICTORS], train["cya_alta"])
        probability = _predict_in_chunks(model, observation.data)
        surface = values_to_surface(observation, probability)
        category = np.digitize(probability, CATEGORY_EDGES).astype("float32")
        category_surface = values_to_surface(observation, category)
        cya_surface = values_to_surface(observation, observation.data["CYA"].to_numpy())
        extent = (
            observation.profile["transform"].c,
            observation.profile["transform"].c + observation.profile["width"] * observation.profile["transform"].a,
            observation.profile["transform"].f + observation.profile["height"] * observation.profile["transform"].e,
            observation.profile["transform"].f,
        )
        raster_path = output_dir / f"{lake}_{observation.date.date()}_probabilidad.tif"
        _write_probability_raster(raster_path, observation, surface)

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
        cya_max = float(np.nanquantile(cya_surface, 0.99))
        image = axes[0].imshow(cya_surface, extent=extent, origin="upper", cmap="magma", vmin=0, vmax=cya_max)
        fig.colorbar(image, ax=axes[0], label=r"CYA ($10^3$ células/mL)")
        image = axes[1].imshow(surface, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(image, ax=axes[1], label="Probabilidad de CYA alta")
        image = axes[2].imshow(category_surface, extent=extent, origin="upper", cmap=category_cmap, norm=category_norm)
        colorbar = fig.colorbar(image, ax=axes[2], ticks=[0, 1, 2])
        colorbar.ax.set_yticklabels(list(CATEGORY_LABELS))
        for ax, title in zip(axes, ("CYA observado (Parte 1)", "Probabilidad predicha", "Categoría operativa")):
            _draw_outline(ax, lake)
            ax.set(title=title, xlabel="Este UTM (m)", ylabel="Norte UTM (m)", aspect="equal")
        fig.suptitle(
            f"{LAKE_NAMES[lake]} — observado vs. modelo (fuera de muestra) — {observation.date.date()}",
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(figures / f"ml_comparacion_observado_predicho_{lake}.png", dpi=190)
        plt.close(fig)

        rows.append(
            {
                "lago": lake,
                "fecha": observation.date.date().isoformat(),
                "pixeles_validos": len(probability),
                "probabilidad_media": float(np.mean(probability)),
                "probabilidad_mediana": float(np.median(probability)),
                "categoria_baja_pct": 100 * float(np.mean(category == 0)),
                "categoria_media_pct": 100 * float(np.mean(category == 1)),
                "categoria_alta_pct": 100 * float(np.mean(category == 2)),
                "esquema_modelo": "reentrenado_sin_esta_fecha",
                "filas_entrenamiento": len(train),
                "corte_baja_media": CATEGORY_EDGES[0],
                "corte_media_alta": CATEGORY_EDGES[1],
                "archivo_raster_local": raster_path.relative_to(ROOT).as_posix(),
            }
        )
    pd.DataFrame(rows).to_csv(tables / "ml_mapas_predictivos_resumen.csv", index=False)


def save_population_metrics(predictions: pd.DataFrame, classes: pd.DataFrame, tables: Path) -> None:
    """Escribe el desempeno esperado sobre los 3,419,056 pixeles validos."""

    counts = (
        predictions.groupby(["lago", "fecha", "error"]).size().unstack(fill_value=0).reset_index()
    )
    population_weighted_metrics(counts, classes).to_csv(
        tables / "ml_metricas_poblacionales.csv", index=False
    )


def save_spatial_error_map(result: ExperimentResult, figures: Path, tables: Path) -> pd.DataFrame:
    predictions = result.spatial_predictions.loc[
        result.spatial_predictions["modelo"] == result.best_model
    ].copy()
    predictions["error"] = classify_errors(predictions["cya_alta"], predictions["prediccion"])
    summary = (
        predictions.groupby(["lago", "fecha", "error"], as_index=False)
        .size()
        .rename(columns={"size": "observaciones"})
    )
    totals = summary.groupby(["lago", "fecha"])["observaciones"].transform("sum")
    summary["porcentaje"] = 100 * summary["observaciones"] / totals
    summary.to_csv(tables / "ml_errores_espaciales.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    for ax, lake in zip(axes, ("atitlan", "amatitlan")):
        subset = predictions.loc[predictions["lago"] == lake]
        for category in ("TN", "TP", "FP", "FN"):
            points = subset.loc[subset["error"] == category]
            if category == "TN" and len(points) > 15_000:
                points = points.sample(15_000, random_state=2026)
            ax.scatter(
                points["x_utm"],
                points["y_utm"],
                s=7 if category in {"FP", "FN"} else 3,
                alpha=0.75 if category in {"FP", "FN"} else 0.18,
                color=ERROR_COLORS[category],
                label=f"{category} (n={len(subset.loc[subset['error'] == category]):,})",
                linewidths=0,
            )
        _draw_outline(ax, lake)
        ax.set(
            title=LAKE_NAMES[lake],
            xlabel="Este UTM (m)",
            ylabel="Norte UTM (m)",
            aspect="equal",
        )
        ax.legend(fontsize=7, loc="best", frameon=True)
    fig.suptitle(
        f"Errores espaciales fuera de muestra — GroupKFold — {result.best_model}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(figures / "ml_errores_espaciales.png", dpi=190)
    plt.close(fig)
    return predictions


def main() -> int:
    args = parse_args()
    configure_style()
    processed = ROOT / "data" / "processed"
    tables = ROOT / "outputs" / "tables"
    figures = ROOT / "outputs" / "figures"
    models = processed / "models"
    for directory in (processed, tables, figures, models):
        directory.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(processed / "dataset_ml.csv.gz", parse_dates=["fecha"])
    classes = pd.read_csv(tables / "ml_clases_por_fecha.csv")
    save_schema_and_counts(data, classes, tables)
    save_eda(data, classes, figures)
    save_block_grid(data, figures, tables)

    result = run_experiments(data)
    aggregate = save_metric_products(result, figures, tables)
    result.random_predictions.to_csv(
        processed / "predicciones_aleatorias.csv.gz", index=False, compression="gzip"
    )
    result.spatial_predictions.to_csv(
        processed / "predicciones_espaciales.csv.gz", index=False, compression="gzip"
    )
    (tables / "ml_mejor_modelo.txt").write_text(result.best_model + "\n", encoding="utf-8")

    save_shap_products(result, figures, tables)
    spatial_errors = save_spatial_error_map(result, figures, tables)
    save_population_metrics(spatial_errors, classes, tables)
    final_model = clone(result.fitted_models[result.best_model]).fit(data[SAFE_PREDICTORS], data["cya_alta"])
    joblib.dump(final_model, models / "modelo_final.joblib")
    (models / "metadata.json").write_text(
        json.dumps(
            {
                "modelo": result.best_model,
                "predictores": SAFE_PREDICTORS,
                "n_entrenamiento_final": len(data),
                "umbral_probabilidad": 0.5,
                "uso": (
                    "Modelo de despliegue ajustado con las 220,000 filas. Los mapas del "
                    "informe NO usan este objeto: cada escena se predice con un modelo "
                    "reentrenado sin su propia fecha para que sea fuera de muestra."
                ),
                "cortes_categoria": list(CATEGORY_EDGES),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_predictive_maps(
        result.fitted_models[result.best_model], data, args.raw_dir, processed, figures, tables
    )

    print(aggregate.to_string(index=False))
    print(f"Modelo final: {result.best_model}")
    print("Resultados finales generados sin usar variables con fuga.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
