"""Ejecuta experimentos y figuras del avance (75 %) de ML y GIS."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4_ml.modeling import run_experiments  # noqa: E402


COLORS = {"atitlan": "#20b6b2", "amatitlan": "#f59e42"}


def style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7fafc",
            "axes.facecolor": "#ffffff",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "font.family": "DejaVu Sans",
        }
    )


def save_eda(data: pd.DataFrame, class_by_date: pd.DataFrame, figures: Path) -> None:
    summary = class_by_date.groupby("lago")[["baja", "alta"]].sum().reset_index()
    long = summary.melt(id_vars="lago", var_name="clase", value_name="observaciones")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(data=long, x="lago", y="observaciones", hue="clase", ax=ax,
                palette={"baja": "#64748b", "alta": "#e45756"})
    ax.set(title="Distribucion completa de la respuesta por lago", xlabel="Lago", ylabel="Pixeles validos")
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    fig.savefig(figures / "ml_distribucion_clases.png", dpi=180)
    plt.close(fig)

    sample = data.sample(n=min(35_000, len(data)), random_state=2026)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for lake, color in COLORS.items():
        subset = sample.loc[sample["lago"] == lake]
        sns.kdeplot(subset["CYA"].clip(upper=subset["CYA"].quantile(0.99)),
                    ax=axes[0], label=lake.title(), color=color, fill=False)
    axes[0].axvline(100, color="#dc2626", linestyle="--", label="Umbral = 100")
    axes[0].set(title="CYA (recorte p99)", xlabel=r"CYA ($10^3$ celulas/mL)")
    axes[0].legend()
    sns.boxplot(data=sample, x="lago", y="NDVI", hue="lago", palette=COLORS, ax=axes[1], legend=False)
    axes[1].set(title="NDVI por lago", xlabel="Lago")
    sns.boxplot(data=sample, x="lago", y="NDWI", hue="lago", palette=COLORS, ax=axes[2], legend=False)
    axes[2].set(title="NDWI por lago", xlabel="Lago")
    fig.tight_layout()
    fig.savefig(figures / "ml_eda_indices_cya.png", dpi=180)
    plt.close(fig)


def save_blocks(data: pd.DataFrame, figures: Path) -> None:
    blocks = data.groupby(["lago", "bloque_1km"], as_index=False).agg(
        x_utm=("x_utm", "mean"), y_utm=("y_utm", "mean"), n=("cya_alta", "size")
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, lake in zip(axes, ("atitlan", "amatitlan")):
        lake_blocks = blocks.loc[blocks["lago"] == lake]
        scatter = ax.scatter(
            lake_blocks["x_utm"], lake_blocks["y_utm"], c=lake_blocks["n"],
            s=90, marker="s", cmap="viridis", edgecolor="white", linewidth=0.5
        )
        ax.set(title=f"Bloques 1 km - {lake.title()}", xlabel="Este UTM (m)", ylabel="Norte UTM (m)", aspect="equal")
        fig.colorbar(scatter, ax=ax, label="Observaciones muestreadas")
    fig.tight_layout()
    fig.savefig(figures / "ml_bloques_espaciales_1km.png", dpi=180)
    plt.close(fig)


def save_metrics(metrics: pd.DataFrame, figures: Path) -> pd.DataFrame:
    aggregate = (
        metrics.groupby(["modelo", "validacion"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"), precision=("precision", "mean"),
            recall=("recall", "mean"), f1=("f1", "mean"), roc_auc=("roc_auc", "mean"),
            folds=("fold", "nunique"), n_evaluaciones=("n", "sum"),
        )
    )
    plot = aggregate.melt(
        id_vars=["modelo", "validacion"],
        value_vars=["precision", "recall", "f1", "roc_auc"],
        var_name="metrica", value_name="valor",
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    validations = ["aleatoria_70_30", "espacial_groupkfold", "temporal_ultima_fecha", "entre_lagos"]
    for ax, validation in zip(axes.flat, validations):
        subset = plot.loc[plot["validacion"] == validation]
        sns.barplot(data=subset, x="metrica", y="valor", hue="modelo", ax=ax)
        ax.set(title=validation.replace("_", " ").title(), xlabel="", ylabel="Valor", ylim=(0, 1.03))
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Desempeno segun esquema de validacion", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "ml_comparacion_validaciones.png", dpi=180)
    plt.close(fig)
    return aggregate


def save_confusions(metrics: pd.DataFrame, figures: Path) -> None:
    subset = metrics.loc[metrics["validacion"] == "aleatoria_70_30"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (_, row) in zip(axes, subset.sort_values("modelo").iterrows()):
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", cbar=False, ax=ax)
        ax.set(title=row.modelo, xlabel="Prediccion", ylabel="Real", xticklabels=["Baja", "Alta"], yticklabels=["Baja", "Alta"])
    fig.tight_layout()
    fig.savefig(figures / "ml_matrices_confusion_aleatoria.png", dpi=180)
    plt.close(fig)


def save_importance(importance: pd.DataFrame, figures: Path) -> None:
    ordered = importance.sort_values("importancia_media")
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.barh(ordered["variable"], ordered["importancia_media"], xerr=ordered["importancia_sd"], color="#2563eb", alpha=0.85)
    ax.set(title=f"Importancia global por permutacion - {ordered['modelo'].iloc[0]}", xlabel="Caida media de ROC-AUC", ylabel="Variable")
    fig.tight_layout()
    fig.savefig(figures / "ml_importancia_global.png", dpi=180)
    plt.close(fig)


def main() -> int:
    processed = ROOT / "data" / "processed"
    tables = ROOT / "outputs" / "tables"
    figures = ROOT / "outputs" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    style()

    data = pd.read_csv(processed / "dataset_ml.csv.gz", parse_dates=["fecha"])
    classes = pd.read_csv(tables / "ml_clases_por_fecha.csv")
    save_eda(data, classes, figures)
    save_blocks(data, figures)
    result = run_experiments(data)
    result.metrics.to_csv(tables / "ml_metricas_detalle.csv", index=False)
    result.parameters.to_csv(tables / "ml_hiperparametros.csv", index=False)
    result.feature_importance.to_csv(tables / "ml_importancia_global.csv", index=False)
    aggregate = save_metrics(result.metrics, figures)
    aggregate.to_csv(tables / "ml_metricas_resumen.csv", index=False)
    save_confusions(result.metrics, figures)
    save_importance(result.feature_importance, figures)
    result.random_predictions.to_csv(
        processed / "predicciones_aleatorias.csv.gz", index=False, compression="gzip"
    )
    result.spatial_predictions.to_csv(
        processed / "predicciones_espaciales.csv.gz", index=False, compression="gzip"
    )
    (tables / "ml_mejor_modelo.txt").write_text(result.best_model + "\n", encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(f"Mejor modelo en particion aleatoria: {result.best_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
