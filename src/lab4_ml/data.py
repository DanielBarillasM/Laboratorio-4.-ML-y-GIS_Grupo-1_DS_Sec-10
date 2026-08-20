"""Construccion del conjunto pixel-fecha para aprendizaje supervisado."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform as transform_coords


CYA_THRESHOLD = 100.0  # 10^3 celulas/mL = 100 000 celulas/mL
ALL_BANDS = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
    "NDVI", "NDWI", "CYA",
]
SAFE_SPECTRAL_PREDICTORS = ["B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SAFE_PREDICTORS = [
    *SAFE_SPECTRAL_PREDICTORS,
    "x_normalizada",
    "y_normalizada",
    "dia_anio_sin",
    "dia_anio_cos",
]
LEAKAGE_EXCLUDED = {
    "CYA": "es la variable continua que define la respuesta binaria",
    "B02": "participa directamente en la ecuacion Se2WaQ de CYA",
    "B03": "participa directamente en CYA y, por tanto, tambien contamina NDWI",
    "B04": "participa directamente en CYA y, por tanto, tambien contamina NDVI",
    "NDVI": "usa B04, una banda empleada para construir CYA",
    "NDWI": "usa B03, una banda empleada para construir CYA",
}


@dataclass(frozen=True)
class DatasetBuildResult:
    data: pd.DataFrame
    inventory: pd.DataFrame
    class_by_date: pd.DataFrame


def _date_from_path(path: Path) -> pd.Timestamp:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    if not match:
        raise ValueError(f"No se encontro fecha ISO en {path.name}")
    return pd.Timestamp(match.group(1))


def discover_stacks(raw_dir: Path) -> list[tuple[str, Path]]:
    """Descubre exactamente un GeoTIFF por lago y fecha."""

    found: list[tuple[str, Path]] = []
    for lake in ("atitlan", "amatitlan"):
        lake_dir = raw_dir / lake
        for path in sorted(lake_dir.glob("*.tif")):
            found.append((lake, path))
    if len(found) != 22:
        raise ValueError(f"Se esperaban 22 GeoTIFF y se encontraron {len(found)} en {raw_dir}")
    keys = {(lake, _date_from_path(path).date()) for lake, path in found}
    if len(keys) != 22:
        raise ValueError("Hay fechas duplicadas en los GeoTIFF multiespectrales")
    return found


def _band_names(src: rasterio.io.DatasetReader) -> list[str]:
    descriptions = [value.strip() if value else "" for value in src.descriptions]
    if descriptions and all(descriptions) and len(set(descriptions)) == src.count:
        return descriptions
    if src.count == len(ALL_BANDS):
        return ALL_BANDS.copy()
    raise ValueError(
        f"{src.name}: se esperaban {len(ALL_BANDS)} bandas con nombre y hay {src.count}"
    )


def build_pixel_dataset(
    raw_dir: Path,
    *,
    max_rows_per_lake_date: int = 10_000,
    random_state: int = 2026,
    threshold: float = CYA_THRESHOLD,
) -> DatasetBuildResult:
    """Limpia, cuenta y muestrea observaciones validas de los 22 rasters.

    El muestreo uniforme por lago-fecha limita el costo computacional sin
    alterar los valores. Los conteos de cobertura se calculan antes de
    muestrear. No se sobremuestrea la clase positiva.
    """

    rng = np.random.default_rng(random_state)
    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict] = []
    class_rows: list[dict] = []
    lake_bounds: dict[str, tuple[float, float, float, float]] = {}

    stacks = discover_stacks(raw_dir)
    for lake, path in stacks:
        with rasterio.open(path) as src:
            names = _band_names(src)
            missing = sorted(set(ALL_BANDS).difference(names))
            if missing:
                raise ValueError(f"{path.name}: faltan bandas {missing}")
            order = [names.index(name) + 1 for name in ALL_BANDS]
            cube = src.read(order, masked=True).filled(np.nan).astype("float32")
            valid = np.isfinite(cube).all(axis=0)
            rows_all, cols_all = np.where(valid)
            valid_count = int(len(rows_all))
            if valid_count == 0:
                raise ValueError(f"{path}: no tiene pixeles validos")
            cya_all = cube[ALL_BANDS.index("CYA")][valid]
            high_all = cya_all >= threshold
            date = _date_from_path(path)
            inventory_rows.append(
                {
                    "lago": lake,
                    "fecha": date.date().isoformat(),
                    "filas_raster": src.height,
                    "columnas_raster": src.width,
                    "pixeles_totales": int(src.height * src.width),
                    "observaciones_validas": valid_count,
                    "observaciones_invalidas": int(src.height * src.width - valid_count),
                    "cobertura_valida_pct": 100.0 * valid_count / (src.height * src.width),
                    "crs": str(src.crs),
                    "resolucion_x_m": abs(float(src.res[0])),
                    "resolucion_y_m": abs(float(src.res[1])),
                }
            )
            class_rows.append(
                {
                    "lago": lake,
                    "fecha": date.date().isoformat(),
                    "baja": int((~high_all).sum()),
                    "alta": int(high_all.sum()),
                    "alta_pct": 100.0 * float(high_all.mean()),
                    "umbral_cya_10e3_cel_ml": threshold,
                }
            )

            take = min(valid_count, max_rows_per_lake_date)
            selected = rng.choice(valid_count, size=take, replace=False)
            rows = rows_all[selected]
            cols = cols_all[selected]
            values = cube[:, rows, cols].T
            xs, ys = xy(src.transform, rows, cols, offset="center")
            xs = np.asarray(xs, dtype="float64")
            ys = np.asarray(ys, dtype="float64")
            lon, lat = transform_coords(src.crs, "EPSG:4326", xs.tolist(), ys.tolist())
            bounds = lake_bounds.setdefault(
                lake, (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            )

        frame = pd.DataFrame(values, columns=ALL_BANDS)
        frame.insert(0, "lago", lake)
        frame.insert(1, "fecha", date)
        frame["x_utm"] = xs
        frame["y_utm"] = ys
        frame["longitud"] = lon
        frame["latitud"] = lat
        frame["x_normalizada"] = (xs - bounds[0]) / (bounds[2] - bounds[0])
        frame["y_normalizada"] = (ys - bounds[1]) / (bounds[3] - bounds[1])
        day = date.dayofyear
        frame["dia_anio_sin"] = np.sin(2 * np.pi * day / 365.25)
        frame["dia_anio_cos"] = np.cos(2 * np.pi * day / 365.25)
        frame["cya_alta"] = (frame["CYA"] >= threshold).astype("int8")
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["bloque_x"] = np.floor(data["x_utm"] / 1000).astype("int32")
    data["bloque_y"] = np.floor(data["y_utm"] / 1000).astype("int32")
    data["bloque_1km"] = (
        data["lago"] + "_" + data["bloque_x"].astype(str) + "_" + data["bloque_y"].astype(str)
    )
    return DatasetBuildResult(
        data=data,
        inventory=pd.DataFrame(inventory_rows).sort_values(["lago", "fecha"]),
        class_by_date=pd.DataFrame(class_rows).sort_values(["lago", "fecha"]),
    )
