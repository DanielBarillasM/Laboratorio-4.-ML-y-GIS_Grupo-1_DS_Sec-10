"""Construye el conjunto reproducible pixel-fecha de la Parte 2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4_ml.data import CYA_THRESHOLD, build_pixel_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "ml_stack")
    parser.add_argument("--max-per-date", type=int, default=10_000)
    parser.add_argument("--threshold", type=float, default=CYA_THRESHOLD)
    args = parser.parse_args()

    processed = ROOT / "data" / "processed"
    tables = ROOT / "outputs" / "tables"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    result = build_pixel_dataset(
        args.raw_dir,
        max_rows_per_lake_date=args.max_per_date,
        threshold=args.threshold,
    )
    result.data.to_csv(processed / "dataset_ml.csv.gz", index=False, compression="gzip")
    result.inventory.to_csv(tables / "ml_inventario_limpieza.csv", index=False)
    result.class_by_date.to_csv(tables / "ml_clases_por_fecha.csv", index=False)
    print(f"Dataset: {len(result.data):,} filas x {result.data.shape[1]} columnas")
    print(f"Alta CYA (muestra): {result.data['cya_alta'].mean():.3%}")
    print(f"Guardado en {processed / 'dataset_ml.csv.gz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
