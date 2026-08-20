"""Descarga el stack multiespectral de la Parte 2 desde CDSE/openEO.

La autenticacion usa el flujo OIDC por codigo de dispositivo. No recibe,
imprime ni guarda usuario o contrasena.

Ejemplo:
    python scripts/download_ml_stack.py --lake all --submit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.config import load_observations  # noqa: E402
from lab4.copernicus import (  # noqa: E402
    ML_SPECTRAL_BANDS,
    authenticate_cdse,
    build_lake_ml_timeseries_cube,
    connect_cdse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", choices=["atitlan", "amatitlan", "all"], default="all")
    parser.add_argument("--resolution", type=float, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "ml_stack",
        help="Directorio local ignorado por Git para resultados y manifiesto.",
    )
    return parser.parse_args()


def append_manifest(path: Path, entry: dict) -> None:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = []
    manifest.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    observations = load_observations()
    lakes = ["atitlan", "amatitlan"] if args.lake == "all" else [args.lake]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"

    connection = connect_cdse()
    cubes: dict[str, tuple] = {}
    for lake in lakes:
        dates = (
            observations.loc[observations["lago"] == lake, "fecha"]
            .dt.date.astype(str).tolist()
        )
        if args.limit is not None:
            dates = dates[: args.limit]
        cube = build_lake_ml_timeseries_cube(
            connection, lake, dates, resolution=args.resolution
        )
        errors = connection.validate_process_graph(cube)
        if errors:
            raise RuntimeError(f"Grafo invalido para {lake}: {errors}")
        cubes[lake] = (cube, dates)
        print(
            f"[validado] {lake}: {len(dates)} fechas; "
            f"{len(ML_SPECTRAL_BANDS)} bandas + NDVI/NDWI/CYA",
            flush=True,
        )

    if not args.submit:
        print("Validacion terminada. Agregue --submit para descargar.")
        return 0

    authenticate_cdse(connection, max_poll_time=1800)
    submitted = []
    for lake, (cube, dates) in cubes.items():
        job = cube.create_job(
            out_format="GTiff",
            title=f"Lab4 ML {lake} stack multiespectral",
            description=(
                "Sentinel-2 L2A: B02-B08, B8A, B11, B12, NDVI, NDWI y CYA; "
                "mascara SCL y contorno lacustre."
            ),
        )
        created = {
            "job_id": job.job_id,
            "lago": lake,
            "fechas": dates,
            "bandas": [*ML_SPECTRAL_BANDS, "NDVI", "NDWI", "CYA"],
            "resolucion_m": args.resolution,
            "creado_utc": datetime.now(timezone.utc).isoformat(),
            "estado": "created",
        }
        append_manifest(manifest_path, created)
        print(f"[creado] {lake}: {job.job_id}", flush=True)
        job.start_job()
        print(f"[iniciado] {lake}: {job.job_id}", flush=True)
        submitted.append((lake, job, created))

    for lake, job, created in submitted:
        errors = 0
        while True:
            try:
                status = job.status()
                errors = 0
            except requests.RequestException as exc:
                errors += 1
                if errors > 10:
                    raise RuntimeError(f"No se pudo consultar {job.job_id}") from exc
                print(f"[red] reintento {errors}/10 en 30 s", flush=True)
                time.sleep(30)
                continue
            print(f"[estado] {lake}: {status}", flush=True)
            if status in {"finished", "error", "canceled"}:
                break
            time.sleep(30)
        if status != "finished":
            append_manifest(manifest_path, {**created, "estado": status})
            continue
        lake_dir = args.output_dir / lake
        lake_dir.mkdir(parents=True, exist_ok=True)
        files = job.get_results().download_files(lake_dir)
        append_manifest(
            manifest_path,
            {**created, "estado": "downloaded", "archivos": [str(p) for p in files]},
        )
        print(f"[descargado] {lake}: {len(files)} archivos en {lake_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
