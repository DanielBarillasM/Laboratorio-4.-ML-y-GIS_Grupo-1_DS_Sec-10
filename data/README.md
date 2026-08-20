# Datos del proyecto

Los datos pesados se conservan localmente y están excluidos de Git. El repositorio versiona configuración, código, tablas resumidas, figuras, notebook e informe.

## Estructura local

```text
data/
├── raw/
│   ├── .gitkeep
│   └── ml_stack/
│       ├── atitlan/        # 11 GeoTIFF
│       └── amatitlan/      # 11 GeoTIFF
└── processed/
    ├── .gitkeep
    ├── dataset_ml.csv.gz
    ├── predicciones_aleatorias.csv.gz
    └── predicciones_espaciales.csv.gz
```

Cada GeoTIFF contiene 13 capas a 20 m:

```text
B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, NDVI, NDWI, CYA
```

`dataset_ml.csv.gz` contiene una muestra reproducible de 10,000 píxeles válidos por lago y fecha. Cada fila incluye identificadores, coordenadas UTM/WGS84, bandas, índices, CYA, etiqueta `cya_alta`, variables estacionales y bloque espacial de 1 km.

## Reconstrucción

```powershell
uv run python scripts/download_ml_stack.py --lake all --submit
uv run python scripts/build_ml_dataset.py
uv run python scripts/run_ml_advance.py
```

La descarga usa autenticación OIDC por código de dispositivo. No se guardan usuario, contraseña ni token de actualización.
