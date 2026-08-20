<div align="center">

# Laboratorio 4 · ML y GIS

### Clasificación geoespacial de cianobacterias en Atitlán y Amatitlán

[![Estado](https://img.shields.io/badge/estado-completo%20100%25-0F9D91?style=for-the-badge)](#cumplimiento)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-L2A-2563EB?style=for-the-badge)](https://dataspace.copernicus.eu/)
[![openEO](https://img.shields.io/badge/openEO-CDSE-F59E0B?style=for-the-badge)](https://openeo.dataspace.copernicus.eu/)

**Universidad del Valle de Guatemala · Data Science · Sección 10 · Grupo 1**

[Informe final](reports/informe_final.pdf) · [Notebook ejecutado](notebooks/02_laboratorio_final_ml_gis.ipynb) · [Ficha del repositorio](reports/Presentacion_Repositorio_Laboratorio_4_ML_GIS_Grupo_1.docx) · [Métricas](outputs/tables/ml_metricas_resumen.csv)

</div>

---

## Descripción

Este proyecto extiende el análisis de sensores remotos de la Parte 1 y usa imágenes **Sentinel-2 L2A** para clasificar concentraciones altas o bajas del proxy de cianobacterias **Se2WaQ** en los lagos de Atitlán y Amatitlán. La unidad analítica es un píxel acuático válido de 20 m asociado con coordenadas, fecha, lago, bandas espectrales e índices.

Se comparan regresión logística, Random Forest y Gradient Boosting bajo cuatro pruebas complementarias: partición aleatoria 70/30, validación espacial GroupKFold con bloques de 1 km, retención de la última fecha y transferencia entre lagos. El modelo seleccionado se interpreta con importancia por permutación y SHAP; finalmente se generan mapas de probabilidad, categorías de riesgo y errores espaciales fuera de muestra.

## Resultados principales

| Indicador | Resultado |
|---|---:|
| Escenas procesadas | 22 |
| Píxeles acuáticos válidos | 3,419,056 |
| Filas de modelado | 220,000 |
| Valores faltantes | 0 |
| Bloques espaciales de 1 km | 203 |
| Mejor ROC-AUC aleatorio | 0.9987 |
| Mejor F1 aleatorio | 0.9626 |
| Mejor recall temporal | 0.9384 |

Gradient Boosting alcanzó ROC-AUC 0.9987 y F1 0.9572 en prueba aleatoria, pero su F1 temporal bajó a 0.6651. La regresión logística mantuvo recall temporal 0.9384. Esta diferencia es uno de los resultados centrales: una división aleatoria no sustituye la validación geográfica y temporal.

SHAP identifica a **B05** y **B08** como las señales de mayor influencia. En la escena más reciente, el mapa de Atitlán clasificó 99.77% de los píxeles válidos con riesgo bajo; Amatitlán presentó 59.58% en categoría alta. Estas categorías expresan probabilidad del modelo y no reemplazan niveles sanitarios ni muestreos de campo.

## Cumplimiento

| Requisito de la Parte 2 | Estado |
|---|:---:|
| Dataset píxel–fecha, limpieza, tipos, faltantes y EDA | ✅ |
| Respuesta binaria, justificación, distribución y desbalance | ✅ |
| Ingeniería de variables y control estricto de fuga | ✅ |
| Tres algoritmos, ajuste y prueba común 70/30 | ✅ |
| Accuracy, precision, recall, F1, ROC-AUC y matrices | ✅ |
| Cuadrícula EPSG:32615 y GroupKFold espacial | ✅ |
| Validación temporal sobre las últimas fechas | ✅ |
| Transferencia Atitlán ↔ Amatitlán | ✅ |
| Importancia global y SHAP global/local | ✅ |
| Mapas predictivos y mapas FP/FN | ✅ |
| Conclusiones, utilidad, limitaciones y datos futuros | ✅ |
| Notebook ejecutado, informe PDF y documentación | ✅ |

## Control de fuga de información

La clase positiva se define como `CYA >= 100`, equivalente a 100,000 células/mL. Es un umbral de cribado inspirado en orientación histórica de la OMS para aguas recreativas; no sustituye conteos microscópicos, identificación taxonómica ni medición de toxinas.

La ecuación Se2WaQ de CYA usa B02, B03 y B04. Para evitar fuga directa o indirecta, el entrenamiento excluye:

```text
CYA · B02 · B03 · B04 · NDVI · NDWI
```

Los modelos usan B05, B06, B07, B08, B8A, B11, B12, coordenadas normalizadas dentro de cada lago y seno/coseno del día del año. El nombre del lago tampoco se entrega como predictor.

## Estructura del repositorio

```text
├── config/                 # Fechas y GeoJSON oficiales
├── data/
│   ├── raw/                # 22 GeoTIFF locales; ignorados por Git
│   ├── processed/          # Dataset, modelos y mapas; ignorados por Git
│   └── README.md           # Diccionario y política de datos
├── notebooks/              # Notebook final estético y ejecutado
├── outputs/
│   ├── figures/            # EDA, métricas, SHAP y cartografía
│   └── tables/             # Resultados reproducibles en CSV
├── reports/                # Informe final .tex/.pdf y ficha .docx
├── scripts/                # Descarga, dataset y análisis final
├── src/
│   ├── lab4/               # Configuración y acceso openEO
│   └── lab4_ml/            # Preparación, modelado y cartografía
└── tests/                  # Pruebas de invariantes críticos
```

## Reproducción

### 1. Instalar el entorno

```powershell
uv sync --extra test
```

También se incluye `requirements.txt` para flujos tradicionales con `pip`.

### 2. Descargar las escenas

```powershell
uv run python scripts/download_ml_stack.py --lake all --submit
```

Copernicus autentica mediante código de dispositivo. El proyecto no recibe ni almacena contraseñas.

### 3. Construir el dataset

```powershell
uv run python scripts/build_ml_dataset.py
```

### 4. Ejecutar el análisis completo y las pruebas

```powershell
uv run python scripts/run_ml_final.py
uv run pytest -q
```

### 5. Compilar el informe

```powershell
cd reports
pdflatex -interaction=nonstopmode -halt-on-error informe_final.tex
pdflatex -interaction=nonstopmode -halt-on-error informe_final.tex
```

## Datos grandes y reproducibilidad

Los GeoTIFF, el dataset comprimido, los modelos serializados y los GeoTIFF de probabilidad se conservan localmente y están excluidos de Git por tamaño. El repositorio sí versiona configuración, código, pruebas, tablas resumidas, figuras, notebook e informe, por lo que los datos pesados pueden regenerarse con los comandos anteriores.

## Integrantes

| Nombre | Carné |
|---|---:|
| Jorge Gabriel Palacios Sales | 231385 |
| Pablo Daniel Barillas Moreno | 22193 |
| Roberto Emiliano Otoniel | 23968 |

## Referencias principales

- [OMS — Guidelines on recreational water quality](https://www.who.int/publications/i/item/9789240031302)
- [EPA — WHO guideline values for cyanobacteria](https://www.epa.gov/habs/world-health-organization-who-1999-guideline-values-cyanobacteria-freshwater)
- [Sentinel Hub — Se2WaQ](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/se2waq/)
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
- [Lundberg y Lee — SHAP](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)

---

<div align="center">
Repositorio académico del Laboratorio 4 · ML y GIS · Parte 2.
</div>
