<div align="center">

# Laboratorio 4 · ML y GIS

### Clasificación geoespacial de cianobacterias en Atitlán y Amatitlán

[![Estado](https://img.shields.io/badge/estado-avance%2075%25-2563eb?style=for-the-badge)](#estado-del-proyecto)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-L2A-0F9D91?style=for-the-badge)](https://dataspace.copernicus.eu/)
[![openEO](https://img.shields.io/badge/openEO-CDSE-F59E0B?style=for-the-badge)](https://openeo.dataspace.copernicus.eu/)

**Grupo 1 · Sección 10 · Data Science**

[Informe PDF](reports/informe_avance.pdf) · [Notebook ejecutado](notebooks/01_avance_ml_gis.ipynb) · [Presentación del repositorio](reports/Presentacion_Repositorio_Laboratorio_4_ML_GIS_Grupo_1.docx) · [Métricas](outputs/tables/ml_metricas_resumen.csv)

</div>

---

## Descripción

Este proyecto utiliza imágenes **Sentinel‑2 L2A** para clasificar concentraciones altas y bajas del proxy de cianobacterias **Se2WaQ** en los lagos de Atitlán y Amatitlán. Cada observación representa un píxel acuático válido de 20 m asociado con coordenadas, fecha, lago, bandas espectrales y variables derivadas.

El avance implementa tres algoritmos y los evalúa bajo cuatro escenarios: partición aleatoria, validación espacial por bloques, retención temporal y transferencia entre lagos.

## Estado del proyecto

| Componente | Estado |
|---|:---:|
| Dataset píxel–fecha y limpieza | ✅ |
| Respuesta binaria y análisis de desbalance | ✅ |
| Ingeniería de variables sin fuga | ✅ |
| Regresión logística, Random Forest y Gradient Boosting | ✅ |
| Validación aleatoria 70/30 | ✅ |
| GroupKFold espacial de 1 km | ✅ |
| Validación temporal | ✅ |
| Transferencia Atitlán ↔ Amatitlán | ✅ |
| Importancia global por permutación | ✅ |
| SHAP global y local | ⏳ Final |
| Mapas predictivos y mapas FP/FN | ⏳ Final |
| Conclusiones integradas | ⏳ Final |

El repositorio corresponde al **avance del 75%**. El trabajo pendiente está delimitado en el informe y al final de este README.

## Resultados del avance

| Indicador | Resultado |
|---|---:|
| Fechas procesadas | 22 |
| Píxeles válidos antes del muestreo | 3,419,056 |
| Filas del dataset de modelado | 220,000 |
| Valores faltantes | 0 |
| Bloques de 1 km | 203 |
| Mejor ROC‑AUC aleatorio | 0.9987 |
| Mejor recall temporal | 0.9384 |

**Gradient Boosting** obtuvo el mejor ROC‑AUC aleatorio. La **regresión logística** conservó el recall más alto ante las fechas más recientes, lo que evidencia que el mejor modelo depende del costo ambiental asignado a los falsos negativos.

## Control de fuga de información

La respuesta se define como `CYA >= 100`, equivalente a 100,000 células/mL. Este valor es un umbral de cribado inspirado en orientación histórica de la OMS para aguas recreativas; no reemplaza conteos microscópicos ni mediciones de toxinas.

La ecuación de CYA utiliza B02, B03 y B04. Para impedir fuga directa o indirecta se excluyen del entrenamiento:

```text
CYA · B02 · B03 · B04 · NDVI · NDWI
```

Los modelos utilizan B05, B06, B07, B08, B8A, B11, B12, coordenadas normalizadas y variables cíclicas del día del año.

## Estructura

```text
├── config/                 # Fechas y geometrías necesarias
├── data/
│   ├── raw/                # GeoTIFF locales; ignorados por Git
│   ├── processed/          # Dataset y predicciones; ignorados por Git
│   └── README.md           # Diccionario y política de datos
├── notebooks/              # Notebook estético y ejecutado
├── outputs/
│   ├── figures/            # Figuras empleadas por notebook e informe
│   └── tables/             # Métricas y resultados reproducibles
├── reports/                # Informe .tex y PDF
├── scripts/                # Descarga, dataset y experimentos
├── src/
│   ├── lab4/               # Configuración y descarga openEO
│   └── lab4_ml/            # Preparación y modelado
└── tests/                  # Pruebas de invariantes críticos
```

## Reproducción

### 1. Instalar el entorno

```powershell
uv sync --extra test
```

También se proporciona `requirements.txt` para flujos tradicionales con `pip`.

### 2. Descargar Sentinel‑2

```powershell
uv run python scripts/download_ml_stack.py --lake all --submit
```

Copernicus solicita autenticación mediante código de dispositivo. El proyecto no recibe ni almacena contraseñas.

### 3. Construir el dataset

```powershell
uv run python scripts/build_ml_dataset.py
```

### 4. Entrenar y evaluar

```powershell
uv run python scripts/run_ml_advance.py
uv run pytest -q
```

### 5. Compilar el informe

```powershell
cd reports
pdflatex informe_avance.tex
pdflatex informe_avance.tex
```

## Archivos locales no versionados

Los 22 GeoTIFF multiespectrales, el dataset comprimido y las predicciones se conservan localmente para terminar SHAP y los mapas. No se suben a GitHub por tamaño y pueden reconstruirse con los scripts anteriores.

## Trabajo pendiente para el 100%

1. Incorporar SHAP global y local, incluida la dirección ambiental de los efectos.
2. Generar mapas de probabilidad y categorías de concentración para ambos lagos.
3. Mapear falsos positivos y falsos negativos y contrastarlos con la Parte 1.
4. Integrar interpretación, limitaciones, necesidades de datos y conclusiones finales.
5. Sustituir el informe y notebook de avance por sus versiones finales y verificar todos los entregables.

## Integrantes

| Nombre | Carné |
|---|---:|
| Jorge Gabriel Palacios Sales | 231385 |
| Pablo Daniel Barillas Moreno | 22193 |
| Roberto Emiliano Otoniel | 23968 |

## Referencias principales

- [OMS — Guidelines on recreational water quality](https://www.who.int/publications/i/item/9789240031302)
- [Sentinel Hub — Se2WaQ](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/se2waq/)
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)

---

<div align="center">
Repositorio académico desarrollado para el Laboratorio 4 de Data Science.
</div>
