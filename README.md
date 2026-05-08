# URBAN FLOOD STRESS

Métodos limpios y directos para cuatro flujos:

- `scripts/download_street_network.py`: descarga la capa LION y guarda CSV + GeoPackage en `data/spatial/vector/streets/`.
- `scripts/build_directed_graph_metrics.py`: prepara la red y calcula métricas de grafo dirigido.
- `scripts/download_311.py`: descarga 311 en parquet mensual para 2010-2019 y 2020-presente en `data/temporal/311/`.
- `notebooks/street-network.ipynb`: carga el LION local y construye métricas de grafo en Jupyter.
- `notebooks/network-connectivity.ipynb`: interpreta componentes, puentes y articulaciones de la red vial.
- `notebooks/download-311.ipynb`: lee el raw local de 311 y empieza el procesamiento en Jupyter.
- `notebooks/census.ipynb`: descarga ACS/TIGER y construye contexto demográfico y socioeconómico.
- `notebooks/noaa-tides.ipynb`: descarga mareas y Thiessen para estaciones NOAA.
- `notebooks/precipitation.ipynb`: descarga precipitación NOAA CDO y Thiessen.

## Estructura de datos

- `data/spatial/vector`: capas vectoriales y salidas geográficas.
- `data/spatial/raster`: rasters.
- `data/temporal`: series, tablas y descargas por fecha.

## Uso rápido

```bash
export NYC_APP_TOKEN=tu_token
python scripts/download_street_network.py
python scripts/build_directed_graph_metrics.py
python scripts/download_311.py
```

En Jupyter:

- abre `notebooks/street-network.ipynb` para descargar calles y generar métricas.
- abre `notebooks/network-connectivity.ipynb` para resumir la conectividad de la red.
- abre `notebooks/download-311.ipynb` para bajar 311 mes a mes.
- abre `notebooks/census.ipynb` para generar el contexto ACS de NYC.

## Paquete

Las funciones reutilizables están en `src/project_name/`:

- `project_name.requests_311`
- `project_name.street_network`
- `project_name.utils`

La parte de calles requiere `geopandas` y `python-igraph`.
