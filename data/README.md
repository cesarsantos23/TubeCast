# Datos de TubeCast

Los datos **no se versionan en Git** (`data/raw/` y `data/processed/` están en `.gitignore`). Se regeneran desde la fuente con el código de `src/entrenamiento/datos.py`.

## Fuente

- **YouTube Data API v3**, canal oficial de la MLB (`channelId = UCoLrcjPV5PbUrUyXq5mjc_A`).
- Endpoints: `channels.list` (playlist de subidas), `playlistItems.list` (IDs de video) y `videos.list` (`snippet`, `statistics`).
- Se descargan hasta 20 000 videos (la playlist de subidas devuelve los más recientes primero).

## Cómo obtenerlos

1. Crear una API key en Google Cloud con la YouTube Data API v3 habilitada y guardarla en `.env` como `YOUTUBE_API_KEY` (ver `.env.example`).
2. Ejecutar el flow (`uv run python -m src.entrenamiento.flow`) o, solo para los datos:

```bash
uv run python -c "from src.entrenamiento.datos import descargar_datos_mlb, construir_serie_diaria; construir_serie_diaria(descargar_datos_mlb())"
```

La descarga consume aproximadamente 1 unidad de cuota por cada 50 videos en cada endpoint (~800 unidades para 20 000 videos, de 10 000 diarias).

## Archivos generados

| Archivo | Unidad | Contenido |
|---|---|---|
| `data/raw/mlb_raw_videos.parquet` | un video | `video_id`, `published_at`, `date`, `views`, `likes`, `comments`, `title` |
| `data/processed/mlb_daily_series.parquet` | un día calendario | `views`: suma de las vistas acumuladas de los videos publicados ese día (días sin publicaciones = 0) |

## Diccionario de datos (crudo)

| Columna | Tipo | Descripción |
|---|---|---|
| `video_id` | str | Identificador público del video |
| `published_at` | datetime | Fecha y hora de publicación (UTC, sin zona) |
| `date` | datetime | `published_at` truncado al día |
| `views` | int | `viewCount` al momento de la descarga |
| `likes` | int | `likeCount` al momento de la descarga |
| `comments` | int | `commentCount` al momento de la descarga |
| `title` | str | Título del video |

## Consideraciones

- **Instantánea, no serie histórica:** la API solo entrega el conteo acumulado *actual*. La serie diaria representa “vistas acumuladas de lo publicado cada día”, no el tráfico real recibido ese día; los días recientes tienen menos tiempo para acumular vistas.
- **Privacidad:** solo se usan métricas públicas a nivel de video de un canal institucional; no se descargan datos personales de usuarios.
- **Muestra de comentarios (entrega 1):** `data/request.ipynb` descarga un video de ejemplo y sus comentarios a `data/raw/*_sample.csv`, usados por los notebooks 01 y 02.
