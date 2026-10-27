import os
from pathlib import Path
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

CHANNEL_ID = "UCoLrcjPV5PbUrUyXq5mjc_A"  # Canal oficial MLB
LOOKBACK = 14
RAW_PARQUET_PATH = Path("data/raw/mlb_raw_videos.parquet")
PROCESSED_PARQUET_PATH = Path("data/processed/mlb_daily_series.parquet")


def descargar_datos_mlb(
    api_key: str | None = None,
    max_videos: int = 20000,
    forzar_descarga: bool = False,
) -> pd.DataFrame:
    """
    Extrae el historial de videos del canal de la MLB desde la API de YouTube v3
    y lo persiste en formato Parquet.
    """
    if RAW_PARQUET_PATH.exists() and not forzar_descarga:
        print(f"Cargando datos crudos desde {RAW_PARQUET_PATH}...")
        df = pd.read_parquet(RAW_PARQUET_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    api_key = api_key or os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("No se encontró YOUTUBE_API_KEY en el archivo .env")

    print(f"Descargando hasta {max_videos} videos del canal MLB desde YouTube API...")
    youtube = build("youtube", "v3", developerKey=api_key)

    channel_resp = youtube.channels().list(
        part="snippet,contentDetails",
        id=CHANNEL_ID,
    ).execute()

    if not channel_resp.get("items"):
        raise ValueError(f"No se encontró el canal con ID {CHANNEL_ID}")

    uploads_playlist = channel_resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    video_ids = []
    next_page_token = None

    while len(video_ids) < max_videos:
        resp = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist,
            maxResults=50,
            pageToken=next_page_token,
        ).execute()

        for item in resp.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
            if len(video_ids) >= max_videos:
                break

        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break

    rows = []
    total_ids = len(video_ids)
    for i in range(0, total_ids, 50):
        batch = video_ids[i : i + 50]
        resp = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(batch),
        ).execute()

        for video in resp.get("items", []):
            try:
                published_at = pd.to_datetime(video["snippet"]["publishedAt"]).tz_localize(None)
                date = published_at.normalize()
                views = int(video["statistics"].get("viewCount", 0))
                likes = int(video["statistics"].get("likeCount", 0))
                comments = int(video["statistics"].get("commentCount", 0))
                title = str(video["snippet"].get("title", ""))

                rows.append({
                    "video_id": video["id"],
                    "published_at": published_at,
                    "date": date,
                    "views": views,
                    "likes": likes,
                    "comments": comments,
                    "title": title,
                })
            except Exception:
                pass

        if (i + 50) % 2000 == 0 or (i + 50) >= total_ids:
            print(f"Procesados: {min(i + 50, total_ids)}/{total_ids} videos")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    RAW_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PARQUET_PATH, index=False)
    print(f"Archivo Parquet guardado en: {RAW_PARQUET_PATH} ({len(df)} registros)")
    return df


def construir_serie_diaria(df_raw: pd.DataFrame) -> pd.Series:
    """Agrupa las vistas por día calendario, rellena días sin subidas con 0 y guarda en Parquet."""
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    daily = df.groupby("date")["views"].sum().sort_index().asfreq("D").fillna(0.0)

    PROCESSED_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_frame(name="views").to_parquet(PROCESSED_PARQUET_PATH)
    print(f"Serie diaria guardada en: {PROCESSED_PARQUET_PATH} ({len(daily)} días)")
    return daily


def crear_ventanas_tabulares(
    series: pd.Series,
    lookback: int = LOOKBACK,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """
    Convierte la serie diaria en matriz de lags (lag_14 ... lag_1).
    """
    valores = series.values
    fechas = series.index
    X_rows, y_vals, target_dates = [], [], []

    feature_names = [f"lag_{i}" for i in range(lookback, 0, -1)]

    for i in range(len(valores) - lookback):
        X_rows.append(valores[i : i + lookback])
        y_vals.append(valores[i + lookback])
        target_dates.append(fechas[i + lookback])

    X_df = pd.DataFrame(X_rows, columns=feature_names, dtype=np.float64)
    y_ser = pd.Series(y_vals, name="target_views", dtype=np.float64)
    return X_df, y_ser, pd.DatetimeIndex(target_dates)


def particionar_serie(X: pd.DataFrame, y: pd.Series, dates: pd.DatetimeIndex) -> dict:
    """Partición estrictamente temporal: 70% Train, 15% Val (HPO), 15% Test (Evaluación final)."""
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return {
        "X_train": X.iloc[:train_end].copy(),
        "y_train": y.iloc[:train_end].copy(),
        "dates_train": dates[:train_end],
        "X_val": X.iloc[train_end:val_end].copy(),
        "y_val": y.iloc[train_end:val_end].copy(),
        "dates_val": dates[train_end:val_end],
        "X_test": X.iloc[val_end:].copy(),
        "y_test": y.iloc[val_end:].copy(),
        "dates_test": dates[val_end:],
    }