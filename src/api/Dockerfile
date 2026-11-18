# Imagen de la API de TubeCast (FastAPI + modelo registrado en MLflow).
# Contexto de construcción: la raíz del repositorio (ver docker-compose.yaml),
# porque la API es el paquete src.api y requirements.txt vive en la raíz.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Dependencias generadas desde uv.lock (uv export); se instalan antes de copiar
# el código para reutilizar esta capa cuando sólo cambia la aplicación.
COPY requirements.txt /code/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Sólo el código que la API necesita. Los secretos (.env) y el modelo no se copian:
# llegan al ejecutar (env_file y, si se usa el respaldo local, el volumen artifacts/).
COPY src/__init__.py /code/src/__init__.py
COPY src/api /code/src/api

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"]
