# Imagen del frontend de TubeCast (Streamlit). No necesita credenciales del modelo:
# sólo la URL de la API, que recibe al ejecutar.
# Contexto de construcción: la raíz del repositorio (ver docker-compose.yaml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /code

# Mismas dependencias bloqueadas que el resto del proyecto (uv export desde uv.lock).
COPY requirements.txt /code/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/frontend /code/src/frontend

EXPOSE 8501
CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
