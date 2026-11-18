import os
import requests
import streamlit as st
from dotenv import load_dotenv

# override=False: respeta API_URL si Docker ya la entregó con env_file.
load_dotenv(override=False)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="TubeCast - MLB Forecast", layout="wide")
st.title("TubeCast: vistas acumuladas por cohorte diaria (canal MLB)")
st.markdown(
    "Cada día es una **cohorte**: los videos publicados ese día. Ingresa las vistas acumuladas "
    "de las **14 cohortes previas** (`t-14` a `t-1`) y el modelo registrado (`@champion`) estima "
    "cuántas vistas acumula la cohorte del día `t` en la misma instantánea. "
    "Es un análisis retrospectivo: no son las vistas que el canal recibe en un día."
)

valores_default = [
    3529320.0, 3816846.0, 1660823.0, 3210395.0, 3791115.0,
    2218409.0, 1034500.0, 3077739.0, 3434652.0, 838785.0,
    1615544.0, 1548594.0, 2100450.0, 1950300.0,
]

with st.sidebar:
    st.header("Configuración de Conexión")
    api_endpoint = st.text_input("URL de la API", value=API_URL)
    if st.button("Verificar estado de la API"):
        try:
            r = requests.get(f"{api_endpoint}/health", timeout=5)
            st.json(r.json())
        except ValueError:
            st.error(f"La API respondió HTTP {r.status_code} sin JSON.")
        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo conectar con la API: {e}")

with st.form("prediction_form"):
    st.subheader("Vistas acumuladas de las 14 cohortes previas")
    cols = st.columns(7)
    lags = []
    for i in range(14):
        col = cols[i % 7]
        etiqueta = f"Cohorte t-{14 - i}"
        val = col.number_input(
            etiqueta,
            value=float(valores_default[i]),
            step=50000.0,
            format="%.0f",
        )
        lags.append(val)

    submitted = st.form_submit_button("Obtener Predicción")

if submitted:
    payload = {"lags_14_dias": lags}
    try:
        response = requests.post(f"{api_endpoint}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            st.success("Predicción calculada correctamente")
            c1, c2, c3 = st.columns(3)
            c1.metric("Vistas estimadas (cohorte t)", data["predicted_views_formatted"])
            c2.metric("Versión del Modelo", f"v{data['model_version']} ({data['model_family']})")
            c3.metric("Modelo Registrado", data["model_name"])

            st.info(data["interpretation"])
            with st.expander("Ver detalle JSON de solicitud y respuesta"):
                st.write("**Solicitud enviada:**", payload)
                st.write("**Respuesta recibida:**", data)
        else:
            st.error(f"La API rechazó la solicitud (HTTP {response.status_code})")
            try:
                st.json(response.json())
            except ValueError:
                st.write(response.text)
    except requests.exceptions.ConnectionError:
        st.error(
            f"Error de conexión: No fue posible comunicarse con la API en {api_endpoint}. "
            "Verifica que el contenedor de la API esté en ejecución y que ya cargó el modelo."
        )
    except requests.exceptions.RequestException as e:
        st.error(f"Error inesperado al consultar la API: {e}")