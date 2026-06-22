"""
app.py — Dashboard interactivo (PC2)
Herramienta de ML para estimación de precios de vivienda.
Equipo «Monos» (Grupo 5) — USIL · Agentes Inteligentes / Análisis de Datos.

Ejecutar:
    streamlit run app.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ===================== Configuración y estilo (paleta USIL) =====================
AZUL = "#002A5C"      # azul institucional oscuro
AZUL2 = "#0061B0"     # azul medio
AZUL3 = "#4F9BD9"     # azul claro
GRIS = "#6E7B8B"
CLARO = "#F5F7FA"

st.set_page_config(page_title="Predicción de Precios de Vivienda — USIL",
                   page_icon="🏠", layout="wide")

st.markdown(f"""
<style>
    .stApp {{ background-color: {CLARO}; }}
    .titulo-usil {{ color:{AZUL}; font-weight:800; font-size:30px; margin-bottom:0; }}
    .subtitulo {{ color:{GRIS}; font-size:15px; margin-top:0; }}
    .kpi-card {{ background:white; border-radius:14px; padding:18px 20px;
        box-shadow:0 2px 10px rgba(0,0,0,0.06); border-left:6px solid {AZUL2}; }}
    .kpi-valor {{ color:{AZUL}; font-size:30px; font-weight:800; margin:0; }}
    .kpi-label {{ color:{GRIS}; font-size:13px; margin:0; }}
    .pred-box {{ background:linear-gradient(135deg,{AZUL},{AZUL2}); color:white;
        border-radius:16px; padding:28px; text-align:center; }}
    .pred-valor {{ font-size:44px; font-weight:800; margin:6px 0; }}
    section[data-testid="stSidebar"] {{ background-color:{AZUL}; }}
    section[data-testid="stSidebar"] * {{ color:#E8EEF6; }}
    h2, h3 {{ color:{AZUL}; }}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def cargar_datos():
    return pd.read_csv(os.path.join(BASE, "data", "house_prices.csv"))


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(os.path.join(BASE, "modelo", "best_model.pkl"))
    meta = json.load(open(os.path.join(BASE, "modelo", "meta.json"), encoding="utf-8"))
    return modelo, meta


df = cargar_datos()
modelo, meta = cargar_modelo()
NUM = ["area", "habitaciones", "banos", "antiguedad", "garage", "piso"]

# ===================== Barra lateral =====================
with st.sidebar:
    st.markdown("## 🏠 PrediCasa USIL")
    st.markdown("Herramienta de Machine Learning para estimar el precio de viviendas.")
    panel = st.radio("Navegación", ["📊 Panel A — Análisis de Datos",
                                     "🤖 Panel B — Predicción"])
    st.markdown("---")
    st.markdown(f"**Mejor modelo:** {meta['mejor_modelo']}")
    r2 = meta["metricas"][meta["mejor_modelo"]]["R2"]
    st.markdown(f"**R² (prueba):** {r2:.3f}")
    st.markdown("---")
    st.caption("Equipo «Monos» (Grupo 5)\nMedina · Roncal · Sánchez · Villegas")

st.markdown('<p class="titulo-usil">Estimación de Precios de Vivienda</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Universidad San Ignacio de Loyola — Práctica Calificada 2</p>',
            unsafe_allow_html=True)
st.write("")


def kpi(col, valor, label):
    col.markdown(f'<div class="kpi-card"><p class="kpi-valor">{valor}</p>'
                 f'<p class="kpi-label">{label}</p></div>', unsafe_allow_html=True)


# ===================== PANEL A — ANÁLISIS DE DATOS =====================
if panel.startswith("📊"):
    st.subheader("Panel A — Análisis exploratorio interactivo")

    # ---- Filtros ----
    c1, c2, c3 = st.columns(3)
    ubic = c1.multiselect("Ubicación", sorted(df.ubicacion.unique()),
                          default=sorted(df.ubicacion.unique()))
    tipos = c2.multiselect("Tipo de vivienda", sorted(df.tipo_vivienda.unique()),
                           default=sorted(df.tipo_vivienda.unique()))
    rango_area = c3.slider("Rango de área (m²)", int(df.area.min()), int(df.area.max()),
                           (int(df.area.min()), int(df.area.max())))

    dff = df[df.ubicacion.isin(ubic) & df.tipo_vivienda.isin(tipos) &
             df.area.between(*rango_area)]
    if dff.empty:
        st.warning("No hay registros con los filtros seleccionados.")
        st.stop()

    # ---- KPIs ----
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, f"{len(dff):,}", "Registros filtrados")
    kpi(k2, f"S/ {dff.precio.mean():,.0f}", "Precio promedio")
    kpi(k3, f"S/ {dff.precio.median():,.0f}", "Precio mediano")
    kpi(k4, f"{dff.area.mean():.0f} m²", "Área promedio")
    st.write("")

    # ---- Gráficos interactivos ----
    g1, g2 = st.columns(2)
    fig1 = px.histogram(dff, x="precio", nbins=30, title="Distribución del precio",
                        color_discrete_sequence=[AZUL2])
    fig1.update_layout(plot_bgcolor="white", bargap=0.05)
    g1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(dff, x="area", y="precio", color="tipo_vivienda",
                      title="Precio vs. área", opacity=0.6,
                      color_discrete_sequence=[AZUL, AZUL2, AZUL3, GRIS])
    fig2.update_layout(plot_bgcolor="white")
    g2.plotly_chart(fig2, use_container_width=True)

    g3, g4 = st.columns(2)
    prom = dff.groupby("ubicacion", as_index=False).precio.mean().sort_values("precio")
    fig3 = px.bar(prom, x="precio", y="ubicacion", orientation="h",
                  title="Precio promedio por ubicación",
                  color_discrete_sequence=[AZUL2])
    fig3.update_layout(plot_bgcolor="white")
    g3.plotly_chart(fig3, use_container_width=True)

    corr = dff[NUM + ["precio"]].corr()
    fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues",
                     title="Matriz de correlación", aspect="auto")
    g4.plotly_chart(fig4, use_container_width=True)

# ===================== PANEL B — PREDICCIÓN =====================
else:
    st.subheader("Panel B — Predicción del precio")
    st.markdown("Ingresa las características de la vivienda y obtén el precio estimado "
                "por el mejor modelo entrenado.")

    col_in, col_out = st.columns([1.2, 1])
    with col_in:
        c1, c2 = st.columns(2)
        area = c1.slider("Área (m²)", *meta["rangos"]["area"], 150)
        antiguedad = c2.slider("Antigüedad (años)", *meta["rangos"]["antiguedad"], 10)
        c3, c4 = st.columns(2)
        habitaciones = c3.slider("Habitaciones", *meta["rangos"]["habitaciones"], 3)
        banos = c4.slider("Baños", *meta["rangos"]["banos"], 2)
        c5, c6 = st.columns(2)
        garage = c5.slider("Estacionamientos", *meta["rangos"]["garage"], 1)
        piso = c6.slider("Piso", *meta["rangos"]["piso"], 5)
        c7, c8 = st.columns(2)
        ubicacion = c7.selectbox("Ubicación", meta["ubicaciones"])
        tipo = c8.selectbox("Tipo de vivienda", meta["tipos_vivienda"])

    entrada = pd.DataFrame([{
        "area": area, "habitaciones": habitaciones, "banos": banos,
        "antiguedad": antiguedad, "garage": garage, "piso": piso,
        "ubicacion": ubicacion, "tipo_vivienda": tipo,
    }])
    pred = float(modelo.predict(entrada)[0])
    mediana = df.precio.median()
    dif = (pred - mediana) / mediana * 100

    with col_out:
        st.markdown(f'<div class="pred-box"><p style="margin:0;font-size:15px">'
                    f'Precio estimado</p><p class="pred-valor">S/ {pred:,.0f}</p>'
                    f'<p style="margin:0;font-size:13px">Modelo: {meta["mejor_modelo"]}</p></div>',
                    unsafe_allow_html=True)
        st.write("")
        comparacion = "por encima" if dif >= 0 else "por debajo"
        st.info(f"Esta vivienda está un **{abs(dif):.0f}% {comparacion}** del precio "
                f"mediano del mercado (S/ {mediana:,.0f}). "
                f"El precio se explica principalmente por el área ({area} m²) y la "
                f"ubicación ({ubicacion}).")

    st.caption("Nota: el modelo es una herramienta de apoyo para la valuación; "
               "no reemplaza una tasación profesional.")
