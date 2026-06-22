"""
train.py — Pipeline de entrenamiento comparativo (PC2)
Tarea: Regresión — predicción del precio de viviendas.

Entrena y compara 5 modelos con el MISMO preprocesamiento, calcula métricas
(R2, RMSE, MAE), genera las figuras del EDA y guarda el mejor modelo.

Uso:
    python train.py
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ----- Paleta USIL -----
AZUL = "#002A5C"; AZUL2 = "#0061B0"; GRIS = "#6E7B8B"; CLARO = "#F5F7FA"
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.edgecolor": GRIS})

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figuras"); os.makedirs(FIG, exist_ok=True)
MOD = os.path.join(BASE, "modelo"); os.makedirs(MOD, exist_ok=True)
DATA = os.path.join(BASE, "data", "house_prices.csv")

NUM = ["area", "habitaciones", "banos", "antiguedad", "garage", "piso"]
CAT = ["ubicacion", "tipo_vivienda"]
TARGET = "precio"


def cargar_datos():
    df = pd.read_csv(DATA)
    return df


def construir_preprocesador():
    """Mismo preprocesamiento para todos los modelos (comparación justa)."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])


def modelos():
    """Diccionario nombre -> estimador base (sin preprocesamiento)."""
    return {
        "Regresión Lineal Múltiple": LinearRegression(),
        "Regresión Polinomial (grado 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lin", LinearRegression()),
        ]),
        "Árbol de Decisión": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
        "Red Neuronal (MLP)": MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=1e-3, max_iter=800, random_state=42),
    }


def evaluar(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


# =================== EDA ===================
def generar_eda(df):
    # 1. Distribución de precios
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(df[TARGET] / 1000, bins=30, color=AZUL2, edgecolor="white")
    ax.set_title("Distribución del precio de viviendas")
    ax.set_xlabel("Precio (miles de soles)"); ax.set_ylabel("Frecuencia")
    fig.tight_layout(); fig.savefig(f"{FIG}/eda1_distribucion_precio.png"); plt.close(fig)

    # 2. Precio vs Área
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.scatter(df["area"], df[TARGET] / 1000, s=12, alpha=0.5, color=AZUL)
    ax.set_title("Relación entre área y precio")
    ax.set_xlabel("Área (m²)"); ax.set_ylabel("Precio (miles de soles)")
    fig.tight_layout(); fig.savefig(f"{FIG}/eda2_precio_area.png"); plt.close(fig)

    # 3. Matriz de correlación
    corr = df[NUM + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="Blues", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    ax.set_title("Matriz de correlación")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{FIG}/eda3_correlacion.png"); plt.close(fig)

    # 4. Precio por ubicación (boxplot)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ubic = df.groupby("ubicacion")[TARGET].median().sort_values().index
    data = [df[df.ubicacion == u][TARGET] / 1000 for u in ubic]
    bp = ax.boxplot(data, tick_labels=ubic, patch_artist=True, vert=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(AZUL2); patch.set_alpha(0.6)
    ax.set_title("Precio según ubicación")
    ax.set_ylabel("Precio (miles de soles)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/eda4_precio_ubicacion.png"); plt.close(fig)
    print("Figuras EDA generadas.")


# =================== MAIN ===================
def main():
    df = cargar_datos()
    X = df[NUM + CAT]
    y = df[TARGET]

    generar_eda(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    pre = construir_preprocesador()
    resultados = {}
    mejores_pred = {}
    mejor = (None, -np.inf, None)

    for nombre, est in modelos().items():
        pipe = Pipeline([("pre", construir_preprocesador()), ("model", est)])
        t0 = time.time()
        pipe.fit(X_train, y_train)
        dt = time.time() - t0
        pred = pipe.predict(X_test)
        met = evaluar(y_test, pred)
        # validación cruzada (R2) sobre todo el conjunto
        cv = cross_val_score(pipe, X, y, cv=KFold(5, shuffle=True, random_state=42),
                             scoring="r2")
        met["CV_R2_media"] = float(cv.mean())
        met["tiempo_s"] = round(dt, 3)
        resultados[nombre] = met
        mejores_pred[nombre] = pred
        print(f"{nombre:32s}  R2={met['R2']:.4f}  RMSE={met['RMSE']:.0f}  MAE={met['MAE']:.0f}  t={dt:.2f}s")
        if met["R2"] > mejor[1]:
            mejor = (nombre, met["R2"], pipe)

    # Guardar el mejor modelo (reentrenado con TODOS los datos)
    nombre_mejor, _, _ = mejor
    pipe_final = Pipeline([("pre", construir_preprocesador()), ("model", modelos()[nombre_mejor])])
    pipe_final.fit(X, y)
    joblib.dump(pipe_final, f"{MOD}/best_model.pkl")

    # Figura: predicho vs real (mejor modelo)
    pred_best = mejores_pred[nombre_mejor]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(y_test / 1000, pred_best / 1000, s=12, alpha=0.5, color=AZUL2)
    lims = [min(y_test.min(), pred_best.min()) / 1000, max(y_test.max(), pred_best.max()) / 1000]
    ax.plot(lims, lims, "--", color=GRIS)
    ax.set_xlabel("Precio real (miles S/)"); ax.set_ylabel("Precio predicho (miles S/)")
    ax.set_title(f"Predicho vs. real — {nombre_mejor}")
    fig.tight_layout(); fig.savefig(f"{FIG}/eda5_pred_vs_real.png"); plt.close(fig)

    # Metadatos para el dashboard (rangos y categorías)
    meta = {
        "mejor_modelo": nombre_mejor,
        "metricas": resultados,
        "ubicaciones": sorted(df.ubicacion.unique().tolist()),
        "tipos_vivienda": sorted(df.tipo_vivienda.unique().tolist()),
        "rangos": {c: [int(df[c].min()), int(df[c].max())] for c in NUM},
        "precio_rango": [int(df.precio.min()), int(df.precio.max())],
    }
    json.dump(meta, open(f"{MOD}/meta.json", "w"), indent=2, ensure_ascii=False)
    json.dump(resultados, open(f"{BASE}/metricas.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nMejor modelo: {nombre_mejor}")
    print("Artefactos guardados en modelo/.")


if __name__ == "__main__":
    main()
