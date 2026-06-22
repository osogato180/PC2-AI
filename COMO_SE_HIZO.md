# Cómo se hizo — Guía para el equipo (PC2)

Esta guía explica, en lenguaje sencillo, todo lo que se construyó para la Práctica
Calificada 2, para que **cualquier integrante pueda entenderlo y defenderlo** en una
sustentación. Léanla completa antes de la entrega.

---

## 1. ¿Qué construimos?
Una herramienta de Machine Learning que **predice el precio de una vivienda** según sus
características (área, habitaciones, baños, antigüedad, garage, piso, ubicación y tipo).
Tiene dos partes:
- Un **motor de ML** (`train.py`) que entrena y compara 5 modelos.
- Un **dashboard interactivo** (`app.py`, hecho en Streamlit) con dos paneles.

La tarea es de **regresión** (predecir un número continuo: el precio).

---

## 2. El flujo, paso a paso (CRISP-DM)
1. **Datos**: usamos el dataset `house_prices.csv` (1200 viviendas) que ya tenían en el Excel.
2. **EDA (exploración)**: calculamos estadísticos, revisamos nulos/duplicados (no hay),
   detectamos outliers (IQR) y vimos correlaciones. Hallazgo clave: **el área es lo que más
   influye en el precio (correlación 0.91)**.
3. **Preprocesamiento**: con un `ColumnTransformer` aplicamos lo MISMO a todos los modelos:
   - One-Hot Encoding a las categóricas (`ubicacion`, `tipo_vivienda`).
   - `StandardScaler` a las numéricas (media 0, varianza 1).
   - División 80/20 (`random_state=42`) + validación cruzada de 5 particiones.
4. **Modelado**: entrenamos 5 modelos (Regresión Lineal, Polinomial, Árbol, Random Forest, Red Neuronal).
5. **Evaluación**: comparamos con R², RMSE y MAE. Ganó la **Regresión Lineal** (R²=0.990).
6. **Despliegue**: el dashboard en Streamlit.

> **Por qué ganó el modelo más simple:** como el precio depende casi linealmente del área,
> la regresión lineal captura la relación sin sobreajustar. Es el *principio de parsimonia*
> (la explicación más simple suele ser la mejor). ¡Esto es un punto fuerte para la defensa!

---

## 3. Cómo ejecutarlo (en cualquier PC)
```bash
pip install -r requirements.txt
python train.py          # entrena y genera figuras + el mejor modelo
streamlit run app.py     # abre el dashboard en el navegador
```

---

## 4. El dashboard (lo que verán)
- **Panel A — Análisis de Datos:** KPIs (registros, precio promedio/mediano, área media),
  filtros por ubicación/tipo/área y 4 gráficos interactivos (distribución, dispersión,
  barras por ubicación y matriz de correlación).
- **Panel B — Predicción:** un formulario donde el usuario ingresa las características y
  obtiene el **precio estimado** + una explicación (cuánto está por encima/debajo de la mediana).

---

## 5. Lo que FALTA hacer antes de entregar (importante)
1. **Subir el proyecto a GitHub** (toda la carpeta `PC2_HousePrice`).
2. **Desplegar en Streamlit Cloud** (https://share.streamlit.io → conectar repo → `app.py`).
3. **Pegar las dos URLs** (dashboard y repositorio) en el informe, sección 3.3, y en la
   sección 2.4 (donde dice `<usuario>`).
4. **(Recomendado) Reemplazar las vistas previas** del informe por capturas reales del
   dashboard ya desplegado, a pantalla completa (mín. 1920×1080). Las que insertamos son
   réplicas fieles con datos reales y sirven, pero la captura del despliegue real suma.
5. Exportar el informe final a **PDF** (ya está hecho: `Informe_PC2_Grupo5.pdf`) y subirlo.

---

## 6. Entregables (checklist de la rúbrica)
- [x] Informe completo respondido → `Informe_PC2_Grupo5.pdf`
- [x] Dataset usado → `data/house_prices.csv` (+ prompt de generación en el informe)
- [x] Código de entrenamiento → `train.py` (+ `app.py`)
- [ ] URL pública del dashboard → **desplegar y pegar**
- [ ] URL del repositorio GitHub → **subir y pegar**
- [ ] (Opcional, bonificación) Video demo ≤ 3 min del dashboard

---

## 7. Preguntas que podrían hacerles (y respuesta corta)
- **¿Por qué la regresión lineal y no la red neuronal?** Porque dio mejor R²/RMSE/MAE; la
  relación es lineal, así que el modelo simple gana y es más interpretable.
- **¿Cómo evitaron la fuga de información?** El escalado se ajusta solo con el train, dentro
  de un Pipeline; el test nunca influye en el preprocesamiento.
- **¿Qué métrica importa más?** En regresión, R² (ajuste) y RMSE/MAE (error en soles).
- **¿Hay sesgos?** Sí: el dataset es sintético y solo cubre distritos de Lima → sesgo
  geográfico; se mitiga ampliando con datos reales y auditando por subgrupo.
