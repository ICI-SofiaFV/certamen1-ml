# 🚗 Certamen 1 · Machine Learning · Grupo 15
**Ensambles, costo de clasificación y umbral óptimo**

🔗 **Dashboard en vivo:** https://certamen1-grupo15.streamlit.app

## ¿Qué hace?
Una aseguradora decide a qué clientes llamar para vender un seguro de vehículo. El dashboard muestra cuánto dinero cuesta cada error del modelo (Random Forest) y cuál es el umbral de decisión que minimiza el costo total.

- **Base:** Health Insurance Cross Sell Prediction (Kaggle) · 380.840 clientes · 12,3% interesados
- **Costos base:** llamada inútil (FP) = US$ 5 · venta perdida (FN) = US$ 50
- **Umbral óptimo:** p* = C_FP / (C_FP + C_FN) = 0,091
- **Ahorro en test vs umbral 0,5:** US$ 320.215 (69,3%)

## Archivos
| Archivo | Descripción |
|---|---|
| `app.py` | Dashboard Streamlit |
| `predicciones_test.csv` | Respuesta real y probabilidades del test (escala real) |
| `requirements.txt` | Librerías necesarias |
| `.streamlit/config.toml` | Colores del dashboard |
| `Certamen1_Grupo_15.ipynb` | Notebook completo (Colab): pipeline, 10 modelos, costos y umbral |
