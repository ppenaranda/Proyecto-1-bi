from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from sklearn.metrics import classification_report

# Cargar el modelo
modelo = joblib.load("modelo_pipeline.pkl")

# Crear la aplicación
app = FastAPI(title="Fake News Detector API")

# Definir los modelos de entrada
class TextoEntrada(BaseModel):
    textos: List[str]

class ReentrenamientoEntrada(BaseModel):
    textos: List[str]
    etiquetas: List[int]  # 0: falsa, 1: verdadera

# Endpoint de predicción
@app.post("/predict")
def predecir_texto(data: TextoEntrada):
    predicciones = modelo.predict(data.textos)
    probabilidades = modelo.predict_proba(data.textos).tolist()

    resultados = [
        {
            "texto": texto,
            "prediccion": int(predicciones[i]),
            "probabilidad_fake": probabilidades[i][0],
            "probabilidad_real": probabilidades[i][1],
        }
        for i, texto in enumerate(data.textos)
    ]

    return {"resultados": resultados}

# Endpoint de reentrenamiento
@app.post("/retrain")
def reentrenar_modelo(data: ReentrenamientoEntrada):
    df = pd.DataFrame({
        "texto": data.textos,
        "etiqueta": data.etiquetas
    })

    global modelo
    modelo.fit(df["texto"], df["etiqueta"])
    joblib.dump(modelo, "modelo_pipeline.pkl")

    y_pred = modelo.predict(df["texto"])
    reporte = classification_report(df["etiqueta"], y_pred, output_dict=True)

    return {
        "precision": reporte["weighted avg"]["precision"],
        "recall": reporte["weighted avg"]["recall"],
        "f1_score": reporte["weighted avg"]["f1-score"]
    }
