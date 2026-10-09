"""
API de predicción de diabetes · Proyecto final MLOps (Maestría en Ciencia de Datos, Universidad Ean)

Recibe las 12 variables que usa el modelo @champion (workspace.default.modelo_diabetes),
las reenvía al endpoint de Databricks Model Serving y devuelve la predicción.

Variables de entorno (se inyectan desde los secretos de GitHub → Azure Container Apps):
  DATABRICKS_ENDPOINT_URL   https://<workspace>/serving-endpoints/<nombre>/invocations
  DATABRICKS_TOKEN          token personal de acceso de Databricks
"""

import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

DATABRICKS_ENDPOINT_URL = os.getenv("DATABRICKS_ENDPOINT_URL", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
TIMEOUT_S = float(os.getenv("DATABRICKS_TIMEOUT", "120"))  # margen por si el endpoint está arrancando

app = FastAPI(
    title="API de predicción de diabetes",
    description="Modelo Gradient Boosting entrenado con el dataset Pima Indians Diabetes (Kaggle) "
                "y servido desde Databricks con el alias @champion.",
    version="1.0.0",
)


class Paciente(BaseModel):
    """Exactamente las 12 variables de la firma del modelo registrado en el NB02."""

    model_config = ConfigDict(
        extra="forbid",  # ni más ni menos: cualquier variable adicional se rechaza
        json_schema_extra={
            "example": {
                "embarazos": 6, "glucosa": 148, "presion_arterial": 72, "grosor_piel": 35,
                "insulina": 125, "imc": 33.6, "historial_familiar": 0.627, "edad": 50,
                "glucosa_alta": 1, "obesidad": 1, "glucosa_x_imc": 49.728,
                "ratio_insulina_glucosa": 0.8446,
            }
        },
    )

    embarazos: float = Field(..., ge=0, description="Número de embarazos")
    glucosa: float = Field(..., gt=0, description="Glucosa plasmática a 2 h (mg/dL)")
    presion_arterial: float = Field(..., gt=0, description="Presión arterial diastólica (mm Hg)")
    grosor_piel: float = Field(..., gt=0, description="Pliegue cutáneo del tríceps (mm)")
    insulina: float = Field(..., gt=0, description="Insulina sérica a 2 h (μU/mL)")
    imc: float = Field(..., gt=0, description="Índice de masa corporal (kg/m²)")
    historial_familiar: float = Field(..., ge=0, description="DiabetesPedigreeFunction")
    edad: float = Field(..., ge=0, description="Edad (años)")
    glucosa_alta: float = Field(..., ge=0, le=1, description="1 si glucosa ≥ 140 mg/dL")
    obesidad: float = Field(..., ge=0, le=1, description="1 si IMC ≥ 30")
    glucosa_x_imc: float = Field(..., description="glucosa × IMC / 100")
    ratio_insulina_glucosa: float = Field(..., description="insulina / glucosa")


# Mismo orden de columnas que la firma del modelo
COLUMNAS = list(Paciente.model_fields.keys())


@app.get("/")
def raiz():
    return {
        "proyecto": "MLOps · Predicción de diabetes (Pima Indians Diabetes)",
        "endpoints": {"salud": "GET /health", "prediccion": "POST /predecir", "documentacion": "GET /docs"},
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predecir")
def predecir(paciente: Paciente):
    if not DATABRICKS_ENDPOINT_URL or not DATABRICKS_TOKEN:
        raise HTTPException(status_code=500, detail="Faltan DATABRICKS_ENDPOINT_URL o DATABRICKS_TOKEN")

    registro = {col: float(getattr(paciente, col)) for col in COLUMNAS}
    try:
        respuesta = httpx.post(
            DATABRICKS_ENDPOINT_URL,
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json={"dataframe_records": [registro]},
            timeout=TIMEOUT_S,
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"No se pudo contactar el endpoint de Databricks: {e}")

    if respuesta.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Databricks respondió {respuesta.status_code}: {respuesta.text[:300]}")

    predicciones = respuesta.json().get("predictions", [])
    return {"predictions": [int(round(float(p))) for p in predicciones]}
