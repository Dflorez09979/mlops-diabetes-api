# API de predicción de diabetes · MLOps en producción

Proyecto final del curso de MLOps (Maestría en Ciencia de Datos, Universidad Ean).

Un modelo **Gradient Boosting** (AUC 0,824) entrenado con el dataset
[Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) de Kaggle,
registrado en Unity Catalog como `workspace.default.modelo_diabetes@champion` y servido desde
Databricks Model Serving. Esta API (FastAPI) lo expone públicamente en **Azure Container Apps**, con
despliegue continuo mediante **GitHub Actions**.

```
Cliente ──► Azure Container Apps (FastAPI) ──► Databricks Model Serving (@champion) ──► predicción
                     ▲
GitHub (push a main) ─┴─ GitHub Actions: build Docker → Azure Container Registry → deploy
```

## Endpoints

| Método | Ruta | Respuesta |
|---|---|---|
| GET | `/health` | `{"status": "ok"}` |
| POST | `/predecir` | `{"predictions": [1]}` (diabetes) o `{"predictions": [0]}` (no diabetes) |
| GET | `/docs` | Documentación interactiva (Swagger) |

## Ejemplo de uso

Paciente real del dataset (registro 259): 11 embarazos, glucosa 155 mg/dL, IMC 33,3, 51 años.

```bash
curl -X POST https://TU_URL/predecir \
  -H "Content-Type: application/json" \
  -d '{"embarazos": 11, "glucosa": 155, "presion_arterial": 76, "grosor_piel": 28, "insulina": 150, "imc": 33.3, "historial_familiar": 1.353, "edad": 51, "glucosa_alta": 1, "obesidad": 1, "glucosa_x_imc": 51.615, "ratio_insulina_glucosa": 0.9677}'
```

Respuesta esperada: `{"predictions":[1]}`

## Variables de entrada

`/predecir` acepta exactamente las 12 variables de la firma del modelo; cualquier variable de más o de
menos se rechaza con error 422.

| Variable | Descripción |
|---|---|
| embarazos | Número de embarazos |
| glucosa | Glucosa plasmática a 2 h (mg/dL) |
| presion_arterial | Presión arterial diastólica (mm Hg) |
| grosor_piel | Pliegue cutáneo del tríceps (mm) |
| insulina | Insulina sérica a 2 h (μU/mL) |
| imc | Índice de masa corporal (kg/m²) |
| historial_familiar | Índice de antecedentes familiares (DiabetesPedigreeFunction) |
| edad | Edad (años) |
| glucosa_alta | 1 si glucosa ≥ 140 mg/dL |
| obesidad | 1 si IMC ≥ 30 |
| glucosa_x_imc | glucosa × IMC / 100 |
| ratio_insulina_glucosa | insulina / glucosa |

## Secretos de GitHub Actions

| Secreto | Contenido |
|---|---|
| `AZURE_CREDENTIALS` | JSON del Service Principal de Azure |
| `ACR_USERNAME` | Usuario del Azure Container Registry |
| `ACR_PASSWORD` | Contraseña del Azure Container Registry |
| `DATABRICKS_ENDPOINT_URL` | URL de invocación del endpoint de Databricks Serving |
| `DATABRICKS_TOKEN` | Token personal de acceso de Databricks |
