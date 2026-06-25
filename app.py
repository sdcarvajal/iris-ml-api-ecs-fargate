from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(
    title="Iris ML API",
    description="API de inferencia para un modelo de clasificacion Iris desplegado en ECS Fargate",
    version="1.0.0",
)

artifact = joblib.load("iris_model.joblib")
model = artifact["model"]
target_names = artifact["target_names"]
model_accuracy = artifact["accuracy"]

class IrisInput(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)

@app.get("/")
def root():
    return {
        "message": "Iris ML API desplegada en AWS ECS Fargate",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: IrisInput):
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]])
    predicted_class = int(model.predict(features)[0])
    predicted_label = target_names[predicted_class]
    probabilities = model.predict_proba(features)[0].tolist()

    return {
        "prediction_class": predicted_class,
        "prediction_label": predicted_label,
        "probabilities": probabilities,
        "model_accuracy": round(float(model_accuracy), 4),
    }
