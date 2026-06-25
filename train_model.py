from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Cargar dataset Iris
iris = load_iris()
X = iris.data
Y = iris.target

# 2. Separar datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# 3. Entrenar un modelo sencillo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluar el modelo
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy del modelo: {accuracy:.4f}")

# 5. Guardar modelo y nombres de clases
artifact = {
    "model": model,
    "target_names": iris.target_names.tolist(),
    "feature_names": iris.feature_names,
    "accuracy": accuracy,
}

joblib.dump(artifact, "iris_model.joblib")
print("Modelo guardado correctamente en iris_model.joblib")
