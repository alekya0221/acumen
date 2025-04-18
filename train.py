import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
import pandas as pd

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Acumen_Model_Training")

# Load sample dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Prepare metadata
input_example = pd.DataFrame([X_test[0]], columns=[f"feature_{i}" for i in range(X.shape[1])])
signature = infer_signature(X_test, y_pred)

# Log to MLflow
with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="AcumenModel",
        input_example=input_example,
        signature=signature
    )

    print(f"✅ Model registered in run {run.info.run_id} with accuracy: {acc:.2f}")
