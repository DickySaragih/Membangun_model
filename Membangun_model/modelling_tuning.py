import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === 1. Load Data ===
data = pd.read_csv("Membangun_model/WineQT_processed.csv")
X = data.drop("quality", axis=1)
y = data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Hyperparameter tuning ===
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [5, 10, None]
}
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# === 3. Konfigurasi MLflow ke DagsHub ===
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("https://dagshub.com/di803805/WineQT-MLFlow")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("1d89765b79436c624cdf3044b2b559daa1da1768")
mlflow.set_tracking_uri("https://dagshub.com/di803805/WineQT-MLFlow.mlflow")

experiment_name = "WineQT_Model_Tuning"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_name)

# === 4. Logging Manual ke MLflow ===
with mlflow.start_run():
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_param("n_estimators", best_model.n_estimators)
    mlflow.log_param("max_depth", best_model.max_depth)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)
    mlflow.sklearn.log_model(best_model, "model")
