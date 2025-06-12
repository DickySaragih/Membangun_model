# modelling_tuning.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from dagshub import dagshub_logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# === 1. Load Dataset ===
data_path = "namadataset_preprocessing/WineQT_processed.csv"
df = pd.read_csv(data_path)

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 3. Model dan Hyperparameter Tuning ===
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# === 4. Konfigurasi MLflow ke DagsHub ===
mlflow.set_tracking_uri("https://dagshub.com/di803805/WineQT-MLFlow")
mlflow.set_experiment("WineQT_Model_Tuning")

# === 5. Evaluasi Manual dan Logging ke MLflow + DagsHub ===
with mlflow.start_run():
    with dagshub_logger() as logger:
        mlflow.log_params(grid_search.best_params_)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", prec)
        mlflow.log_metric("recall_macro", rec)
        mlflow.log_metric("f1_macro", f1)

        logger.log_metrics({
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1
        })

        # Confusion Matrix
        unique_labels = sorted(np.unique(np.concatenate((y_test, y_pred))))
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Simpan model
        mlflow.sklearn.log_model(best_model, "random_forest_model")

print("Model tuning selesai dan telah dilog ke MLflow dan DagsHub.")
