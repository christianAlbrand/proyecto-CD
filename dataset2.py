# -*- coding: utf-8 -*-

# === IMPORTS ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.inspection import permutation_importance

from pymongo import MongoClient


# === CONEXIÓN A MONGODB ===
client = MongoClient("mongodb+srv://christianalbrand:3scOCBf8tXeV7uxj@christian.go2j6da.mongodb.net/?appName=Christian")
db = client["diabetes_db"]
coleccion = db["dataset2"]


# === CARGA DEL DATASET ===
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="progresion")
df = pd.concat([X, y], axis=1)

# Insertar datos iniciales SOLO si la colección está vacía
if coleccion.count_documents({}) == 0:
    coleccion.insert_many(df.to_dict(orient="records"))
    print("Dataset almacenado en MongoDB")
else:
    print("Los datos ya existían en MongoDB, no se insertaron duplicados.")


# === REGRESIÓN LINEAL (SPLIT CORRECTO) ===
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_lr = LinearRegression()
model_lr.fit(X_train_reg, y_train_reg)
y_pred = model_lr.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred)
print("MSE regresión lineal:", mse)


# === CLUSTERING (K-Means) ===
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

df["cluster"] = clusters

# === GUARDAR LOS CLUSTERS EN MONGODB (VERSIÓN CORRECTA) ===
docs = list(coleccion.find())  # traer los docs en el mismo orden que insertaste

for i, doc in enumerate(docs):
    coleccion.update_one(
        {"_id": doc["_id"]},
        {"$set": {"cluster": int(clusters[i])}}
    )

print("Clusters guardados en MongoDB Atlas correctamente.")

print(df["cluster"].value_counts())
print("Silhouette score:", silhouette_score(X, clusters))


# === CLASIFICACIÓN (KNN) — SPLIT SEPARADO ===
y_class = pd.qcut(y, q=3, labels=["baja", "media", "alta"])

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_cls, y_train_cls)

acc = knn.score(X_test_cls, y_test_cls)
print("Accuracy K-NN:", acc)


# === PERMUTATION IMPORTANCE (YA FUNCIONA) ===
result = permutation_importance(
    model_lr, X_test_reg, y_test_reg, n_repeats=10, random_state=42
)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": result.importances_mean
}).sort_values("importance", ascending=False)

print(importance_df)


# === VALIDACIÓN CRUZADA ===
scores_lr = cross_val_score(model_lr, X, y, cv=5)
scores_knn = cross_val_score(knn, X, y_class, cv=5)

print("CV Regresión Lineal:", scores_lr.mean())
print("CV KNN:", scores_knn.mean())


# === GRÁFICAS ===
plt.scatter(y_test_reg, y_pred)
plt.xlabel("Real")
plt.ylabel("Predicción")
plt.title("Regresión lineal – diabetes")
plt.show()

plt.scatter(X["bmi"], X["bp"], c=df["cluster"])
plt.xlabel("BMI")
plt.ylabel("BP")
plt.title("Clusters K-Means – Diabetes")
plt.show()

train_sizes, train_scores, test_scores = learning_curve(knn, X, y_class, cv=5)

plt.plot(train_sizes, train_scores.mean(axis=1), label="train")
plt.plot(train_sizes, test_scores.mean(axis=1), label="test")
plt.legend()
plt.title("Curva de aprendizaje – KNN")
plt.xlabel("Tamaño del entrenamiento")
plt.ylabel("Score")
plt.show()

plt.barh(importance_df["feature"], importance_df["importance"])
plt.title("Importancia de variables – Diabetes")
plt.gca().invert_yaxis()
plt.show()

plt.bar(["Linear Regression", "KNN"], [scores_lr.mean(), scores_knn.mean()])
plt.ylabel("Score promedio CV")
plt.title("Comparación de modelos ML")
plt.show()
