# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos desde el CSV local
df = pd.read_csv("heart.csv")

print("Primeras filas del dataset:")
print(df.head(), "\n")

# -------------------------------------------------------
# 1. Medidas de tendencia central
# -------------------------------------------------------
cols = ['age', 'chol', 'trestbps']

central = pd.DataFrame({
    'mean': df[cols].mean(),
    'median': df[cols].median(),
    'mode': df[cols].mode().iloc[0]
})

print("Medidas de tendencia central:")
print(central, "\n")

# -------------------------------------------------------
# 2. Medidas de dispersión
# -------------------------------------------------------
dispersion = df.describe().loc[['std', 'min', 'max']]
print("Medidas de dispersión:")
print(dispersion, "\n")

# -------------------------------------------------------
# 3. Distribución por género y presencia de enfermedad
# -------------------------------------------------------
print("Distribución por género y target:")
print(df.groupby(['sex', 'target']).size(), "\n")

tabla_porcentaje = pd.crosstab(df['sex'], df['target'], normalize='index') * 100
print("Tabla de porcentaje por género y enfermedad:")
print(tabla_porcentaje, "\n")

# -------------------------------------------------------
# 4. Correlación entre variables
# -------------------------------------------------------
corr = df.corr(numeric_only=True)
print("Correlaciones:")
print(corr, "\n")

print("Correlaciones mayores a 0.5:")
print(corr[corr.abs() > 0.5], "\n")

# -------------------------------------------------------
# 5. Outliers en presión arterial y colesterol
# -------------------------------------------------------
def outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series[(series < lower) | (series > upper)]

out_bp = outliers_iqr(df['trestbps'])
out_chol = outliers_iqr(df['chol'])

print(f"Outliers en presión arterial: {len(out_bp)}")
print(f"Outliers en colesterol: {len(out_chol)}")

# =======================================================
#                G  R  Á  F  I  C  A  S
# =======================================================

# 1. Histograma de edades
plt.figure()
df['age'].hist()
plt.title("Histograma de edades")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

# 2. Boxplot de presión arterial
plt.figure()
plt.boxplot(df['trestbps'])
plt.title("Boxplot de presión arterial")
plt.ylabel("Presión arterial")
plt.show()

# 3. Boxplot de colesterol
plt.figure()
plt.boxplot(df['chol'])
plt.title("Boxplot de colesterol")
plt.ylabel("Colesterol")
plt.show()

# 4. Mapa de calor de correlaciones
plt.figure()
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.title("Mapa de calor de correlaciones")
plt.show()
