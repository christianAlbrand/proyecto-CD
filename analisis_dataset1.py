# -*- coding: utf-8 -*-
import pandas as pd

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
