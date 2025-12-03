import pandas as pd
import pyodbc

# 1. Cargar el dataset heart.csv
df = pd.read_csv("heart.csv")

# 2. Conexión a SQL Server
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=localhost\\SQLEXPRESS08;"
    "Database=proyectoCD;"
    "Trusted_Connection=yes;"
)

cursor = conn.cursor()

# 3. Insertar datos fila por fila
for index, row in df.iterrows():
    cursor.execute("""
        INSERT INTO HeartData (
            age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal, target
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    row.age, row.sex, row.cp, row.trestbps, row.chol, row.fbs,
    row.restecg, row.thalach, row.exang, row.oldpeak,
    row.slope, row.ca, row.thal, row.target)

conn.commit()
print("Datos insertados correctamente en SQL Server.")
