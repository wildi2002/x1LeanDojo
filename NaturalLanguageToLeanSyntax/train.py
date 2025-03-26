import pandas as pd

# Lade das Dataset
df = pd.read_parquet("hf://datasets/internlm/Lean-Workbook/wkbk_1009.parquet")

# Schau dir die ersten paar Zeilen an, um zu verstehen, wie die Daten strukturiert sind
print(df.head())
print(df.columns)