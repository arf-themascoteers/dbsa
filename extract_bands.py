import pandas as pd

df = pd.read_csv("archive/True_sigmoid_False_False_False_0.0001_False.csv")
first_row_band_columns = df.loc[0, df.columns[df.columns.str.startswith('band')]]
ar = [f"{a}" for a in first_row_band_columns]
print(ar)

first_row_band_columns = df.loc[len(df)-1, df.columns[df.columns.str.startswith('band')]]
ar = [f"{a}" for a in first_row_band_columns]
print(ar)