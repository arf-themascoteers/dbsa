import pandas as pd

df = pd.read_csv("results/results.csv")
result = df.groupby('')['Value'].sum()

print(result)