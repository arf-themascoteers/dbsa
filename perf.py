import pandas as pd

df = pd.read_csv("results/results.csv")
result = df.groupby('sis')['r2_test'].mean()

print(result)