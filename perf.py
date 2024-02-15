import pandas as pd

#df = pd.read_csv("archive/4/results.csv")
df = pd.read_csv("archive/3/results.csv")
result = df.groupby('sis')['r2_test'].mean()
print(result)

df = df[df["r2_test"] > 0.3]
result = df.groupby('sis')['r2_test'].mean()
print(result)