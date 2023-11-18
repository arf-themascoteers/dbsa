import pandas as pd

df = pd.read_csv("data/dataset_r.csv")
df = df.sample(frac=0.1)
df.to_csv("data/dataset_min.csv", index=False)

df = pd.read_csv("data/dataset_66.csv")
df = df.sample(frac=0.1)
df.to_csv("data/dataset_66_min.csv", index=False)