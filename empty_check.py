import pandas as pd
import constants

df = pd.read_csv(constants.FULL_DATASET)
empties = []
for i in range(len(df)):
    row = df.iloc[i]
    x = row.isna().sum()
    if x > 0:
        print(f"Found empty {row['id']}")


