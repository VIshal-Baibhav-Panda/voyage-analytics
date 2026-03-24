import pandas as pd

df = pd.read_csv("data/raw/flights.csv")
print(df.iloc[0])