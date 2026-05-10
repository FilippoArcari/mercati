import pandas as pd
df = pd.read_parquet("cache/data_1d.parquet")
print("Original unscaled data from Cache:")
print(df.iloc[:5, :3])
