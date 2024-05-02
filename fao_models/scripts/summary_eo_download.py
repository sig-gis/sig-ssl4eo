import pandas as pd

# 2    93105
# 0     1384
checked_csv = "/Volumes/External/pc530/training/checked_locations.csv"
df = pd.read_csv(checked_csv, header=None)
print(df)
a = df[3].value_counts()
print(a)
