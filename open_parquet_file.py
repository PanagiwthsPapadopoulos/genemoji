import pandas as pd

# Load the parquet file into a DataFrame
df = pd.read_parquet('/Users/panagiwths/Desktop/assignments/genemoji/emoji_features.parquet')

# Preview the data
print(df.head())

import pandas as pd
df = pd.read_parquet("/Users/panagiwths/Desktop/assignments/genemoji/training_data.parquet")
df.to_csv("/Users/panagiwths/Desktop/assignments/genemoji/training_data.csv", index=False)