import pandas as pd
import json
import numpy as np

# Load the parquet file into a DataFrame
df = pd.read_parquet('/Users/panagiwths/Desktop/assignments/genemoji/train-00000-of-00001.parquet')

# Preview the data
print(df.head())

# Step 1: Extract all unique tags into a set
all_tags = set(tag for tags in df['tags'] for tag in tags)

# Sort the vocabulary alphabetically
vocab = sorted(all_tags)

# Save vocabulary to vocab.txt
with open("/Users/panagiwths/Desktop/assignments/genemoji/vocabulary.txt", "w", encoding="utf-8") as f:
    for word in vocab:
        f.write(word + "\n")
print("Vocabulary saved to vocabulary.txt.")

# Create multi-hot encoded vectors and prepare JSON entries
emoji_feature_list = []



for _, row in df.iterrows():
    binary_vector = np.array([1 if tag in row["tags"] else 0 for tag in vocab], dtype=int).tolist()
    emoji_entry = {
        "emoji": row["character"],
        "unicode": row["unicode"],
        "feature_vector": binary_vector
    }
    emoji_feature_list.append(emoji_entry)

print("Done creating emoji feature list.")
    
# Convert emoji_feature_list to DataFrame
emoji_df = pd.DataFrame(emoji_feature_list)

# Write to Parquet file
emoji_df.to_parquet("/Users/panagiwths/Desktop/assignments/genemoji/emoji_features.parquet", engine='pyarrow', index=False)

print("✅ Parquet file created successfully.")