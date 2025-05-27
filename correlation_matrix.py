import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("12136_pairs_without_dropping_nan_descr.csv")

# Drop irrelevant columns
df = df.drop(columns=["url_key", "thumb_url", "small_image_url", "description", 'Unnamed: 0'])

# Encode categorical features
cat_cols = ["name", "title", "brand", "model"]
for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

# Convert release_date to ordinal
df["release_date"] = pd.to_datetime(df["release_date"])
df["release_date"] = (df["release_date"] - pd.Timestamp("2020-01-01")).dt.days

# Compute Spearman correlation
corr = df.corr(method="spearman")

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Spearman Correlation Matrix")
plt.show()
