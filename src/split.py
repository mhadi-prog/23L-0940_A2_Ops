"""Stratified train/val/test split from the clean manifest."""
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
df = pd.read_csv("reports/clean_manifest.csv")
df["label"] = (df["class"] == "AiArtData").astype(int)   # 1 = AI-generated, 0 = real

train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

for name, part in [("train", train), ("val", val), ("test", test)]:
    part[["path", "class", "label"]].to_csv(f"reports/{name}.csv", index=False)
    print(name, len(part), part["label"].value_counts().to_dict())