"""Build the clean manifest: drop corrupt files and extra duplicate copies."""
import pandas as pd

df = pd.read_csv("reports/audit.csv")
clean = df[(~df["corrupt"]) & (~df["duplicate"])]
clean.to_csv("reports/clean_manifest.csv", index=False)
print(len(clean), clean["class"].value_counts().to_dict())