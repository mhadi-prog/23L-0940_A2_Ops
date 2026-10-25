"""Dataset audit: per-image metadata, corrupt files, exact duplicates."""
import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


rows = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path.name.startswith("."):
        continue
    if path.suffix.lower() not in EXTS:
        continue
    cls = path.parent.name   # class = top-level folder
    row = {"path": str(path), "class": cls,
           "file_size": path.stat().st_size, "sha256": sha256(path),
           "width": None, "height": None, "mode": None,
           "format": None, "corrupt": False, "error": ""}
    try:
        with Image.open(path) as im:
            im.verify()                 # cheap integrity check
        with Image.open(path) as im:    # verify() invalidates the handle
            im.load()                   # full decode catches truncation
            row.update(width=im.width, height=im.height,
                       mode=im.mode, format=im.format)
    except Exception as e:
        row["corrupt"] = True
        row["error"] = str(e)[:100]
    rows.append(row)

df = pd.DataFrame(rows)
df["duplicate"] = df.duplicated("sha256", keep="first")  # extra copies only
df.to_csv(OUT / "audit.csv", index=False)

ok = df[~df["corrupt"]]
print("\n=== Images per class ===")
print(df.groupby("class").size())
print("\n=== Corrupt files ===", int(df["corrupt"].sum()))
print(df[df["corrupt"]][["path", "error"]].to_string(index=False))
print("\n=== Exact duplicates (extra copies) ===", int(df["duplicate"].sum()))
print(df[df.duplicated("sha256", keep=False)]
      .sort_values("sha256")[["path", "class", "sha256"]].to_string(index=False))

dup_groups = df[df.duplicated("sha256", keep=False)].groupby("sha256")["class"].nunique()
print("Duplicate groups spanning both classes (label leakage):",
      int((dup_groups > 1).sum()))

print("\n=== Width / height / file size (min, median, max) per class ===")
stats = ok.groupby("class")[["width", "height", "file_size"]].agg(["min", "median", "max"])
print(stats.to_string())
stats.to_csv(OUT / "audit_stats.csv")

print("\n=== Colour modes ===")
print(ok.groupby(["class", "mode"]).size())
print("\n=== Formats ===")
print(ok.groupby(["class", "format"]).size())