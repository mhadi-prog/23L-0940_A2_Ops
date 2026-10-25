"""Visual EDA: sample grids per class + histograms of width, height, file size."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # save to files, no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
GRID = 4                                   # 4 x 4 = 16 samples per class
FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("reports/clean_manifest.csv")
classes = sorted(df["class"].unique())
colors = {"AiArtData": "tab:red", "RealArt": "tab:blue"}


def load_thumb(path, size=256):
    with Image.open(path) as im:
        im = im.convert("RGB")             # handles L, P, RGBA
        im.thumbnail((size, size))
        return np.asarray(im)


# ---- Figure 1: sample grids (one figure per class) ----
for cls in classes:
    sample = df[df["class"] == cls].sample(GRID * GRID, random_state=SEED)
    fig, axes = plt.subplots(GRID, GRID, figsize=(10, 10))
    for ax, (_, r) in zip(axes.ravel(), sample.iterrows()):
        ax.imshow(load_thumb(r["path"]))
        ax.set_title(f'{r["width"]:.0f}x{r["height"]:.0f}', fontsize=7)
        ax.axis("off")
    fig.suptitle(f"Random samples: {cls} (seed={SEED})")
    fig.tight_layout()
    fig.savefig(FIG / f"samples_{cls}.png", dpi=150)
    plt.close(fig)

# ---- Figure 2: histograms of width, height, file size per class ----
df["file_kb"] = df["file_size"] / 1024
df["aspect"] = df["width"] / df["height"]

panels = [("width", "Width (px)", False),
          ("height", "Height (px)", False),
          ("file_kb", "File size (KB, log scale)", True),
          ("aspect", "Aspect ratio (w/h)", False)]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (col, label, log) in zip(axes.ravel(), panels):
    lo, hi = df[col].min(), df[col].quantile(0.99)   # clip extreme outliers in the plot
    bins = np.geomspace(lo, hi, 40) if log else np.linspace(lo, hi, 40)
    for cls in classes:
        ax.hist(df.loc[df["class"] == cls, col].clip(upper=hi), bins=bins,
                alpha=0.55, label=cls, color=colors.get(cls))
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(label)
    ax.set_ylabel("Number of images")
    ax.legend()
fig.suptitle("Image properties by class (x-axis clipped at the 99th percentile)")
fig.tight_layout()
fig.savefig(FIG / "histograms.png", dpi=150)
plt.close(fig)

# Numbers to quote in the captions
print(df.groupby("class")[["width", "height", "file_kb", "aspect"]]
        .describe(percentiles=[.5]).T.round(2).to_string())
print("\nSaved to", FIG.resolve())