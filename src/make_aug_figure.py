import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import make_dataset, set_seed

set_seed()
orig = next(iter(make_dataset("reports/train.csv", training=True, shuffle=False, augment=False)))
aug = next(iter(make_dataset("reports/train.csv", training=True, shuffle=False, augment=True)))

n = 8
fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.6))
for i in range(n):
    for row, (imgs, title) in enumerate([(orig, "original"), (aug, "augmented")]):
        axes[row, i].imshow(imgs[0][i].numpy().astype("uint8").clip(0, 255))
        axes[row, i].set_title(f"{title}\n{'AI' if imgs[1][i] else 'real'}", fontsize=8)
        axes[row, i].axis("off")
fig.suptitle("Training batch: before (top) and after (bottom) augmentation")
fig.tight_layout()
fig.savefig("reports/figures/augmented_batch.png", dpi=150)
print("saved")