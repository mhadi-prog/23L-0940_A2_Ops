"""Train baseline or transfer model. Never touches the test split."""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from data import SEED, make_dataset, set_seed
from models import (build_baseline_cnn, build_transfer, trainable_layer_names,
                    unfreeze_last_block)

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["baseline", "transfer"], required=True)
p.add_argument("--run-name", required=True)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--ft-lr", type=float, default=1e-5)      # stage 2 (transfer only)
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--ft-epochs", type=int, default=15)
p.add_argument("--patience", type=int, default=6)
p.add_argument("--dropout", type=float, default=None)
p.add_argument("--no-class-weight", action="store_true")
p.add_argument("--mlflow", action="store_true")
args = p.parse_args()

os.environ["PYTHONHASHSEED"] = str(SEED)
set_seed(SEED)
for d in ("models", "reports/figures"):
    Path(d).mkdir(parents=True, exist_ok=True)

train_ds = make_dataset("reports/train.csv", training=True)
val_ds = make_dataset("reports/val.csv", training=False)

labels = pd.read_csv("reports/train.csv")["label"].values
class_weight = None
if not args.no_class_weight:
    n = len(labels)
    class_weight = {0: n / (2 * (labels == 0).sum()), 1: n / (2 * (labels == 1).sum())}
print("class_weight:", class_weight)


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return "unknown"


def dvc_md5():
    try:
        return re.search(r"md5:\s*(\S+)", Path("data/raw.dvc").read_text()).group(1)
    except Exception:
        return "unknown"


# ---------- optional MLflow (manual logging via callback) ----------
class MlflowLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.offset = 0

    def on_epoch_end(self, epoch, logs=None):
        import mlflow
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()},
                           step=self.offset + epoch)


logger = MlflowLogger()
if args.mlflow:
    import mlflow
    from dotenv import load_dotenv
    load_dotenv()                      # reads MLFLOW_* from the git-ignored .env
    mlflow.set_experiment("23L-0940_A2_Ops")
    mlflow.start_run(run_name=args.run_name)
    mlflow.set_tags({"git_commit": sh("git rev-parse HEAD"), "dvc_data_md5": dvc_md5(),
                     "seed": SEED, "logging": "manual", "model": args.model})
    mlflow.log_params({"model": args.model, "lr": args.lr, "ft_lr": args.ft_lr,
                       "epochs": args.epochs, "ft_epochs": args.ft_epochs,
                       "patience": args.patience, "dropout": args.dropout,
                       "batch": 32, "img_size": 160,
                       "class_weight": not args.no_class_weight})


def plot_curves(hist, best_epoch, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key in zip(ax, ["loss", "accuracy"]):
        a.plot(range(1, len(hist[key]) + 1), hist[key], label="train")
        a.plot(range(1, len(hist[key]) + 1), hist["val_" + key], label="validation")
        a.axvline(best_epoch, color="k", ls="--", label=f"best epoch = {best_epoch}")
        a.set_xlabel("epoch"); a.set_ylabel(key); a.legend(); a.set_title(f"{title}: {key}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_stage(model, stage, epochs):
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(f"models/{args.run_name}_{stage}.keras",
                                           monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=max(2, args.patience // 2), min_lr=1e-7),
    ]
    if args.mlflow:
        cbs.append(logger)
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  class_weight=class_weight, callbacks=cbs, verbose=2).history
    best = int(np.argmin(h["val_loss"])) + 1
    print(f"[{stage}] best epoch = {best}, val_loss = {min(h['val_loss']):.4f}, "
          f"val_acc = {h['val_accuracy'][best - 1]:.4f}, val_auc = {h['val_auc'][best - 1]:.4f}")
    plot_curves(h, best, f"{args.run_name} {stage}", f"reports/figures/{args.run_name}_{stage}.png")
    json.dump({k: [float(v) for v in vs] for k, vs in h.items()},
              open(f"reports/{args.run_name}_{stage}_history.json", "w"))
    logger.offset += len(h["loss"])
    if args.mlflow:
        mlflow.log_metrics({f"{stage}_best_epoch": best, f"{stage}_best_val_loss": min(h["val_loss"]),
                            f"{stage}_best_val_acc": h["val_accuracy"][best - 1],
                            f"{stage}_best_val_auc": h["val_auc"][best - 1]})
    return best


if args.model == "baseline":
    model = build_baseline_cnn(lr=args.lr, **({"dropout": args.dropout} if args.dropout else {}))
    run_stage(model, "baseline", args.epochs)
else:
    model, base = build_transfer(lr=args.lr, **({"dropout": args.dropout} if args.dropout else {}))
    t1 = trainable_layer_names(model, base)
    Path(f"reports/{args.run_name}_trainable_stage1.txt").write_text("\n".join(t1))
    print("Stage 1 trainable layers:", t1)
    run_stage(model, "stage1_frozen", args.epochs)

    unfreeze_last_block(model, base, lr=args.ft_lr)
    t2 = trainable_layer_names(model, base)
    Path(f"reports/{args.run_name}_trainable_stage2.txt").write_text("\n".join(t2))
    print("Stage 2 trainable layers:", t2)
    run_stage(model, "stage2_finetune", args.ft_epochs)

if args.mlflow:
    mlflow.end_run()