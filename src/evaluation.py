
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from data import make_dataset

FIG = Path("reports/figures")


def predict(model, csv):
    return model.predict(make_dataset(csv, training=False), verbose=0).ravel()


def metrics_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": roc_auc_score(y, p)}


def pick_threshold(y, p, beta=1.0):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y, (p >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def misclassified_grid(df, p, thr, path, n=10):
    df = df.copy()
    df["prob_ai"] = p
    df["pred"] = (p >= thr).astype(int)
    wrong = df[df["pred"] != df["label"]].copy()
    wrong["conf"] = np.where(wrong["pred"] == 1, wrong["prob_ai"], 1 - wrong["prob_ai"])
    wrong = wrong.sort_values("conf", ascending=False).head(n)   # most confident mistakes first
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
        with Image.open(r["path"]) as im:
            im = im.convert("RGB"); im.thumbnail((300, 300))
            ax.imshow(im)
        ax.set_title(f'true: {"AI" if r["label"] else "real"}\npred: {"AI" if r["pred"] else "real"} '
                     f'(p_ai={r["prob_ai"]:.2f})', fontsize=8)
    fig.suptitle(f"Most confident misclassified test images (threshold {thr:.2f})")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    wrong.to_csv("reports/misclassified.csv", index=False)


def evaluate_and_log(model, run_name, use_mlflow=False, eval_test=False, beta=1.0):
    FIG.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    yv = pd.read_csv("reports/val.csv")["label"].values
    pv = predict(model, "reports/val.csv")
    thr = pick_threshold(yv, pv, beta)
    val_m = metrics_at(yv, pv, thr)
    print(f"threshold (chosen on val, F{beta}) = {thr:.2f}")
    print("val metrics:", {k: round(v, 4) for k, v in val_m.items()})

    model_path = f"models/{run_name}_final.keras"
    model.save(model_path)
    files = [model_path, "reports/train.csv", "reports/val.csv", "reports/test.csv"]

    freeze = subprocess.check_output("pip freeze", shell=True, text=True)
    Path("reports/pip_freeze.txt").write_text(freeze)
    files.append("reports/pip_freeze.txt")

    test_m = None
    if eval_test:                                   # the one and only test evaluation
        tdf = pd.read_csv("reports/test.csv")
        yt = tdf["label"].values
        pt = predict(model, "reports/test.csv")
        test_m = metrics_at(yt, pt, thr)            # threshold fixed from val
        print("TEST metrics:", {k: round(v, 4) for k, v in test_m.items()})

        cm = confusion_matrix(yt, (pt >= thr).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["real", "AI"]).plot(cmap="Blues")
        plt.title(f"{run_name}: test confusion matrix (thr {thr:.2f})")
        plt.savefig(FIG / f"{run_name}_confusion.png", dpi=150); plt.close()

        fpr, tpr, _ = roc_curve(yt, pt)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f'AUC = {test_m["roc_auc"]:.3f}')
        plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate")
        plt.ylabel("True positive rate"); plt.legend(); plt.title(f"{run_name}: test ROC")
        plt.savefig(FIG / f"{run_name}_roc.png", dpi=150); plt.close()

        pd.DataFrame({"path": tdf["path"], "label": yt, "prob_ai": pt}).to_csv(
            f"reports/{run_name}_test_predictions.csv", index=False)
        misclassified_grid(tdf, pt, thr, FIG / f"{run_name}_misclassified.png")
        files += [FIG / f"{run_name}_confusion.png", FIG / f"{run_name}_roc.png",
                  FIG / f"{run_name}_misclassified.png", "reports/misclassified.csv",
                  f"reports/{run_name}_test_predictions.csv"]

    if use_mlflow:
        import mlflow
        mlflow.log_param("threshold", thr)
        mlflow.log_metrics({f"val_{k}_at_thr": v for k, v in val_m.items()})
        if test_m:
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        for f in files:
            mlflow.log_artifact(str(f))
        for f in Path("reports").glob(f"{run_name}_*_trainable_*.txt"):
            mlflow.log_artifact(str(f))
        for f in FIG.glob(f"{run_name}_*.png"):
            mlflow.log_artifact(str(f))