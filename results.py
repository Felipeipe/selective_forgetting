"""Aggregate the runs in lightning_logs/ into figures and summary stats (mean ± std)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from main import LitMLP, fetch_adult, load_adult, split_indices

LOG_DIR = Path("lightning_logs")
FIG_DIR = Path("report/figures")
LABELS = ["<=50K", ">50K"]

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2 = "#0b0b0b", "#52514e"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def run_dirs():
    return sorted(LOG_DIR.glob("version_*"), key=lambda p: int(p.name.split("_")[1]))


def epoch_curves(run):
    df = pd.read_csv(run / "metrics.csv")
    train = df.dropna(subset=["train_loss"]).set_index("epoch")[["train_loss", "train_acc"]]
    val = df.dropna(subset=["val_loss"]).set_index("epoch")[["val_loss", "val_acc"]]
    return pd.concat([train, val], axis=1)


def plot_band(ax, curves, col, color, label):
    stacked = np.stack([c[col].to_numpy() for c in curves])
    mean, std = stacked.mean(0), stacked.std(0, ddof=1)
    epochs = curves[0].index.to_numpy() + 1
    ax.plot(epochs, mean, color=color, lw=1.5, label=label)
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.25, lw=0)


def single_column_axes():
    # sized for one IEEE column (3.5 in)
    fig, ax = plt.subplots(figsize=(3.4, 1.5))
    ax.grid(axis="y", color="#e5e5e2", lw=0.5)
    ax.set_xlabel("Época")
    return fig, ax


def plot_loss_curves(curves, path):
    fig, ax = single_column_axes()
    plot_band(ax, curves, "train_loss", BLUE, "Entrenamiento")
    plot_band(ax, curves, "val_loss", ORANGE, "Validación")
    ax.set_ylabel("Entropía cruzada")
    ax.legend()
    fig.savefig(path, bbox_inches="tight")


def plot_accuracy_curves(curves, path):
    fig, ax = single_column_axes()
    plot_band(ax, curves, "train_acc", BLUE, "Entrenamiento")
    plot_band(ax, curves, "val_acc", ORANGE, "Validación")
    ax.set_ylabel("Exactitud")
    ax.legend(loc="lower right")
    fig.savefig(path, bbox_inches="tight")


@torch.no_grad()
def test_predictions(run, test):
    ckpt = next((run / "checkpoints").glob("*.ckpt"))
    model = LitMLP.load_from_checkpoint(ckpt, map_location="cpu").eval()
    x, _ = test.tensors
    return model(x).argmax(dim=1).numpy()


def confusion_matrix(y, pred):
    # rows: true class, columns: predicted class
    return np.bincount(2 * y + pred, minlength=4).reshape(2, 2)


def test_sex():
    # sex of each test row, using the same split as load_adult
    X, _ = fetch_adult()
    _, _, test_idx = split_indices(len(X))
    return X["sex"].to_numpy()[test_idx]


def draw_confusion_matrix(ax, cms):
    # normalized per true class, averaged over runs
    rates = cms / cms.sum(axis=2, keepdims=True)
    mean, std = rates.mean(0), rates.std(0, ddof=1)

    ax.imshow(mean, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            color = "white" if mean[i, j] > 0.5 else INK
            ax.text(
                j, i, f"{mean[i, j]:.3f}\n± {std[i, j]:.3f}",
                ha="center", va="center", color=color,
            )
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Clase predicha")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def plot_confusion_matrix(cms, path):
    fig, ax = plt.subplots(figsize=(2.0, 1.8))
    draw_confusion_matrix(ax, cms)
    ax.set_ylabel("Clase real")
    fig.savefig(path, bbox_inches="tight")


def plot_confusion_matrix_by_group(cms_by_group, path):
    fig, axes = plt.subplots(1, len(cms_by_group), figsize=(3.5, 1.8), sharey=True)
    for ax, (name, cms) in zip(axes, cms_by_group.items()):
        draw_confusion_matrix(ax, cms)
        ax.set_title(f"{name} (n = {cms[0].sum()})", fontsize=8)
    axes[0].set_ylabel("Clase real")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def group_metrics(cms):
    # cms: (runs, 2, 2) counts -> mean ± std over runs
    tn, fp, fn, tp = cms[:, 0, 0], cms[:, 0, 1], cms[:, 1, 0], cms[:, 1, 1]
    n = tn + fp + fn + tp
    metrics = {
        "exactitud": (tp + tn) / n,
        "tasa >50K real": (tp + fn) / n,
        "tasa >50K predicha": (tp + fp) / n,
        "TPR (recall >50K)": tp / (tp + fn),
        "FPR": fp / (fp + tn),
    }
    return pd.DataFrame({k: [v.mean(), v.std(ddof=1)] for k, v in metrics.items()}, index=["mean", "std"])


def test_metrics(run):
    df = pd.read_csv(run / "metrics.csv")
    return df.dropna(subset=["test_acc"]).iloc[-1][["test_acc", "test_loss"]]


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = run_dirs()
    print(f"{len(runs)} runs: {[r.name for r in runs]}")

    metrics = pd.DataFrame([test_metrics(r) for r in runs])
    print(metrics.agg(["mean", "std"]))

    curves = [epoch_curves(r) for r in runs]
    plot_loss_curves(curves, FIG_DIR / "loss_curves.pdf")
    plot_accuracy_curves(curves, FIG_DIR / "accuracy_curves.pdf")

    _, _, test = load_adult()
    y = test.tensors[1].numpy()
    preds = [test_predictions(r, test) for r in runs]
    cms = np.stack([confusion_matrix(y, p) for p in preds])
    plot_confusion_matrix(cms, FIG_DIR / "confusion_matrix.pdf")
    print("confusion matrices (counts):\n", cms)

    sex = test_sex()
    cms_by_sex = {
        label: np.stack([confusion_matrix(y[sex == value], p[sex == value]) for p in preds])
        for label, value in [("Mujeres", "Female"), ("Hombres", "Male")]
    }
    plot_confusion_matrix_by_group(cms_by_sex, FIG_DIR / "confusion_matrix_by_sex.pdf")
    for label, group_cms in cms_by_sex.items():
        print(f"\n{label}:\n", group_metrics(group_cms).T.round(3))
