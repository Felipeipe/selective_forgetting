"""Group evaluation of the pruned models in pruned_models/ against the original model.

For every model (original + each pruned one) it computes, on the test split:
- global accuracy;
- per group of sex, race and age: confusion matrix, accuracy, TPR, FPR and rate of
  predicted >50K;
- the gap between groups (max - min) of TPR, FPR, predicted >50K rate and macro F1.

Outputs:
- pruned_models/fairness_metrics.csv: one row per model;
- pruned_models/graphs/heatmap_accuracy.pdf, heatmap_brecha_tpr.pdf and heatmap_brecha_f1.pdf: grid of lambda vs p
  for each pruned attribute. The column p = 0 is the original (unpruned) model;
- confusion_matrix_by_group.pdf inside the folder of each model.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from main import LitMLP, load_adult
from mutual_information import RUN, load_model, protected_attributes
from results import INK, LABELS, confusion_matrix

ROOT_DIR = Path("pruned_models")
GRAPH_DIR = ROOT_DIR / "graphs"
ORIGINAL_DIR = Path("experiments")
CSV_PATH = ROOT_DIR / "fairness_metrics.csv"
SPLIT = "test"  # evaluation only: the pruning decisions must not depend on this split

# the models were trained without these attributes as input
REMOVED_ATTRS = ["sex", "race", "age"]
# folder name of the pruned attribute -> column of the protected attribute
ATTRIBUTE_COLUMN = {"Sexo": "sex", "Raza": "race", "Edad": "age"}
ATTRIBUTE_LABEL = {"sex": "Sexo", "race": "Raza", "age": "Edad"}


def group_labels(Z):
    """Group of each test row, for each protected attribute."""
    sex = Z["sex"].map({"Female": "Mujeres", "Male": "Hombres"})
    # the other races are about 15 % of the data and some have fewer than 100 test rows,
    # so they are joined into one group
    race = np.where(Z["race"] == "White", "Blanca", "No blanca")
    age = pd.cut(Z["age"], bins=[0, 29, 49, np.inf], labels=["17-29", "30-49", "50+"])
    return {"sex": sex.to_numpy(), "race": race, "age": age.astype(str).to_numpy()}


@torch.no_grad()
def predict(model, x):
    return model(x).argmax(dim=1).numpy()


def f1(tp, fp, fn):
    # F1 of one class; 0 if the class is never predicted nor present
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0


def rates(cm):
    (tn, fp), (fn, tp) = cm
    return {
        "acc": (tp + tn) / cm.sum(),
        "tpr": tp / (tp + fn),
        "fpr": fp / (fp + tn),
        "pos": (tp + fp) / cm.sum(),  # rate of predicted >50K
        # mean of the F1 of class >50K and the F1 of class <=50K (negatives taken as positives)
        "f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
    }


def evaluate(y, pred, groups):
    """Global accuracy, group metrics and gaps. Returns (row for the CSV, group confusion matrices)."""
    row = {"accuracy": (y == pred).mean()}
    cms = {}
    for attr, labels in groups.items():
        names = sorted(np.unique(labels))
        cms[attr] = {g: confusion_matrix(y[labels == g], pred[labels == g]) for g in names}
        group_rates = {g: rates(cm) for g, cm in cms[attr].items()}
        for g, r in group_rates.items():
            for metric, value in r.items():
                row[f"{metric}_{attr}_{g}"] = value
        for metric in ["tpr", "fpr", "pos", "f1"]:
            values = [r[metric] for r in group_rates.values()]
            row[f"brecha_{metric}_{attr}"] = max(values) - min(values)
    return row, cms


def plot_confusion_by_group(cms, title, path):
    """One row per protected attribute, one confusion matrix per group, normalized per true class."""
    n_cols = max(len(c) for c in cms.values())
    fig, axes = plt.subplots(len(cms), n_cols, figsize=(1.6 * n_cols, 1.7 * len(cms)), squeeze=False)
    for row, (attr, group_cms) in enumerate(cms.items()):
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(group_cms):
                ax.axis("off")
                continue
            name, cm = list(group_cms.items())[col]
            norm = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                            color="white" if norm[i, j] > 0.5 else INK)
            ax.set_title(f"{name} (n = {cm.sum()})", fontsize=7)
            ax.set_xticks(range(2), LABELS if row == len(cms) - 1 else [])
            ax.set_yticks(range(2), LABELS if col == 0 else [])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(f"{ATTRIBUTE_LABEL[attr]}\nClase real")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_grid(df, original, value_col, label, cmap, path, gap_of_pruned=False):
    """One heatmap per pruned attribute: rows lambda, columns p. The column p = 0 is the
    original model (the same value for every lambda, since nothing is pruned)."""
    attributes = list(ATTRIBUTE_COLUMN)
    tables = {}
    for name in attributes:
        col = value_col.format(attr=ATTRIBUTE_COLUMN[name]) if gap_of_pruned else value_col
        table = df[df["attribute"] == name].pivot(index="scalar", columns="percentage", values=col)
        table.insert(0, 0.0, original[col])  # p = 0: original model
        tables[name] = table.sort_index().sort_index(axis=1)
    all_values = np.concatenate([t.to_numpy().ravel() for t in tables.values()])
    vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

    fig, axes = plt.subplots(1, len(attributes), figsize=(7.0, 2.3), sharey=True)
    for ax, name in zip(axes, attributes):
        table = tables[name]
        values = table.to_numpy(dtype=float)
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                dark = (values[i, j] - vmin) / (vmax - vmin) > 0.6
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                        color="white" if dark else INK)
        # line that separates the original model from the pruned ones
        ax.axvline(0.5, color="white", lw=2)
        ax.set_xticks(range(values.shape[1]), ["0\n(original)"] + [f"{p:g}" for p in table.columns[1:]])
        ax.set_yticks(range(values.shape[0]), [f"{s:g}" for s in table.index])
        ax.set_xlabel("Fracción podada $p$")
        title = f"Poda por {name.lower()}"
        if gap_of_pruned:
            title += f"\n(brecha entre grupos de {name.lower()})"
        ax.set_title(title, fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel(r"$\lambda$")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label(label)
    cbar.outline.set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": 0, "val": 1, "test": 2}
    x, y = load_adult(protected_attributes=REMOVED_ATTRS)[splits[SPLIT]].tensors
    y = y.numpy()
    groups = group_labels(protected_attributes(SPLIT))

    # original model
    model = load_model(RUN, x.shape[1])
    original, cms = evaluate(y, predict(model, x), groups)
    plot_confusion_by_group(cms, "Modelo original", ORIGINAL_DIR / "figures/confusion_matrix_by_group.pdf")
    rows = [{"attribute": "original", "scalar": np.nan, "percentage": 0.0, **original}]

    # pruned models
    for path in sorted(ROOT_DIR.rglob("pruned_model.pth")):
        params = json.loads((path.parent / "parameters.json").read_text())
        model = LitMLP(x.shape[1], hidden_dims=model.hparams.hidden_dims)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        row, cms = evaluate(y, predict(model, x), groups)
        title = f"Poda por {params['attribute'].lower()}, $\\lambda$ = {params['scalar']:g}, p = {params['percentage']:g}"
        plot_confusion_by_group(cms, title, path.parent / "confusion_matrix_by_group.pdf")
        rows.append({**params, **row})

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"saved {CSV_PATH} ({len(df)} models)")

    pruned = df[df["attribute"] != "original"]
    plot_grid(pruned, original, "accuracy", "Exactitud", "Blues",
              GRAPH_DIR / "heatmap_accuracy.pdf")
    plot_grid(pruned, original, "brecha_tpr_{attr}", "Brecha de TPR (máx $-$ mín)", "Oranges",
              GRAPH_DIR / "heatmap_brecha_tpr.pdf", gap_of_pruned=True)
    plot_grid(pruned, original, "brecha_f1_{attr}", "Brecha de F1 macro (máx $-$ mín)", "Oranges",
              GRAPH_DIR / "heatmap_brecha_f1.pdf", gap_of_pruned=True)
    print(f"saved heatmaps in {GRAPH_DIR}")

    cols = ["accuracy", "brecha_tpr_sex", "brecha_tpr_race", "brecha_tpr_age",
            "brecha_f1_sex", "brecha_f1_race", "brecha_f1_age",
            "brecha_pos_sex", "brecha_pos_race", "brecha_pos_age"]
    print("\noriginal model:")
    print(pd.Series(original)[cols].round(3).to_string())
    print("\npruned models:")
    print(pruned.set_index(["attribute", "scalar", "percentage"])[cols].round(3).to_string())
