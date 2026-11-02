"""Mutual information between the internal representations of the report's model and
the target Y or the protected attributes Z.

Two analyses:
- per neuron, with scikit-learn's nearest-neighbor (KSG) estimators -> heatmap;
- per layer (whole representation), with MINE (torch_mist) -> line plot.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score
from torch import nn
from torch_mist import estimate_mi

from main import LitMLP, fetch_adult, load_adult, split_indices
from results import INK  # importing results also applies the shared matplotlib style

RUN = Path("lightning_logs/mutual_info_model")  # model analyzed in the report
OUT_DIR = Path("experiments")
NEURON_CSV = OUT_DIR / "mi_per_neuron.csv"
OUTPUT_CSV = OUT_DIR / "mi_output.csv"
MINE_CSV = OUT_DIR / "mi_mine_por_capa.csv"
HEATMAP_FIG = OUT_DIR / "figures/mi_heatmap.pdf"
MINE_FIG = OUT_DIR / "figures/mi_mine_por_capa.pdf"

SPLITS = {"train": 0, "val": 1, "test": 2}
# Both analyses use the same split so they can be compared. Once these estimates are
# used to decide what to prune, switch to "train": the test set must not guide interventions.
SPLIT = "test"
PROTECTED = {"sex": "discrete", "race": "discrete", "age": "continuous"}

# MINE plot: variable -> (label, color, marker); colors from matplotlib's tab10
MINE_VARIABLES = {
    "sex": ("Sexo", "#1f77b4", "o"),
    "race": ("Raza", "#ff7f0e", "s"),
    "age": ("Edad", "#2ca02c", "^"),
    "Y": ("$Y$ (ingreso)", "#d62728", "D"),
}


def load_model(run, input_dim):
    ckpt = next((run / "checkpoints").glob("*.ckpt"))
    model = LitMLP.load_from_checkpoint(ckpt, map_location="cpu").eval()
    if model.hparams.input_dim != input_dim:
        raise ValueError(
            f"{run.name} expects {model.hparams.input_dim} input features but the data has "
            f"{input_dim}: main.PROTECTED_ATTRS does not match the attributes it was trained without"
        )
    return model


@torch.no_grad()
def hidden_activations(model, x):
    """Post-ReLU activations of each hidden layer, as (n_samples, n_neurons) arrays."""
    acts = []
    h = x
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            acts.append(h.numpy())
    return acts


def protected_attributes(split):
    # the protected attributes are not model inputs, so read them from the raw data
    X, _ = fetch_adult(protected_attributes=[])
    idx = split_indices(len(X))[SPLITS[split]]
    return X.iloc[idx][list(PROTECTED)].reset_index(drop=True)


# ---------------------------------------------------------------- KSG, per neuron


def mutual_information(U, target, kind, seed=0):
    """I(U_i; target) in bits, for every column U_i of U."""
    if kind == "discrete":
        codes, _ = pd.factorize(target)
        mi = mutual_info_classif(U, codes, discrete_features=False, random_state=seed)
    else:
        mi = mutual_info_regression(U, target, discrete_features=False, random_state=seed)
    return mi / np.log(2)


def pairwise_mi(a, a_kind, b, b_kind, seed=0):
    """I(a; b) in bits between two 1-D variables, picking the estimator by variable type."""
    if a_kind == "discrete" and b_kind == "discrete":
        return mutual_info_score(pd.factorize(a)[0], pd.factorize(b)[0]) / np.log(2)
    if a_kind == "discrete":
        a, a_kind, b, b_kind = b, b_kind, a, a_kind
    return mutual_information(a[:, None], b, b_kind, seed)[0]


@torch.no_grad()
def output_mi_table(model, x, y, Z):
    """MI between the network output and Y / the protected attributes.

    The output is taken two ways: the predicted probability P(>50K) (continuous) and
    the predicted class (discrete). The true label is included as a reference."""
    prob = torch.softmax(model(x), dim=1)[:, 1].numpy()
    outputs = {
        "P(>50K)": (prob, "continuous"),
        "clase predicha": ((prob > 0.5).astype(int), "discrete"),
        "Y real (referencia)": (y, "discrete"),
    }
    variables = {"Y": (y, "discrete")} | {
        name: (Z[name].to_numpy(), kind) for name, kind in PROTECTED.items()
    }
    rows = []
    for out_name, (out, out_kind) in outputs.items():
        row = {"output": out_name}
        for var_name, (var, var_kind) in variables.items():
            row[f"I(out;{var_name})"] = pairwise_mi(out, out_kind, var, var_kind)
        rows.append(row)
    return pd.DataFrame(rows)


def mi_table(model, x, y, Z):
    tables = []
    for layer, U in enumerate(hidden_activations(model, x)):
        df = pd.DataFrame(
            {
                "layer": layer,
                "neuron": np.arange(U.shape[1]),
                "dead": U.std(axis=0) == 0,
                "I(U;Y)": mutual_information(U, y, "discrete"),
            }
        )
        for name, kind in PROTECTED.items():
            df[f"I(U;{name})"] = mutual_information(U, Z[name].to_numpy(), kind)
        df.loc[df["dead"], df.columns.str.startswith("I(")] = 0.0
        tables.append(df)
    return pd.concat(tables, ignore_index=True)


# ---------------------------------------------------------------- MINE, per layer


def one_hot(values):
    codes, uniques = pd.factorize(values)
    return np.eye(len(uniques), dtype=np.float32)[codes]


def mine_mi(x, y):
    """I(x; y) in bits estimated with MINE. x and y are (n_samples, dim) arrays."""
    # fixed seed: the critic's training is random, so results would change on every run
    torch.manual_seed(0)
    out = estimate_mi(
        data=(x.astype(np.float32), y.astype(np.float32)),
        estimator="mine",
        max_iterations=1000,
        hidden_dims=[256, 128, 64],  # critic architecture
        verbose=False,
    )
    mi = out[0] if isinstance(out, tuple) else out
    return float(mi) / np.log(2)  # torch_mist returns nats


def mine_table(model, x, y, Z):
    """MINE estimate for each hidden layer and the output logits (rows) and each
    variable (columns, named by the plot labels)."""
    # no torch.no_grad() here: it would also stop the MINE critic from training
    with torch.no_grad():
        logits = model(x).numpy()
    layers = hidden_activations(model, x) + [logits]
    layer_names = [f"Capa {i + 1}" for i in range(len(layers) - 1)] + ["Salida"]
    variables = {
        "sex": one_hot(Z["sex"].to_numpy()),
        "race": one_hot(Z["race"].to_numpy()),
        "age": Z["age"].to_numpy()[:, None],
        "Y": one_hot(y),
    }
    table = pd.DataFrame(index=layer_names, columns=[v[0] for v in MINE_VARIABLES.values()], dtype=float)
    for key, (label, _, _) in MINE_VARIABLES.items():
        for name, U in zip(layer_names, layers):
            table.loc[name, label] = mine_mi(U, variables[key])
            print(f"MINE {name} y {label}: {table.loc[name, label]:.4f} bits")
    return table


# ---------------------------------------------------------------- plots


def plot_mi_heatmap(result, output_result, path):
    """Grid of strips: one row per hidden layer, one column per variable. Each strip
    shows the neurons of that layer in architecture order, left to right. A last row
    shows the MI of the network output P(>50K)."""
    mi_cols = [c for c in result.columns if c.startswith("I(")]
    labels = ["$Y$", "sexo", "raza", "edad"]
    layers = sorted(result["layer"].unique())

    out = output_result[output_result["output"] == "P(>50K)"]
    out_values = out[[c.replace("I(U;", "I(out;") for c in mi_cols]].to_numpy()[0]
    vmax = max(result[mi_cols].to_numpy().max(), out_values.max())

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#d9d9d6")

    fig, axes = plt.subplots(
        len(layers) + 1, len(mi_cols), figsize=(7.0, 2.3),
        gridspec_kw={"hspace": 0.9, "wspace": 0.12}, squeeze=False,
    )

    def style(ax, row, col, row_label):
        ax.set_yticks([])
        ax.tick_params(length=0, pad=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row == 0:
            ax.set_title(labels[col], fontsize=8)
        if col == 0:
            ax.set_ylabel(row_label, rotation=0, ha="right", va="center")

    for row, layer in enumerate(layers):
        df = result[result["layer"] == layer].sort_values("neuron")
        size = len(df)
        for col, mi_col in enumerate(mi_cols):
            ax = axes[row, col]
            values = np.array(df[mi_col], dtype=float)[None, :]
            values[:, df["dead"].to_numpy()] = np.nan
            im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            ticks = [0, size // 2 - 1, size - 1]
            ax.set_xticks(ticks, [t + 1 for t in ticks])
            first, *_, last = ax.get_xticklabels()
            first.set_ha("left")
            last.set_ha("right")
            style(ax, row, col, f"Capa {layer + 1}")

    for col, value in enumerate(out_values):
        ax = axes[-1, col]
        ax.imshow([[value]], aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.text(0, 0, f"{value:.3f}", ha="center", va="center", color="white" if value > vmax / 2 else INK)
        ax.set_xticks([])
        style(ax, len(layers), col, "Salida\n$P(>$50K$)$")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("Información mutua (bits)")
    cbar.outline.set_visible(False)
    fig.savefig(path, bbox_inches="tight")


def plot_mine(table, path):
    """table: one row per layer and one column per variable label (see mine_table)."""
    # drawn wide and scaled down to the column width, so lines, markers and text
    # stay thin and small in the report
    with plt.rc_context({"font.size": 10}):
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        for label, color, marker in MINE_VARIABLES.values():
            ax.plot(table.index, table[label], color=color, marker=marker, lw=1.5, ms=6, label=label)
        ax.set_xlabel("Representación interna")
        ax.set_ylabel("Información mutua (bits)")
        ax.set_ylim(bottom=0)
        ax.grid(color="#e5e5e2", lw=0.8)
        ax.legend(loc="upper right")
        fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    HEATMAP_FIG.parent.mkdir(parents=True, exist_ok=True)

    x, y = load_adult()[SPLITS[SPLIT]].tensors
    y = y.numpy()
    Z = protected_attributes(SPLIT)
    model = load_model(RUN, x.shape[1])
    print(f"run: {RUN}, split: {SPLIT} ({len(y)} rows)")

    # KSG, per neuron
    result = mi_table(model, x, y, Z)
    result.to_csv(NEURON_CSV, index=False)
    output_result = output_mi_table(model, x, y, Z)
    output_result.to_csv(OUTPUT_CSV, index=False)
    plot_mi_heatmap(result, output_result, HEATMAP_FIG)
    print(f"saved {NEURON_CSV}, {OUTPUT_CSV}, {HEATMAP_FIG}")

    mi_cols = [c for c in result.columns if c.startswith("I(")]
    print("\nmean MI per layer (bits), averaged over neurons:")
    print(result.groupby("layer")[mi_cols + ["dead"]].mean().round(4))
    print("\nMI between the network output and each variable (bits):")
    print(output_result.set_index("output").round(4))

    # MINE, per layer
    print()
    mine = mine_table(model, x, y, Z)
    mine.to_csv(MINE_CSV)
    plot_mine(mine, MINE_FIG)
    print(f"saved {MINE_CSV}, {MINE_FIG}")
    print("\nMINE per layer (bits):")
    print(mine.round(4))
