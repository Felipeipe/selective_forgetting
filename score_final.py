import json
import pandas as pd
from pathlib import Path
from prune import get_parameters
import matplotlib.pyplot as plt
import numpy as np
ROOT_DIR = Path("pruned_models")

with open("experiments/metrics.json", "r") as f:
    metrics_original = json.load(f)
# Cargar csv con datos de MI por cada y por neurona
mi_table_layers_original = pd.read_csv("experiments/mi_mine_por_capa.csv")
mi_table_final_neuronas_original = pd.read_csv("experiments/mi_output.csv")
accuracy_original = metrics_original["balanced_accuracy"]
f1_original = metrics_original["f1_macro"]

def generate_heatmaps(results, graph_dir):
    df = pd.DataFrame(results)

    score_columns = {
        "score_model": "General Score",
        "comparison_MI_neurons": "Comparison MI Neurons",
        "comparison_MI_layers": "Comparison MI Layers"
    }

    attributes = df["attribute"].unique()

    for attribute in attributes:

        df_attribute = df[df["attribute"] == attribute]

        output_dir = graph_dir / attribute
        output_dir.mkdir(parents=True, exist_ok=True)

        for score_column, title in score_columns.items():

            heatmap_data = df_attribute.pivot(
                index="scalar",
                columns="percentage",
                values=score_column
            )

            heatmap_data = heatmap_data.sort_index()
            heatmap_data = heatmap_data.reindex(
                sorted(heatmap_data.columns), axis=1
            )

            values = heatmap_data.to_numpy(dtype=float)

            fig, ax = plt.subplots(figsize=(8, 6))

            # Crear heatmap
            im = ax.imshow(
                values,
                cmap="RdYlGn",
                aspect="auto"
            )

            # Agregar valores a cada celda
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):

                    if np.isfinite(values[i, j]):
                        ax.text(
                            j, i,
                            f"{values[i, j]:.3f}",
                            ha="center",
                            va="center",
                            color="black"
                        )

            # Configurar ejes
            ax.set_xticks(range(len(heatmap_data.columns)))
            ax.set_xticklabels(heatmap_data.columns)

            ax.set_yticks(range(len(heatmap_data.index)))
            ax.set_yticklabels(heatmap_data.index)

            ax.set_xlabel("Pruning Percentage")
            ax.set_ylabel("Pruning Scalar")
            ax.set_title(f"{title} - {attribute}")

            # Barra de color
            fig.colorbar(im, ax=ax, label=title)

            fig.tight_layout()

            fig.savefig(
                output_dir / f"{score_column}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

    print(f"Heatmaps guardados en: {graph_dir}")

def MI_attribute(mi_table_neurons, mi_table_layers, attribute_neurons, attribute_layers):
    mi_neurons = mi_table_neurons[attribute_neurons].iloc[0]
    mi_layers = mi_table_layers[attribute_layers].sum()
    return mi_neurons, mi_layers

def score_model (original_accuracy, original_F1, original_MI_neurons, original_MI_layers, pruned_accuracy, pruned_F1, pruned_MI_neurons, pruned_MI_layers):
    accuracy_score = (pruned_accuracy - original_accuracy) / original_accuracy
    F1_score = (pruned_F1 - original_F1) / original_F1
    MI_score_neurons = - (pruned_MI_neurons - original_MI_neurons) / (original_MI_neurons)
    MI_score_layers = - (pruned_MI_layers - original_MI_layers) / original_MI_layers
    return MI_score_neurons + MI_score_layers + accuracy_score + F1_score


def comparison_MI (original_MI_selected, pruned_MI_selected, original_MI_attribute1, pruned_MI_attribute1, original_MI_attribute2, pruned_MI_attribute2):
    # Se aplica independientemente para neuronas y se puede usar en otra instancia para capas.
    MI_selected = - (pruned_MI_selected - original_MI_selected) / original_MI_selected
    MI_attribute1 = (pruned_MI_attribute1 - original_MI_attribute1) / original_MI_attribute1
    MI_attribute2 = (pruned_MI_attribute2 - original_MI_attribute2) / original_MI_attribute2
    return 2 * MI_selected + MI_attribute1 + MI_attribute2

if __name__ == "__main__":

    _, _,_, attribute_names_layers, atribute_names_outputs = get_parameters()

    # Cargar MI original
    list_mi_neurons_original = []
    list_mi_layers_original = []

    for attribute_neuron, attribute_layer in zip(
        atribute_names_outputs, attribute_names_layers
    ):
        mi_neurons, mi_layers = MI_attribute(mi_table_final_neuronas_original, mi_table_layers_original, attribute_neuron, attribute_layer)

        list_mi_neurons_original.append(mi_neurons)
        list_mi_layers_original.append(mi_layers)

    # Evaluar modelos podados
    results = []
    GRAPH_DIR = ROOT_DIR / "graphs"
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    for RUN in ROOT_DIR.rglob("pruned_model.pth"):
        OUT_DIR = RUN.parent
        print ("OUT DIR:", OUT_DIR)
        OUTPUT_CSV = OUT_DIR / "mi_output.csv"
        MINE_CSV = OUT_DIR / "mi_mine_por_capa.csv"
        metrics_path = OUT_DIR / "metrics.json"
        parameters_path = OUT_DIR / "parameters.json"

        with open(parameters_path, "r") as f:
            parameters = json.load(f)
        attribute_selected = parameters ["attribute"]
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        accuracy_pruned = metrics ["balanced_accuracy"]
        F1_pruned = metrics ["f1_macro"]
        idx_selected = attribute_names_layers.index(attribute_selected)
        mi_table_layers = pd.read_csv(MINE_CSV)
        mi_table_final_neurons = pd.read_csv(OUTPUT_CSV)

        list_mi_neurons = []
        list_mi_layers = []

        for attribute_neuron, attribute_layer in zip(atribute_names_outputs, attribute_names_layers):

            mi_neurons, mi_layers = MI_attribute(mi_table_final_neurons,mi_table_layers,attribute_neuron,attribute_layer)
            list_mi_neurons.append(mi_neurons)
            list_mi_layers.append(mi_layers)

        # Score general
        score_model_prune = score_model(accuracy_original, f1_original, list_mi_neurons_original[idx_selected], list_mi_layers_original[idx_selected], accuracy_pruned, F1_pruned, list_mi_neurons[idx_selected], list_mi_layers[idx_selected])
        print("Score:", score_model_prune)

        metrics ["score model"] = score_model_prune
        # Índices de los atributos restantes
        other_attributes_idx = [i for i in range(len(atribute_names_outputs)) if i != idx_selected]

        # Comparación MI de neuronas
        comparison_score_MI_neurons = comparison_MI(
            list_mi_neurons_original[idx_selected],
            list_mi_neurons[idx_selected],

            list_mi_neurons_original[other_attributes_idx[0]],
            list_mi_neurons[other_attributes_idx[0]],

            list_mi_neurons_original[other_attributes_idx[1]],
            list_mi_neurons[other_attributes_idx[1]]
        )
        metrics ["Comparison score MI Neurons"] = comparison_score_MI_neurons
        print("Comparison MI neurons:", comparison_score_MI_neurons)

        # Comparación MI de capas
        comparison_score_MI_layer = comparison_MI(
            list_mi_layers_original[idx_selected],
            list_mi_layers[idx_selected],

            list_mi_layers_original[other_attributes_idx[0]],
            list_mi_layers[other_attributes_idx[0]],

            list_mi_layers_original[other_attributes_idx[1]],
            list_mi_layers[other_attributes_idx[1]]
        )
        metrics ["Comparison score MI layers"] = comparison_score_MI_layer
        print("Comparison MI layers:", comparison_score_MI_layer)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        results.append({
            "attribute": parameters ["attribute"],
            "scalar": parameters ["scalar"],
            "percentage": parameters ["percentage"],
            "score_model": float(score_model_prune),
            "comparison_MI_neurons": float(comparison_score_MI_neurons),
            "comparison_MI_layers": float(comparison_score_MI_layer)
        })
    generate_heatmaps(results, GRAPH_DIR)