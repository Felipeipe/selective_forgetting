import pandas as pd
import numpy as np
import torch
from main import LitMLP, load_adult
from mutual_information import SPLIT, SPLITS, RUN, protected_attributes, load_model
from pathlib import Path
import json
# Cargar csv con datos de MI por cada y por neurona
mi_capa = pd.read_csv("experiments/mi_mine_por_capa.csv")
mi_neurona = pd.read_csv("experiments/mi_per_neuron.csv")
mi_output = pd.read_csv("experiments/mi_output.csv")


# Proporción de MI de cada atributo con respecto al total de MI de la capa
def MI_relativo_capa (mi_capa, attribute):
    MI_attribute = mi_capa [attribute].head(3)
    total_MI = MI_attribute.sum()
    return MI_attribute.values / total_MI

# Función para obtener los parámetros de los csv
def get_parameters ():
    layer_names = mi_neurona ["layer"].unique()
    neurons_per_layer = []
    atribute_names_neurons = list (mi_neurona.columns) [-3:]
    atribute_names_layers = list (mi_capa.columns) [1:4]
    atribute_names_outputs = list (mi_output.columns) [1:4]
    for layer in layer_names: 
        neurons_per_layer.append(len(mi_neurona [mi_neurona ["layer"] == layer]))
    return list (layer_names), neurons_per_layer, atribute_names_neurons, atribute_names_layers, atribute_names_outputs

# Función para obtener el score de cada neurona
def score_neuron (idx, scalar, attribute, layer):
    neuron_layer = mi_neurona [mi_neurona ["layer"] == layer]
    neuron_stats = neuron_layer [neuron_layer ["neuron"] == idx]
    target_MI = neuron_stats ["I(U;Y)"].values[0]
    attribute_MI = neuron_stats [attribute].values[0]
    score = target_MI - scalar * attribute_MI
    return score


# Función para obtener el score de cada neurona en una capa
def score_layer (num_neurons, layer, scalar, attribute):
    scores = []
    for idx in range(num_neurons):
        score = score_neuron (idx, scalar, attribute, layer)
        scores.append(score)
    return scores

# Función para obtener el número de neuronas a podar por capa
def prune_proportion(total_neurons, percentage_prune, attribute, layer_names):
    total_prune = round(total_neurons * percentage_prune)
    relative_MI = MI_relativo_capa(mi_capa, attribute)
    exact = relative_MI * total_prune
    prune_neurons = np.floor(exact).astype(int)
    remainder = total_prune - prune_neurons.sum()
    order = np.argsort(-(exact - prune_neurons))
    prune_neurons[order[:remainder]] += 1
    return prune_neurons

# Función para obtener los índices de las neuronas a podar por capa
def prune_idx (num_neurons, neurons_prune, layer, scalar,attribute):
    scores = score_layer (num_neurons, layer, scalar, attribute)
    idx_prune = np.argsort (scores)
    return idx_prune [:neurons_prune].copy()


def prune_neurons(model, idx_prune, layer):
    layers_model = model._modules["net"]

    with torch.no_grad():
        layers_model[layer].weight[idx_prune] = 0
        layers_model[layer].bias[idx_prune] = 0

    return model

def full_prune_model(model, layer_names, neurons_per_layer, percentage_prune, attribute_layer, attribute_neuron,scalar=1.0):
    neurons_prune = prune_proportion (sum(neurons_per_layer), percentage_prune, attribute_layer, layer_names)
    print ("Neurons to prune:", neurons_prune)
    for layer in layer_names:   
            idx_prune = prune_idx (int (neurons_per_layer [layer]), int (neurons_prune [layer]), layer, scalar, attribute_neuron)
            prune_neurons(model, idx_prune, layer * 2)
            print ("Layer:", layer)
            print ("Indices to prune:", idx_prune)
    return model

if __name__ == "__main__":
    x, y = load_adult()[SPLITS[SPLIT]].tensors
    y = y.numpy()
    Z = protected_attributes(SPLIT)
    model = load_model(RUN, x.shape[1])
    layer_names, neurons_per_layer, atribute_names_neurons, atribute_names_layers, _ = get_parameters()
    print ("Layer names:", layer_names)
    print ("Neurons per layer:", neurons_per_layer)
    percentage_prune = 0.2
    attribute_layer = atribute_names_layers [0]
    attribute_neuron = atribute_names_neurons [0]
    print ("Attribute layer:", attribute_layer)
    print ("Attribute neuron:", attribute_neuron)
    
    # Guardar modelo podado
    for i in range (len(atribute_names_layers)):
        attribute_layer = atribute_names_layers [i]
        attribute_neuron = atribute_names_neurons [i]
        for scalar in [0.5, 1.0, 1.5, 2.0]:
            for percentage_prune in [0.1, 0.2, 0.3, 0.4]:
                print ("Pruning with scalar:", scalar, "and percentage:", percentage_prune)
                #model = load_model(RUN, x.shape[1])
                #model = full_prune_model(model, layer_names, neurons_per_layer, percentage_prune, attribute_layer, attribute_neuron, scalar)
                save_path = Path(f"pruned_models/attribute_{attribute_layer}_scalar_{scalar}_percentage_{percentage_prune}")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                #torch.save(model.state_dict(), save_path / "pruned_model.pth")
                print(f"Modelo podado guardado en: {save_path}")
                parameters= {
                    "attribute": attribute_layer,
                    "scalar": scalar,
                    "percentage": percentage_prune
                }
                with open(save_path / "parameters.json", "w") as f:
                    json.dump(parameters, f, indent=4)