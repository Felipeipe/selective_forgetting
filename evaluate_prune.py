from main import LitMLP, load_adult, plot_confusion_matrix
import torch
from torch.utils import data
import lightning as L
from pathlib import Path
from mutual_information import HEATMAP_FIG, SPLIT, SPLITS, mi_table, output_mi_table, protected_attributes, plot_mi_heatmap, mine_table, plot_mine
import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT_DIR = Path("pruned_models")


def load_final_model(model, path):
    print ("loading model from", path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


if __name__ == "__main__":
    train, val, test = load_adult()
    input_dim = train.tensors[0].shape[1]

    trainer = L.Trainer(accelerator="auto", devices="auto", logger=False)

    x, y = load_adult()[SPLITS[SPLIT]].tensors
    y = y.numpy()
    Z = protected_attributes(SPLIT)

    for RUN in ROOT_DIR.rglob("pruned_model.pth"):

        OUT_DIR = RUN.parent
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        NEURON_CSV = OUT_DIR / "mi_per_neuron.csv"
        OUTPUT_CSV = OUT_DIR / "mi_output.csv"
        MINE_CSV = OUT_DIR / "mi_mine_por_capa.csv"
        HEATMAP_FIG = OUT_DIR / "figures/mi_heatmap.pdf"
        MINE_FIG = OUT_DIR / "figures/mi_mine_por_capa.pdf"
        HEATMAP_FIG.parent.mkdir(parents=True, exist_ok=True)

        model = LitMLP(input_dim, hidden_dims=[64, 32, 16], lr=1e-3)
        model = load_final_model (model, RUN)

        results = trainer.test(model=model, dataloaders=data.DataLoader( test, batch_size=1024,num_workers=0))
        cm = model.confmat.cpu().numpy()

        y_true = []
        y_pred = []
    
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                y_true.extend([i] * cm[i, j])
                y_pred.extend([j] * cm[i, j])
    
        accuracy = np.trace(cm) / np.sum(cm)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        
        metrics = {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "f1_macro": float(f1_macro),
            "confusion_matrix": cm.tolist()
        }

        with open(OUT_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(model.confmat)
        plot_confusion_matrix(
            model.confmat,
            ["<=50K", ">50K"],
            path=OUT_DIR / "confusion_matrix.pdf",  # type:ignore
        )

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
