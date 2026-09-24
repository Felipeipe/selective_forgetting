from pathlib import Path

# module that avoids all of the rewriting of the training pipeline. Does logging, automatic validation and checkpoint storing
import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torchmetrics.classification import MulticlassConfusionMatrix
from ucimlrepo import fetch_ucirepo

# for now we're training without these attributes. Should be changed according to the methodology.
PROTECTED_ATTRS = ["sex", "race", "age"]


class LitMLP(L.LightningModule):
    def __init__(self, input_dim, hidden_dims, num_classes=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        dims = [input_dim, *hidden_dims]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):  # type: ignore
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)
        self.test_cm = MulticlassConfusionMatrix(num_classes)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def _eval_step(self, batch, prefix):
        x, y = batch
        logits = self(x)
        self.log(f"{prefix}_loss", F.cross_entropy(logits, y), prog_bar=True)
        self.log(
            f"{prefix}_acc", (logits.argmax(dim=1) == y).float().mean(), prog_bar=True
        )
        return logits

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        logits = self._eval_step(batch, "test")
        self.test_cm.update(logits.argmax(dim=1), batch[1])

    def on_test_epoch_end(self):
        # rows: true class, columns: predicted class
        self.confmat = self.test_cm.compute().cpu()
        self.test_cm.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)  # type:ignore


def fetch_adult(cache=Path("data/adult.csv"), protected_attributes=PROTECTED_ATTRS):
    if not cache.exists():
        adult = fetch_ucirepo(id=2)
        cache.parent.mkdir(exist_ok=True)
        # creates a local copy of dataset
        pd.concat([adult.data.features, adult.data.targets], axis=1).to_csv(  # type:ignore
            cache, index=False
        )
    df = pd.read_csv(cache)
    print(df.columns)
    classes = protected_attributes + ["income"]
    return df.drop(columns=classes), df["income"]


def load_adult(val_frac=0.15, test_frac=0.15, seed=0):
    X, y = fetch_adult()
    y = y.str.strip().str.rstrip(".") == ">50K"

    num_cols = X.select_dtypes(include="number").columns

    cat_cols = X.select_dtypes(include="object").columns
    X[cat_cols] = X[cat_cols].fillna("?")
    X = pd.get_dummies(X, columns=list(cat_cols), dtype=float)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g).numpy()
    n_val, n_test = int(len(X) * val_frac), int(len(X) * test_frac)
    val_idx = perm[:n_val]
    test_idx = perm[n_val : n_val + n_test]
    train_idx = perm[n_val + n_test :]

    mean, std = X.iloc[train_idx][num_cols].mean(), X.iloc[train_idx][num_cols].std()
    X[num_cols] = (X[num_cols] - mean) / std

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.long)
    return (
        data.TensorDataset(X[train_idx], y[train_idx]),
        data.TensorDataset(X[val_idx], y[val_idx]),
        data.TensorDataset(X[test_idx], y[test_idx]),
    )


def plot_confusion_matrix(cm, labels, path="confusion_matrix.png"):
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    train, val, test = load_adult()
    input_dim = train.tensors[0].shape[1]

    model = LitMLP(input_dim, hidden_dims=[64, 32, 16], lr=1e-3)
    trainer = L.Trainer(max_epochs=31, log_every_n_steps=10)
    trainer.fit(
        model,
        data.DataLoader(train, batch_size=1024, shuffle=True, num_workers=0),
        data.DataLoader(val, batch_size=1024, num_workers=0),
    )
    trainer.test(model, data.DataLoader(test, batch_size=1024, num_workers=0))

    print(model.confmat)
    plot_confusion_matrix(model.confmat, ["<=50K", ">50K"])
