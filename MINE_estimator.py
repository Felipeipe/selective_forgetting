import numpy as np
from pathlib import Path
from torch_mist import estimate_mi
from sklearn.datasets import load_iris
# module that avoids all of the rewriting of the training pipeline. Does logging, automatic validation and checkpoint storing
import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
from main import LitMLP, fetch_adult, load_adult, split_indices
from results import FIG_DIR, run_dirs
from sklearn.preprocessing import OneHotEncoder
# Load the Iris Dataset as a pandas DataFrame
@torch.no_grad()
def output_for_layer(run, test):
    ckpt = next((run / "checkpoints").glob("*.ckpt"))
    model = LitMLP.load_from_checkpoint(ckpt, map_location="cpu").eval()
    x, _ = test.tensors
    x1 = model.net [1] (model.net[0](x))
    x2 = model.net [3] (model.net[2](x1))
    x3 = model.net [5] (model.net[4](x2))
    x4 = model.net (x)
    return x1.numpy(), x2.numpy(), x3.numpy(), x4.numpy()

def test_money():
    # money of each test row, using the same split as load_adult
    _, y = fetch_adult(cache=Path("data/adult.csv"), protected_attributes=[])
    y = y.str.strip().str.rstrip(".") == ">50K"
    _, _, test_idx = split_indices(len(y))
    return y.to_numpy()[test_idx].astype(int)



def test_sex():
    # sex of each test row, using the same split as load_adult
    X, _ = fetch_adult(cache=Path("data/adult.csv"), protected_attributes=[])
    _, _, test_idx = split_indices(len(X))
    X.loc[X["sex"] == "Mujeres", "sex"] = "Female"
    X.loc[X["sex"] == "Hombres", "sex"] = "Male"
    return X["sex"].to_numpy()[test_idx]

def test_race():
    # race of each test row, using the same split as load_adult
    X, _ = fetch_adult(cache=Path("data/adult.csv"), protected_attributes=[])
    _, _, test_idx = split_indices(len(X))
    return X["race"].to_numpy()[test_idx]

def test_age():
    # age of each test row, using the same split as load_adult
    X, _ = fetch_adult(cache=Path("data/adult.csv"), protected_attributes=[])
    _, _, test_idx = split_indices(len(X))
    return X["age"].to_numpy()[test_idx]

def test_native_country():
    # native country of each test row, using the same split as load_adult
    X, _ = fetch_adult(cache=Path("data/adult.csv"), protected_attributes=[])
    _, _, test_idx = split_indices(len(X))
    return X["native-country"].to_numpy()[test_idx]

def encoded_list (array):
    keys_list = list(dict.fromkeys(array))
    keys_list = [[key] for key in keys_list]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(keys_list)
    encoded_array = enc.transform(array.reshape(-1, 1)).toarray()
    return encoded_array.tolist()


def estimador_MINE (data, x_key, y_key):
    output = estimate_mi(
        data=data,          # The dataset (as a pandas.DataFrame, many other formats are supported)
        x_key=x_key,  # Consider the 'petal length' column as x
        y_key=y_key,   # And the 'petal width` as y
        estimator='mine',           # Use the MINE mutual information estimator
        max_iterations=1000,        # Number of maximum train iterations 
        hidden_dims=[256, 128, 64],          # Hidden layers for the critic architecture.
    )
    return output
    

if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = run_dirs()
    print(f"{len(runs)} runs: {[r.name for r in runs]}")

    train, val, test = load_adult()
    X1_list, X2_list, X3_list, X4_list = [], [], [], []
    print ("runs length:",runs)
    x1, x2, x3, x4 = output_for_layer(runs [0], test)
    print ("Shape primer elemento:",x1[0].shape)
    for i in range(len(x1)):
        X1_list.append(x1[i])
        X2_list.append(x2[i])
        X3_list.append(x3[i])
        X4_list.append(x4[i])

    gender = test_sex()
    race = test_race()
    age = test_age()
    money = test_money()
    native_country = test_native_country()
    df = pd.DataFrame({'Layer1': X1_list, 'Layer2': X2_list, 'Layer3': X3_list, 'Layer4': X4_list, 'Gender': encoded_list(gender), 'Race': encoded_list(race), 'Age': list (age), 'Native Country': encoded_list(native_country), "Money": list (money)})
    print(df)

    output1_money = estimador_MINE (df, 'Layer1', 'Money')
    output2_money = estimador_MINE (df, 'Layer2', 'Money')
    output3_money = estimador_MINE (df, 'Layer3', 'Money')
    output4_money = estimador_MINE (df, 'Layer4', 'Money')

    output1_gender = estimador_MINE (df, 'Layer1', 'Gender')
    output2_gender = estimador_MINE (df, 'Layer2', 'Gender')
    output3_gender = estimador_MINE (df, 'Layer3', 'Gender')
    output4_gender = estimador_MINE (df, 'Layer4', 'Gender')

    output1_race = estimador_MINE (df, 'Layer1', 'Race')
    output2_race = estimador_MINE (df, 'Layer2', 'Race')
    output3_race = estimador_MINE (df, 'Layer3', 'Race')
    output4_race = estimador_MINE (df, 'Layer4', 'Race')

    output1_age = estimador_MINE (df, 'Layer1', 'Age')
    output2_age = estimador_MINE (df, 'Layer2', 'Age')
    output3_age = estimador_MINE (df, 'Layer3', 'Age')
    output4_age = estimador_MINE (df, 'Layer4', 'Age')
    
    print ("Estimador MINE para Layer1 y Gender:", output1_gender [0])
    print ("Estimador MINE para Layer2 y Gender:", output2_gender [0])
    print ("Estimador MINE para Layer3 y Gender:", output3_gender [0])
    print ("Estimador MINE para Layer4 y Gender:", output4_gender [0])
    
    print ("Estimador MINE para Layer1 y Race:", output1_race [0])
    print ("Estimador MINE para Layer2 y Race:", output2_race [0])
    print ("Estimador MINE para Layer3 y Race:", output3_race [0])
    print ("Estimador MINE para Layer4 y Race:", output4_race [0])

    print ("Estimador MINE para Layer1 y Age:", output1_age [0])
    print ("Estimador MINE para Layer2 y Age:", output2_age [0])
    print ("Estimador MINE para Layer3 y Age:", output3_age [0])
    print ("Estimador MINE para Layer4 y Age:", output4_age [0])

    print ("Estimador MINE para Layer1 y Money:", output1_money [0])
    print ("Estimador MINE para Layer2 y Money:", output2_money [0])
    print ("Estimador MINE para Layer3 y Money:", output3_money [0])
    print ("Estimador MINE para Layer4 y Money:", output4_money [0])

    layer = ['Layer1', 'Layer2', 'Layer3', 'Output Layer']
    output_gender = [next(iter(output1_gender[0].values())), next(iter(output2_gender[0].values())), next(iter(output3_gender[0].values())), next(iter(output4_gender[0].values()))]
    output_race = [next(iter(output1_race[0].values())), next(iter(output2_race[0].values())), next(iter(output3_race[0].values())), next(iter(output4_race[0].values()))]
    output_age = [next(iter(output1_age[0].values())), next(iter(output2_age[0].values())), next(iter(output3_age[0].values())), next(iter(output4_age[0].values()))]
    output_money = [next(iter(output1_money[0].values())), next(iter(output2_money[0].values())), next(iter(output3_money[0].values())), next(iter(output4_money[0].values()))]

    fig, ax = plt.subplots()
    ax.plot(layer, output_gender, marker='o', label="Gender")
    ax.plot(layer, output_race, marker='s', label="Race")
    ax.plot(layer, output_age, marker='^', label="Age")
    ax.plot(layer, output_money, marker='d', label="Money")
    ax.set_title("Basic Components of Matplotlib Figure")
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")
    ax.legend() 
    plt.show()