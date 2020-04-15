from pathlib import Path
from model import Model
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import click
import json
import sys

project_dir = Path(__file__).resolve().parents[1]

model_dir = "{}/{}".format(project_dir, "models")
params_dir = "{}/{}".format(project_dir, "params")
raw_dir = "{}/{}".format(project_dir, "data/raw")

default_model_json = "{}/{}".format(params_dir, "init_model.json")

TARGET_COLUMN = "SeriousDlqin2yrs"


@click.command()
@click.option("--model_json", default=default_model_json)
@click.option("--split_ratio", default=1, type=float)
def train(model_json,
          split_ratio):
    """
    Train a model given its parameters and train/validation sets

    Parameters:
    model_json (str): Filepath to JSON containing model
    split_ratio (float): Gives split proportion to generate
    train and validation datasets
    """

    dataset_filepath = "{}/{}".format(raw_dir, "cs-training.csv")

    with open(model_json, "r") as f:
        model_dict = json.loads(f.read())
    model = Model.get_model_from_dict(model_dict)

    model_filepath = "{}/{}.pkl".format(model_dir, model.name)

    df_dataset = pd.read_csv(dataset_filepath, index_col=0)
    df_dataset.fillna(value=-1, inplace=True)
    df_dataset = df_dataset.sample(frac=1, random_state=0)

    columns = df_dataset.columns.tolist()
    feature_columns = sorted([column for column in columns if column != TARGET_COLUMN])

    # Normalize inputs
    df_X = df_dataset[feature_columns]
    df_X.fillna(value=0, inplace=True)
    df_X = (df_X - df_X.mean(axis=0)) / df_X.std(axis=0)

    X = df_X.values
    Y = df_dataset[TARGET_COLUMN].values

    train_idxs = [0, int((split_ratio) * df_dataset.shape[0])]
    validation_idxs = [int((split_ratio) * df_dataset.shape[0]), df_dataset.shape[0]]

    X_train = X[train_idxs[0]:train_idxs[1]]
    y_train = Y[train_idxs[0]:train_idxs[1]]
    X_validation = X[validation_idxs[0]:validation_idxs[1]]
    y_validation = Y[validation_idxs[0]:validation_idxs[1]]

    if model.clf == xgb.XGBClassifier:
        scale_pos_weight = y_train[np.where(y_train == 0)[0]].shape[0] / y_train[np.where(y_train == 1)[0]].shape[0]
        model.params.update({"scale_pos_weight": scale_pos_weight})

    model, AUCs = model.fit(X=X_train,
                            y=y_train,
                            eval_set=[(X_train, y_train),
                                      (X_validation, y_validation)])

    pickle.dump(model, open(model_filepath, "wb"))


if __name__ == "__main__":
    train()
