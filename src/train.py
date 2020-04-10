from pathlib import Path
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import click
import json

project_dir = Path(__file__).resolve().parents[1]

model_dir = "{}/{}".format(project_dir, "models")
params_dir = "{}/{}".format(project_dir, "params")
raw_dir = "{}/{}".format(project_dir, "data/raw")

default_model_json = "{}/{}".format(params_dir, "init_model.json")

TARGET_COLUMN = "SeriousDlqin2yrs"


@click.command()
@click.option('--model_json', default=default_model_json)
@click.option('--split_ratio', default=1, type=float)
def train(model_json,
          split_ratio):

    dataset_filepath = "{}/{}".format(raw_dir, "cs-training.csv")

    with open(model_json, "r") as f:
        model_dict = json.loads(f.read())

    model_name = model_dict["name"]
    model_filepath = "{}/{}.pkl".format(model_dir, model_name)
    params_dict = model_dict["params"]

    df_dataset = pd.read_csv(dataset_filepath, index_col=0)
    df_dataset = df_dataset.sample(frac=1, random_state=0)

    columns = df_dataset.columns.tolist()
    feature_columns = [column for column in columns if column != TARGET_COLUMN]

    X = df_dataset[feature_columns].values
    Y = df_dataset[TARGET_COLUMN].values

    train_idxs = [0, int((split_ratio) * df_dataset.shape[0])]
    validation_idxs = [int((split_ratio) * df_dataset.shape[0]), df_dataset.shape[0]]

    X_train = X[train_idxs[0]:train_idxs[1]]
    y_train = Y[train_idxs[0]:train_idxs[1]]
    X_validation = X[validation_idxs[0]:validation_idxs[1]]
    y_validation = Y[validation_idxs[0]:validation_idxs[1]]

    model, train_AUC, val_AUC = train_model(X_train=X_train,
                                            y_train=y_train,
                                            X_validation=X_validation,
                                            y_validation=y_validation,
                                            params_dict=params_dict)

    print("Train AUC: {}".format(train_AUC))
    print("Validation AUC: {}".format(val_AUC))

    pickle.dump(model, open(model_filepath, "wb"))


def train_model(X_train,
                y_train,
                X_validation,
                y_validation,
                params_dict,
                verbose=True):

    eval_set = [(X_train, y_train)]
    if X_validation.shape[0] > 0:  # In case split_ratio is 1 and we are not keeping any data for validation
        eval_set.append((X_validation, y_validation))

    eval_metric = ["auc", "error"]
    scale_pos_weight = y_train[np.where(y_train == 0)[0]].shape[0] / y_train[np.where(y_train == 1)[0]].shape[0]

    params_dict["scale_pos_weight"] = scale_pos_weight

    model = xgb.XGBClassifier(**params_dict) \
               .fit(X=X_train,
                    y=y_train,
                    eval_metric=eval_metric,
                    eval_set=eval_set,
                    verbose=verbose)

    train_AUC = model.__dict__["evals_result_"]["validation_0"]["auc"][-1]
    val_AUC = 0
    if X_validation.shape[0] > 0:
        val_AUC = model.__dict__["evals_result_"]["validation_1"]["auc"][-1]

    return model, train_AUC, val_AUC


if __name__ == "__main__":
    train()
