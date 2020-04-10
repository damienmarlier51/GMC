from pathlib import Path
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, \
                     Trials, \
                     fmin, \
                     hp, \
                     tpe
from src.train import train_model
from sklearn.model_selection import KFold
import pickle
import os

project_dir = Path(__file__).resolve().parents[1]

model_dir = "{}/{}".format(project_dir, "models")
params_dir = "{}/{}".format(project_dir, "params")
raw_dir = "{}/{}".format(project_dir, "data/raw")

dataset_filepath = "{}/{}".format(raw_dir, "cs-training.csv")
default_model_json = "{}/{}".format(params_dir, "init_model.json")
trials_filepath = "{}/{}".format(model_dir, "opt_trials.pkl")

NFOLDS = 10
TARGET_COLUMN = "SeriousDlqin2yrs"

df_dataset = pd.read_csv(dataset_filepath, index_col=0)
df_dataset.fillna(value=-1, inplace=True)
df_dataset = df_dataset.sample(frac=1, random_state=0)

columns = df_dataset.columns.tolist()
feature_columns = sorted([column for column in columns if column != TARGET_COLUMN])

X = df_dataset[feature_columns].values
Y = df_dataset[TARGET_COLUMN].values

trials = Trials()
max_mean_val_AUC = 0
best_params = {}

if os.path.exists(trials_filepath) is True:

    trials = pickle.load(open(trials_filepath, "rb"))
    trial_list = [trial for i, trial in enumerate(trials) if "loss" in trial["result"]]

    if len(trial_list) > 0:

        best_trial = trial_list[np.argmin([trial["result"]["loss"] for trial in trial_list])]
        max_mean_val_AUC = 1 - best_trial["result"]["loss"]
        best_params = pd.DataFrame.from_dict(best_trial["misc"]["vals"]) \
                                  .to_dict(orient="records")[0]

        print("Use existing Trial object")
        print("Nb trials executed: {}".format(len(trials)))
        print("Current Max Mean Val AUC: {}".format(max_mean_val_AUC))
        print("Current Best params: {}".format(best_params))


def get_model_loss(params_dict):
    """
    Train model and compute loss (1-AUC) on fold validation sets

    Parameters:
    params_dict (dict): Model parameters

    Returns:
    dict: Dict with loss and round status
    """

    global max_mean_val_AUC
    global best_params

    params_dict["n_estimators"] = int(params_dict["n_estimators"])

    kf = KFold(n_splits=NFOLDS)

    sum_val_AUC = 0

    for i, (train_idxs, validation_idxs) in enumerate(kf.split(X)):

        X_train, X_validation = X[train_idxs], X[validation_idxs]
        y_train, y_validation = Y[train_idxs], Y[validation_idxs]

        model, train_AUC, val_AUC = train_model(X_train=X_train,
                                                y_train=y_train,
                                                X_validation=X_validation,
                                                y_validation=y_validation,
                                                params_dict=params_dict,
                                                verbose=False)

        sum_val_AUC += val_AUC

        print("Fold {} Val AUC: {}".format(i, val_AUC))

    mean_val_AUC = sum_val_AUC / NFOLDS

    loss = 1 - mean_val_AUC

    pickle.dump(trials, open(trials_filepath, "wb"))

    if max_mean_val_AUC < mean_val_AUC:
        max_mean_val_AUC = mean_val_AUC
        best_params = params_dict

    print("Round Mean Val AUC: {}".format(mean_val_AUC))
    print("Max Mean Val AUC: {}".format(max_mean_val_AUC))
    print("Best params: {}".format(best_params))
    print("Nb trials executed: {}".format(len(trials)))

    return {"loss": loss, "status": STATUS_OK}


def optimize(random_state=0):
    """
    Run parameter optimization

    Parameters:
    random_state (int): To enable reproducibility

    Returns:
    dict: Best set of parameters
    """

    space = {
        "n_estimators": hp.quniform("n_estimators", 100, 400, 10),
        "eta": hp.quniform("eta", 0.01, 0.5, 0.001),
        "max_depth":  hp.choice("max_depth", np.arange(1, 8, dtype=int)),
        "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
        "subsample": hp.quniform("subsample", 0.5, 1, 0.01),
        "gamma": hp.quniform("gamma", 0.5, 1, 0.01),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
        "eval_metric": "auc",
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "exact",
        "silent": 1,
        "job": 1,
        "seed": random_state
    }

    best = fmin(fn=get_model_loss,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=100)

    pickle.dump(trials, open(trials_filepath, "wb"))

    return best


if __name__ == "__main__":

    best = optimize()
