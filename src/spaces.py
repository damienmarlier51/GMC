from hyperopt import hp
import numpy as np

random_state = 0

et_space = {
    "n_estimators": hp.quniform("n_estimators", 100, 400, 10),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.01),
    "max_depth": hp.choice("max_depth", np.arange(1, 8, dtype=int)),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 0.1, 1, 0.01),
    "n_jobs": -1,
    "warm_start": True,
    "class_weight": "balanced",
    "verbose": 0
}

rf_space = {
    "n_estimators": hp.quniform("n_estimators", 100, 400, 10),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.01),
    "max_depth": hp.choice("max_depth", np.arange(1, 8, dtype=int)),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 0.1, 1, 0.01),
    "n_jobs": -1,
    "warm_start": True,
    "class_weight": "balanced",
    "verbose": 0
}

lg_space = {
    "penalty": hp.choice("penalty", ["l1", "l2", "elasticnet"]),
    "C": hp.quniform("C", 0.1, 1, 0.01),
    "class_weight": "balanced",
    "verbose": 0
}

xgb_space = {
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
    "job": -1,
    "seed": random_state
}

ffnn_space = {
    "fc1_dim": hp.quniform("fc1_dim", 1, 256, 1),
    "fc2_dim": hp.quniform("fc2_dim", 1, 256, 1),
    "learning_rate": hp.quniform("learning_rate", 0.0001, 0.1, 0.0001),
    "batch_size": hp.choice("batch_size", [16, 32, 64]),
    "weight_decay": hp.quniform("weight_decay", 0, 1, 0.1),
    "eps": hp.quniform("eps", 1e-8, 1e-4, 1e-4),
    "num_epochs": hp.choice("num_epochs", [10, 15, 20])
}
