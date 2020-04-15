import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import (RandomForestClassifier,
                              ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from typing import get_type_hints
import numpy as np
import inspect


class Model(object):

    def __init__(self,
                 name,
                 clf,
                 seed=0,
                 params={}):
        params["random_state"] = seed
        self.name = name
        self.params = params
        self.clf = clf
        self.fitted_model = None

    @staticmethod
    def get_model_from_dict(model_dict):

        name = model_dict["name"]
        clf = model_dict["clf"]
        params = model_dict["params"]

        if clf == "xgb":
            clf = XGBClassifier
        elif clf == "rf":
            clf = RandomForestClassifier
        elif clf == "lg":
            clf = LogisticRegression
        elif clf == "ffnn":
            clf = FFNN
        elif clf == "et":
            clf = ExtraTreesClassifier

        return Model(name=name,
                     clf=clf,
                     params=params)

    def fit(self, X, y, eval_set=[]):

        args = {}
        args["X"] = X
        args["y"] = y

        model = self.clf(**self.params)
        if "eval_set" in inspect.getargspec(model.fit).args:
            args.update({"eval_set": eval_set})
        print(inspect.getargspec(model.fit).args)
        if "eval_metric" in inspect.getargspec(model.fit).args:
            args.update({"eval_metric": ["auc"]})

        print(model)

        model = model.fit(**args)

        AUCs = []
        for i, eval_tuple in enumerate(eval_set):
            y_pred = model.predict_proba(eval_tuple[0])
            AUC = roc_auc_score(y_true=eval_tuple[1],
                                y_score=y_pred[:, 1])
            AUCs.append(AUC)

        print("AUCs: {}".format(AUCs))

        self.fitted_model = model

        return model, AUCs

    def predict(self, X):
        return self.fitted_model.predict(X)

    def predict_proba(self, X):
        return self.fitted_model.predict_proba(X)


class PrepareData(Dataset):

    def __init__(self, X, y):

        self.X = torch.from_numpy(X).type(torch.FloatTensor)
        self.y = torch.from_numpy(y).type(torch.FloatTensor)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FFNN(torch.nn.Module):

    def __init__(self,
                 input_dim=10,
                 output_dim=2,
                 fc1_dim=10,
                 fc2_dim=10,
                 num_epochs=20,
                 batch_size=16,
                 learning_rate=0.001,
                 eps=1e-8,
                 weight_decay=0,
                 random_state=0):

        super().__init__()

        self.fc1 = nn.Linear(in_features=input_dim, out_features=fc1_dim, bias=True)
        self.fc2 = nn.Linear(in_features=fc1_dim, out_features=fc2_dim, bias=True)
        self.fc3 = nn.Linear(in_features=fc2_dim, out_features=2, bias=True)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def forward(self, x):

        x = F.relu(input=self.fc1(x))
        x = F.relu(input=self.fc2(x))
        x = self.fc3(x)

        return x

    def predict_proba(self, X):

        self.eval()
        with torch.no_grad():
            yhat = F.softmax(self(torch.from_numpy(X).type(torch.FloatTensor)), dim=1).numpy()
        return yhat

    def fit(self, X, y, eval_set=[]):

        num_epochs = self.num_epochs
        batch_size = self.batch_size
        lr = self.learning_rate
        eps = self.eps
        weight_decay = self.weight_decay

        _y = np.zeros((y.shape[0], 2))
        _y[[np.where(y == 0)[0]], 0] = 1
        _y[[np.where(y == 1)[0]], 1] = 1

        pos_weight = y[np.where(y == 0)[0]].shape[0] / y[np.where(y == 1)[0]].shape[0]
        ds = PrepareData(X=X, y=_y)
        ds = DataLoader(ds, batch_size=batch_size, shuffle=True)

        cost_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1, pos_weight]))
        optimizer = torch.optim.Adam(params=self.parameters(),
                                     lr=lr,
                                     eps=eps,
                                     weight_decay=weight_decay,
                                     amsgrad=True)

        self.eval()
        with torch.no_grad():
            yhat = self(torch.from_numpy(X).type(torch.FloatTensor))
            _y = torch.from_numpy(_y).type(torch.FloatTensor)
            loss = cost_func(yhat, _y)

        print("Init loss: {}".format(loss.item()))

        for e in range(num_epochs):

            losses = []

            self.train()

            for ix, (_x, _y) in enumerate(ds):

                optimizer.zero_grad()

                yhat = self(_x)

                loss = cost_func(yhat, _y)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            print("Nb batches: {}".format(len(losses)))

            AUCs = []
            for i, eval_tuple in enumerate(eval_set):
                y_pred = self.predict_proba(eval_tuple[0])
                y_pred = np.concatenate((y_pred, eval_tuple[1].reshape(-1, 1)), axis=1)
                AUC = roc_auc_score(y_true=eval_tuple[1],
                                    y_score=y_pred[:, 1])
                AUCs.append(AUC)

            print("[{}/{}], loss: {}, AUCs: {}".format(e,
                                                       num_epochs,
                                                       np.mean(losses),
                                                       AUCs))

        return self
