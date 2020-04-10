

## Introduction

Objective of the assignment is to predict which borrowers will experience financial distress in the next two years.

In this assignment, following steps are covered:

 - EDA
 - Training initial model (with default parameters)
 - Assessing model
 - Submitting predictions using initial model
 - Optimizing model to reach higher score in Kaggle Leaderboard
 - Submitting predictions using optimal model

## 1) Repository structure

Current repository in inspired from https://drivendata.github.io/cookiecutter-data-science/
```
/data:
  /output -> Predictions
  /raw -> Raw data (training and test datasets)
/models:
  *.pkl -> Pickled trained models / Hyperopt trials
/notebooks
  EDA.ipynb -> Use for EDA
  assess.ipynb -> Use for assessing model performance
/src
  train.py --> Train model
  predict.py --> Predict model
  optimize.py --> Run Hyperopt with TPE
Makefile -> Use to simplify pipeline execution
requirements.txt -> Packages to install
setup.py -> Use to install current package
```

## 2) Quickstart

If you would like to get started ASAP, run these make commands in the following order:<br/>
```make venv``` --> Set-up python virtual environment<br/>
```make train_def``` --> Train initial model<br/>
```make predict_def``` --> Predict using initial model <br/>
```make optimize``` --> Run Hyperparameter optimizer (Hyperopt)<br/>
```make train_opt``` --> Train model with optimized parameters<br/>
```make predict_opt``` --> Predict using optimized model

## 3) Set-up Environment

Run the following command:<br/>
```make venv```<br/>
It will install all necessary packages used in this assignment.

## 4) Exploratory Data Analysis

EDA is available in notebook ```notebooks/EDA.ipynb```.<br/>

In this notebook, we are looking there at any missing data in the dataset, what feature type is and what its distribution is. We also looking at correlation between our target class and the different features.

## 5) Train initial classifier

First model is a XGBoost model. <br/>
This model was chosen as it yields good results without much data transformation (such as normalization, clipping etc...) required. Its parameters can be found in ```params/def_xgb_model.json```. <br/>


```
{
  "name": "def_xgb_model",
  "params": {
	    "max_depth": 4,
	    "n_estimators": 100,
	    "learning_rate": 0.05,
	    "n_jobs": -1,
	    "objective": "binary:logistic",
	    "colsample_bytree": 0.5,
	    "gamma": 1
  }
}
```

### a) Training

We use 90% of the data for training and the remaining 10% for validation to assess that model doesn't overfit and generalizes well to new data.

Run either:<br/>
```make train_def```<br/>
Or:<br/>
```python3 train.py --model_json ../params/def_xgb_model.json --split_ratio 0.9```

### b) Prediction

You can now then use the model for prediction.<br/>

Run either:<br/>
```make predict_def```<br/>
Or:<br/>
```python3 predict.py --model_json ../params/def_xgb_model.json```

### c) Model assessment

Model is assessed in notebook ```notebooks/assess.ipynb```.<br/>

In this notebook, we are investigating different plots and metrics to assess model performance. We are also looking at feature importance for model interpretability.

### d) Submission results

Private Score: 0.86645 (150th)<br/>
Public Score: 0.85998 (211th)

## 6) Reaching top 100

(Done outside of the 3-hour time assignment window given)<br/>

To reach a higher score, I considered two options:<br/>

 - Hyperparameter tuning<br/>
 - Ensemble / Stacking ensemble<br/>

Because of time constraints, I opted for the first option.

### a) Run optimizer

I used ```hyperopt``` package to find a set of XGBoost parameters such as model mean validation AUC over K-Folds is maximized.
TPE algorithm is picked to search the space for the best parameters.<br/>

Search history is dumped at every round in ```models/opt_trials.pkl``` so that optimizer can be stopped and resumed  anytime.

Run either:<br/>
```make optimize```<br/>
Or:<br/>
```python3 optimize.py```<br/>

Optimizer may take a while to run.<br/>
Best set of parameters retrieved from the optimization process is the following one:

```
{
  "name": "opt_xgb_model",
  "params": {
    "booster": "gbtree",
    "colsample_bytree": 0.65,
    "eta": 0.065,
    "gamma": 0.84,
    "max_depth": 5,
    "min_child_weight": 5.0,
    "n_estimators": 130,
    "subsample": 0.84
  }
}
```

### b) Submission results

Private Score: 0.86800 (59th)<br/>
Public Score: 0.86173 (80th)
