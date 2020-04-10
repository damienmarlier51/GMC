.PHONY: train_def predict_def run_def train_opt predict_opt run_opt optimize

VENV_NAME = venv_dm_fp

# Makefile
venv: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: requirements.txt
	test -d $(VENV_NAME) || virtualenv $(VENV_NAME)
	$(VENV_NAME)/bin/pip install -r requirements.txt
	$(VENV_NAME)/bin/python3 setup.py develop
	ipython kernel install --user --name=$(VENV_NAME)
	touch $(VENV_NAME)/bin/activate


train_def:
	cd src; ../$(VENV_NAME)/bin/python3 train.py --model_json ../params/def_xgb_model.json \
																						   --split_ratio 0.9

predict_def:
	cd src; ../$(VENV_NAME)/bin/python3 predict.py --model_json ../params/def_xgb_model.json

optimize:
	cd src; ../$(VENV_NAME)/bin/python3 optimize.py

train_opt:
	cd src; ../$(VENV_NAME)/bin/python3 train.py --model_json ../params/opt_xgb_model.json \
																							 --split_ratio 1

predict_opt:
	cd src; ../$(VENV_NAME)/bin/python3 predict.py --model_json ../params/opt_xgb_model.json

run_def: train_def predict_def

run_opt: train_opt predict_opt
