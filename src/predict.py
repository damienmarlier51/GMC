from pathlib import Path
import pandas as pd
import pickle
import click
import json

project_dir = Path(__file__).resolve().parents[1]
raw_data_dir = "{}/{}".format(project_dir, "data/raw")
output_dir = "{}/{}".format(project_dir, "data/output")
model_dir = "{}/{}".format(project_dir, "models")
params_dir = "{}/{}".format(project_dir, "params")

test_filepath = "{}/{}".format(raw_data_dir, "cs-test.csv")
default_model_json = "{}/{}".format(params_dir, "init_model.json")

TARGET_COLUMN = "SeriousDlqin2yrs"


@click.command()
@click.option('--model_json', default=default_model_json)
def predict(model_json):
    """
    Predict class probabilities

    Parameters:
    model_json (str): Filepath to JSON containing model
    parameters
    """

    with open(model_json, "r") as f:
        model_dict = json.loads(f.read())

    model_name = model_dict["name"]
    model_filepath = "{}/{}.pkl".format(model_dir, model_name)
    output_filepath = "{}/{}_predictions.csv".format(output_dir, model_name)

    model = pickle.load(open(model_filepath, "rb"))
    df_test = pd.read_csv(test_filepath, index_col=0)
    df_test.fillna(value=-1, inplace=True)

    columns = df_test.columns.tolist()
    feature_columns = sorted([column for column in columns if column != TARGET_COLUMN])

    X_test = df_test[feature_columns].values
    Y_pred = model.predict_proba(X_test)[:, 1]

    df_test["Probability"] = Y_pred
    df_test = df_test[["Probability"]].reset_index()
    df_test.columns = ["Id", "Probability"]

    df_test.to_csv(output_filepath, index=None)


if __name__ == "__main__":
    predict()
