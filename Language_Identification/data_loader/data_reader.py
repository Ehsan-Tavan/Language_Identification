# ============================ Third Party libs ============================
import json
import pandas as pd


def read_csv(path: str, columns: list = None, names: list = None) -> pd.DataFrame:
    dataframe = pd.read_csv(path, usecols=columns) if columns else pd.read_csv(path)
    return dataframe.rename(columns=dict(zip(columns, names))) if names else dataframe


def read_json(path):
    with open(path) as file:
        return json.load(file)
