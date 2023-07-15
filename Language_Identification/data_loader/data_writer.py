# ============================ Third Party libs ============================
import json
import pandas as pd


def write_csv(dataframe: pd.DataFrame, path: str) -> None:
    dataframe.to_csv(path, index=False)


def write_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf8") as outfile:
        json.dump(data, outfile)
