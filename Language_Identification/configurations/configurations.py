# ============================ Third Party libs ============================
import argparse
from pathlib import Path


class BaseConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def add_path(self) -> None:
        self.parser.add_argument("--raw_data_dir",
                                 type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Raw")

        self.parser.add_argument("--train_data_file",
                                 type=str,
                                 default="train.csv")

        self.parser.add_argument("--valid_data_file",
                                 type=str,
                                 default="valid.csv")

        self.parser.add_argument("--test_data_file",
                                 type=str,
                                 default="test.csv")

    def get_config(self) -> argparse.Namespace:
        self.add_path()
        return self.parser.parse_args()
