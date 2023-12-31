# ============================ Third Party libs ============================
import argparse
from pathlib import Path


class BaseConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--model_name",
                                 type=str,
                                 default="xlm-roberta-base")

        self.parser.add_argument("--train_batch_size",
                                 type=int,
                                 default=128)

        self.parser.add_argument("--inference_batch_size",
                                 type=int,
                                 default=512)

        self.parser.add_argument("--n_epochs",
                                 type=int,
                                 default=50)

        self.parser.add_argument("--pooling_method",
                                 type=str,
                                 default="mean")

        self.parser.add_argument("--save_top_k",
                                 type=int,
                                 default=1)

        self.parser.add_argument("--num_workers",
                                 type=int,
                                 default=4)

        self.parser.add_argument("--max_len",
                                 type=int,
                                 default=50)

        self.parser.add_argument("--lr",
                                 type=float,
                                 default=2e-5)

        self.parser.add_argument("--device",
                                 type=str,
                                 default="cuda:0")

        self.parser.add_argument("--ml_classifier",
                                 type=str,
                                 help="GaussianNB or SVC",
                                 default="GaussianNB")
        self.parser.add_argument("--ml_vectorizer",
                                 type=str,
                                 help="CountVectorizer or TfidfVectorizer",
                                 default="TfidfVectorizer")

        self.parser.add_argument("--use_char",
                                 type=bool,
                                 default=True)

        self.parser.add_argument("--using_char_threshold",
                                 type=float,
                                 default=0.2)

        self.parser.add_argument("--use_token_length",
                                 type=bool,
                                 default=True)

        self.parser.add_argument("--using_token_length_threshold",
                                 type=float,
                                 default=0.2)

        self.parser.add_argument("--char_embedding_dim",
                                 type=int,
                                 default=20)

        self.parser.add_argument("--n_filters",
                                 type=int,
                                 default=64)
        self.parser.add_argument("--filter_sizes",
                                 type=list,
                                 default=[3, 5, 7])

    def add_path(self) -> None:
        self.parser.add_argument("--assets_dir",
                                 type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/")

        self.parser.add_argument("--raw_data_dir",
                                 type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Raw")

        self.parser.add_argument("--processed_data_dir",
                                 type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Processed")

        self.parser.add_argument("--plots_dir",
                                 type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/plots")
        self.parser.add_argument("--sample_length_analysis_folder",
                                 type=str,
                                 default="sample_length")

        self.parser.add_argument("--label2index_file",
                                 type=str,
                                 default="label2index.json")

        self.parser.add_argument("--index2label_file",
                                 type=str,
                                 default="index2label.json")

        self.parser.add_argument("--train_data_file",
                                 type=str,
                                 default="train.csv")

        self.parser.add_argument("--valid_data_file",
                                 type=str,
                                 default="valid.csv")

        self.parser.add_argument("--test_data_file",
                                 type=str,
                                 default="test.csv")

        self.parser.add_argument("--lm_path",
                                 type=str,
                                 default="/mnt/disk2/LanguageModels/mt5-base-en")

        self.parser.add_argument("--saved_model_dir",
                                 type=str,
                                 default=Path(__file__).parents[
                                             2].__str__() + "/assets/saved_models/")

        self.parser.add_argument("--best_model_path",
                                 type=str,
                                 default="mt5-base-en/mean/checkpoints/"
                                         "QTag-epoch=07-dev_loss=0.02.ckpt")

    def get_config(self) -> argparse.Namespace:
        self.add_path()
        return self.parser.parse_args()
