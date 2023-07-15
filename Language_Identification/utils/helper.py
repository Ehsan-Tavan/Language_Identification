# ============================ Third Party libs ============================
import os
from typing import List, Tuple
import logging
import pandas
from sklearn import preprocessing
# ============================ My packages ============================
from Language_Identification.data_loader import read_csv, write_csv
from Language_Identification.data_preparation import normalize_text
from .InputExample import InputExample

logging.basicConfig(level=logging.DEBUG)


def load_dataset(processed_train_file: str,
                 processed_valid_file: str,
                 processed_test_file: str,
                 raw_train_file: str,
                 raw_valid_file: str,
                 raw_test_file: str) -> Tuple[pandas.DataFrame,
                                              pandas.DataFrame,
                                              pandas.DataFrame]:
    if os.path.exists(processed_train_file) and os.path.exists(
            processed_valid_file) and os.path.exists(processed_test_file):
        logging.info("Loading processed data ...")
        train_data = read_csv(processed_train_file)
        valid_data = read_csv(processed_valid_file)
        test_data = read_csv(processed_test_file)
    else:
        logging.info("Loading raw data ...")
        train_data = read_csv(raw_train_file)
        valid_data = read_csv(raw_valid_file)
        test_data = read_csv(raw_test_file)

        logging.info("Normalizing all data ...")
        train_data["text"] = train_data["text"].apply(normalize_text)
        valid_data["text"] = valid_data["text"].apply(normalize_text)
        test_data["text"] = test_data["text"].apply(normalize_text)

        logging.info("Saving processed data ...")
        write_csv(train_data, path=processed_train_file)
        write_csv(valid_data, path=processed_valid_file)
        write_csv(test_data, path=processed_test_file)
    return train_data, valid_data, test_data


def encode_labels(labels: List[str]):
    label2index, index2label = {}, {}
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)

    for label in label_encoder.classes_:
        label2index[label] = int(label_encoder.transform([label])[0])
        index2label[int(label_encoder.transform([label])[0])] = label

    return encoded_labels, label2index, index2label, label_encoder


def prepare_example(data_frame, mode: str = "train") -> list:
    data = []
    if mode in ["train", "test"]:
        for _, row in data_frame.iterrows():
            data.append(InputExample(text=row["text"],
                                     label=row["labels"]))
    elif mode == "inference":
        for _, row in data_frame.iterrows():
            data.append(InputExample(text=row["text"]))
    return data
