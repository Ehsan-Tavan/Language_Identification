# ============================ Third Party libs ============================
import os
import logging
import torch
# ============================ My packages ============================
from Language_Identification.configurations import BaseConfig
from Language_Identification.data_loader import write_json
from Language_Identification.utils import prepare_example, encode_labels, load_dataset
from Language_Identification.models.lm_model import LmModel

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    TRAIN_DATA, VALID_DATA, TEST_DATA = load_dataset(
        processed_train_file=os.path.join(ARGS.processed_data_dir, ARGS.train_data_file),
        processed_valid_file=os.path.join(ARGS.processed_data_dir, ARGS.valid_data_file),
        processed_test_file=os.path.join(ARGS.processed_data_dir, ARGS.test_data_file),
        raw_train_file=os.path.join(ARGS.raw_data_dir, ARGS.train_data_file),
        raw_valid_file=os.path.join(ARGS.raw_data_dir, ARGS.valid_data_file),
        raw_test_file=os.path.join(ARGS.raw_data_dir, ARGS.test_data_file)
    )

    logging.info("Train set contain %s sample ...", len(TRAIN_DATA))
    logging.info("Valid set contain %s sample ...", len(VALID_DATA))
    logging.info("Test set contain %s sample ...", len(TEST_DATA))

    logging.info("Normalizing all data ...")
    encoded_labels, label2index, index2label, label_encoder = encode_labels(
        labels=list(TRAIN_DATA.labels))
    write_json(label2index, path=os.path.join(ARGS.assets_dir, ARGS.label2index_file))
    write_json(index2label, path=os.path.join(ARGS.assets_dir, ARGS.index2label_file))
    TRAIN_DATA.labels = encoded_labels
    VALID_DATA.labels = label_encoder.transform(list(VALID_DATA.labels))
    TEST_DATA.labels = label_encoder.transform(list(TEST_DATA.labels))

    TRAIN_SAMPLES = prepare_example(TRAIN_DATA)
    VALID_SAMPLES = prepare_example(VALID_DATA)
    TEST_SAMPLES = prepare_example(TEST_DATA)

    MODEL = LmModel(model_path=ARGS.lm_path, config=ARGS, loss_fct=torch.nn.CrossEntropyLoss(),
                    pooling_methods=[ARGS.pooling_method], num_classes=len(label2index),
                    train_data=TRAIN_SAMPLES, dev_data=VALID_SAMPLES, test_data=TEST_SAMPLES)

    MODEL.fit(check_point_monitor="dev_loss", check_point_mode="min",
              early_stopping_monitor="dev_loss", early_stopping_patience=5)
    MODEL.test()
