# ============================ Third Party libs ============================
import os
import logging
import torch
# ============================ My packages ============================
from Language_Identification.configurations import BaseConfig
from Language_Identification.data_loader import write_json
from Language_Identification.utils import prepare_example, encode_labels, load_dataset, \
    calculate_token_length
from Language_Identification.models.lm_model import Classifier
from Language_Identification.utils import Trainer
from Language_Identification.data_preparation import TokenIndexer

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

    CHARACTERS = [char for sample in TRAIN_DATA.text for char in str(sample)]
    CHAR_INDEXER = TokenIndexer(vocabs=CHARACTERS)
    CHAR_INDEXER.build_vocab2idx()
    CHAR_INDEXER.build_idx2vocab()

    CHAR_INDEXER.save(vocab2idx_path=os.path.join(ARGS.assets_dir, "char2idx.json"),
                      idx2vocab_path=os.path.join(ARGS.assets_dir, "idx2char.json"))

    TRAIN_CHARS = [[char for char in str(sample)] for sample in TRAIN_DATA.text]
    VALID_CHARS = [[char for char in str(sample)] for sample in VALID_DATA.text]
    TEST_CHARS = [[char for char in str(sample)] for sample in TEST_DATA.text]

    TRAIN_CHARS = CHAR_INDEXER.convert_samples_to_token_indexes(TRAIN_CHARS)
    VALID_CHARS = CHAR_INDEXER.convert_samples_to_token_indexes(VALID_CHARS)
    TEST_CHARS = CHAR_INDEXER.convert_samples_to_token_indexes(TEST_CHARS)

    TRAIN_DATA_TOKEN_LENGTH = calculate_token_length(list(TRAIN_DATA.text), list(TRAIN_DATA.labels))
    VALID_DATA_TOKEN_LENGTH = calculate_token_length(list(VALID_DATA.text), list(VALID_DATA.labels))
    TEST_DATA_TOKEN_LENGTH = calculate_token_length(list(TEST_DATA.text), list(TEST_DATA.labels))

    ENCODED_LABELS, LABEL2INDEX, INDEX2LABEL, LABEL_ENCODER = encode_labels(
        labels=list(TRAIN_DATA.labels))
    write_json(LABEL2INDEX, path=os.path.join(ARGS.assets_dir, ARGS.label2index_file))
    write_json(INDEX2LABEL, path=os.path.join(ARGS.assets_dir, ARGS.index2label_file))
    TRAIN_DATA.labels = ENCODED_LABELS
    VALID_DATA.labels = LABEL_ENCODER.transform(list(VALID_DATA.labels))
    TEST_DATA.labels = LABEL_ENCODER.transform(list(TEST_DATA.labels))

    TRAIN_SAMPLES = prepare_example(TRAIN_DATA, TRAIN_CHARS, TRAIN_DATA_TOKEN_LENGTH)
    VALID_SAMPLES = prepare_example(VALID_DATA, VALID_CHARS, VALID_DATA_TOKEN_LENGTH)
    TEST_SAMPLES = prepare_example(TEST_DATA, TEST_CHARS, TEST_DATA_TOKEN_LENGTH)

    CLASSIFIER = Classifier(config=ARGS, loss_fct=torch.nn.CrossEntropyLoss(),
                            num_classes=len(LABEL2INDEX),
                            num_characters=len(CHAR_INDEXER.get_vocab2idx()),
                            char_embedding_dim=ARGS.char_embedding_dim,
                            pooling_methods=[ARGS.pooling_method])

    TRAINER = Trainer(model_path=ARGS.lm_path, config=ARGS, classifier=CLASSIFIER,
                      train_data=TRAIN_SAMPLES, dev_data=VALID_SAMPLES, test_data=TEST_SAMPLES)

    TRAINER.fit(check_point_monitor="dev_loss", check_point_mode="min",
                early_stopping_monitor="dev_loss", early_stopping_patience=5)
    TRAINER.test()
