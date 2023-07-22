# ============================ Third Party libs ============================
import os
import logging
from sklearn.metrics import classification_report
# ============================ My packages ============================
from Language_Identification.configurations import BaseConfig
from Language_Identification.data_loader import write_json
from Language_Identification.utils import encode_labels, load_dataset, CLASSIFIER2OBJECT, \
    VECTORIZER2OBJECT

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

    ENCODED_LABELS, LABEL2INDEX, INDEX2LABEL, LABEL_ENCODER = encode_labels(
        labels=list(TRAIN_DATA.labels))
    write_json(LABEL2INDEX, path=os.path.join(ARGS.assets_dir, ARGS.label2index_file))
    write_json(LABEL2INDEX, path=os.path.join(ARGS.assets_dir, ARGS.index2label_file))
    TRAIN_DATA.labels = ENCODED_LABELS
    VALID_DATA.labels = LABEL_ENCODER.transform(list(VALID_DATA.labels))
    TEST_DATA.labels = LABEL_ENCODER.transform(list(TEST_DATA.labels))

    VECTORIZER = VECTORIZER2OBJECT[ARGS.ml_vectorizer]
    TRAIN_TEXT_VECTORIZED = VECTORIZER.fit_transform(TRAIN_DATA.text).toarray()
    VALID_TEXT_VECTORIZED = VECTORIZER.transform(VALID_DATA.text).toarray()
    TEST_TEXT_VECTORIZED = VECTORIZER.transform(TEST_DATA.text).toarray()

    CLASSIFIER = CLASSIFIER2OBJECT[ARGS.ml_classifier]

    CLASSIFIER.fit(TRAIN_TEXT_VECTORIZED, TRAIN_DATA.labels)

    PREDICTED_LABELS = CLASSIFIER.predict(TEST_TEXT_VECTORIZED)

    REPORT = classification_report(list(TEST_DATA["labels"]), PREDICTED_LABELS,
                                   target_names=LABEL2INDEX.keys(), digits=4)
    print(REPORT)
