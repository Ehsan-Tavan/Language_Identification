# ============================ Third Party libs ============================
import os
from sklearn.metrics import classification_report

# ============================ My packages ============================
from Language_Identification.configurations import BaseConfig
from Language_Identification.data_loader import read_csv, read_json
from Language_Identification.utils import prepare_example, calculate_token_length
from Language_Identification.models.lm_model import Classifier
from Language_Identification.data_preparation import TokenIndexer

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    TEST_DATA = read_csv(os.path.join(ARGS.processed_data_dir, ARGS.test_data_file))
    LABEL2INDEX = read_json(os.path.join(ARGS.assets_dir, ARGS.label2index_file))
    CHAR2INDEX = read_json(os.path.join(ARGS.assets_dir, "char2idx.json"))
    TEST_DATA["labels"] = TEST_DATA["labels"].apply(lambda label: LABEL2INDEX[label])
    CHAR_INDEXER = TokenIndexer()
    CHAR_INDEXER.load(vocab2idx_path=os.path.join(ARGS.assets_dir, "char2idx.json"),
                      idx2vocab_path=os.path.join(ARGS.assets_dir, "idx2char.json"))

    TEST_CHARS = [[char for char in str(sample)] for sample in TEST_DATA.text]
    TEST_CHARS = CHAR_INDEXER.convert_samples_to_token_indexes(TEST_CHARS)

    TEST_DATA_TOKEN_LENGTH = calculate_token_length(list(TEST_DATA.text), list(TEST_DATA.labels))
    TEST_SAMPLES = prepare_example(TEST_DATA, TEST_CHARS, TEST_DATA_TOKEN_LENGTH)

    MODEL_PATH = os.path.join(ARGS.saved_model_dir, ARGS.best_model_path)
    MODEL = Classifier.load_from_checkpoint(MODEL_PATH).to(ARGS.device)

    PROBABILITIES, PREDICTED_LABELS = MODEL.predict(TEST_SAMPLES,
                                                    batch_size=ARGS.inference_batch_size,
                                                    max_len=ARGS.max_len)

    REPORT = classification_report(list(TEST_DATA["labels"]), PREDICTED_LABELS.cpu(),
                                   target_names=LABEL2INDEX.keys(), digits=4)
    print(REPORT)
